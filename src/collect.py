"""과거 시즌 문자중계 대량 수집기.

사용 예)
  python -m src.collect --start 2025-03-22 --end 2025-10-01
  python -m src.collect --season 2024
  python -m src.collect --season 2024 --workers 3
결과: data/raw/events_<범위>_shardNNN.parquet  (투구 단위 이벤트)
      data/raw/games_<범위>_shardNNN.parquet   (경기 단위 메타 + 결과 + 사전정보)

이미 수집한 game_id 는 건너뛰므로 중단 후 재실행해도 안전하다.
40경기마다 샤드로 저장하므로 중간에 끊겨도 받은 데이터는 남는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_RAW, MAX_INNINGS                     # noqa: E402
from src import naver_api as api                             # noqa: E402
from src.parser import flatten_relay, game_result, parse_preview  # noqa: E402

CKPT = DATA_RAW / "_done_games.json"
LOCK = DATA_RAW / "_collect.lock"


class _Lock:
    """수집기 동시 실행을 막는다.

    두 프로세스가 같이 돌면 체크포인트를 서로 덮어쓴다. 각자 시작할 때 읽은
    목록에 자기 몫만 더해 저장하므로, 나중에 저장한 쪽이 상대 기록을 지운다.
      실측: 그렇게 157경기가 체크포인트에서 사라졌다. parquet 에는 남아 있어
            그대로 재개하면 같은 경기를 두 번 받는다.
    """

    def __enter__(self):
        import os
        try:
            with open(LOCK, "x") as f:
                f.write(str(os.getpid()))
        except FileExistsError:
            raise SystemExit(
                f"[collect] 이미 다른 수집기가 실행 중입니다 "
                f"(PID {LOCK.read_text().strip()}).\n"
                f"          정말 아무것도 안 돌고 있다면 {LOCK} 를 지우고 다시 실행하세요.")
        return self

    def __exit__(self, *exc):
        LOCK.unlink(missing_ok=True)
        return False


def rebuild_checkpoint() -> int:
    """parquet 에 실제로 들어 있는 경기로 체크포인트를 다시 만든다."""
    import glob
    fs = sorted(DATA_RAW.glob("events_*.parquet"))
    if not fs:
        return 0
    gids = set()
    for f in fs:
        gids |= set(pd.read_parquet(f, columns=["game_id"]).game_id.unique())
    _save_done(gids)
    return len(gids)


def _load_done() -> set:
    return set(json.loads(CKPT.read_text())) if CKPT.exists() else set()


def _save_done(done: set) -> None:
    CKPT.write_text(json.dumps(sorted(done)))


def _flush(ev_frames: list, game_rows: list, start: str, end: str) -> None:
    """중간 결과를 샤드 파일로 저장 (중단되어도 데이터가 남도록)."""
    if not ev_frames:
        return
    tag = f"{start}_{end}"
    n = len(list(DATA_RAW.glob(f"events_{tag}_shard*.parquet")))
    pd.concat(ev_frames, ignore_index=True).to_parquet(
        DATA_RAW / f"events_{tag}_shard{n:03d}.parquet", index=False)
    pd.DataFrame(game_rows).to_parquet(
        DATA_RAW / f"games_{tag}_shard{n:03d}.parquet", index=False)


def _list_games(start: str, end: str) -> list[dict]:
    """일정 조회는 한 번에 약 310경기까지만 응답한다 → 월 단위로 쪼개서 모은다."""
    from datetime import date, timedelta
    y0, m0, d0 = (int(x) for x in start.split("-"))
    y1, m1, d1 = (int(x) for x in end.split("-"))
    cur, last = date(y0, m0, d0), date(y1, m1, d1)
    seen, out = set(), []
    while cur <= last:
        nxt = (date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1) - timedelta(days=1))
        chunk_end = min(nxt, last)
        for g in api.fetch_schedule(cur.isoformat(), chunk_end.isoformat()):
            if g["gameId"] not in seen:
                seen.add(g["gameId"])
                out.append(g)
        cur = chunk_end + timedelta(days=1)
    return out


def _final_inning(g: dict) -> int | None:
    """일정 API의 statusInfo('9회말', '7회초'…)에서 실제 종료 이닝을 읽는다."""
    import re
    m = re.match(r"\s*(\d+)\s*회", str(g.get("statusInfo") or ""))
    return int(m.group(1)) if m else None


def _fetch_one(g: dict, with_preview: bool) -> tuple[pd.DataFrame, dict] | None:
    """한 경기의 중계 + 메타를 가져온다. 워커 스레드에서 호출된다."""
    gid = g["gameId"]
    innings = api.fetch_relay_full(gid, MAX_INNINGS)
    if not innings:
        return None
    df = flatten_relay(innings, gid)
    if df.empty:
        return None
    # 이닝 번호에 구멍이 있으면 전송이 중간에 끊긴 것이다.
    # TransientError 로 대부분 걸리지만, 이중으로 막는다.
    # 실측: DNS 블립 한 번에 6회가 통째로 빠진 경기가 정상인 척 저장됐다.
    inns = sorted(int(i) for i in df["inning"].unique())
    if inns and set(range(1, max(inns) + 1)) - set(inns):
        raise api.TransientError(f"{gid}: 이닝 누락 {sorted(set(range(1, max(inns)+1)) - set(inns))}")
    # 마지막 이닝은 '9회'로 고정하면 안 된다. 강우 콜드게임이 실재한다.
    #   실측: 20260526LGLT02026 은 statusCode=RESULT, statusInfo='7회말' 인
    #         정상 종료 경기다. 9회 기준으로 검사하면 영영 수집하지 못한다.
    # 일정 API가 알려주는 실제 종료 이닝과 비교한다.
    expect = _final_inning(g)
    if inns and expect and max(inns) < expect:
        raise api.TransientError(f"{gid}: {max(inns)}회까지만 수집됨 (실제 {expect}회 종료)")
    meta = {
        "game_id": gid,
        "game_date": g["gameDate"],
        "season": int(gid[-4:]),
        "stadium": g.get("stadium"),
        "home_team": g.get("homeTeamName"),
        "away_team": g.get("awayTeamName"),
        "home_code": g.get("homeTeamCode"),
        "away_code": g.get("awayTeamCode"),
        **game_result(innings, g),
    }
    if with_preview:
        # preview 는 보조 정보다. 여기서 실패해도 경기 전체를 버릴 이유는 없다.
        # (중계 실패는 TransientError 로 올라가 경기가 재시도된다)
        try:
            meta.update({k: v for k, v in parse_preview(api.fetch_preview(gid)).items()
                         if k not in ("home_team", "away_team", "stadium")})
        except api.TransientError:
            pass
    return df, meta


def collect_range(start: str, end: str, with_preview: bool = True,
                  limit: int | None = None, shard_every: int = 40,
                  workers: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """기간 내 종료된 경기를 모두 수집한다.

    workers > 1 이면 경기 단위로 병렬 처리한다. 전역 레이트리밋이 총 요청 수를
    묶어두므로 서버 부담은 늘지 않고, 요청 왕복 지연만 겹쳐서 처리량이 올라간다.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    games = _list_games(start, end)
    games = [g for g in games if g.get("statusCode") == "RESULT" and not g.get("cancel")]
    if limit:
        games = games[:limit]
    done = _load_done()
    todo = [g for g in games if g["gameId"] not in done]
    print(f"[collect] {start}~{end}: 종료 경기 {len(games)}건 "
          f"/ 이미 수집 {len(games) - len(todo)}건 / 남은 {len(todo)}건 (워커 {workers})")

    ev_frames, game_rows = [], []
    n_ok = n_fail = 0

    def flush_if_needed() -> None:
        nonlocal ev_frames, game_rows
        if len(game_rows) >= shard_every:
            _flush(ev_frames, game_rows, start, end)
            _save_done(done)
            print(f"[collect]  샤드 저장 · 누적 {len(done)}경기 "
                  f"(성공 {n_ok} / 실패 {n_fail})")
            ev_frames, game_rows = [], []

    if workers <= 1:
        for i, g in enumerate(todo, 1):
            r = _fetch_one(g, with_preview)
            if r is None:
                n_fail += 1
                continue
            df, meta = r
            ev_frames.append(df)
            game_rows.append(meta)
            done.add(g["gameId"])
            n_ok += 1
            flush_if_needed()
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_fetch_one, g, with_preview): g for g in todo}
            for fut in as_completed(futs):
                g = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:                       # noqa: BLE001
                    print(f"[collect]  {g['gameId']} 오류: {e}")
                    r = None
                if r is None:
                    n_fail += 1
                    continue
                df, meta = r
                ev_frames.append(df)
                game_rows.append(meta)
                done.add(g["gameId"])
                n_ok += 1
                flush_if_needed()

    if game_rows:
        _flush(ev_frames, game_rows, start, end)
    _save_done(done)
    print(f"[collect] 완료 · 성공 {n_ok} / 실패 {n_fail}")

    tag = f"{start}_{end}"
    ev_files = sorted(DATA_RAW.glob(f"events_{tag}_shard*.parquet"))
    gm_files = sorted(DATA_RAW.glob(f"games_{tag}_shard*.parquet"))
    events = pd.concat([pd.read_parquet(f) for f in ev_files], ignore_index=True) if ev_files else pd.DataFrame()
    gmeta = pd.concat([pd.read_parquet(f) for f in gm_files], ignore_index=True) if gm_files else pd.DataFrame()
    return events, gmeta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--season", type=int, help="지정 시 해당 시즌 전체(3/1~11/30)")
    ap.add_argument("--limit", type=int, default=None, help="테스트용 경기 수 제한")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--rebuild-checkpoint", action="store_true",
                    help="parquet 기준으로 체크포인트를 다시 만들고 종료")
    ap.add_argument("--workers", type=int, default=3,
                    help="병렬 워커 수. 전역 레이트리밋이 총 요청량을 고정하므로 서버 부담은 동일")
    a = ap.parse_args()

    if a.rebuild_checkpoint:
        print(f"[collect] 체크포인트 재구성 완료: {rebuild_checkpoint()}경기")
        return

    if a.season:
        a.start, a.end = f"{a.season}-03-01", f"{a.season}-11-30"
    if not (a.start and a.end):
        ap.error("--start/--end 또는 --season 이 필요합니다")

    with _Lock():
        ev, gm = collect_range(a.start, a.end, not a.no_preview, a.limit, workers=a.workers)
    if ev.empty:
        print("[collect] 새로 수집된 데이터 없음")
        return
    # 샤드 파일이 원본이다. 합본을 따로 쓰지 않는다 (학습 시 중복 로드 방지).
    print(f"[collect] 누적 이벤트 {len(ev):,}행 / 경기 {len(gm):,}건 "
          f"(샤드 {len(list(DATA_RAW.glob(f'events_{a.start}_{a.end}_shard*.parquet')))}개)")


if __name__ == "__main__":
    main()
