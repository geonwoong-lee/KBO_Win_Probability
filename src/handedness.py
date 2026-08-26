"""선수 좌우(투타) 조회표 만들기.

  python -m src.handedness --games 60

좌우는 **선수의 고정 속성**이다. 경기마다 다시 받을 이유가 없다.
그래서 표본 경기 몇 개의 라인업에서 pcode → (던지는 손, 치는 손) 을 뽑아
models/player_hand.csv 로 저장하고, 이미 수집해 둔 데이터에 붙여 쓴다.
재수집(50분) 없이 5분이면 끝난다.

hitType 형식: '우투좌타' = 오른손 투구 / 왼손 타격. 양손타자는 '양타'.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODEL_DIR, MAX_INNINGS                       # noqa: E402
from src import naver_api as api                                # noqa: E402
from src.collect import _list_games                             # noqa: E402

OUT = MODEL_DIR / "player_hand.csv"


def parse_hit_type(s: str | None) -> tuple[str | None, str | None]:
    """'우투좌타' → ('R', 'L').  양손타자는 타격이 'S'."""
    if not s or len(s) < 4:
        return None, None
    throws = {"우": "R", "좌": "L", "양": "S"}.get(s[0])
    bats = {"우": "R", "좌": "L", "양": "S"}.get(s[2])
    return throws, bats


def collect_map(n_games: int = 60, season: int = 2026) -> pd.DataFrame:
    games = [g for g in _list_games(f"{season}-03-01", f"{season}-11-30")
             if g.get("statusCode") == "RESULT" and not g.get("cancel")]
    # 시즌 전체에 고르게 퍼뜨려 뽑는다 (엔트리가 시즌 중 바뀌므로)
    step = max(1, len(games) // n_games)
    picked = games[::step][:n_games]
    print(f"[hand] {season} 시즌 {len(games)}경기 중 {len(picked)}경기 표본으로 조회표 생성")

    rows: dict[str, dict] = {}
    for i, g in enumerate(picked, 1):
        try:
            innings = api.fetch_relay_full(g["gameId"], MAX_INNINGS)
        except Exception as e:                                   # noqa: BLE001
            print(f"  [{i}/{len(picked)}] {g['gameId']} 실패: {e}")
            continue
        for data in innings:
            for grp in ("homeLineup", "awayLineup"):
                blk = data.get(grp) or {}
                for role in ("pitcher", "batter"):
                    for x in (blk.get(role) or []):
                        code = str(x.get("pcode", ""))
                        if not code or code in rows:
                            continue
                        th, ba = parse_hit_type(x.get("hitType"))
                        if th or ba:
                            rows[code] = {"pcode": code, "name": x.get("name"),
                                          "throws": th, "bats": ba}
        if i % 15 == 0:
            print(f"  [{i}/{len(picked)}] 선수 {len(rows)}명 수집")

    df = pd.DataFrame(rows.values())
    return df


def load_map() -> pd.DataFrame | None:
    return pd.read_csv(OUT, dtype={"pcode": str}) if OUT.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--season", type=int, default=2026)
    a = ap.parse_args()

    df = collect_map(a.games, a.season)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n[hand] 선수 {len(df)}명 → {OUT}")
    print(df.groupby(["throws"]).size().rename("투수 손").to_string())
    print(df.groupby(["bats"]).size().rename("타격 손").to_string())


if __name__ == "__main__":
    main()
