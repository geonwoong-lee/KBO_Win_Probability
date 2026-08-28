"""실시간 승률 엔진 — 경기 시작부터 종료까지 폴링하며 승률과 근거를 갱신한다.

  python -m src.live --auto                  # 오늘 진행 중인 경기 자동 선택
  python -m src.live --game 20260825LTHT02026
  python -m src.live --replay 20260823HTWO02026   # 끝난 경기를 처음부터 재생

폴링 비용을 줄이는 방법
  * 지나간 이닝의 중계는 변하지 않으므로 한 번만 받아 캐시한다.
  * 매 폴링에서는 '현재 이닝'만 다시 받는다 → 요청 1~2회/주기.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import REPORT_DIR, MAX_INNINGS                     # noqa: E402
from src import naver_api as api                               # noqa: E402
from src.parser import flatten_relay, parse_preview            # noqa: E402
from src.features import build_features                        # noqa: E402
from src.explain import WinProbabilityExplainer                # noqa: E402


class LiveGame:
    """한 경기의 중계를 증분 수집하고 승률 시계열을 유지한다."""

    def __init__(self, game_id: str, explainer: WinProbabilityExplainer | None = None):
        self.game_id = game_id
        self.exp = explainer or WinProbabilityExplainer()
        self.inning_cache: dict[int, dict] = {}
        self.finished_innings: set[int] = set()
        self.meta = self._load_meta()
        self.history: list[dict] = []

    # ── 메타 (경기 전 사전정보) ───────────────────────────────────────
    def _load_meta(self) -> dict:
        sched = api.fetch_game_status(self.game_id) or {}
        pv = parse_preview(api.fetch_preview(self.game_id))
        return {
            "game_id": self.game_id,
            "season": int(self.game_id[-4:]),
            "game_date": sched.get("gameDate", self.game_id[:8]),
            "home_team": sched.get("homeTeamName") or pv.get("home_team") or "홈",
            "away_team": sched.get("awayTeamName") or pv.get("away_team") or "원정",
            "stadium": sched.get("stadium"),
            "status": sched.get("statusCode"),
            "status_info": sched.get("statusInfo"),
            **{k: v for k, v in pv.items() if k not in ("home_team", "away_team", "stadium")},
            # 라벨 자리 (실시간에는 미정 → build_features 통과용 더미)
            "home_win": 1, "home_score_final": 0, "away_score_final": 0,
        }

    # ── 증분 수집 ────────────────────────────────────────────────────
    def refresh(self) -> pd.DataFrame:
        """캐시를 갱신하고 전체 이벤트 DataFrame 을 반환."""
        inn = 1
        while inn <= MAX_INNINGS:
            if inn not in self.finished_innings:
                data = api.fetch_relay_inning(self.game_id, inn)
                if not data or not data.get("textRelays"):
                    break
                self.inning_cache[inn] = data
            # 다음 이닝에 데이터가 있으면 이 이닝은 끝난 것 → 다시 받지 않는다
            has_next = (inn + 1) in self.inning_cache
            if not has_next:
                nxt = api.fetch_relay_inning(self.game_id, inn + 1)
                if nxt and nxt.get("textRelays"):
                    self.inning_cache[inn + 1] = nxt
                    has_next = True
            if not has_next:
                break                      # 여기가 현재 진행 중인 이닝
            self.finished_innings.add(inn)
            inn += 1
        innings = [self.inning_cache[i] for i in sorted(self.inning_cache)]
        return flatten_relay(innings, self.game_id)

    # ── 현재 승률 + 설명 ─────────────────────────────────────────────
    def snapshot(self) -> dict | None:
        ev = self.refresh()
        if ev.empty:
            return None
        games = pd.DataFrame([self.meta])
        feats = build_features(ev, games, None)
        if feats.empty:
            return None
        feats = self.exp.add_win_probability(feats)
        cur = feats.iloc[-1]

        info = self.exp.explain(cur, self.meta["home_team"], self.meta["away_team"])
        info.update({
            "game_id": self.game_id,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "inning": int(cur["inning"]),
            "is_bottom": int(cur["is_bottom"]),
            "home_score": int(cur["home_score"]),
            "away_score": int(cur["away_score"]),
            "last_text": str(cur["text"]),
            "wpa_last": float(cur["wpa"]),
            "curve": feats[["event_idx", "inning", "is_bottom", "wp_home",
                            "wpa", "text", "event_type"]].to_dict("records"),
            "key_plays": self.exp.key_plays(feats).to_dict("records"),
        })
        self.history.append({"ts": info["ts"], "wp_home": info["wp_home"],
                             "inning": info["inning"], "text": info["last_text"]})
        return info


# ── 콘솔 출력 ────────────────────────────────────────────────────────
def render(info: dict) -> str:
    h, a = info["home_team"], info["away_team"]
    bar_n = int(round(info["wp_home"] * 30))
    bar = "█" * bar_n + "░" * (30 - bar_n)
    half = "말" if info["is_bottom"] else "초"
    lines = [
        "─" * 66,
        f"  {a} {info['away_score']} : {info['home_score']} {h}"
        f"   |  {info['inning']}회{half}   |  {info['ts'][11:]}",
        f"  최근: {info['last_text']}"
        + (f"   (직전 상황으로 승률 {info['wpa_last']*100:+.1f}%p)"
           if abs(info["wpa_last"]) > 0.001 else ""),
        "",
        f"  {a} {info['wp_away']*100:5.1f}%  {bar}  {info['wp_home']*100:5.1f}% {h}",
        "",
        "  [지금 승률을 만든 요인]",
    ]
    for s in info["shap"][:4]:
        arrow = "▲" if s["delta_pp"] > 0 else "▼"
        lines.append(f"    {arrow} {s['label']:<14} {s['delta_pp']:+6.1f}%p   (값 {s['value']:g})")
    lines.append("")
    lines.append(f"  [만약에]  (모두 {h} 기준 승률)")
    for c in info["counterfactuals"][:3]:
        lines.append(f"    · {c['label']:<18} → {c['wp']*100:5.1f}%  ({c['delta_pp']:+.1f}%p)")
    lev = info["leverage"]
    lines.append("")
    lines.append(f"  이 타석의 중요도: 평균 타석의 {lev['leverage_index']}배 "
                 f"(결과에 따라 승률 약 {lev['swing_pp']:.1f}%p 움직일 국면)")
    lines.append("─" * 66)
    return "\n".join(lines)


def pick_live_game() -> str | None:
    """오늘 진행 중인 경기를 고른다. 없으면 다음 경기 일정을 보여준다."""
    from datetime import timedelta
    today = datetime.now()
    games = api.fetch_schedule(today.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
    started = [g for g in games if g.get("statusCode") in ("STARTED", "LIVE")]
    if started:
        return started[0]["gameId"]

    print(f"[live] {today:%Y-%m-%d} 진행 중인 경기가 없습니다.")
    if not games:
        # 오늘 편성이 아예 없으면(월요일 등) 이후 5일 안의 다음 경기를 찾아 보여준다
        for d in range(1, 6):
            nxt = today + timedelta(days=d)
            games = api.fetch_schedule(nxt.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d"))
            if games:
                print(f"[live] 다음 경기 일정 ({nxt:%Y-%m-%d})")
                break
    for g in games:
        print(f"   {g['gameId']}  {g['gameDateTime'][11:16]}  "
              f"{g['awayTeamName']} @ {g['homeTeamName']}  [{g.get('statusInfo')}]")
    if games:
        print()
        print("   특정 경기를 보려면:  python -m src.live --game <gameId>")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--replay", help="종료된 경기를 처음부터 재생")
    ap.add_argument("--interval", type=int, default=30, help="폴링 주기(초)")
    ap.add_argument("--speed", type=float, default=0.12, help="리플레이 속도(초/이벤트)")
    a = ap.parse_args()

    exp = WinProbabilityExplainer()

    # ── 리플레이 모드: 끝난 경기를 처음부터 되감아 보여준다 ──
    if a.replay:
        lg = LiveGame(a.replay, exp)
        ev = flatten_relay(api.fetch_relay_full(a.replay, MAX_INNINGS), a.replay)
        feats = build_features(ev, pd.DataFrame([lg.meta]), None)
        feats = exp.add_win_probability(feats)
        out = REPORT_DIR / f"wp_{a.replay}.csv"
        cols = [c for c in ("inning", "is_bottom", "outs", "base_state", "away_score",
                            "home_score", "text", "wp_home", "wpa", "naver_wp_home")
                if c in feats.columns]
        feats[cols].to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[replay] 승률 시계열 저장 → {out}")

        from src.viz import win_probability_chart
        png = win_probability_chart(feats, lg.meta["home_team"], lg.meta["away_team"],
                                    REPORT_DIR / f"wp_{a.replay}.png")
        print(f"[replay] 승률 곡선 저장 → {png}")

        from src.flow import (half_inning_track, turning_points,
                              render_track, render_turning_points)
        h, aw = lg.meta["home_team"], lg.meta["away_team"]
        track = half_inning_track(feats)
        track.to_csv(REPORT_DIR / f"track_{a.replay}.csv", index=False, encoding="utf-8-sig")

        print(f"\n[회별 승률 트랙]  {aw} @ {h}\n")
        print(render_track(track, h, aw))

        print("\n\n[흐름이 바뀐 지점]\n")
        print(render_turning_points(turning_points(feats), h, aw))

        print("\n\n[승부를 가른 플레이 TOP 5]  — 전환점과는 다른 개념(큰 플레이 ≠ 흐름 전환)\n")
        for r in exp.key_plays(feats, top=5).itertuples():
            half = "말" if r.is_bottom else "초"
            print(f"  {r.inning}회{half} {r.away_score}-{r.home_score}  "
                  f"승률 {r.wp_home*100:5.1f}%  (이 타석으로 {r.wpa*100:+6.1f}%p)  |  {r.text}")
        print()
        print(render({**exp.explain(feats.iloc[-1], lg.meta["home_team"], lg.meta["away_team"]),
                      "ts": datetime.now().isoformat(timespec="seconds"),
                      "inning": int(feats.iloc[-1]["inning"]),
                      "is_bottom": int(feats.iloc[-1]["is_bottom"]),
                      "home_score": int(feats.iloc[-1]["home_score"]),
                      "away_score": int(feats.iloc[-1]["away_score"]),
                      "last_text": str(feats.iloc[-1]["text"]),
                      "wpa_last": float(feats.iloc[-1]["wpa"])}))
        return

    gid = a.game or (pick_live_game() if a.auto else None)
    if not gid:
        return

    lg = LiveGame(gid, exp)
    print(f"[live] {lg.meta['away_team']} @ {lg.meta['home_team']} "
          f"({lg.meta.get('stadium')})  폴링 {a.interval}초")
    last_text = None
    while True:
        try:
            info = lg.snapshot()
            if info and info["last_text"] != last_text:
                print(render(info))
                last_text = info["last_text"]
            st = api.fetch_game_status(gid) or {}
            if st.get("statusCode") == "RESULT":
                print("[live] 경기 종료")
                if lg.history:
                    pd.DataFrame(lg.history).to_csv(
                        REPORT_DIR / f"live_{gid}.csv", index=False, encoding="utf-8-sig")
                break
        except KeyboardInterrupt:
            print("\n[live] 중단")
            break
        except Exception as e:                     # noqa: BLE001
            print(f"[live] 오류: {e}")
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
