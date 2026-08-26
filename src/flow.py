"""경기 흐름 분석 — 반이닝별 승률 추이와 '흐름이 바뀐 지점'.

이 파일은 두 가지를 구분한다. 섞으면 안 되는 서로 다른 개념이다.

  반이닝 트랙 (half_inning_track)
      1회초부터 9회말까지 각 반이닝의 시작·종료 승률과 그 사이의 변화량.
      x축이 '이벤트 번호'가 아니라 '회(초/말)'가 되므로 야구적으로 읽힌다.
      한 경기가 18~20행으로 요약되고, 각 행이 실제 야구 단위와 1:1 대응한다.

  흐름 전환점 (turning_points)
      |WPA| 상위 플레이와는 **다른 것**이다.
      8-1 상황의 홈런은 WPA가 커도 흐름을 바꾸지 않는다. 이미 결정된 경기다.
      반대로 5-4에서 나온 평범한 안타 하나가 주도권을 넘길 수 있다.
      그래서 승률을 다섯 구간으로 나누고 **구간이 바뀐 순간**만 잡는다.

          홈 결정적  wp >= 0.85
          홈 우세    0.65 <= wp < 0.85
          접전       0.35 <  wp < 0.65
          원정 우세  0.15 <  wp <= 0.35
          원정 결정적 wp <= 0.15

      승률이 경계에서 떨거나 잠깐 넘었다 돌아오는 것까지 전환으로 잡으면
      의미 없는 잡음이 쏟아진다. 그래서 **새 구간이 연속 N개 이벤트 이상
      유지될 때만** 전환으로 인정한다(히스테리시스).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 승률 구간 경계 (홈 관점). 값이 클수록 홈에 유리.
ZONES = [
    (0.85, 1.01, "홈 결정적"),
    (0.65, 0.85, "홈 우세"),
    (0.35, 0.65, "접전"),
    (0.15, 0.35, "원정 우세"),
    (-0.01, 0.15, "원정 결정적"),
]
ZONE_ORDER = ["원정 결정적", "원정 우세", "접전", "홈 우세", "홈 결정적"]

RESULT_EVENTS = {13, 23, 24, 14}     # 타석 결과 · 아웃 · 득점 · 주자 진루


def zone_of(wp: float) -> str:
    for lo, hi, name in ZONES:
        if lo < wp <= hi:
            return name
    return "접전"


def half_label(inning: int, is_bottom: int) -> str:
    return f"{int(inning)}회{'말' if is_bottom else '초'}"


# ── 반이닝별 승률 트랙 ────────────────────────────────────────────────
def half_inning_track(feats: pd.DataFrame) -> pd.DataFrame:
    """1회초부터 경기 종료까지 반이닝 단위 승률 추이.

    종료 승률은 '다음 반이닝의 시작 승률'로 잡는다. 그래야 변화량의 합이
    경기 전체 승률 변화와 정확히 일치하고, 반이닝 사이에 틈이 생기지 않는다.
    """
    d = feats.sort_values("event_idx").reset_index(drop=True)
    d["half_key"] = d["inning"] * 2 + d["is_bottom"]

    blocks, res_frames = [], []
    for key, blk in d.groupby("half_key", sort=True):
        blk = blk.sort_values("event_idx")
        first = blk.iloc[0]
        bat_is_home = int(first["is_bottom"]) == 1
        score_col = "home_score" if bat_is_home else "away_score"
        runs = int(blk[score_col].iloc[-1]) - int(blk[score_col].iloc[0])

        # 그 반이닝을 규정한 플레이.
        # 단순 max|WPA| 를 쓰면 안 된다 — 공격팀이 2점 낸 이닝인데
        # '이닝을 끝낸 아웃'(반대 방향)이 뽑혀 득점 표시와 어긋난다.
        # 그래서 방향은 아래에서 순변화가 정해진 뒤 _pick_defining 이 고른다.
        # half_over 행은 직전 반이닝의 플레이가 넘어온 것 → 대표 플레이에서 제외
        res_frames.append(blk[blk["event_type"].isin(RESULT_EVENTS)
                              & (blk.get("half_over", 0) == 0)])

        blocks.append({
            "half_key": int(key),
            "inning": int(first["inning"]),
            "is_bottom": int(first["is_bottom"]),
            "half": half_label(first["inning"], first["is_bottom"]),
            "공격": "홈" if bat_is_home else "원정",
            "start_idx": int(first["event_idx"]),
            "end_idx": int(blk["event_idx"].iloc[-1]),
            "wp_start": float(first["wp_home"]),
            "wp_end_internal": float(blk["wp_home"].iloc[-1]),
            "runs": runs,
            "away_score": int(blk["away_score"].iloc[-1]),
            "home_score": int(blk["home_score"].iloc[-1]),
            "n_events": len(blk),
        })

    t = pd.DataFrame(blocks)
    if t.empty:
        return t
    # 종료 승률 = 다음 반이닝의 시작 승률 (마지막은 경기 최종값)
    t["wp_end"] = t["wp_start"].shift(-1)
    t.loc[t.index[-1], "wp_end"] = t["wp_end_internal"].iloc[-1]
    t["delta"] = t["wp_end"] - t["wp_start"]
    t["zone_end"] = t["wp_end"].map(zone_of)

    # DataFrame 을 컬럼에 담아 두면 itertuples 가 이름을 위치 이름으로 바꿔버린다.
    # 별도 리스트로 들고 다니는 편이 안전하다.
    picks = [_pick_defining(rf, dv) for rf, dv in zip(res_frames, t["delta"])]
    t["top_play"] = [p[0] for p in picks]
    t["top_wpa"] = [p[1] for p in picks]
    return t.drop(columns=["wp_end_internal"])


def _pick_defining(res: pd.DataFrame, delta: float) -> tuple[str, float]:
    """반이닝을 규정한 플레이 — 순변화와 같은 방향 중 가장 큰 것."""
    if res is None or not len(res):
        return "", 0.0
    same = res[np.sign(res["wpa"]) == np.sign(delta)] if delta else res
    pick = same if len(same) else res
    top = pick.iloc[pick["wpa"].abs().values.argmax()]
    return str(top["text"]), float(top["wpa"])


# ── 흐름 전환점 ──────────────────────────────────────────────────────
def turning_points(feats: pd.DataFrame, hold: int = 6, margin: float = 0.04,
                   min_delta: float = 0.10) -> pd.DataFrame:
    """승률 구간이 바뀐 순간만 추출한다.

    잡음을 세 겹으로 막는다. 셋 다 실측으로 필요성을 확인했다.

      margin    구간을 나갈 때는 경계를 이만큼 넘어야 한다(히스테리시스).
                이게 없으면 0.35 경계에서 승률이 오르내릴 때마다 전환이 찍힌다.
                실측: 3~5회가 사실상 한 국면인데 전환 4건이 잡혔다.
      hold      새 구간이 이만큼 연속 유지돼야 인정. 순간적 이탈을 거른다.
      min_delta 전환 시점까지의 누적 승률 변화가 이보다 작으면 무시.
    """
    d = feats.sort_values("event_idx").reset_index(drop=True)
    if len(d) < hold + 1:
        return pd.DataFrame()

    wp = d["wp_home"].values
    n = len(wp)

    def zone_with_margin(v: float, cur: str) -> str:
        """현재 구간에 머물러 있으면 경계를 margin 만큼 넓혀서 판정한다."""
        for lo, hi, name in ZONES:
            if name == cur:
                if lo - margin < v <= hi + margin:
                    return cur
                break
        return zone_of(v)

    points = []
    cur_zone = zone_of(wp[0])
    cur_start = 0

    i = 1
    while i < n:
        z = zone_with_margin(wp[i], cur_zone)
        if z == cur_zone:
            i += 1
            continue
        # 새 구간이 hold 만큼 유지되는지 (margin 없이 엄격하게) 확인.
        # 경기 막판에는 남은 이벤트가 hold 보다 적다 — 그때는 남은 전부로 판정한다.
        # 이걸 빠뜨리면 끝내기 역전처럼 **경기 최대 전환점이 통째로 누락된다.**
        window = wp[i:i + hold]
        if len(window) == 0 or not all(zone_of(v) == z for v in window):
            i += 1
            continue

        delta = float(wp[i] - wp[cur_start])
        if abs(delta) < min_delta:
            cur_zone, cur_start = z, i
            i += 1
            continue

        row = d.iloc[i]
        trigger, trig_wpa, t_inn, t_bot = _find_trigger(d, cur_start, i, delta)
        points.append({
            "event_idx": int(row["event_idx"]),
            "inning": t_inn,
            "is_bottom": t_bot,
            "half": half_label(t_inn, t_bot),
            "away_score": int(row["away_score"]),
            "home_score": int(row["home_score"]),
            "from_zone": cur_zone,
            "to_zone": z,
            "kind": _classify(cur_zone, z),
            "wp_before": float(wp[cur_start]),
            "wp_after": float(wp[i]),
            "delta": delta,
            "trigger": trigger,
            "trigger_wpa": trig_wpa,
        })
        cur_zone, cur_start = z, i
        i += 1

    return pd.DataFrame(points)


def _find_trigger(d: pd.DataFrame, lo: int, hi: int, delta: float) -> tuple[str, float, int, int]:
    """전환을 만든 플레이를 찾는다.

    단순히 '직전 이벤트'를 쓰면 안 된다 — 실측에서 -19.7%p 전환의 계기가
    +4.2%p 플레이로 잡혔다. 전환 구간 안에서 **전환과 같은 방향으로**
    가장 크게 움직인 결과성 플레이를 찾아야 한다.
    """
    seg = d.iloc[lo:hi + 1]
    seg = seg[seg["event_type"].isin(RESULT_EVENTS)]
    if seg.empty:
        row = d.iloc[hi]
        return str(row["text"]), float(row["wpa"]), _disp(row)[0], _disp(row)[1]
    same = seg[np.sign(seg["wpa"]) == np.sign(delta)]
    pick = same if len(same) else seg
    top = pick.iloc[pick["wpa"].abs().values.argmax()]
    inn, bot = _disp(top)
    return str(top["text"]), float(top["wpa"]), inn, bot


def _disp(row) -> tuple[int, int]:
    """플레이가 실제로 일어난 반이닝. 정규화로 옮겨진 행은 disp_* 를 봐야 한다."""
    inn = row["disp_inning"] if "disp_inning" in row.index else row["inning"]
    bot = row["disp_is_bottom"] if "disp_is_bottom" in row.index else row["is_bottom"]
    return int(inn), int(bot)


def _classify(frm: str, to: str) -> str:
    """구간 전환에 야구 해설로 쓸 수 있는 이름을 붙인다."""
    i0, i1 = ZONE_ORDER.index(frm), ZONE_ORDER.index(to)
    # 어느 편인가: -1 원정 우위, 0 접전, +1 홈 우위
    side0 = 0 if i0 == 2 else (1 if i0 > 2 else -1)
    side1 = 0 if i1 == 2 else (1 if i1 > 2 else -1)

    if side0 * side1 == -1:
        return "주도권 교체"          # 홈 우위 ↔ 원정 우위 직접 전환
    if to.endswith("결정적"):
        return "승부 기울어짐"
    if frm.endswith("결정적"):
        return "승부 되살아남"
    if to == "접전":
        return "접전 복귀"
    if frm == "접전":
        return "균형 깨짐"
    return "우위 확대" if abs(i1 - 2) > abs(i0 - 2) else "우위 축소"


# ── 콘솔 렌더링 ──────────────────────────────────────────────────────
def render_track(track: pd.DataFrame, home: str, away: str, width: int = 26) -> str:
    """반이닝 트랙을 텍스트 막대로 그린다."""
    lines = [f"  {'회':<6} {'공격':<4} {'스코어':>7}  {'승률(' + home + ')':>10}  승률변화   흐름",
             "  " + "─" * (width + 46)]
    for r in track.itertuples():
        filled = int(round(r.wp_end * width))
        bar = "█" * filled + "·" * (width - filled)
        arrow = "▲" if r.delta > 0.005 else ("▼" if r.delta < -0.005 else " ")
        run = f" (+{r.runs})" if r.runs else ""
        lines.append(
            f"  {r.half:<6} {getattr(r, '공격'):<4} "
            f"{r.away_score:>3}-{r.home_score:<3}  {r.wp_end*100:>8.1f}%  "
            f"{arrow}{abs(r.delta)*100:>5.1f}%p  {bar}{run}")
    return "\n".join(lines)


def render_turning_points(tp: pd.DataFrame, home: str, away: str) -> str:
    if tp.empty:
        return "  (구간을 바꿀 만한 전환점이 없었습니다 — 시종 한쪽으로 흐른 경기)"
    lines = []
    for r in tp.itertuples():
        lines.append(
            f"  [{r.kind}] {r.half}  {r.away_score}-{r.home_score}\n"
            f"      {r.from_zone} → {r.to_zone}   "
            f"{home} 승률 {r.wp_before*100:.1f}% → {r.wp_after*100:.1f}% "
            f"({r.delta*100:+.1f}%p)\n"
            f"      계기: {r.trigger} (WPA {r.trigger_wpa*100:+.1f}%p)")
    return "\n\n".join(lines)
