"""이벤트 스트림 → 승률 모델 학습용 특성 행렬.

설계 원칙
  1) 누수(leakage) 금지 — 그 시점 이후에만 알 수 있는 정보는 절대 넣지 않는다.
     안타/볼넷 누적처럼 '지금까지'의 값만 사용하고, 최종 스코어는 라벨로만 쓴다.
  2) 상태의 정규화 — 3아웃 행은 '다음 반이닝 시작(무사 주자없음)'으로 옮긴다.
     그래야 같은 국면이 항상 같은 특성 벡터가 된다.
  3) 야구의 구조를 특성으로 — 남은 아웃카운트, 기대득점(RE), 마지막 공격 여부처럼
     '규칙에서 유도되는 값'을 명시적으로 넣어야 트리 모델이 적은 데이터로도 잘 배운다.
  4) 라벨은 홈팀 승리(1) / 패배(0). 무승부 경기는 제외.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 상태 변화를 만드는 이벤트만 사용
STATE_EVENTS = {1, 8, 13, 14, 23, 24}

BASE_STATE_NAME = {
    0: "주자 없음", 1: "1루", 2: "2루", 3: "1·2루",
    4: "3루", 5: "1·3루", 6: "2·3루", 7: "만루",
}

# ── 특성 목록 ────────────────────────────────────────────────────────
# 유효 표본은 '행 수'가 아니라 '경기 수'다. 한 경기의 400여 행은 모두 같은 라벨을
# 공유하므로, 경기 내에서 값이 변하지 않는 특성(선발 ERA 차, 팀 승률 차)은
# 사실상 '경기 식별자'로 작동한다. 경기 수가 적을 때 이걸 넣으면 모델이
# 야구 국면 대신 "어느 팀 경기인지"를 외워버린다.
#   → 실측: 183경기로 학습했을 때 두 사전정보 특성이 gain의 38.6%를 차지하고
#     이닝·아웃·주자 특성은 사실상 무시됐다. 점수차 로지스틱보다 성능이 나빴다.
# 그래서 사전정보는 기본적으로 끄고, 경기 수가 충분할 때만 켠다.
USE_PREGAME_PRIOR = False
MIN_GAMES_FOR_PRIOR = 1200          # 이 정도는 있어야 사전정보가 신호로 작동한다
                                    # (791경기에서 재검증했으나 여전히 성능을 떨어뜨렸다)

# 리그 평균 (부호 있는 선수 특성의 중심값). 대략적인 KBO 수준이면 충분하다.
LEAGUE_ERA, LEAGUE_HRA = 4.50, 0.270

# 아래 목록은 추측이 아니라 **절제 실험(reports/ablation.md)으로 확정한 것**이다.
#
# 1,103경기 4-fold GroupKFold OOF log loss:
#   점수+시간(6개)                      0.4577
#   +주자·카운트(14개)  ★ 채택          0.4479   ← 주자·카운트는 실제 신호가 있다
#   +선수특성 전체(19개)                 0.4486
#   +투수ERA/타자타율만(16개)            0.4479   ← 있으나 없으나 완전히 동일
#   +투구수/불펜만(16개)                 0.4487
#   +박스스코어(22개)                    0.4509   ← 해롭다
#   +사전정보(21개)                      0.4577   ← 해롭다
#
# 선수 특성을 왜 전부 뺐나 — 이게 이 프로젝트에서 가장 중요한 발견이다.
#   처음엔 선수 특성이 가장 값진 줄 알았다. 빼면 +0.0116 으로 손실이 제일 컸다.
#   그런데 그 값은 파서 버그의 산물이었다. currentPlayersInfo 가 이벤트마다
#   바뀌지 않는 필드라, 투수 ERA·타자 타율이 **경기 내 상수**로 들어가 있었다.
#   즉 모델은 야구를 배운 게 아니라 '경기 식별자'를 외우고 있었다(§4 ① 의 재발).
#   올바른 출처로 고쳐 다시 재보니 **기여가 정확히 0** 이었다.
#   이건 야구적으로도 말이 된다 — 남은 이닝 동안 양 팀 타선이 모두 돌기 때문에
#   경기 중 승률은 개별 선수보다 '점수와 남은 아웃'이 압도적으로 지배한다.
#
# 빠진 것들의 근거
#   pitch_adv_home / bat_adv_home  : 제대로 고친 뒤에도 기여 0
#   fatigue_adv_home / bullpen_adv : 기여 0 (오히려 미세하게 나빠짐)
#   bat_order                      : 기여 0
#   hit_diff / bb_diff / err_diff  : 점수와 중복이면서 잡음만 더한다
#   inning / half_index            : outs_left_bat/field 와 사실상 같은 정보
STATE_FEATURES = [
    # 점수 x 시간 — 승률의 대부분을 설명하는 축
    "score_diff", "score_x_progress", "score_per_out_left",
    "outs_left_bat", "outs_left_field", "is_bottom",
    # 주자 · 카운트 · 결정적 국면
    "base_state", "run_exp", "run_potential", "outs", "balls", "strikes",
    "is_walkoff_chance", "is_last_at_bat",
]
PREGAME_FEATURES = ["prior_home_wra_diff", "prior_starter_era_diff"]

FEATURES = STATE_FEATURES + (PREGAME_FEATURES if USE_PREGAME_PRIOR else [])

CATEGORICAL = ["base_state"]


# ── 기대득점(RE): 베이스-아웃 상태별 잔여 기대득점 ──────────────────────
def build_run_expectancy(events: pd.DataFrame) -> pd.DataFrame:
    """데이터에서 직접 24개 베이스-아웃 상태의 잔여 기대득점을 추정."""
    df = events.copy()
    df["half_id"] = df["game_id"] + "_" + df["inning"].astype(str) + "_" + df["is_bottom"].astype(str)
    df["bat_score"] = np.where(df["is_bottom"] == 1, df["home_score"], df["away_score"])
    # 반이닝 종료 시점 득점 - 현재 득점 = 이 상태 이후 추가 득점
    end = df.groupby("half_id")["bat_score"].transform("max")
    df["runs_rest"] = end - df["bat_score"]
    valid = df[(df["outs"] < 3) & df["event_type"].isin(STATE_EVENTS)]
    re = (valid.groupby(["base_state", "outs"])["runs_rest"]
          .agg(["mean", "count"]).reset_index()
          .rename(columns={"mean": "run_exp"}))
    return re


DEFAULT_RUN_EXP = {  # 표본이 없을 때 쓰는 KBO 근사치 (2019-2024 리그 평균 수준)
    (0, 0): 0.49, (0, 1): 0.26, (0, 2): 0.10,
    (1, 0): 0.87, (1, 1): 0.52, (1, 2): 0.22,
    (2, 0): 1.11, (2, 1): 0.68, (2, 2): 0.32,
    (3, 0): 1.46, (3, 1): 0.92, (3, 2): 0.43,
    (4, 0): 1.36, (4, 1): 0.95, (4, 2): 0.36,
    (5, 0): 1.77, (5, 1): 1.16, (5, 2): 0.48,
    (6, 0): 1.95, (6, 1): 1.37, (6, 2): 0.57,
    (7, 0): 2.28, (7, 1): 1.55, (7, 2): 0.72,
}


def normalize_states(df: pd.DataFrame) -> pd.DataFrame:
    """3아웃 행을 다음 반이닝 시작 상태로 옮기고, 중복 상태를 제거."""
    d = df[df["event_type"].isin(STATE_EVENTS)].copy()

    # 3아웃 행은 '상태'로는 다음 반이닝 시작이지만 'text'는 직전 반이닝의 플레이다.
    # 정규화 전 소속을 남겨 두지 않으면, 예컨대 2회초를 끝낸 아웃이
    # 2회말(상대팀 공격)의 대표 플레이로 표시된다. 실측으로 확인한 버그다.
    d["disp_inning"] = d["inning"]
    d["disp_is_bottom"] = d["is_bottom"]

    rolled = d["outs"] >= 3
    d.loc[rolled, ["outs", "on_1b", "on_2b", "on_3b", "balls", "strikes"]] = 0
    # 말(bottom)에서 3아웃 → 다음 이닝 초, 초(top)에서 3아웃 → 같은 이닝 말
    d.loc[rolled & (d["is_bottom"] == 1), "inning"] += 1
    d.loc[rolled, "is_bottom"] = np.where(d.loc[rolled, "is_bottom"] == 1, 0, 1)

    d["base_state"] = d["on_1b"] * 1 + d["on_2b"] * 2 + d["on_3b"] * 4
    key = ["game_id", "inning", "is_bottom", "outs", "base_state",
           "balls", "strikes", "home_score", "away_score"]
    dup = d[key].eq(d[key].shift()).all(axis=1)
    return d[~dup].reset_index(drop=True)


# ── 좌우 매치업 ──────────────────────────────────────────────────────
# 좌우는 선수 고정 속성이라 경기마다 받을 필요가 없다.
# models/player_hand.csv (src.handedness 로 생성) 를 pcode 로 조회해 붙인다.
_HAND: dict | None = None


def _hand_map() -> dict:
    global _HAND
    if _HAND is None:
        from config import MODEL_DIR
        f = MODEL_DIR / "player_hand.csv"
        if f.exists():
            t = pd.read_csv(f, dtype={"pcode": str})
            _HAND = {r.pcode: (r.throws, r.bats) for r in t.itertuples()}
        else:
            _HAND = {}
    return _HAND


def platoon_advantage(bats: str | None, throws: str | None) -> float:
    """타자 관점의 좌우 우위. +1 유리, -1 불리, 0 알 수 없음.

    좌타자 vs 우투수, 우타자 vs 좌투수면 타자가 유리하다(플래툰 어드밴티지).
    양손타자는 항상 반대편에서 치므로 늘 유리하다.
    """
    if not bats or not throws or throws == "S":
        return 0.0
    if bats == "S":
        return 1.0
    return 1.0 if bats != throws else -1.0


SEASON_START_DOY = 70        # 3월 11일 전후가 KBO 개막
TRUST_RAMP_DAYS = 45         # 개막 후 이만큼 지나면 시즌 성적을 그대로 믿는다


def _season_trust(d: pd.DataFrame) -> np.ndarray:
    """시즌 성적을 얼마나 믿을지 (0~1). game_id 앞 8자리가 경기 날짜다."""
    if "game_id" not in d.columns:
        return np.ones(len(d))
    try:
        doy = pd.to_datetime(d["game_id"].astype(str).str[:8],
                             format="%Y%m%d", errors="coerce").dt.dayofyear
        days_in = (doy - SEASON_START_DOY).clip(lower=0).fillna(TRUST_RAMP_DAYS)
        return (days_in / TRUST_RAMP_DAYS).clip(0, 1).to_numpy()
    except Exception:                                  # noqa: BLE001
        return np.ones(len(d))


def add_derived(d: pd.DataFrame, re_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """기본 상태(이닝·초말·아웃·주자·점수차)에서 모든 파생 특성을 계산한다.

    **파생 특성의 정의는 여기 한 곳에만 있다.** 반사실·레버리지에서 상태를 바꾼 뒤에도
    반드시 이 함수를 다시 호출해야 한다. 그러지 않으면 score_per_out_left 처럼
    gain 비중이 큰 특성이 낡은 값으로 남아 "만약 1점 더 앞섰다면" 예측이 틀린다.
    """
    d = d.copy()
    d["score_diff"] = d["home_score"] - d["away_score"]
    d["half_index"] = (d["inning"] - 1) * 2 + d["is_bottom"]

    # ── 남은 아웃카운트: 야구 승률의 가장 강력한 구조적 신호 ──
    # 3점 차는 1회에는 별것 아니지만 9회에는 사실상 끝이다. 이 차이는 '이닝'이 아니라
    # '각 팀에게 남은 공격 아웃 수'로 표현된다. 9회 기준 각 팀 27아웃.
    #   초(top)  진행 중  : 원정은 3*(i-1)+outs 소모, 홈은 3*(i-1) 소모
    #   말(bottom) 진행 중: 원정은 3*i 소모,        홈은 3*(i-1)+outs 소모
    inn, outs, bot = d["inning"], d["outs"], d["is_bottom"]
    away_used = np.where(bot == 1, inn * 3, (inn - 1) * 3 + outs)
    home_used = np.where(bot == 1, (inn - 1) * 3 + outs, (inn - 1) * 3)
    outs_left_home = np.clip(27 - home_used, 0, None)
    outs_left_away = np.clip(27 - away_used, 0, None)
    d["outs_left_bat"] = np.where(bot == 1, outs_left_home, outs_left_away)
    d["outs_left_field"] = np.where(bot == 1, outs_left_away, outs_left_home)

    # ── 기대득점(RE) ──
    if re_table is not None and len(re_table):
        m = {(int(r.base_state), int(r.outs)): float(r.run_exp) for r in re_table.itertuples()}
    else:
        m = DEFAULT_RUN_EXP
    d["run_exp"] = [m.get((int(b), min(2, int(o))), DEFAULT_RUN_EXP.get((int(b), min(2, int(o))), 0.5))
                 for b, o in zip(d["base_state"], d["outs"])]
    innings_left = np.maximum(0, 9 - d["inning"]) + (1 - d["is_bottom"]) * 0.5
    d["run_potential"] = d["run_exp"] + innings_left * 0.55

    # ── 점수차 x 경기 진행도 ──
    # 같은 3점차도 1회에는 거의 무의미하고 9회에는 사실상 승부가 끝난 것이다.
    progress = 1.0 - (d["outs_left_bat"] + d["outs_left_field"]) / 54.0
    d["score_x_progress"] = d["score_diff"] * progress
    d["score_per_out_left"] = d["score_diff"] / (1.0 + d["outs_left_bat"] + d["outs_left_field"]) * 27.0

    # ── 결정적 국면 플래그 ──
    d["is_last_at_bat"] = ((d["inning"] >= 9) & (d["is_bottom"] == 1)).astype(int)
    d["is_walkoff_chance"] = (
        (d["inning"] >= 9) & (d["is_bottom"] == 1) &
        (d["score_diff"] >= -1) & (d["base_state"] > 0)
    ).astype(int)

    # ── 선수 특성의 부호 고정 ──
    # pitcher_season_era 는 '지금 던지는 투수'의 ERA다. 초에는 홈 투수, 말에는 원정 투수.
    # 즉 같은 컬럼이 반이닝마다 라벨에 정반대로 작용한다. 이대로 넣으면 모델이
    # is_bottom 으로 부호를 추론해야 하고, 경기 수가 적을 때 그 추론이 편향된다.
    #   실측: 동점·무사·주자없음에서 모델의 초→말 승률 상승폭이 +9%p 였으나
    #         같은 국면의 실제 홈 승률 차이는 +5%p 였다 (초/말 효과를 2배 과대평가).
    # 그래서 모든 선수 특성을 '홈팀에게 유리한 방향이 +' 로 재정의한다.
    # 부수 효과로 네 특성 모두에 단조 증가 제약을 걸 수 있다.
    # 부호는 **선수 데이터가 실제로 속한 반이닝**(disp_is_bottom)으로 정한다.
    # 정규화된 3아웃 행은 상태(is_bottom)가 다음 반이닝으로 넘어가지만
    # pitcher_season_era / batter_season_hra 는 직전 반이닝 선수의 값 그대로다.
    # 여기서 is_bottom 을 쓰면 선수는 그대로인데 부호만 뒤집혀
    # **반이닝 경계마다 인위적인 승률 점프**가 생긴다.
    #   실측: 2-2 동점에서 4회말 2아웃 33.3% → 5회초 0아웃 58.6% (+25%p).
    #         bat_adv_home 한 특성이 -16.3%p → +12.8%p 로 혼자 29%p를 흔들었다.
    sign_src = d["disp_is_bottom"] if "disp_is_bottom" in d.columns else d["is_bottom"]
    home_sign = np.where(sign_src == 1, 1.0, -1.0)   # 말=원정이 던짐 → 홈에 유리

    # 시즌 초반 성적은 표본이 없다. 개막전에는 타율 0.000 · ERA 0.00 으로 찍힌다.
    #   실측: 3월 경기의 시즌 ERA 평균이 0.30 (4월은 4.64).
    #   그대로 쓰면 개막 한 달 내내 모든 투수가 특급, 모든 타자가 최악이 된다.
    # API가 시즌 누적 표본 수를 주지 않으므로(ab/inn 은 그날 경기 기록이다)
    # **시즌 경과일**을 대리 변수 삼아 리그 평균 쪽으로 축소한다.
    # 개막 후 6주쯤이면 성적을 그대로 신뢰한다.
    trust = _season_trust(d)

    if "pitcher_season_era" in d.columns:
        era = d["pitcher_season_era"].replace(0.0, np.nan).fillna(LEAGUE_ERA)
        d["pitch_adv_home"] = (era - LEAGUE_ERA) * trust * home_sign
    if "batter_season_hra" in d.columns:
        hra = d["batter_season_hra"].replace(0.0, np.nan).fillna(LEAGUE_HRA)
        d["bat_adv_home"] = (hra - LEAGUE_HRA) * trust * home_sign

    # ── 좌우 매치업 ──
    # 상대 전적(vsHra)은 홈팀 타자만 제공되어 쓸 수 없었다(§4). 좌우는 양 팀 모두 있고
    # 표본도 타자당 수백 타석이라 잡음이 훨씬 작다.
    if {"batter_id", "pitcher_id"} <= set(d.columns):
        hm = _hand_map()
        if hm:
            bats = d["batter_id"].astype(str).map(lambda c: (hm.get(c) or (None, None))[1])
            thr = d["pitcher_id"].astype(str).map(lambda c: (hm.get(c) or (None, None))[0])
            adv = np.array([platoon_advantage(b, t) for b, t in zip(bats, thr)])
            d["platoon_adv_home"] = adv * home_sign

    # 상대 전적 — 이 타자가 '이 투수에게' 시즌 평균보다 얼마나 잘/못 치는가.
    # 원시 vsHra 를 그대로 쓰면 대부분 타자 실력을 다시 인코딩할 뿐이므로,
    # 시즌 타율 대비 **증분**만 남긴다. 기록 없으면 0(중립).
    if {"batter_vs_hra", "batter_season_hra"} <= set(d.columns):
        vs = d["batter_vs_hra"]
        base_hra = d["batter_season_hra"].replace(0.0, np.nan)
        d["matchup_adv_home"] = ((vs - base_hra).fillna(0.0) * home_sign)
        d["matchup_raw_home"] = ((vs.fillna(LEAGUE_HRA) - LEAGUE_HRA) * home_sign)
    if "pitcher_pitches" in d.columns:
        # 투구수가 많은 투수는 지쳐 있다 → 그 투수를 상대하는 공격팀에 유리
        d["fatigue_adv_home"] = d["pitcher_pitches"] / 100.0 * home_sign
    if {"away_pitchers_used", "home_pitchers_used"} <= set(d.columns):
        # 불펜을 더 많이 쓴 팀이 불리하다. home/away 는 경기 내내 의미가 고정돼 있다.
        d["bullpen_adv_home"] = d["away_pitchers_used"] - d["home_pitchers_used"]
    return d


def build_features(events: pd.DataFrame, games: pd.DataFrame,
                   re_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """이벤트 + 경기메타 → 학습용 데이터프레임 (특성 + 라벨 + 식별자)."""
    d = normalize_states(events)
    d["hit_diff"] = d["home_hit"] - d["away_hit"]
    d["bb_diff"] = d["home_bb"] - d["away_bb"]
    d["err_diff"] = d["home_err"] - d["away_err"]

    d = add_derived(d, re_table)

    # ── 경기 전 사전정보 병합 ──
    gcols = ["game_id", "season", "game_date", "home_team", "away_team", "home_win",
             "home_score_final", "away_score_final"]
    opt = [c for c in ("home_team_wra", "away_team_wra", "home_starter_era", "away_starter_era")
           if c in games.columns]
    g = games[gcols + opt].copy()
    d = d.merge(g, on="game_id", how="inner")

    d["prior_home_wra_diff"] = (d.get("home_team_wra", np.nan) - d.get("away_team_wra", np.nan)) \
        if "home_team_wra" in d.columns else 0.0
    d["prior_starter_era_diff"] = (d.get("away_starter_era", np.nan) - d.get("home_starter_era", np.nan)) \
        if "home_starter_era" in d.columns else 0.0
    d[["prior_home_wra_diff", "prior_starter_era_diff"]] = \
        d[["prior_home_wra_diff", "prior_starter_era_diff"]].fillna(0.0)

    d = d[d["home_win"].notna()].copy()          # 무승부 제외
    d["home_win"] = d["home_win"].astype(int)
    for c in FEATURES:
        if c not in d.columns:
            d[c] = 0.0
    return d


def describe_state(row) -> str:
    """사람이 읽는 국면 설명 문자열."""
    half = "말" if row["is_bottom"] else "초"
    base = BASE_STATE_NAME.get(int(row["base_state"]), "?")
    sd = int(row["score_diff"])
    lead = f"홈 {sd}점 리드" if sd > 0 else (f"원정 {-sd}점 리드" if sd < 0 else "동점")
    return (f"{int(row['inning'])}회{half} {int(row['outs'])}아웃 {base}, "
            f"{lead} ({int(row['away_score'])}-{int(row['home_score'])}), "
            f"카운트 {int(row['balls'])}B-{int(row['strikes'])}S")
