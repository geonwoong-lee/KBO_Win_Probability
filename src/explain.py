"""승률을 '왜 그렇게 봤는지' 설명하는 XAI 레이어.

설명을 세 겹으로 쌓는다. 각 겹이 답하는 질문이 다르다.

  1) WPA (Win Probability Added)  — "그 타석이 승률을 얼마나 바꿨나?"
     야구계가 이미 쓰는 언어. 모델과 무관하게 정의되므로 가장 신뢰받는다.

  2) SHAP (TreeSHAP)              — "지금 이 승률은 어떤 요인이 만들었나?"
     현재 상태의 각 특성이 기준 승률(50%)에서 몇 %p씩 밀어올렸는지 분해한다.
     LightGBM 내장 pred_contrib 을 쓰므로 별도 라이브러리 없이도 동작한다.

  3) 반사실 (Counterfactual)      — "만약 무사였다면? 만약 주자가 없었다면?"
     상태를 하나만 바꿔 다시 예측한다. 현장에서 가장 설득력 있는 형태이며,
     '지금 이 상황의 진짜 레버리지가 무엇인지'를 직접 보여준다.

추가로 레버리지(이 타석 하나가 승률을 흔들 수 있는 폭)를 함께 제공한다.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODEL_DIR                                   # noqa: E402
from src.features import (FEATURES, BASE_STATE_NAME, describe_state,   # noqa: E402
                          DEFAULT_RUN_EXP, add_derived)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# 특성 → 한국어 설명 (SHAP 결과를 사람 말로 옮길 때 사용)
FEATURE_KO = {
    "inning": "이닝",
    "is_bottom": "홈팀 공격(말) 여부",
    "outs": "아웃카운트",
    "base_state": "주자 상황",
    "balls": "볼 카운트",
    "strikes": "스트라이크 카운트",
    "score_diff": "점수차",
    "half_index": "경기 진행도",
    "score_x_progress": "점수차x경기진행도",
    "score_per_out_left": "점수차/잔여아웃",
    "outs_left_bat": "공격팀 남은 아웃",
    "outs_left_field": "수비팀 남은 아웃",
    "run_exp": "기대득점(RE)",
    "run_potential": "잔여 득점 잠재력",
    "is_last_at_bat": "홈팀 마지막 공격",
    "is_walkoff_chance": "끝내기 기회",
    "bat_order": "타순",
    "pitch_adv_home": "투수 매치업 우위",
    "bat_adv_home": "타자 매치업 우위",
    "fatigue_adv_home": "투수 피로 우위",
    "bullpen_adv_home": "불펜 소모 우위",
    "pitcher_pitches": "현재 투수 투구수",
    "pitcher_season_era": "투수 시즌 ERA",
    "batter_season_hra": "타자 시즌 타율",
    "hit_diff": "안타 차",
    "bb_diff": "볼넷 차",
    "err_diff": "실책 차",
    "prior_home_wra_diff": "시즌 승률 차(홈-원정)",
    "prior_starter_era_diff": "선발 ERA 차",
}

# 다음 타석에서 나올 수 있는 결과와 대략적 발생 확률 (KBO 리그 평균 수준)
OUTCOMES = [
    ("아웃",   0.680),
    ("볼넷",   0.090),
    ("단타",   0.150),
    ("2루타",  0.050),
    ("홈런",   0.030),
]


class WinProbabilityExplainer:
    """학습된 모델을 불러와 승률 예측 + 3중 설명을 생성한다."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        import lightgbm as lgb
        self.booster = lgb.Booster(model_file=str(model_dir / "wp_model.txt"))
        cal = model_dir / "calibrator.pkl"
        self.calibrator = pickle.loads(cal.read_bytes()) if cal.exists() else None
        meta = model_dir / "meta.json"
        self.meta = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        re_csv = model_dir / "run_exp.csv"
        self.re_table = pd.read_csv(re_csv) if re_csv.exists() else None
        if self.re_table is not None:
            self.run_exp = {(int(r.base_state), int(r.outs)): float(r.run_exp)
                         for r in self.re_table.itertuples()}
        else:
            self.run_exp = dict(DEFAULT_RUN_EXP)
        self._li_ref = None

    # ── 상태 변경 후 파생 특성 재계산 ────────────────────────────────
    def _respec(self, rows: list) -> pd.DataFrame:
        """반사실·레버리지에서 바꾼 상태로부터 파생 특성을 다시 계산한다.

        이걸 빠뜨리면 score_per_out_left(gain 36.5%) 같은 특성이 낡은 값으로 남아
        "만약 1점 더 앞섰다면" 같은 예측이 통째로 틀린다.
        """
        df = pd.DataFrame(rows)
        return add_derived(df, self.re_table)

    # ── 예측 ─────────────────────────────────────────────────────────
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """홈팀 승리 확률 (보정 적용)."""
        raw = self.booster.predict(X[FEATURES])
        return np.clip(self.calibrator.predict(raw) if self.calibrator is not None else raw,
                       0.001, 0.999)

    # ── 1) WPA ───────────────────────────────────────────────────────
    def add_win_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        """이벤트 시계열에 승률과 WPA를 붙인다.

        WPA 는 두 가지를 계산한다.

          wpa     이벤트 단위 차분. 승률 곡선을 그리는 데 쓴다.
          wpa_pa  **타석 단위**. 야구에서 말하는 WPA 는 원래 이쪽이다
                  (타석 시작 승률 → 타석 종료 승률).

        타석 단위가 필요한 이유는 홈런 하나가 여러 이벤트로 쪼개지기 때문이다.
        만루 홈런이면 '홈런' 한 줄과 '주자 홈인' 세 줄로 나뉘고, 점수는 홈인 줄에서
        올라간다. 이벤트 단위로만 보면 **끝내기 만루 홈런의 대표 플레이가
        '3루주자 홈인'으로 잡힌다.** 실측 사례:

            홈런 (7-5)            +4.2%p
            1루주자 홈인 (7-6)     -2.2%p   ← 득점했는데 음수 (만루가 사라져서)
            2루주자 홈인 (7-7)    +43.7%p
            3루주자 홈인 (7-8)    +45.7%p
            ────────────────────────────
            타석 합계             +91.4%p   ← 이게 그 홈런의 WPA다
        """
        d = df.copy()
        d["wp_home"] = self.predict(d)
        d["wpa"] = d["wp_home"].diff().fillna(0.0)

        # 타석 합계를 그 타석의 '결과' 이벤트에 귀속시킨다.
        d["wpa_pa"] = 0.0
        if "pa_no" in d.columns:
            d = d.sort_values("event_idx")
            tot = d.groupby("pa_no")["wpa"].transform("sum")
            is_res = d["event_type"].isin([13, 23])
            # 결과 이벤트가 여럿이면 첫 줄에만 (희생플라이 + 득점 등)
            first_res = is_res & ~is_res.groupby(d["pa_no"]).cumsum().gt(1)
            d.loc[first_res, "wpa_pa"] = tot[first_res]

        d["wpa_batting_team"] = np.where(d["is_bottom"] == 1, d["wpa"], -d["wpa"])
        return d

    def key_plays(self, df: pd.DataFrame, top: int = 8) -> pd.DataFrame:
        """승률을 가장 크게 흔든 플레이 — 타석 단위 WPA 상위.

        이벤트 단위가 아니라 타석 단위로 본다. 그러지 않으면 만루 홈런의
        대표가 '주자 홈인'으로 잡힌다(add_win_probability 참고).
        """
        d = self.add_win_probability(df) if "wpa_pa" not in df.columns else df
        d = d[d["wpa_pa"] != 0].copy()
        d["wpa"] = d["wpa_pa"]
        # 3아웃 정규화로 다음 반이닝에 옮겨진 행은 소속 표기를 원래대로 되돌린다.
        # 그러지 않으면 '4회초(원정 공격)에 홈팀 타자'처럼 말이 안 되는 줄이 나온다.
        if "disp_inning" in d.columns:
            d["inning"] = d["disp_inning"]
            d["is_bottom"] = d["disp_is_bottom"]
        out = d.reindex(d["wpa"].abs().sort_values(ascending=False).index).head(top)
        return out[["inning", "is_bottom", "outs", "base_state", "away_score",
                    "home_score", "text", "wp_home", "wpa"]]

    # ── 2) SHAP ──────────────────────────────────────────────────────
    def shap_breakdown(self, row: pd.Series, top: int = 6) -> list[dict]:
        """현재 승률을 특성별 기여도(%p)로 분해.

        LightGBM 의 pred_contrib 은 log odds 공간의 기여도를 준다.
        기여도가 큰 순서대로 하나씩 더해가며 확률 공간으로 변환하면
        각 항의 합이 정확히 (예측확률 - 기준확률) 이 되도록 분해할 수 있다.
        """
        X = pd.DataFrame([row[FEATURES].values], columns=FEATURES).astype(float)
        contrib = self.booster.predict(X, pred_contrib=True)[0]
        base_logit, feat_logit = contrib[-1], contrib[:-1]

        order = np.argsort(-np.abs(feat_logit))
        cum = base_logit
        items = []
        for i in order:
            new = cum + feat_logit[i]
            items.append({
                "feature": FEATURES[i],
                "label": FEATURE_KO.get(FEATURES[i], FEATURES[i]),
                "value": float(row[FEATURES[i]]),
                "delta_pp": float((_sigmoid(new) - _sigmoid(cum)) * 100),
            })
            cum = new
        items.sort(key=lambda x: -abs(x["delta_pp"]))
        return items[:top]

    # ── 3) 반사실 ────────────────────────────────────────────────────
    def counterfactuals(self, row: pd.Series) -> list[dict]:
        """상태를 하나씩만 바꿔 다시 예측한다 — '만약 ~였다면'."""
        base_p = float(self.predict(self._respec([row]))[0])
        cases: list[tuple[str, dict]] = []

        # 기본 상태만 바꾼다. 파생 특성은 _respec 이 다시 계산한다.
        if row["outs"] > 0:
            cases.append(("무사였다면", {"outs": 0}))
        if row["outs"] < 2:
            cases.append(("2아웃이었다면", {"outs": 2}))
        if row["base_state"] != 0:
            cases.append(("주자가 없었다면", {"base_state": 0}))
        if row["base_state"] != 7:
            cases.append(("만루였다면", {"base_state": 7}))
        cases.append(("홈팀이 1점 더 앞섰다면", {"home_score": row["home_score"] + 1}))
        cases.append(("홈팀이 1점 더 뒤졌다면", {"away_score": row["away_score"] + 1}))
        if row["inning"] < 9:
            cases.append(("같은 상황이 9회였다면", {"inning": 9}))

        rows, labels = [], []
        for label, patch in cases:
            r = row.copy()
            for k, v in patch.items():
                r[k] = v
            rows.append(r)
            labels.append(label)
        if not rows:
            return []
        ps = self.predict(self._respec(rows))
        out = [{"label": l, "wp": float(p), "delta_pp": float((p - base_p) * 100)}
               for l, p in zip(labels, ps)]
        out.sort(key=lambda x: -abs(x["delta_pp"]))
        return out

    # ── 레버리지 ─────────────────────────────────────────────────────
    def leverage(self, row: pd.Series) -> dict:
        """이 타석 하나가 승률을 흔들 수 있는 기대 폭 (%p)."""
        base_p = float(self.predict(self._respec([row]))[0])
        rows, ws, names = [], [], []
        for name, w in OUTCOMES:
            rows.append(self._apply_outcome(row, name))
            ws.append(w)
            names.append(name)
        ps = self.predict(self._respec(rows))
        swing = float(np.sum(np.array(ws) * np.abs(ps - base_p)) * 100)
        detail = [{"outcome": n, "wp": float(p), "delta_pp": float((p - base_p) * 100)}
                  for n, p in zip(names, ps)]
        li_ref = self.meta.get("league_mean_swing_pp", 3.2)
        return {"swing_pp": round(swing, 2),
                "leverage_index": round(swing / li_ref, 2) if li_ref else None,
                "detail": detail}

    def _apply_outcome(self, row: pd.Series, outcome: str) -> pd.Series:
        """가상의 타석 결과를 반영한 상태를 만든다 (근사)."""
        r = row.copy()
        b = int(row["base_state"])
        on1, on2, on3 = b & 1, (b >> 1) & 1, (b >> 2) & 1
        bat_key = "home_score" if row["is_bottom"] == 1 else "away_score"
        runs = 0

        if outcome == "아웃":
            r["outs"] = min(3, int(row["outs"]) + 1)
        elif outcome == "볼넷":
            if on1 and on2 and on3:
                runs = 1                              # 만루 밀어내기
            elif on1 and on2:
                on3 = 1
            elif on1:
                on2 = 1
            on1 = 1
            r["base_state"] = on1 + on2 * 2 + on3 * 4
        elif outcome == "단타":
            runs = on3 + on2
            r["base_state"] = 1 + (on1 * 2)           # 타자 1루, 기존 1루주자 2루
        elif outcome == "2루타":
            runs = on1 + on2 + on3
            r["base_state"] = 2
        elif outcome == "홈런":
            runs = 1 + on1 + on2 + on3
            r["base_state"] = 0

        r[bat_key] = row[bat_key] + runs              # 파생(score_diff 등)은 _respec 이 계산
        r["balls"], r["strikes"] = 0, 0
        return r

    # ── 종합 설명 ────────────────────────────────────────────────────
    def explain(self, row: pd.Series, home_team: str = "홈", away_team: str = "원정") -> dict:
        wp = float(self.predict(pd.DataFrame([row]))[0])
        shap = self.shap_breakdown(row)
        cf = self.counterfactuals(row)
        lev = self.leverage(row)
        return {
            # 주의: shap/counterfactuals/leverage 의 수치는 모두 '홈팀 기준'이다.
            # 원정팀 관점으로 보여줄 때는 (1 - wp), (-delta) 로 변환해야 한다.
            "wp_home": wp,
            "wp_away": 1 - wp,
            "home_team": home_team,
            "away_team": away_team,
            "state": describe_state(row),
            "shap": shap,
            "counterfactuals": cf,
            "leverage": lev,
            "narrative": self.narrate(wp, row, shap, cf, lev, home_team, away_team),
        }

    @staticmethod
    def narrate(wp: float, row: pd.Series, shap: list[dict], cf: list[dict],
                lev: dict, home_team: str, away_team: str) -> str:
        """숫자를 사람이 읽는 문장으로 옮긴다."""
        lead, prob = (home_team, wp) if wp >= .5 else (away_team, 1 - wp)
        sign = 1 if wp >= .5 else -1          # 우세팀 관점으로 부호 정렬

        L = [f"**{lead} 승리 확률 {prob*100:.1f}%**  —  {describe_state(row)}"]

        up = [s for s in shap if s["delta_pp"] * sign > 0][:3]
        dn = [s for s in shap if s["delta_pp"] * sign < 0][:2]
        if up:
            L.append("이렇게 본 이유: " + ", ".join(
                f"{s['label']}({_fmt_val(s['feature'], s['value'])}) {abs(s['delta_pp']):+.1f}%p"
                for s in up))
        if dn:
            L.append("반대로 깎아내린 요인: " + ", ".join(
                f"{s['label']}({_fmt_val(s['feature'], s['value'])}) -{abs(s['delta_pp']):.1f}%p"
                for s in dn))
        if cf:
            # cf 의 wp/delta_pp 는 항상 '홈 기준'이다. 헤드라인이 원정팀이면 기준을 맞춘다.
            c = cf[0]
            wp_lead = c["wp"] if sign > 0 else 1 - c["wp"]
            d_lead = c["delta_pp"] * sign
            L.append(f"만약 {c['label']} → {lead} {wp_lead*100:.1f}% ({d_lead:+.1f}%p)")
        L.append(f"이 타석의 레버리지: 평균 대비 {lev['leverage_index']}배 "
                 f"(기대 승률 변동 폭 {lev['swing_pp']:.1f}%p)")
        return "\n".join(L)


def _fmt_val(feature: str, v: float) -> str:
    if feature == "base_state":
        return BASE_STATE_NAME.get(int(v), str(int(v)))
    if feature in ("is_bottom", "is_last_at_bat", "is_walkoff_chance"):
        return "예" if v else "아니오"
    if feature in ("run_exp", "run_potential", "pitcher_season_era", "batter_season_hra",
                   "pitch_adv_home", "bat_adv_home", "fatigue_adv_home",
                   "score_x_progress", "score_per_out_left",
                   "prior_home_wra_diff", "prior_starter_era_diff"):
        return f"{v:.2f}"
    return str(int(v))
