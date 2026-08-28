"""확률 보정기 — 승률 곡선의 '매끄러움'까지 함께 본다.

트리 모델의 출력 확률은 극단으로 몰리는 경향이 있어 보정이 필요하다.
보통은 isotonic regression 을 쓰지만, **승률 예측에서는 그게 함정이 된다.**

  실측: isotonic 은 9,973개의 서로 다른 예측값을 50단계로 뭉갰다.
  그 결과 승률 곡선이 계단 함수가 되고, 4회 동점 상황의 평범한 아웃 하나에
  WPA +36.9%p 같은 가짜 점프가 생겼다. WPA 는 승률의 차분이므로
  보정기의 계단이 그대로 '승부처'로 잘못 잡힌다.

그래서 세 후보를 모두 학습해 검증셋 log loss 로 비교하고, 동시에
**매끄러움(고유값 개수)** 을 함께 기록해 선택 근거를 남긴다.
isotonic 은 Platt 보다 뚜렷하게 나을 때만 채택한다.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6
CLIP_LO, CLIP_HI = 0.001, 0.999


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class IdentityCalibrator:
    """보정하지 않음 (원본 확률이 이미 잘 맞을 때)."""
    kind = "identity"

    def fit(self, p, y):
        return self

    def predict(self, p):
        return np.clip(np.asarray(p, dtype=float), CLIP_LO, CLIP_HI)


class PlattCalibrator:
    """Platt scaling — log odds 에 1차 로지스틱을 씌운다. 단조 + 매끄러움."""
    kind = "platt"

    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, p, y):
        from sklearn.linear_model import LogisticRegression
        z = _logit(p).reshape(-1, 1)
        lr = LogisticRegression(max_iter=1000).fit(z, np.asarray(y))
        self.a = float(lr.coef_[0][0])
        self.b = float(lr.intercept_[0])
        return self

    def predict(self, p):
        z = _logit(p)
        out = 1.0 / (1.0 + np.exp(-np.clip(self.a * z + self.b, -30, 30)))
        return np.clip(out, CLIP_LO, CLIP_HI)


class IsotonicCalibrator:
    """비모수 단조 보정. 유연하지만 계단이 생긴다."""
    kind = "isotonic"

    def __init__(self):
        self.iso = None

    def fit(self, p, y):
        from sklearn.isotonic import IsotonicRegression
        self.iso = IsotonicRegression(out_of_bounds="clip").fit(np.asarray(p), np.asarray(y))
        return self

    def predict(self, p):
        return np.clip(self.iso.predict(np.asarray(p, dtype=float)), CLIP_LO, CLIP_HI)


def _log_loss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _smoothness(cal) -> int:
    """[0,1] 을 2001점으로 훑어 서로 다른 출력값이 몇 개인지 — 계단 수."""
    return int(len(np.unique(np.round(cal.predict(np.linspace(0.001, 0.999, 2001)), 6))))


# isotonic 이 자유롭게 만드는 계단 수는 보통 30~60개다. 계단 하나를 신뢰하려면
# 독립 표본(=경기)이 20개쯤 필요하므로 최소 1,000경기는 있어야 한다.
# 실측으로 두 번 확인했다.
#   · 홀드아웃 검증 52경기 → isotonic 검증 0.499(최고), 테스트 0.440 (원본 0.410보다 나쁨)
#   · OOF 320경기        → isotonic OOF 0.4552(최고), 테스트 0.4948 (원본 0.4917보다 나쁨)
# 두 번 다 '검증 점수가 가장 좋은 보정기'가 테스트에서 가장 나빴다.
MIN_VALID_GAMES_FOR_ISOTONIC = 1000


def select_calibrator(p_valid, y_valid, n_valid_games: int | None = None,
                      isotonic_margin: float = 0.01):
    """세 후보를 비교해 하나를 고른다.

    선택 규칙
      1) 검증 경기 수가 부족하면 isotonic 은 후보에서 제외한다 (계단 과적합).
      2) 남은 후보 중 검증 log loss 가 가장 좋은 것을 고른다.
      3) isotonic 이 후보일 때도 Platt 보다 `isotonic_margin` 이상 나아야 채택한다.
         승률 곡선의 연속성(WPA 해석 가능성)이 소수점 셋째 자리보다 중요하다.
    """
    cands = [IdentityCalibrator().fit(p_valid, y_valid),
             PlattCalibrator().fit(p_valid, y_valid),
             IsotonicCalibrator().fit(p_valid, y_valid)]
    iso_ok = n_valid_games is None or n_valid_games >= MIN_VALID_GAMES_FOR_ISOTONIC

    report = []
    for c in cands:
        eligible = iso_ok or c.kind != "isotonic"
        report.append({"kind": c.kind,
                       "valid_log_loss": round(_log_loss(y_valid, c.predict(p_valid)), 4),
                       "distinct_outputs": _smoothness(c),
                       "eligible": eligible})

    by = {r["kind"]: r["valid_log_loss"] for r in report}
    chosen = min(("identity", "platt"), key=lambda k: by[k])
    if iso_ok and by["isotonic"] < by[chosen] - isotonic_margin:
        chosen = "isotonic"
    picked = next(c for c in cands if c.kind == chosen)
    for r in report:
        r["chosen"] = (r["kind"] == chosen)
    return picked, report
