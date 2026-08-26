"""승률 모델 학습 · 보정 · 평가.

  python -m src.train                      # data/raw 의 모든 데이터로 학습
  python -m src.train --test-season 2026   # 특정 시즌을 테스트셋으로 고정

산출물: models/wp_model.txt, models/calibrator.pkl, models/meta.json,
        models/re24.csv, reports/eval.md, reports/reliability.png

────────────────────────────────────────────────────────────────────────
이 파일의 설계를 지배하는 하나의 사실: **유효 표본은 경기 수다.**

한 경기는 400여 행을 만들지만 그 행들은 모두 같은 라벨(홈 승/패)을 공유한다.
따라서 행이 20만 개여도 독립 표본은 경기 수(수백 개)뿐이다. 이걸 무시하면
세 군데에서 연달아 사고가 난다. 셋 다 실측으로 겪었다.

  1) 특성 — 경기 내 상수인 사전정보가 '경기 식별자'로 작동해 gain의 38.6%를 차지.
            모델이 야구가 아니라 "어느 팀 경기인지"를 외웠다. → features.py 에서 분리
  2) 모델 용량 — 다만 실측해보니 용량은 병목이 아니었다(§tune). 병목은 특성이었다.
  3) 보정기 — 검증 52경기에 isotonic 을 맞추면 검증 log loss 는 최고(0.499)지만
            테스트에서는 원본보다 나빠진다(0.440 vs 0.410).
            → 홀드아웃 대신 GroupKFold out-of-fold 예측으로 보정기를 학습
────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_RAW, DATA_PROC, MODEL_DIR, REPORT_DIR  # noqa: E402
from src.features import (FEATURES, build_features,            # noqa: E402
                          build_run_expectancy, normalize_states)

N_FOLDS = 5


# ── 데이터 로드 ──────────────────────────────────────────────────────
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = [pd.read_parquet(f) for f in sorted(DATA_RAW.glob("events_*.parquet"))]
    gm = [pd.read_parquet(f) for f in sorted(DATA_RAW.glob("games_*.parquet"))]
    if not ev:
        raise SystemExit("data/raw 에 수집 데이터가 없습니다. 먼저 python -m src.collect 를 실행하세요.")
    events = pd.concat(ev, ignore_index=True).drop_duplicates(["game_id", "event_idx"])
    games = pd.concat(gm, ignore_index=True).drop_duplicates("game_id")
    return events, games


# ── 베이스라인들 ─────────────────────────────────────────────────────
def baseline_empirical(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """전통적 방식: (이닝, 초/말, 베이스-아웃, 점수차) 구간별 실제 승률 조회표."""
    def key(d):
        return list(zip(d["inning"].clip(1, 10), d["is_bottom"], d["base_state"],
                        d["outs"].clip(0, 2), d["score_diff"].clip(-7, 7)))
    tab = pd.DataFrame({"k": key(train), "y": train["home_win"].values})
    look = tab.groupby("k")["y"].agg(["mean", "count"])
    prior = train["home_win"].mean()
    # 표본이 적은 칸은 전체 평균 쪽으로 축소(shrinkage)
    look["p"] = (look["mean"] * look["count"] + prior * 30) / (look["count"] + 30)
    mp = look["p"].to_dict()
    return np.array([mp.get(k, prior) for k in key(test)])


def baseline_scorediff(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """점수차 + 남은 아웃만 쓰는 로지스틱 회귀 (사람이 머릿속으로 하는 계산)."""
    from sklearn.linear_model import LogisticRegression
    cols = ["score_diff", "outs_left_bat", "is_bottom"]

    def mat(d):
        X = d[cols].values.astype(float)
        inter = d["score_diff"].values * (27 - d["outs_left_bat"].values) / 27.0
        return np.column_stack([X, inter])

    lr = LogisticRegression(max_iter=1000).fit(mat(train), train["home_win"])
    return lr.predict_proba(mat(test))[:, 1]


# ── 평가 ─────────────────────────────────────────────────────────────
def evaluate(y, p, name: str) -> dict:
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {"model": name,
            "log_loss": round(log_loss(y, p), 4),
            "brier": round(brier_score_loss(y, p), 4),
            "auc": round(roc_auc_score(y, p), 4),
            "acc": round(accuracy_score(y, p > 0.5), 4),
            "n": len(y)}


def reliability_plot(y, p, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = np.linspace(0, 1, 21)
    idx = np.digitize(p, bins) - 1
    xs, ys, ns = [], [], []
    for b in range(20):
        m = idx == b
        if m.sum() >= 30:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
            ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], "--", c="gray", lw=1, label="perfect calibration")
    if ns:
        ax.scatter(xs, ys, s=np.array(ns) / max(ns) * 220 + 15, alpha=.75, label="observed")
    ax.set_xlabel("predicted win prob")
    ax.set_ylabel("actual win rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def lgb_params(n_rows: int, n_games: int) -> dict:
    """하이퍼파라미터는 src.tune 의 OOF 탐색으로 확정한 값이다.

    탐색 결과(713경기, 6개 설정): OOF log loss 0.4647~0.4661 — 폭이 0.0014뿐이었다.
    즉 **용량은 병목이 아니다.** 규제를 세게 걸어 신호를 누르고 있다는 가설은 틀렸다.
    병목은 특성이었고, 그건 절제 실험(reports/ablation.md)으로 해결했다.
    그래서 여기서는 무난한 중간값을 쓰고 잎 크기만 '경기당 행 수' 기준으로 맞춘다.
    """
    mono_up = {"score_diff", "score_x_progress", "score_per_out_left",
               # 부호를 홈팀 유리 방향으로 고정했으므로 모두 단조 증가여야 한다
               "pitch_adv_home", "bat_adv_home", "fatigue_adv_home", "bullpen_adv_home"}
    per_game = n_rows / max(n_games, 1)
    return dict(objective="binary", metric="binary_logloss",
                learning_rate=0.02,
                num_leaves=63,
                # 잎 하나에 최소 '1경기 분량'의 행 — 유효 표본이 경기 수라는 사실을 반영
                min_data_in_leaf=max(100, int(per_game)),
                feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1,
                lambda_l1=1.0, lambda_l2=10.0,
                min_gain_to_split=1e-4,
                monotone_constraints=[1 if f in mono_up else 0 for f in FEATURES],
                verbose=-1, num_threads=0)


# ── 메인 ─────────────────────────────────────────────────────────────
def main() -> None:
    import lightgbm as lgb
    from sklearn.model_selection import GroupShuffleSplit, GroupKFold
    from sklearn.metrics import brier_score_loss, log_loss
    from src.calibrate import select_calibrator

    ap = argparse.ArgumentParser()
    ap.add_argument("--test-season", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    a = ap.parse_args()

    events, games = load_raw()
    print(f"[train] 이벤트 {len(events):,}행 / 경기 {len(games):,}건 "
          f"/ 시즌 {sorted(int(s) for s in games.season.unique())}")

    # 1) 경기 단위 분할 — 같은 경기의 행이 train/test 에 섞이면 라벨이 새어나간다.
    #    별도 검증 홀드아웃은 두지 않는다. 학습 경기 안에서 교차검증으로 해결한다.
    gid = games["game_id"].values
    if a.test_season and (games.season == a.test_season).any():
        test_g = set(games.loc[games.season == a.test_season, "game_id"])
        train_g = set(gid) - test_g
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tr_i, te_i = next(gss.split(gid, groups=gid))
        train_g, test_g = set(gid[tr_i]), set(gid[te_i])
    print(f"[train] 경기 분할  train {len(train_g)} / test {len(test_g)}")

    # 2) RE24 는 학습 경기에서만 추정 (테스트 정보 유입 차단)
    re_tab = build_run_expectancy(normalize_states(events[events.game_id.isin(train_g)].copy()))
    re_tab.to_csv(MODEL_DIR / "re24.csv", index=False)

    data = build_features(events, games, re_tab)
    data.to_parquet(DATA_PROC / "model_frame.parquet", index=False)
    tr = data[data.game_id.isin(train_g)].reset_index(drop=True)
    te = data[data.game_id.isin(test_g)].reset_index(drop=True)
    print(f"[train] 행 수  train {len(tr):,} / test {len(te):,}")

    params = lgb_params(len(tr), len(train_g))
    print(f"[train] min_data_in_leaf = {params['min_data_in_leaf']} "
          f"(경기당 평균 {len(tr)/max(len(train_g),1):.0f}행)")

    # 3) GroupKFold 교차검증
    #    (a) 조기종료 라운드 수를 폴드 중앙값으로 결정
    #    (b) 학습 경기 '전체'에 대한 out-of-fold 예측을 만들어 보정기 학습에 사용
    #        → 보정기가 소수 경기(52개)에 과적합되는 사고를 원천 차단
    n_folds = min(a.folds, max(2, len(train_g) // 20))
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros(len(tr))
    oof_lr = np.zeros(len(tr))      # 베이스라인도 같은 폴드로 OOF 예측을 만든다
    oof_emp = np.zeros(len(tr))
    fold_rows = []
    for k, (i_tr, i_va) in enumerate(gkf.split(tr[FEATURES], tr["home_win"], groups=tr["game_id"]), 1):
        f_tr, f_va = tr.loc[i_tr], tr.loc[i_va]
        dtr = lgb.Dataset(f_tr[FEATURES], f_tr["home_win"])
        dva = lgb.Dataset(f_va[FEATURES], f_va["home_win"], reference=dtr)
        b = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[i_va] = b.predict(f_va[FEATURES], num_iteration=b.best_iteration)
        oof_lr[i_va] = baseline_scorediff(f_tr, f_va)
        oof_emp[i_va] = baseline_empirical(f_tr, f_va)
        ll = log_loss(f_va["home_win"], np.clip(oof[i_va], 1e-6, 1 - 1e-6))
        fold_rows.append({"fold": k, "n_games": f_va["game_id"].nunique(),
                          "best_iter": b.best_iteration, "log_loss": round(ll, 4)})
        print(f"[train]  fold {k}/{n_folds}  경기 {fold_rows[-1]['n_games']:>3}  "
              f"iter {b.best_iteration:>4}  log loss {ll:.4f}")

    n_rounds = int(np.median([f["best_iter"] for f in fold_rows]))
    oof_ll = log_loss(tr["home_win"], np.clip(oof, 1e-6, 1 - 1e-6))
    print(f"[train] OOF log loss {oof_ll:.4f}  ·  최종 부스팅 라운드 {n_rounds}")

    # 4) 보정기 — 학습 경기 전체의 OOF 예측으로 학습
    cal, cal_report = select_calibrator(oof, tr["home_win"].values,
                                        n_valid_games=len(train_g))
    print()
    print("[train] 보정기 비교 (OOF 기준)")
    for r in cal_report:
        mark = " <- 채택" if r["chosen"] else ""
        elig = "" if r["eligible"] else "  (경기 수 부족으로 제외)"
        print(f"   {r['kind']:<10} OOF log loss {r['valid_log_loss']:.4f}  "
              f"고유 출력값 {r['distinct_outputs']:>5}개{mark}{elig}")

    # 5) 최종 모델 — 학습 경기 전체 사용
    booster = lgb.train(params, lgb.Dataset(tr[FEATURES], tr["home_win"]),
                        num_boost_round=n_rounds)

    p_te_raw = booster.predict(te[FEATURES])
    p_te = cal.predict(p_te_raw)
    y_te = te["home_win"].values

    rows = [evaluate(y_te, np.full(len(y_te), tr["home_win"].mean()), "1. 상수(홈 승률 평균)"),
            evaluate(y_te, baseline_scorediff(tr, te), "2. 점수차 로지스틱"),
            evaluate(y_te, baseline_empirical(tr, te), "3. 경험적 조회표"),
            evaluate(y_te, p_te_raw, "4. LightGBM (보정 전)"),
            evaluate(y_te, p_te, f"5. LightGBM + {cal.kind} 보정 [최종]")]

    # 네이버 자체 승률과의 정면 비교 — 같은 행에서만 공정하게 비교한다.
    # 0.0 / 100.0 은 네이버 쪽 결측 자리표시자로 보이므로 제외한다.
    if "naver_wp_home" in te.columns:
        nv = te["naver_wp_home"] / 100.0
        ok = (nv.notna() & (nv > 0.0) & (nv < 1.0)).values
        if ok.sum() > 500:
            rows.append(evaluate(y_te[ok], nv.values[ok], f"[참고] 네이버 자체 승률 (n={int(ok.sum()):,})"))
            rows.append(evaluate(y_te[ok], p_te[ok], "[참고] 우리 모델 (동일 구간)"))
    res = pd.DataFrame(rows)

    # ── OOF 기준 비교 (주 비교표) ──
    # 테스트 홀드아웃은 경기 수가 적어 분할 운에 따라 순위가 뒤집힌다.
    # 같은 폴드에서 만든 OOF 예측으로 비교하면 학습 경기 전체가 표본이 되어 안정적이다.
    y_tr = tr["home_win"].values
    oof_rows = [evaluate(y_tr, np.full(len(y_tr), y_tr.mean()), "1. 상수(홈 승률 평균)"),
                evaluate(y_tr, oof_lr, "2. 점수차 로지스틱"),
                evaluate(y_tr, oof_emp, "3. 경험적 조회표"),
                evaluate(y_tr, oof, "4. LightGBM [최종]")]
    if "naver_wp_home" in tr.columns:
        nv_tr = tr["naver_wp_home"] / 100.0
        ok_tr = (nv_tr.notna() & (nv_tr > 0.0) & (nv_tr < 1.0)).values
        if ok_tr.sum() > 500:
            oof_rows.append(evaluate(y_tr[ok_tr], nv_tr.values[ok_tr],
                                     f"[참고] 네이버 자체 승률 (n={int(ok_tr.sum()):,})"))
            oof_rows.append(evaluate(y_tr[ok_tr], oof[ok_tr], "[참고] 우리 모델 (동일 구간)"))
    oof_res = pd.DataFrame(oof_rows)

    print()
    print(f"── OOF 비교 (학습 {len(train_g)}경기 / {len(tr):,}행) ──")
    print(oof_res.to_string(index=False))
    print()
    print(f"── 테스트 홀드아웃 ({len(test_g)}경기 / {len(te):,}행) ──")
    print(res.to_string(index=False))

    # 이닝별 성능
    te2 = te.copy()
    te2["p"] = p_te
    by_inn = []
    for innv, d in te2.groupby("inning"):
        if len(d) < 50 or d.home_win.nunique() < 2:
            continue
        by_inn.append({"inning": int(innv), "n": len(d),
                       "brier": round(brier_score_loss(d.home_win, d.p), 4),
                       "mean_pred": round(float(d.p.mean()), 3),
                       "actual": round(float(d.home_win.mean()), 3)})
    by_inn = pd.DataFrame(by_inn)

    reliability_plot(y_te, p_te, REPORT_DIR / "reliability.png", "Calibration (test set)")

    imp = pd.DataFrame({"feature": FEATURES,
                        "gain": booster.feature_importance("gain")}).sort_values("gain", ascending=False)
    imp["share"] = (imp.gain / max(imp.gain.sum(), 1)).round(4)

    # 6) 저장
    booster.save_model(str(MODEL_DIR / "wp_model.txt"))
    with open(MODEL_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump(cal, f)
    meta = {"features": FEATURES,
            "n_rounds": n_rounds, "folds": fold_rows, "oof_log_loss": round(oof_ll, 4),
            "oof_metrics": oof_rows,
            "calibration": cal_report,
            "n_train_games": len(train_g), "n_test_games": len(test_g),
            "seasons": sorted(int(s) for s in games.season.unique()),
            "metrics": rows, "importance": imp.to_dict("records")}
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    # 7) 레버리지 지수의 기준값 — 리그 평균 타석이 승률을 흔드는 폭(%p)
    from src.explain import WinProbabilityExplainer
    ex = WinProbabilityExplainer()
    samp = tr[tr.event_type == 8]
    samp = samp.sample(min(600, len(samp)), random_state=0) if len(samp) else tr.sample(min(600, len(tr)))
    meta["league_mean_swing_pp"] = round(float(np.mean([ex.leverage(r)["swing_pp"]
                                                        for _, r in samp.iterrows()])), 3)
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"[train] 리그 평균 타석 변동폭 = {meta['league_mean_swing_pp']}%p (레버리지 기준값)")

    md = ["# 승률 모델 평가 리포트", "",
          f"- 학습 경기 {len(train_g)} / 테스트 경기 {len(test_g)}",
          f"- 테스트 행 수 {len(te):,} (투구·타석 단위 상태)",
          f"- {n_folds}-fold GroupKFold OOF log loss **{oof_ll:.4f}** · 최종 라운드 {n_rounds}", "",
          f"## 모델 비교 — OOF 기준 (주 비교표, {len(train_g)}경기)", "",
          oof_res.to_markdown(index=False), "",
          "> 테스트 홀드아웃은 경기 수가 적어 분할 운에 따라 순위가 뒤집힌다. "
          "같은 폴드에서 만든 OOF 예측으로 비교하면 학습 경기 전체가 표본이 되어 안정적이다.", "",
          f"## 모델 비교 — 테스트 홀드아웃 ({len(test_g)}경기)", "",
          res.to_markdown(index=False), "",
          "> Log loss / Brier 는 낮을수록 좋다. 승부예측에서 중요한 것은 정답률이 아니라 확률의 정확도다.", "",
          "## 교차검증 폴드", "", pd.DataFrame(fold_rows).to_markdown(index=False), "",
          "## 보정기 선택", "", pd.DataFrame(cal_report).to_markdown(index=False), "",
          "> 계단형 보정기(isotonic)는 log loss 는 좋아 보여도 승률 곡선을 계단으로 만들어 "
          "WPA 에 가짜 점프를 만든다. 그리고 경기 수가 적으면 검증셋을 외워버린다. "
          "그래서 뚜렷한 차이가 없으면 매끄러운 쪽을 택한다.", "",
          "## 이닝별 성능", "", by_inn.to_markdown(index=False), "",
          "## 특성 중요도 (gain 기준)", "", imp.head(20).to_markdown(index=False), "",
          "## 보정 곡선", "", "![reliability](reliability.png)", ""]
    (REPORT_DIR / "eval.md").write_text("\n".join(md), encoding="utf-8")
    print("[train] 저장 완료 → models/  ·  reports/eval.md")


if __name__ == "__main__":
    main()
