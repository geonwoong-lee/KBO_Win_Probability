"""하이퍼파라미터 탐색 · 특성 세트 절제(ablation) — 둘 다 OOF log loss 기준.

  python -m src.tune                # 하이퍼파라미터 후보군 탐색
  python -m src.tune --ablate       # 특성 세트 절제 실험

왜 OOF 로 고르는가
  테스트 홀드아웃은 경기 수가 적어(140경기 수준) 분할 운에 따라 순위가 뒤집힌다.
  GroupKFold OOF 는 학습 경기 전체(수백 경기)가 표본이므로 훨씬 안정적이다.

하이퍼파라미터 탐색 결과 (713경기)
  6개 설정이 OOF log loss 0.4647~0.4661 안에 모두 들어왔다. 폭이 0.0014 다.
  **용량은 병목이 아니다.** 규제가 신호를 누르고 있다는 가설은 틀렸다.
  병목은 특성이므로, 특성 세트를 직접 잘라보는 절제 실험이 필요하다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODEL_DIR, REPORT_DIR                        # noqa: E402
from src.features import FEATURES, build_features, build_run_expectancy, normalize_states  # noqa: E402
from src.train import load_raw                                  # noqa: E402

MONO_UP = {"score_diff", "score_x_progress", "score_per_out_left",
           "pitch_adv_home", "bat_adv_home", "fatigue_adv_home",
           "bullpen_adv_home", "platoon_adv_home"}


def grid(n_rows: int, n_games: int) -> list[dict]:
    """경기당 평균 행 수를 기준으로 후보를 만든다."""
    per_game = n_rows / max(n_games, 1)
    return [
        # (이름, num_leaves, 잎당 최소 '경기 분량', lambda_l2, learning_rate, feature_fraction)
        {"name": "A 현재설정",      "leaves": 31,  "games_per_leaf": 2.0, "l2": 20.0, "lr": 0.02, "ff": 0.70},
        {"name": "B 용량↑",        "leaves": 63,  "games_per_leaf": 1.0, "l2": 10.0, "lr": 0.02, "ff": 0.80},
        {"name": "C 용량↑↑",       "leaves": 127, "games_per_leaf": 0.5, "l2": 5.0,  "lr": 0.02, "ff": 0.80},
        {"name": "D 얕고 오래",     "leaves": 15,  "games_per_leaf": 1.0, "l2": 5.0,  "lr": 0.01, "ff": 0.90},
        {"name": "E 규제↓",        "leaves": 63,  "games_per_leaf": 0.5, "l2": 1.0,  "lr": 0.03, "ff": 1.00},
        {"name": "F 중간",         "leaves": 63,  "games_per_leaf": 1.5, "l2": 10.0, "lr": 0.015, "ff": 0.75},
    ]


def to_params(c: dict, per_game: float) -> dict:
    return dict(objective="binary", metric="binary_logloss",
                learning_rate=c["lr"], num_leaves=c["leaves"],
                min_data_in_leaf=max(50, int(per_game * c["games_per_leaf"])),
                feature_fraction=c["ff"], bagging_fraction=0.8, bagging_freq=1,
                lambda_l1=1.0, lambda_l2=c["l2"], min_gain_to_split=1e-4,
                monotone_constraints=[1 if f in MONO_UP else 0 for f in FEATURES],
                verbose=-1, num_threads=0)


def feature_sets() -> dict[str, list[str]]:
    """무엇이 실제로 신호를 갖는지 확인하기 위한 특성 세트들."""
    from src.features import STATE_FEATURES, PREGAME_FEATURES
    base = list(STATE_FEATURES)
    score_core = ["score_diff", "score_x_progress", "score_per_out_left",
                  "outs_left_bat", "outs_left_field", "is_bottom"]
    baseout = ["base_state", "run_exp", "run_potential", "outs", "balls", "strikes",
               "is_walkoff_chance", "is_last_at_bat"]
    player = ["pitch_adv_home", "bat_adv_home", "fatigue_adv_home",
              "bullpen_adv_home", "bat_order"]
    box = ["hit_diff", "bb_diff", "err_diff"]
    return {
        # 5차: 좌우 매치업(플래툰)이 승률에 기여하는가.
        #   상대 전적(vsHra)은 홈팀만 제공돼 쓸 수 없었다.
        #   좌우는 양 팀 모두 있고 타자당 표본이 수백 타석이라 잡음이 훨씬 작다.
        "A 현재(14)":          base,
        "B +좌우매치업":        base + ["platoon_adv_home"],
        "C +좌우+시즌성적":      base + ["platoon_adv_home", "pitch_adv_home", "bat_adv_home"],
        "D +시즌성적만":        base + ["pitch_adv_home", "bat_adv_home"],
    }


def run_oof(data, feats: list[str], params_fn, splits, rounds: int):
    """주어진 특성 세트로 OOF 예측을 만든다."""
    import lightgbm as lgb
    import numpy as np
    params = params_fn(feats)
    oof = np.zeros(len(data))
    iters = []
    for i_tr, i_va in splits:
        f_tr, f_va = data.iloc[i_tr], data.iloc[i_va]
        dtr = lgb.Dataset(f_tr[feats], f_tr["home_win"])
        dva = lgb.Dataset(f_va[feats], f_va["home_win"], reference=dtr)
        b = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[i_va] = b.predict(f_va[feats], num_iteration=b.best_iteration)
        iters.append(b.best_iteration)
    return oof, int(np.median(iters))


def main() -> None:
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import log_loss, brier_score_loss

    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ablate", action="store_true", help="특성 세트 절제 실험")
    a = ap.parse_args()

    events, games = load_raw()
    re_tab = build_run_expectancy(normalize_states(events.copy()))
    data = build_features(events, games, re_tab)
    n_games = data["game_id"].nunique()
    per_game = len(data) / n_games
    print(f"[tune] {n_games}경기 / {len(data):,}행 (경기당 {per_game:.0f}행)")

    gkf = GroupKFold(n_splits=a.folds)
    splits = list(gkf.split(data[FEATURES], data["home_win"], groups=data["game_id"]))

    # ── 특성 세트 절제 ────────────────────────────────────────────────
    if a.ablate:
        def pf(feats):
            return dict(objective="binary", metric="binary_logloss",
                        learning_rate=0.02, num_leaves=63,
                        min_data_in_leaf=max(50, int(per_game)),
                        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                        lambda_l1=1.0, lambda_l2=10.0, min_gain_to_split=1e-4,
                        monotone_constraints=[1 if f in MONO_UP else 0 for f in feats],
                        verbose=-1, num_threads=0)

        arows = []
        for name, feats in feature_sets().items():
            feats = [f for f in feats if f in data.columns]
            oof, it = run_oof(data, feats, pf, splits, a.rounds)
            ll = log_loss(data["home_win"], np.clip(oof, 1e-6, 1 - 1e-6))
            br = brier_score_loss(data["home_win"], np.clip(oof, 1e-6, 1 - 1e-6))
            arows.append({"feature_set": name, "n_feat": len(feats),
                          "iter_median": it, "oof_log_loss": round(ll, 4),
                          "oof_brier": round(br, 4)})
            print(f"[ablate] {name:<16} 특성 {len(feats):>2}개 → OOF log loss {ll:.4f}")

        ares = pd.DataFrame(arows)
        base_ll = float(ares.loc[ares.feature_set == "A 현재(14)", "oof_log_loss"].iloc[0])
        ares["vs_현재"] = (ares["oof_log_loss"] - base_ll).round(4)
        ares = ares.sort_values("oof_log_loss").reset_index(drop=True)
        print()
        print(ares.to_string(index=False))
        md = ["# 특성 세트 절제 실험 (OOF log loss 기준)", "",
              f"- 대상 {n_games}경기 / {len(data):,}행 · {a.folds}-fold GroupKFold", "",
              ares.to_markdown(index=False), "",
              "> `vs_현재` 는 현재 19개 특성 세트 대비 차이. 음수면 더 좋다.", ""]
        (REPORT_DIR / "ablation.md").write_text("\n".join(md), encoding="utf-8")
        print("[ablate] 저장 → reports/ablation.md")
        return

    cands = grid(len(data), n_games)
    if a.quick:
        cands = cands[:3]

    rows = []
    for c in cands:
        params = to_params(c, per_game)
        oof = np.zeros(len(data))
        iters = []
        for i_tr, i_va in splits:
            f_tr, f_va = data.iloc[i_tr], data.iloc[i_va]
            dtr = lgb.Dataset(f_tr[FEATURES], f_tr["home_win"])
            dva = lgb.Dataset(f_va[FEATURES], f_va["home_win"], reference=dtr)
            b = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            oof[i_va] = b.predict(f_va[FEATURES], num_iteration=b.best_iteration)
            iters.append(b.best_iteration)
        ll = log_loss(data["home_win"], np.clip(oof, 1e-6, 1 - 1e-6))
        br = brier_score_loss(data["home_win"], np.clip(oof, 1e-6, 1 - 1e-6))
        rows.append({"config": c["name"], "leaves": c["leaves"],
                     "min_leaf": params["min_data_in_leaf"], "l2": c["l2"],
                     "lr": c["lr"], "ff": c["ff"],
                     "iter_median": int(np.median(iters)),
                     "oof_log_loss": round(ll, 4), "oof_brier": round(br, 4)})
        print(f"[tune] {c['name']:<12} leaves={c['leaves']:>3} min_leaf={params['min_data_in_leaf']:>4} "
              f"l2={c['l2']:>5} lr={c['lr']:<5} → OOF log loss {ll:.4f}")

    res = pd.DataFrame(rows).sort_values("oof_log_loss").reset_index(drop=True)
    print()
    print(res.to_string(index=False))
    best = res.iloc[0]
    print(f"\n[tune] 최적: {best['config']}  (OOF log loss {best['oof_log_loss']})")

    (MODEL_DIR / "tune_result.json").write_text(
        json.dumps(res.to_dict("records"), ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "tune.md").write_text(
        "# 하이퍼파라미터 탐색 (OOF log loss 기준)\n\n"
        f"- 대상 {n_games}경기 / {len(data):,}행 · {a.folds}-fold GroupKFold\n\n"
        + res.to_markdown(index=False)
        + "\n\n> 테스트 홀드아웃이 아니라 OOF 로 고르는 이유: 홀드아웃은 경기 수가 적어 "
          "분할 운에 따라 순위가 뒤집힌다.\n",
        encoding="utf-8")
    print("[tune] 저장 → models/tune_result.json · reports/tune.md")


if __name__ == "__main__":
    main()
