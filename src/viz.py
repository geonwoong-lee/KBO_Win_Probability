"""승률 곡선 시각화 — 한 경기의 흐름을 한 장으로 보여준다.

x축을 '이벤트 번호'가 아니라 **회(초/말)** 로 읽히게 만드는 것이 핵심이다.
이벤트 번호는 야구적 의미가 없다 — 1회가 40칸, 5회가 15칸을 차지하는 식이라
곡선의 가로 길이가 '그 이닝이 얼마나 중요했나'와 무관하게 들쭉날쭉해진다.

  위 패널   투구 단위 승률 곡선 + 흐름 전환점(수직선)
            눈금과 격자를 반이닝 경계에 두고 1초/1말/2초… 로 라벨링한다.
            말(홈 공격)은 옅은 음영으로 구분한다.
  아래 패널 반이닝별 승률 변화량 막대
            막대의 **폭이 그 반이닝의 이벤트 수**라 이닝 길이가 그대로 보인다.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

from src.flow import half_inning_track, turning_points   # noqa: E402

# 윈도우 한글 폰트
for _f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"):
    if any(_f in f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

HOME_C, AWAY_C = "#C0413C", "#2C5AAD"
KIND_C = {
    "주도권 교체": "#8E44AD",
    "승부 기울어짐": "#C0413C",
    "승부 되살아남": "#1F7A5C",
    "균형 깨짐": "#D68910",
    "접전 복귀": "#1F7A5C",
}


def win_probability_chart(feats: pd.DataFrame, home: str, away: str,
                          out_path: Path, naver: bool = True) -> Path:
    """승률 곡선 + 반이닝 트랙 + 흐름 전환점."""
    d = feats.sort_values("event_idx").reset_index(drop=True)
    x = np.arange(len(d))
    wp = d["wp_home"].values * 100

    track = half_inning_track(d)
    tps = turning_points(d)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(14, 8.4), height_ratios=[2.6, 1], sharex=True,
        gridspec_kw={"hspace": 0.06}, layout="constrained")

    # ── 반이닝 경계와 음영 ──
    pos = {}                       # half_key → (시작 x, 폭)
    for r in track.itertuples():
        i0 = int(d.index[d["event_idx"] == r.start_idx][0])
        i1 = int(d.index[d["event_idx"] == r.end_idx][0])
        pos[r.half_key] = (i0, max(1, i1 - i0 + 1))
        if r.is_bottom:            # 말 = 홈 공격 → 옅은 음영
            for a in (ax, ax2):
                a.axvspan(i0, i1 + 1, color="#000000", alpha=.035, lw=0, zorder=0)
        ax.axvline(i0, color="#C8CCD2", lw=.7, zorder=1)

    ticks = [pos[r.half_key][0] + pos[r.half_key][1] / 2 for r in track.itertuples()]
    labels = [f"{r.inning}{'말' if r.is_bottom else '초'}" for r in track.itertuples()]

    # ── 위: 승률 곡선 ──
    ax.axhline(50, color="#888", lw=.9, ls="--", zorder=2)
    ax.fill_between(x, 50, wp, where=wp >= 50, color=HOME_C, alpha=.20, zorder=3)
    ax.fill_between(x, 50, wp, where=wp < 50, color=AWAY_C, alpha=.20, zorder=3)
    ax.plot(x, wp, color="#15181D", lw=1.8, zorder=5, label="우리 모델")

    if naver and "naver_wp_home" in d.columns:
        nv = d["naver_wp_home"].where(lambda s: (s > 0) & (s < 100))
        if nv.notna().sum() > 10:
            ax.plot(x, nv.ffill().values, color="#00A97F", lw=1.2, ls=":",
                    alpha=.9, zorder=4, label="네이버 자체 승률")

    # ── 흐름 전환점 ──
    for k, r in enumerate(tps.itertuples()):
        i = int(d.index[d["event_idx"] == r.event_idx][0])
        c = KIND_C.get(r.kind, "#555")
        ax.axvline(i, color=c, lw=1.6, ls="--", alpha=.85, zorder=6)
        ax.plot(i, r.wp_after * 100, "o", ms=7, mfc="white", mec=c, mew=2, zorder=8)
        # 그림 가장자리에서는 라벨을 안쪽으로 밀어 잘리지 않게 한다
        frac = i / max(len(d) - 1, 1)
        dx = -58 if frac > 0.88 else (58 if frac < 0.10 else 0)
        ha = "right" if dx < 0 else ("left" if dx > 0 else "center")
        dy = 30 if r.delta > 0 else -44
        if r.wp_after * 100 > 88:          # 위쪽에 붙으면 아래로
            dy = -46
        elif r.wp_after * 100 < 14:
            dy = 32
        ax.annotate(f"{r.kind}\n{r.delta*100:+.0f}%p",
                    xy=(i, r.wp_after * 100),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, ha=ha, zorder=9, color=c, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=c, lw=1.1),
                    arrowprops=dict(arrowstyle="-", lw=.8, color=c, alpha=.6)
                    if dx else None)

    ax.set_ylim(-14, 114)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(f"{home} 승리 확률 (%)", fontsize=10)
    fin = d.iloc[-1]
    ax.set_title(f"{away} {int(fin['away_score'])} : {int(fin['home_score'])} {home}"
                 f"   —   회별 승률 추이와 흐름 전환점", fontsize=13, pad=10)
    ax.legend(loc="lower left", fontsize=9, framealpha=.92)
    ax.grid(axis="y", alpha=.18)

    # ── 아래: 반이닝별 변화량 (막대 폭 = 그 반이닝의 이벤트 수) ──
    for r in track.itertuples():
        i0, w = pos[r.half_key]
        dv = r.delta * 100
        ax2.bar(i0, dv, width=w, align="edge",
                color=HOME_C if dv >= 0 else AWAY_C, alpha=.85, zorder=3)
        if abs(dv) >= 8:
            ax2.text(i0 + w / 2, dv + (2.5 if dv > 0 else -2.5), f"{dv:+.0f}",
                     ha="center", va="bottom" if dv > 0 else "top",
                     fontsize=8, zorder=4)
    ax2.axhline(0, color="#666", lw=.9)
    lo, hi = track["delta"].min() * 100, track["delta"].max() * 100
    ax2.set_ylim(min(lo * 1.30, -8), max(hi * 1.18, 8))   # 막대 위아래 라벨 자리
    ax2.set_ylabel("반이닝 변화 (%p)", fontsize=10)
    ax2.set_xlabel("회 (초 / 말) — 칸 너비는 그 반이닝의 이벤트 수", fontsize=9.5)
    ax2.grid(axis="y", alpha=.18)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_xlim(0, len(d))

    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
