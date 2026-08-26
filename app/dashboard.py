"""실시간 승률 대시보드.

  streamlit run app/dashboard.py

경기를 고르면 승률 흐름 · 현재 국면 · 근거 · 만약에 · 승부처를 보여준다.
자동 갱신은 st.fragment 로 **화면 일부만** 다시 그린다.
브라우저를 통째로 새로고침하면(meta refresh) Streamlit 세션이 초기화돼
고른 경기가 매번 풀려버린다 — 실제로 겪은 문제다.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import MAX_INNINGS                                  # noqa: E402
from src import naver_api as api                                # noqa: E402
from src.parser import flatten_relay                            # noqa: E402
from src.features import build_features                         # noqa: E402
from src.explain import WinProbabilityExplainer                 # noqa: E402
from src.live import LiveGame                                   # noqa: E402
from src.flow import half_inning_track, turning_points          # noqa: E402

st.set_page_config(page_title="KBO 실시간 승률", page_icon="⚾", layout="wide")


@st.cache_resource
def get_explainer() -> WinProbabilityExplainer:
    return WinProbabilityExplainer()


@st.cache_data(ttl=60)
def get_games(d: str) -> list[dict]:
    return api.fetch_schedule(d, d)


@st.cache_data(ttl=20, show_spinner=False)
def compute(game_id: str) -> dict | None:
    exp = get_explainer()
    lg = LiveGame(game_id, exp)
    ev = flatten_relay(api.fetch_relay_full(game_id, MAX_INNINGS), game_id)
    if ev.empty:
        return None
    feats = exp.add_win_probability(build_features(ev, pd.DataFrame([lg.meta]), None))
    return {"meta": lg.meta, "feats": feats}


@st.cache_data(ttl=300)
def latest_date_with_relay(back: int = 8) -> date:
    """중계 데이터가 실제로 있는 가장 최근 날짜.

    '어제'로 잡으면 월요일(휴식일)에 빈 화면이 되고,
    '오늘'로 잡으면 경기 시작 전에는 볼 게 없다.
    """
    today = date.today()
    for d in range(0, back):
        cand = today - timedelta(days=d)
        gs = get_games(cand.strftime("%Y-%m-%d"))
        if any(g.get("statusCode") in ("RESULT", "STARTED", "LIVE") for g in gs):
            return cand
    return today


# ── 사이드바 ─────────────────────────────────────────────────────────
st.sidebar.header("경기 선택")
sel_date = st.sidebar.date_input("날짜", value=latest_date_with_relay())
games = get_games(sel_date.strftime("%Y-%m-%d"))
if not games:
    st.warning(f"{sel_date:%Y-%m-%d} 에는 KBO 경기가 없습니다. "
               "왼쪽 사이드바에서 다른 날짜를 선택하세요. (월요일은 대체로 휴식일입니다)")
    st.stop()

labels = {g["gameId"]: f"{g['awayTeamName']} @ {g['homeTeamName']}  [{g.get('statusInfo') or g['statusCode']}]"
          for g in games}
# key 를 주면 선택이 세션에 남아 자동 갱신 후에도 유지된다
gid = st.sidebar.selectbox("경기", list(labels), format_func=lambda k: labels[k],
                           key=f"game_{sel_date:%Y%m%d}")
auto = st.sidebar.toggle("30초마다 자동 갱신", value=False, key="auto_refresh",
                         help="진행 중인 경기에서만 의미가 있습니다. "
                              "화면 일부만 다시 그리므로 고른 경기는 그대로 유지됩니다.")

st.sidebar.divider()
with st.sidebar.expander("용어 설명", expanded=False):
    st.markdown(
        "**승률** — 지금 이 상황에서 그 팀이 최종 승리할 확률입니다.\n\n"
        "**%p (퍼센트포인트)** — 확률의 *차이*를 나타내는 단위입니다. "
        "40%에서 55%로 올랐다면 +15%p 입니다.\n\n"
        "**승률 변화** — 그 플레이 하나로 승률이 얼마나 움직였는지입니다. "
        "야구 통계에서 WPA라고 부릅니다.\n\n"
        "**중요도(레버리지)** — 지금 타석이 승부를 흔들 수 있는 폭이, "
        "평범한 타석의 몇 배인지입니다. 3배면 여기가 승부처라는 뜻입니다.")

try:
    get_explainer()
except Exception:
    st.error("학습된 모델이 없습니다. 먼저 `python -m src.collect` → `python -m src.train` 을 실행하세요.")
    st.stop()


# ── 본문 (자동 갱신 대상) ────────────────────────────────────────────
@st.fragment(run_every=30 if auto else None)
def live_panel(game_id: str) -> None:
    res = compute(game_id)
    if not res:
        st.info("아직 중계 데이터가 없습니다 (경기 전).")
        return

    meta, feats = res["meta"], res["feats"]
    exp = get_explainer()
    cur = feats.iloc[-1]
    info = exp.explain(cur, meta["home_team"], meta["away_team"])
    home, away = meta["home_team"], meta["away_team"]
    half = "말" if cur["is_bottom"] else "초"

    # ── 헤더 ──
    st.title(f"⚾ {away} @ {home}")
    if auto:
        st.caption("자동 갱신 켜짐 · 30초마다 이 화면만 다시 그립니다")
    c1, c2, c3, c4 = st.columns([1.3, 1.3, 1.6, 1.8])
    c1.metric(away, int(cur["away_score"]))
    c2.metric(home, int(cur["home_score"]))
    c3.metric("현재 국면", f"{int(cur['inning'])}회{half} {int(cur['outs'])}아웃")
    c4.metric(f"{home} 승률", f"{info['wp_home']*100:.1f}%",
              delta=f"{cur['wpa']*100:+.1f}%p",
              help="아래 숫자는 직전 플레이로 승률이 움직인 폭입니다.")

    st.progress(info["wp_home"])
    st.caption(f"← {away} {info['wp_away']*100:.1f}%     |     {home} {info['wp_home']*100:.1f}% →")
    st.info(f"**최근 상황**  {cur['text']}  \n{info['state']}")

    # ── 회별 승률 추이 ──
    st.subheader("회별 승률 추이")
    st.caption(f"1회초부터 지금까지 {home} 승률이 어떻게 움직였는지. "
               "50%보다 위면 홈팀이, 아래면 원정팀이 유리하다는 뜻입니다.")
    track = half_inning_track(feats)
    if not track.empty:
        # st.line_chart 는 문자열 인덱스를 가나다순으로 정렬해 '1회말'이 '1회초'보다
        # 앞에 온다. Altair 로 순서를 명시해야 x축이 뒤섞이지 않는다.
        import altair as alt
        cd = track[["half", "wp_end", "delta", "away_score", "home_score"]].copy()
        cd["승률"] = (cd["wp_end"] * 100).round(1)
        cd["승률 변화"] = (cd["delta"] * 100).round(1)
        cd["스코어"] = cd["away_score"].astype(str) + "-" + cd["home_score"].astype(str)
        order = track["half"].tolist()
        base = alt.Chart(cd).encode(
            x=alt.X("half:N", sort=order, title=None,
                    axis=alt.Axis(labelAngle=-45, labelFontSize=11)))
        line = base.mark_line(point=True, strokeWidth=2.5).encode(
            y=alt.Y("승률:Q", title=f"{home} 승률 (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=["half", "스코어", "승률", "승률 변화"])
        rule = alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(
            strokeDash=[5, 4], color="gray").encode(y="y:Q")
        st.altair_chart((rule + line).properties(height=260), use_container_width=True)

        tt = track.copy()
        tt["스코어"] = tt["away_score"].astype(str) + "-" + tt["home_score"].astype(str)
        tt["이 회 득점"] = tt["runs"].map(lambda v: f"{v}점" if v else "")
        tt[f"{home} 승률"] = (tt["wp_end"] * 100).map(lambda v: f"{v:.1f}%")
        tt["승률 변화"] = tt["delta"].map(lambda v: f"{v*100:+.1f}%p")
        tt["이 회 가장 큰 플레이"] = tt.apply(
            lambda r: f"{r['top_play']} ({r['top_wpa']*100:+.1f}%p)" if r["top_play"] else "", axis=1)
        st.table(tt[["half", "공격", "스코어", "이 회 득점", f"{home} 승률",
                     "승률 변화", "이 회 가장 큰 플레이"]]
                 .rename(columns={"half": "회"}).set_index("회"))
        st.caption(f"**승률 변화**는 그 반이닝 동안 {home} 승률이 움직인 폭입니다. "
                   "괄호 안 숫자도 같은 의미이며, +면 홈팀에게, −면 원정팀에게 유리해진 것입니다.")

    # ── 흐름이 바뀐 지점 ──
    st.subheader("흐름이 바뀐 지점")
    st.caption("승률이 크게 움직인 플레이와는 다릅니다. 8-1로 앞선 상황의 홈런은 "
               "점수는 벌리지만 이미 기운 승부를 바꾸지는 못합니다. "
               "여기서는 **누가 유리한지 자체가 바뀐 순간**만 골라냅니다.")
    tp = turning_points(feats)
    if tp.empty:
        st.info("아직 판세가 바뀔 만한 순간이 없었습니다 — 한쪽으로 흐르고 있는 경기입니다.")
    else:
        for r in tp.itertuples():
            c1, c2 = st.columns([1, 3.2])
            with c1:
                st.metric(r.kind, f"{r.delta*100:+.1f}%p",
                          help=f"{r.from_zone} → {r.to_zone} 로 판세가 넘어갔습니다.")
            with c2:
                st.markdown(
                    f"**{r.half}  {r.away_score}-{r.home_score}** &nbsp; "
                    f"`{r.from_zone} → {r.to_zone}`\n\n"
                    f"{home} 승률 **{r.wp_before*100:.1f}% → {r.wp_after*100:.1f}%**\n\n"
                    f"결정적이었던 플레이 — {r.trigger} "
                    f"(이 플레이로 {r.trigger_wpa*100:+.1f}%p)")
            st.divider()

    with st.expander("투구 단위 승률 곡선 (원자료)"):
        curve = feats[["event_idx", "wp_home"]].copy()
        curve[f"{home} 승률(%)"] = curve["wp_home"] * 100
        st.line_chart(curve.set_index("event_idx")[[f"{home} 승률(%)"]], height=240)

    # ── 근거 · 만약에 ──
    left, right = st.columns(2)

    with left:
        st.subheader("지금 승률을 만든 요인")
        st.caption("아무 정보도 없을 때의 기준 승률에서 출발해, 각 요인이 "
                   f"{home} 쪽으로 몇 %p씩 밀어올렸는지 나눈 값입니다. 모두 더하면 현재 승률이 됩니다.")
        sh = pd.DataFrame(info["shap"])
        sh["기여"] = sh["delta_pp"].map(lambda v: f"{v:+.1f}%p")
        sh["값"] = sh["value"].map(lambda v: f"{v:,.2f}")
        st.table(sh[["label", "값", "기여"]].rename(columns={"label": "요인"}).set_index("요인"))

    with right:
        st.subheader("만약에")
        st.caption("지금 상황에서 딱 하나만 바꿔보면 승률이 어떻게 달라지는지입니다. "
                   "변화가 클수록 그 조건이 지금 승부에 중요하다는 뜻입니다.")
        cf = pd.DataFrame(info["counterfactuals"])
        cf[f"{home} 승률"] = (cf["wp"] * 100).map(lambda v: f"{v:.1f}%")
        cf["변화"] = cf["delta_pp"].map(lambda v: f"{v:+.1f}%p")
        st.table(cf[["label", f"{home} 승률", "변화"]]
                 .rename(columns={"label": "가정"}).set_index("가정"))
        lev = info["leverage"]
        st.metric("이 타석의 중요도", f"평균의 {lev['leverage_index']}배",
                  help=f"이 타석 결과에 따라 승률이 평균 {lev['swing_pp']:.1f}%p 움직일 것으로 봅니다. "
                       "3배가 넘으면 승부처라 보고 불펜·대타를 고민할 자리입니다.")

    st.subheader("다음 타자가 어떻게 되느냐에 따라")
    st.caption(f"다음 타석 결과별로 {home} 승률이 어떻게 바뀌는지 미리 계산한 값입니다.")
    sc = pd.DataFrame(info["leverage"]["detail"])
    sc[f"{home} 승률"] = (sc["wp"] * 100).map(lambda v: f"{v:.1f}%")
    sc["변화"] = sc["delta_pp"].map(lambda v: f"{v:+.1f}%p")
    st.table(sc[["outcome", f"{home} 승률", "변화"]]
             .rename(columns={"outcome": "결과"}).set_index("결과"))

    # ── 승부처 ──
    st.subheader("승률을 가장 크게 흔든 플레이")
    st.caption("플레이 직후 승률이 얼마나 움직였는지 기준 상위입니다. "
               "야구 통계에서 WPA(Win Probability Added)라고 부르는 값입니다.")
    kp = exp.key_plays(feats, top=10).copy()
    kp["이닝"] = kp["inning"].astype(str) + kp["is_bottom"].map({1: "회말", 0: "회초"})
    kp["스코어"] = kp["away_score"].astype(str) + "-" + kp["home_score"].astype(str)
    kp[f"{home} 승률"] = (kp["wp_home"] * 100).map(lambda v: f"{v:.1f}%")
    kp["이 플레이로"] = (kp["wpa"] * 100).map(lambda v: f"{v:+.1f}%p")
    st.table(kp[["이닝", "스코어", "text", f"{home} 승률", "이 플레이로"]]
             .rename(columns={"text": "플레이"}).set_index("이닝"))

    with st.expander("모델 정보"):
        st.json({k: v for k, v in exp.meta.items() if k != "importance"})


live_panel(gid)
