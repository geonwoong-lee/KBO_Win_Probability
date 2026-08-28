"""문자중계 원본 JSON → 투구 단위 tidy 이벤트 테이블.

핵심 관찰(실측으로 확인):
  * textRelays 는 이닝 안에서 '최신순'으로 내려온다 → 정렬 필요.
  * 정렬 키는 (textRelay.no, textOption.seqno). no 는 경기 전체에서 단조 증가하는
    타석 인덱스, seqno 는 이닝 내 이벤트 인덱스다.
  * 모든 textOption 에 currentGameState 가 붙어 있고, 이는 '해당 이벤트 직후'의 상태다.
  * base1/base2/base3 는 0 이면 비어 있고, 0 이 아니면 그 주자의 '타순 번호'가 들어간다.
  * homeOrAway == "1" 이면 홈팀 공격(말), "0" 이면 원정팀 공격(초).
  * textRelay.metricOption 에 **네이버 자체 승률**(타석 단위)이 들어 있다.
    우리 모델의 직접 비교 대상(벤치마크)으로 쓴다.
  * currentPlayersInfo 는 **이벤트마다 바뀌지 않는다.** 응답 전체에 고정된 한 쌍이다.
    실측: 1회초와 1회말이 똑같이 home:batter(hra=0.271) / away:pitcher(era=7.21) 였다.
    즉 여기서 뽑은 투수 ERA·타자 타율은 **경기 내 상수** — '경기 식별자'로 작동한다.
    선수 성적은 반드시 아래 두 곳에서 가져와야 한다.
      타자 시즌 타율 : batterRecord.seasonHra (타석 시작 행에만 있으므로 타석 안에서 이월)
      투수 시즌 ERA  : home/awayLineup.pitcher[] 의 pcode → seasonEra 조회표
      상대 전적      : batterRecord.vsHra — 이 타자의 '현재 상대 투수' 상대 타율.
                       0.000 은 대부분 '기록 없음'이라 결측으로 처리한다.
                       표본이 통산 8타수 수준이라(pitcherVsBatterCareerStats) 잡음이 크다.

이벤트 type 코드
   0 이닝 시작 | 1 투구 | 2 선수교체 | 7 기타(마운드 방문 등)
   8 타자 등장 | 13 타석 결과 | 14 주자 진루 | 23 아웃 처리 | 24 득점 | 99 구분선
"""
from __future__ import annotations

import pandas as pd

EV_INNING_START, EV_PITCH, EV_SUB, EV_ETC = 0, 1, 2, 7
EV_BATTER, EV_RESULT, EV_ADVANCE, EV_OUT, EV_SCORE, EV_SEP = 8, 13, 14, 23, 24, 99

PITCH_RESULT = {"B": "볼", "T": "스트라이크", "S": "헛스윙", "F": "파울", "H": "타격"}


def _i(v, default=0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _f(v, default=None):
    try:
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return default


def _pitcher_era_map(relay_innings: list[dict]) -> dict[str, float]:
    """pcode → 시즌 ERA. 라인업에 처음 등장했을 때의 값을 쓴다.

    마지막 이닝의 값을 쓰면 그 경기에서 얻어맞은 결과가 ERA에 반영돼
    '이 투수가 오늘 부진했다'는 미래 정보가 새어든다.
    """
    m: dict[str, float] = {}
    for data in relay_innings:
        for grp in ("homeLineup", "awayLineup"):
            for pit in ((data.get(grp) or {}).get("pitcher") or []):
                code, era = str(pit.get("pcode", "")), _f(pit.get("seasonEra"))
                if code and era is not None and code not in m:
                    m[code] = era
    return m


def _batter_hra_map(relay_innings: list[dict]) -> dict[str, float]:
    """pcode → 시즌 타율. batterRecord 가 없는 행의 보조 수단."""
    m: dict[str, float] = {}
    for data in relay_innings:
        for grp in ("homeLineup", "awayLineup"):
            for bat in ((data.get(grp) or {}).get("batter") or []):
                code, hra = str(bat.get("pcode", "")), _f(bat.get("seasonHra"))
                if code and hra is not None and code not in m:
                    m[code] = hra
    return m


def flatten_relay(relay_innings: list[dict], game_id: str) -> pd.DataFrame:
    """이닝별 textRelayData 리스트 → 시간순 이벤트 DataFrame."""
    era_map = _pitcher_era_map(relay_innings)
    hra_map = _batter_hra_map(relay_innings)

    rows: list[dict] = []
    for data in relay_innings:
        for relay in data.get("textRelays", []):
            for opt in relay.get("textOptions", []):
                rows.append({"_no": relay.get("no", 0), "_seq": opt.get("seqno", 0),
                             "inning": relay.get("inn"), "relay": relay, "opt": opt})
    if not rows:
        return pd.DataFrame()

    rows.sort(key=lambda r: (r["_no"], r["_seq"]))

    out: list[dict] = []
    pitch_count: dict[str, int] = {}
    used_pitchers: dict[str, set] = {"H": set(), "A": set()}
    cur_bat_order = {"H": 0, "A": 0}
    cur_bat_hra: dict[str, float | None] = {"H": None, "A": None}
    cur_vs_hra: dict[str, float | None] = {"H": None, "A": None}

    for idx, r in enumerate(rows):
        opt, relay = r["opt"], r["relay"]
        gs = opt.get("currentGameState") or {}
        etype = opt.get("type")
        hoa = str(relay.get("homeOrAway", "0"))
        is_bottom = hoa == "1"

        pit, bat = str(gs.get("pitcher", "")), str(gs.get("batter", ""))
        fielding = "A" if is_bottom else "H"
        batting = "H" if is_bottom else "A"

        if etype == EV_PITCH and pit:
            pitch_count[pit] = pitch_count.get(pit, 0) + 1
        if pit:
            used_pitchers[fielding].add(pit)

        brec = opt.get("batterRecord") or {}
        if brec.get("batOrder"):
            cur_bat_order[batting] = _i(brec["batOrder"])

        metric = relay.get("metricOption") or {}
        # 타자 시즌 타율: batterRecord 우선, 없으면 라인업 조회표
        if brec.get("seasonHra") is not None:
            cur_bat_hra[batting] = _f(brec.get("seasonHra"))
        elif bat and bat in hra_map:
            cur_bat_hra[batting] = hra_map[bat]
        # 상대 전적 타율. 0.000 은 '기록 없음'으로 본다.
        if brec:
            v = _f(brec.get("vsHra"))
            cur_vs_hra[batting] = v if (v is not None and v > 0.0) else None

        out.append({
            "game_id": game_id,
            "event_idx": idx,
            "pa_no": r["_no"],
            "seqno": r["_seq"],
            "inning": _i(r["inning"], 1),
            "is_bottom": int(is_bottom),
            "event_type": etype,
            "text": opt.get("text", ""),
            "home_score": _i(gs.get("homeScore")),
            "away_score": _i(gs.get("awayScore")),
            "outs": _i(gs.get("out")),
            "balls": _i(gs.get("ball")),
            "strikes": _i(gs.get("strike")),
            "on_1b": int(_i(gs.get("base1")) > 0),
            "on_2b": int(_i(gs.get("base2")) > 0),
            "on_3b": int(_i(gs.get("base3")) > 0),
            "home_hit": _i(gs.get("homeHit")),
            "away_hit": _i(gs.get("awayHit")),
            "home_bb": _i(gs.get("homeBallFour")),
            "away_bb": _i(gs.get("awayBallFour")),
            "home_err": _i(gs.get("homeError")),
            "away_err": _i(gs.get("awayError")),
            "pitcher_id": pit,
            "batter_id": bat,
            "bat_order": cur_bat_order[batting],
            "pitcher_pitches": pitch_count.get(pit, 0),
            "home_pitchers_used": len(used_pitchers["H"]),
            "away_pitchers_used": len(used_pitchers["A"]),
            "pitcher_season_era": era_map.get(pit),
            "batter_season_hra": cur_bat_hra[batting],
            "batter_vs_hra": cur_vs_hra[batting],
            "batter_hit_type": brec.get("hitType"),
            "pitch_speed": _f(opt.get("speed")),
            "pitch_type": opt.get("stuff"),
            "pitch_result": opt.get("pitchResult"),
            # 네이버 자체 승률 (벤치마크용, 0~100)
            # 결측 판별은 '값이 0이냐'가 아니라 **홈+원정 합이 100이냐**로 한다.
            #   정상  h=50,  a=50  → 합 100
            #   결측  h=0,   a=0   → 합 0
            #   홈 확정 h=100, a=0 → 합 100 (진짜 예측이므로 살려야 한다)
            # 0 을 전부 결측으로 보면 네이버가 확신하고 맞힌 행을 통째로 버려
            # 네이버 점수가 실제보다 나쁘게 나온다.
            "naver_wp_home": _f(metric.get("homeTeamWinRate")),
            "naver_wp_away": _f(metric.get("awayTeamWinRate")),
            "naver_wpa_plate": _f(metric.get("wpaByPlate")),
        })

    df = pd.DataFrame(out)
    df["half_over"] = (df["outs"] >= 3).astype(int)
    df["balls"] = df["balls"].clip(0, 3)
    df["strikes"] = df["strikes"].clip(0, 2)
    return df


def game_result(relay_innings: list[dict], schedule_row: dict | None = None) -> dict:
    """최종 스코어와 홈 승리 여부. 무승부는 home_win=None."""
    if schedule_row and schedule_row.get("statusCode") == "RESULT":
        hs, as_ = _i(schedule_row["homeTeamScore"]), _i(schedule_row["awayTeamScore"])
    else:
        gs = (relay_innings[-1].get("currentGameState") or {}) if relay_innings else {}
        hs, as_ = _i(gs.get("homeScore")), _i(gs.get("awayScore"))
    return {"home_score_final": hs, "away_score_final": as_,
            "home_win": 1 if hs > as_ else (0 if hs < as_ else None)}


def parse_preview(preview: dict | None) -> dict:
    """경기 전 사전 정보 → 평평한 특성 dict (0-0 상황의 사전확률용)."""
    if not preview:
        return {}
    o: dict = {}
    for side in ("home", "away"):
        st = preview.get(side + "Standings") or {}
        o[side + "_team_wra"] = _f(st.get("wra"))
        o[side + "_team_era"] = _f(st.get("era"))
        o[side + "_team_hra"] = _f(st.get("hra"))
        o[side + "_team_rank"] = _i(st.get("rank"), 0) or None
        sp = ((preview.get(side + "Starter") or {}).get("currentSeasonStats") or {})
        o[side + "_starter_era"] = _f(sp.get("era"))
        o[side + "_starter_whip"] = _f(sp.get("whip"))
    gi = preview.get("gameInfo") or {}
    o["stadium"] = gi.get("stadium")
    o["home_team"] = gi.get("hName")
    o["away_team"] = gi.get("aName")
    return o
