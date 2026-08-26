"""네이버 스포츠 비공식 API 클라이언트.

확인된 엔드포인트 (2026-08 기준, 모두 GET / 인증 불필요):
  /schedule/games?upperCategoryId=kbaseball&fromDate=&toDate=   경기 일정·결과
  /schedule/games/{gameId}/relay?inning=N                        이닝별 문자중계(투구 단위)
  /schedule/games/{gameId}/preview                               선발·라인업·팀순위(경기 전)
  /schedule/games/{gameId}/record                                박스스코어

gameId 형식: YYYYMMDD + 원정팀코드 + 홈팀코드 + 더블헤더플래그 + 시즌
             예) 20260823HTWO02026  → 8/23 KIA(원정) @ 키움(홈)
"""
from __future__ import annotations

import time
import logging
import threading
from typing import Any

import requests
from requests.adapters import HTTPAdapter

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import API_BASE, HEADERS, REQUEST_DELAY, MAX_RETRY, TIMEOUT

log = logging.getLogger(__name__)


class TransientError(RuntimeError):
    """네트워크 문제로 응답을 못 받았다. '데이터 없음'과 반드시 구분해야 한다.

    구분하지 않으면 DNS 블립 한 번에 그 이닝이 통째로 빠진 채
    경기가 조용히 잘린다 — 6회가 사라진 경기가 정상 경기로 저장된다.
    """
_session = requests.Session()
_session.headers.update(HEADERS)
_session.mount("https://", HTTPAdapter(pool_connections=8, pool_maxsize=8))

# 전역 레이트리밋. 워커를 늘려도 '초당 총 요청 수'는 REQUEST_DELAY 로 고정된다.
# 병목은 요청당 왕복 지연(약 1.4초)이므로, 총량을 묶어둔 채 대기만 겹쳐도 크게 빨라진다.
_rate_lock = threading.Lock()
_last_call = [0.0]


def _reserve_slot() -> None:
    """다음 요청 시각을 예약하고, 그 시각까지 기다린다."""
    with _rate_lock:
        now = time.time()
        start = max(now, _last_call[0] + REQUEST_DELAY)
        _last_call[0] = start
    wait = start - time.time()
    if wait > 0:
        time.sleep(wait)


def _get(path: str, **params) -> dict[str, Any] | None:
    """레이트리밋을 지키며 GET 요청. 실패 시 None."""
    for attempt in range(MAX_RETRY):
        _reserve_slot()
        try:
            r = _session.get(f"{API_BASE}{path}", params=params, timeout=TIMEOUT)
            if r.status_code == 404:
                return None                      # 존재하지 않는 경기
            r.raise_for_status()
            js = r.json()
            if not js.get("success"):
                return None
            return js.get("result")
        except Exception as e:                   # noqa: BLE001
            log.warning("GET %s 실패(%d/%d): %s", path, attempt + 1, MAX_RETRY, e)
            time.sleep(1.5 * (attempt + 1))
    raise TransientError(path)


# ── 공개 함수 ────────────────────────────────────────────────────────
def fetch_schedule(from_date: str, to_date: str) -> list[dict]:
    """기간 내 KBO 정규리그 경기 목록. 날짜는 'YYYY-MM-DD'."""
    res = _get(
        "/schedule/games",
        fields="basic,superCategoryId,categoryName,stadium,statusNum",
        upperCategoryId="kbaseball",
        fromDate=from_date,
        toDate=to_date,
        size=500,
    )
    if not res:
        return []
    return [g for g in res.get("games", []) if g.get("categoryId") == "kbo"]


def fetch_relay_inning(game_id: str, inning: int) -> dict | None:
    """한 이닝의 문자중계 원본(textRelayData)."""
    res = _get(f"/schedule/games/{game_id}/relay", inning=inning)
    return res.get("textRelayData") if res else None


def fetch_relay_full(game_id: str, max_inning: int = 15) -> list[dict]:
    """1회부터 데이터가 끊길 때까지 전 이닝 문자중계를 모아 반환.

    전송 실패는 TransientError 로 올려보낸다. 여기서 삼키면 중간 이닝이 빠진
    반쪽 경기가 정상인 척 저장된다.
    """
    out, misses = [], 0
    for inn in range(1, max_inning + 1):
        data = fetch_relay_inning(game_id, inn)      # 실패 시 TransientError
        if not data or not data.get("textRelays"):
            misses += 1
            if misses >= 2:                      # 연속 2이닝 비면 종료
                break
            continue
        misses = 0
        out.append(data)
    return out


def fetch_preview(game_id: str) -> dict | None:
    """선발투수·라인업·팀 순위 등 경기 전 정보."""
    res = _get(f"/schedule/games/{game_id}/preview")
    return res.get("previewData") if res else None


def fetch_record(game_id: str) -> dict | None:
    """최종 박스스코어."""
    return _get(f"/schedule/games/{game_id}/record")


def fetch_game_status(game_id: str) -> dict | None:
    """실시간 폴링용: 현재 상태 + 마지막 이닝 중계만 가볍게 조회."""
    date = f"{game_id[:4]}-{game_id[4:6]}-{game_id[6:8]}"
    for g in fetch_schedule(date, date):
        if g["gameId"] == game_id:
            return g
    return None
