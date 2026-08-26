"""프로젝트 전역 설정."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
for _d in (DATA_RAW, DATA_PROC, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── 네이버 스포츠 비공식 API ───────────────────────────────────────────
API_BASE = "https://api-gw.sports.naver.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://m.sports.naver.com/",
    "Accept": "application/json",
}
REQUEST_DELAY = 0.4      # 초. 서버 부담을 줄이기 위한 최소 간격
MAX_RETRY = 3
TIMEOUT = 15

# ── 도메인 상수 ──────────────────────────────────────────────────────
TEAMS = {
    "HT": "KIA", "SS": "삼성", "LG": "LG", "OB": "두산", "WO": "키움",
    "LT": "롯데", "SK": "SSG", "NC": "NC", "KT": "KT", "HH": "한화",
}
REGULATION_INNINGS = 9
MAX_INNINGS = 15          # KBO 정규시즌 연장 상한(시즌별 상이) 대비 여유
SCORE_DIFF_CLIP = 10      # 점수차 특성 클리핑 범위

# 학습/검증/테스트 시즌 분리 (시간 기반 홀드아웃)
SEASON_SPLIT = {
    "train": [2021, 2022, 2023, 2024],
    "valid": [2025],
    "test":  [2026],
}
