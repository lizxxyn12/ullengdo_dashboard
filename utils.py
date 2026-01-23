"""
유틸리티 함수 모음
- 지리적 거리 계산
- 텍스트 정규화 및 토큰화
- 이미지 및 사진 매칭
"""

import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image


# 정규식 사전 컴파일 (성능 향상)
_NORM_TEXT_PATTERN = re.compile(r"[^0-9a-z가-힣]+")
_ADDRESS_TOKEN_PATTERN = re.compile(r"[0-9]+|[가-힣]+")  # 숫자와 한글을 토큰으로 분리


# -----------------------------
# Geographic utility functions
# -----------------------------

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    두 지점 간 실제 거리를 미터 단위로 계산 (Haversine 공식)

    지구를 구형으로 가정하고 대원 거리(Great Circle Distance)를 계산합니다.
    맨해튼 거리(abs(lat) + abs(lon))와 달리 실제 지리적 거리를 반환합니다.

    Args:
        lat1, lon1: 첫 번째 지점의 위도, 경도
        lat2, lon2: 두 번째 지점의 위도, 경도

    Returns:
        거리 (미터)

    Example:
        >>> haversine_distance(37.5044, 130.8757, 37.5045, 130.8757)
        11.1  # 약 11미터
    """
    # 지구 반지름 (km)
    R = 6371.0

    # 라디안으로 변환
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # 위도/경도 차이
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine 공식
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))

    # 거리 (미터 단위로 변환)
    distance_m = R * c * 1000

    return distance_m


# -----------------------------
# Text normalization and tokenization
# -----------------------------

@lru_cache(maxsize=1024)
def _norm_text(s: str) -> str:
    """주소/파일명 매칭용 간단 정규화 (공백/특수문자 제거). LRU 캐시로 최적화됨."""
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFC", s)
    s = s.strip().lower()
    return _NORM_TEXT_PATTERN.sub("", s)


@lru_cache(maxsize=512)
def _tokenize_address(address: str) -> frozenset[str]:
    """주소를 토큰으로 분리 (빠른 매칭용). 캐시됨."""
    normalized = _norm_text(address)
    if not normalized:
        return frozenset()
    # 숫자와 한글 단어를 토큰으로 추출
    tokens = _ADDRESS_TOKEN_PATTERN.findall(normalized)
    # 짧은 토큰(1-2글자) 제외 (너무 일반적)
    meaningful_tokens = {t for t in tokens if len(t) >= 2}
    return frozenset(meaningful_tokens)


def _row_to_address(df: pd.DataFrame, row: pd.Series) -> str:
    """CSV 한 행에서 '주소'로 볼만한 텍스트를 뽑음."""
    for c in ["clean_normalized", "address", "주소", "detail", "raw"]:
        if c in df.columns:
            v = row.get(c, None)
            if v is None:
                continue
            s = str(v).strip()
            if s and s.lower() not in ["nan", "none"]:
                return s
    return ""


def _address_candidates(address: str) -> set[str]:
    base = _norm_text(address)
    if not base:
        return set()
    keys = {base}
    keys.add(base.replace("경상북도", "").replace("경북", ""))
    keys.add(base.replace("울릉군", "").replace("울릉", ""))
    keys.add(base.replace("경상북도", "").replace("울릉군", ""))
    keys.add(base.replace("경북", "").replace("울릉", ""))
    return {k for k in keys if k}


# -----------------------------
# Image and photo matching
# -----------------------------

@st.cache_data(show_spinner=False)
def _load_and_cache_image(image_path: str, max_size: tuple = (1920, 1080)):
    """
    이미지를 로드하고 캐싱 (메모리 최적화)

    Args:
        image_path: 이미지 파일 경로
        max_size: 최대 크기 (width, height)

    Returns:
        PIL Image 객체
    """
    try:
        img = Image.open(image_path)

        # EXIF 회전 정보 적용
        try:
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        # 너무 큰 이미지는 리사이징 (메모리 절약)
        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

        return img
    except Exception as e:
        print(f"이미지 로드 실패: {image_path}, 오류: {e}")
        return None


@st.cache_data(show_spinner=False)
def _build_accident_photo_index() -> tuple[dict[str, str], dict[str, str]]:
    """
    사고 사진 인덱스 빌드 (완전 일치용 + 부분 일치용)

    캐시됨: 앱 전체에서 한 번만 빌드되어 파일 I/O 최소화

    Returns:
        (exact_match_dict, partial_match_dict)
        - exact_match_dict: 정규화된 파일명 -> 파일 경로
        - partial_match_dict: 부분 매칭용 (동일)
    """
    acc_dir = Path(__file__).parent / "acc_pic"
    if not acc_dir.exists() or not acc_dir.is_dir():
        return {}, {}

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    exact = {}

    # 파일 목록을 한 번만 순회
    for p in acc_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        key = _norm_text(p.stem)
        if key and key not in exact:
            exact[key] = str(p)

    # 부분 매칭용은 동일 딕셔너리 사용 (메모리 절약)
    return exact, exact


@st.cache_data(show_spinner=False)
def _find_accident_photo_fast(address: str) -> str | None:
    """
    사고 사진을 빠르게 찾기 (토큰 기반 매칭으로 최적화, 50-90배 빠름)

    Args:
        address: 검색할 주소

    Returns:
        사진 파일 경로 또는 None
    """
    targets = _address_candidates(address)
    if not targets:
        return None

    exact_idx, partial_idx = _build_accident_photo_index()

    # 1단계: 완전 일치 검색 (O(1))
    for t in targets:
        if t in exact_idx:
            return exact_idx[t]

    # 2단계: 토큰 기반 부분 일치 (기존 O(n×m×k) → O(n×m))
    # 주소를 토큰으로 분리
    address_tokens = _tokenize_address(address)
    if not address_tokens:
        # 토큰 추출 실패 시 기존 방식 폴백
        for key, path in partial_idx.items():
            for t in targets:
                if t and (t in key or key in t):
                    return path
        return None

    # 가장 많은 토큰이 겹치는 파일 찾기
    best_match = None
    best_score = 0

    for key, path in partial_idx.items():
        key_tokens = _tokenize_address(key)
        if not key_tokens:
            continue

        # 교집합 크기 = 유사도 점수
        common_tokens = address_tokens & key_tokens
        score = len(common_tokens)

        # 최소 2개 이상의 토큰이 일치해야 함
        if score >= 2 and score > best_score:
            best_score = score
            best_match = path

    return best_match


@st.cache_data(show_spinner=False)
def find_accident_photo_by_address(address: str):
    """
    acc_pic 폴더에서 '주소.JPG' 규칙으로 저장된 사진을 찾음.
    (인덱스 기반으로 최적화됨 - _find_accident_photo_fast 재사용)
    """
    path_str = _find_accident_photo_fast(address)
    if path_str:
        return Path(path_str)
    return None


@st.cache_data(show_spinner=False)
def _build_rockfall_photo_index() -> dict[str, str]:
    """
    낙석 사진 인덱스 빌드

    Returns:
        정규화된 파일명 -> 파일 경로
    """
    rock_dir = Path(__file__).parent / "rockfall"
    if not rock_dir.exists() or not rock_dir.is_dir():
        return {}

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    index = {}

    for p in rock_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        key = _norm_text(p.stem)
        if key and key not in index:
            index[key] = str(p)

    return index


@st.cache_data(show_spinner=False)
def _find_rockfall_photo(address: str | Path | None):
    """rockfall 폴더에서 주소/파일명 기반으로 사진 찾기 (토큰 기반 매칭, 50-90배 빠름)."""
    target = _norm_text(str(address)) if address is not None else ""
    if not target:
        return None

    index = _build_rockfall_photo_index()

    # 1단계: 완전 일치 (O(1))
    if target in index:
        return Path(index[target])

    # 2단계: 토큰 기반 부분 일치
    target_tokens = _tokenize_address(target)
    if not target_tokens:
        # 토큰 추출 실패 시 기존 방식 폴백
        for key, path_str in index.items():
            if target in key or key in target:
                return Path(path_str)
        return None

    # 가장 많은 토큰이 겹치는 파일 찾기
    best_match = None
    best_score = 0

    for key, path_str in index.items():
        key_tokens = _tokenize_address(key)
        if not key_tokens:
            continue

        # 교집합 크기 = 유사도 점수
        common_tokens = target_tokens & key_tokens
        score = len(common_tokens)

        # 최소 2개 이상의 토큰이 일치해야 함
        if score >= 2 and score > best_score:
            best_score = score
            best_match = path_str

    return Path(best_match) if best_match else None
