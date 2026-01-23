"""
데이터 로딩 관련 함수들

이 모듈은 app.py에서 데이터를 로드하는 함수들을 포함합니다.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import re
import math
import unicodedata
from functools import lru_cache

# utils.py에서 필요한 함수들 import
from utils import (
    _norm_text,
    _tokenize_address,
    _address_candidates,
    _row_to_address,
    _build_accident_photo_index,
    _find_accident_photo_fast,
    find_accident_photo_by_address,
    _build_rockfall_photo_index,
    _find_rockfall_photo,
)

# -----------------------------
# SMS 키워드 상수 (app.py에서 가져옴)
# -----------------------------
SMS_SHIP_KEYWORDS = [
    "금광해운",
    "대저해운",
    "대저해운 도착시간",
    "에이치해운",
    "미래해운",
    "우성해운",
    "주식회사태성해운",
    "태성해운 도착시간",
    "한국해운",
]

SMS_SHIP_VESSEL_KEYWORDS = [
    "금광11호",
    "미래15호",
]

SMS_PEOPLE_KEYWORDS = [
    "대저페리",
    "썬라이즈 도착시간",
    "씨스포빌",
    "씨스포빌 도착시간",
    "울릉크루즈",
    "제이에이치페리",
    "제이에이치페리 도착시간",
]

SMS_PEOPLE_VESSEL_KEYWORDS = [
    "씨스타11호",
    "씨스타1호",
    "씨스타5호",
    "뉴씨다오펄호",
    "뉴시다오펄호",
    "썬라이즈호",
    "퀸스타2호",
]

SMS_PASSENGER_KEYWORDS = ["탑승인원", "여객", "승객", "승선", "크루즈"]
SMS_CARGO_KEYWORDS = ["화물", "차량", "선적", "택배", "물류"]

SMS_CANCEL_KEYWORDS = ["결항", "취소", "출항 취소", "운항 취소"]
SMS_CONTROL_KEYWORDS = ["운항 통제", "운항통제", "운항이 통제", "통제되었습니다"]
SMS_CHANGE_KEYWORDS = ["시간 변경", "시간변경", "시간 변경된", "시간변경된"]

SMS_ARRIVE_KEYWORDS = [
    "입항",
    "입항 예정",
    "입항 예정시간",
    "입항입니다",
    "도착",
    "도착시간",
]

SMS_DEPART_KEYWORDS = [
    "출항",
    "출발",
    "운항예정",
    "운항 예정",
    "정상운항",
    "운항합니다",
    "출항 예정",
    "정상출항",
    "출항합니다",
    "출발합니다",
]

SMS_ARRIVE_ROUTE_PATTERNS = [
    r"포항.*?(→|->|➡|>).*?울릉",
    r"포항\(영일만항\).*?→.*?울릉\(사동항\)",
]

SMS_DEPART_ROUTE_PATTERNS = [
    r"울릉.*?(→|->|➡|>).*?포항",
    r"울릉\(사동항\).*?→.*?포항\(영일만항\)",
]

# 텍스트 정규화 관련 함수는 utils.py에서 import
from utils import (
    _norm_text,
    _tokenize_address,
    _address_candidates,
    _build_accident_photo_index,
    _find_accident_photo_fast,
    find_accident_photo_by_address,
    _build_rockfall_photo_index,
    _find_rockfall_photo,
)


# -----------------------------
# 데이터 로딩 함수들
# -----------------------------

def _accident_files_signature() -> tuple:
    """사고 CSV 변경 감지를 위한 시그니처."""
    data_dir = Path(__file__).parent
    sig_items = []
    for f in data_dir.iterdir():
        if not f.is_file():
            continue
        name = unicodedata.normalize("NFC", f.name)
        if not name.endswith(".csv"):
            continue
        if "교통계" in name and "교통사고" in name and "년도" in name:
            stat = f.stat()
            sig_items.append((name, stat.st_mtime, stat.st_size))
            continue
        if re.search(r"ulleung_accidents_with_coords_20\d{2}\.csv", name):
            stat = f.stat()
            sig_items.append((name, stat.st_mtime, stat.st_size))
    fallback = data_dir / "ulleung_accidents_with_coords.csv"
    if fallback.exists():
        stat = fallback.stat()
        sig_items.append((fallback.name, stat.st_mtime, stat.st_size))
    return tuple(sorted(sig_items))


@st.cache_data(show_spinner=False)
def load_accidents_csv(file_signature: tuple | None = None) -> pd.DataFrame:
    """사고 좌표 CSV를 로드(연도별 파일 우선)."""

    def _read_csv_safely(path: Path) -> pd.DataFrame:
        for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue
        return pd.read_csv(path)

    def _parse_year_from_name(name: str):
        m = re.search(r"(20\d{2})년도", name)
        if not m:
            m = re.search(r"ulleung_accidents_with_coords_(20\d{2})\.csv", name)
            if not m:
                return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    data_dir = Path(__file__).parent
    year_files = []
    year_with_coords = []
    for f in data_dir.iterdir():
        if not f.is_file():
            continue
        name = unicodedata.normalize("NFC", f.name)
        if not name.endswith(".csv"):
            continue
        if "교통계" in name and "교통사고" in name and "년도" in name:
            year_files.append(f)
            if name.endswith("_with_coords.csv"):
                year_with_coords.append(f)
            continue
        if re.search(r"ulleung_accidents_with_coords_20\d{2}\.csv", name):
            year_with_coords.append(f)

    target_files = year_with_coords if year_with_coords else year_files
    df_list = []
    for f in sorted(target_files):
        name = unicodedata.normalize("NFC", f.name)
        year = _parse_year_from_name(name)
        if year is None:
            continue

        temp = _read_csv_safely(f)
        temp.columns = [str(c).strip() for c in temp.columns]

        lat_col = next(
            (c for c in ["latitude", "Latitude", "lat", "위도"] if c in temp.columns),
            None,
        )
        lon_col = next(
            (c for c in ["longitude", "Longitude", "lon", "경도"] if c in temp.columns),
            None,
        )
        if not lat_col or not lon_col:
            continue

        addr_col = next(
            (c for c in temp.columns if "사고" in c and "장소" in c),
            None,
        )
        type_col = next(
            (
                c
                for c in temp.columns
                if ("종별" in c)
                or (c in ["type", "accident_type", "사고유형", "사고_type"])
            ),
            None,
        )

        temp["latitude"] = pd.to_numeric(temp[lat_col], errors="coerce")
        temp["longitude"] = pd.to_numeric(temp[lon_col], errors="coerce")
        temp = temp.dropna(subset=["latitude", "longitude"])

        if addr_col:
            temp["raw"] = temp[addr_col].astype(str)
            temp["detail"] = temp[addr_col].astype(str)
        if type_col:
            temp["type"] = temp[type_col].astype(str)
        temp["year"] = year

        cols = [
            c
            for c in [
                "clean_normalized",
                "raw",
                "detail",
                "latitude",
                "longitude",
                "type",
                "year",
            ]
            if c in temp.columns
        ]
        if cols:
            df_list.append(temp[cols])

    if df_list:
        return pd.concat(df_list, ignore_index=True)

    csv_path = data_dir / "ulleung_accidents_with_coords.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    df = _read_csv_safely(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return pd.DataFrame()

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df["year"] = 2025
    return df


@st.cache_data(show_spinner=False)
def load_ev_charger_points() -> list[tuple[float, float, str]]:
    """울릉군 전기차 충전소 좌표 로드 (마커용)."""
    points, _ = load_ev_charger_data()
    return points


@st.cache_data(show_spinner=False)
def load_ev_charger_data() -> tuple[list[tuple[float, float, str]], list[dict]]:
    """울릉군 전기차 충전소 좌표 및 메타데이터 로드."""
    csv_path = Path(__file__).parent / "울릉군 전기차 충전소 2020-07-13.csv"
    if not csv_path.exists():
        return [], []

    def _read_csv_safely(path: Path):
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            try:
                return pd.read_csv(path, encoding="utf-8")
            except Exception:
                return pd.read_csv(path)

    def _clean_text(val) -> str:
        if val is None:
            return ""
        s = str(val).strip()
        if not s or s.lower() in ["nan", "none"]:
            return ""
        return s

    def _first_text(*vals: str) -> str:
        for v in vals:
            if v:
                return v
        return ""

    df = _read_csv_safely(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    lat_col = next(
        (c for c in ["위도", "latitude", "Latitude", "lat"] if c in df.columns), None
    )
    lon_col = next(
        (c for c in ["경도", "longitude", "Longitude", "lon"] if c in df.columns), None
    )
    if not lat_col or not lon_col:
        return [], []

    df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    # 완전 벡터화된 텍스트 정리 (apply 제거)
    for col_name, target_col in [
        ("충전소명", "name_clean"),
        ("충전소위치상세", "detail_clean"),
        ("소재지도로명주소", "road_addr_clean"),
        ("소재지지번주소", "lot_addr_clean")
    ]:
        if col_name in df.columns:
            # str 변환 → strip → nan/none 제거
            df[target_col] = (
                df[col_name].astype(str).str.strip()
                .replace(["nan", "none", "None", ""], "")
            )
        else:
            df[target_col] = ""

    # 주소 우선순위 선택
    def select_address(row):
        return _first_text(row["road_addr_clean"], row["lot_addr_clean"], row["detail_clean"])

    df["address"] = df.apply(select_address, axis=1)
    # 벡터화된 조건 처리 (fillna 사용)
    df["label_name"] = df["name_clean"].fillna("충전소").replace("", "충전소")
    df["label_addr"] = df["address"].fillna("주소 미상").replace("", "주소 미상")
    df["label"] = "충전소 : " + df["label_name"] + "<br/>주소 : " + df["label_addr"]

    # 결과를 리스트로 변환
    points = list(zip(df["lat"], df["lon"], df["label"]))

    # 메타데이터 생성
    meta_list = []
    for i, row in df.iterrows():
        meta = {
            "idx": int(i) if isinstance(i, (int, float)) else len(meta_list),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "name": row["label_name"],
            "address": row["label_addr"],
            "detail": _clean_text(row.get("충전소위치상세", "")),
            "slow_charger": _clean_text(row.get("완속충전기대수", "0")),
            "fast_charger": _clean_text(row.get("급속충전기대수", "0")),
            "slow_available": _clean_text(row.get("완속충전가능여부", "")),
            "fast_available": _clean_text(row.get("급속충전가능여부", "")),
            "fast_type": _clean_text(row.get("급속충전타입구분", "")),
            "open_time": _clean_text(row.get("이용가능시작시각", "")),
            "close_time": _clean_text(row.get("이용가능종료시각", "")),
            "parking_fee": _clean_text(row.get("주차료부과여부", "")),
            "operator": _clean_text(row.get("관리업체명", "")),
            "phone": _clean_text(row.get("관리업체전화번호", "")),
        }
        meta_list.append(meta)

    return points, meta_list


@st.cache_data(show_spinner=False)
def load_rockfall_points() -> tuple[list[tuple[float, float, str]], list[dict]]:
    """rockfall 폴더 사진명(주소) 기반으로 좌표 매칭."""
    rock_dir = Path(__file__).parent / "rockfall"
    if not rock_dir.exists():
        return [], []

    coords_final_path = Path(__file__).parent / "rockfall_coords_final.csv"

    def _read_csv_safely(path: Path):
        try:
            return pd.read_csv(path, encoding="utf-8")
        except Exception:
            try:
                return pd.read_csv(path, encoding="utf-8-sig")
            except Exception:
                return pd.read_csv(path)

    def _build_from_coords_df(df_coords: pd.DataFrame):
        if df_coords.empty:
            return [], []

        # 컬럼명 정규화
        df_coords.columns = [str(c).strip() for c in df_coords.columns]

        lat_col = next(
            (
                c
                for c in ["latitude", "Latitude", "lat", "위도"]
                if c in df_coords.columns
            ),
            None,
        )
        lon_col = next(
            (
                c
                for c in ["longitude", "Longitude", "lon", "경도"]
                if c in df_coords.columns
            ),
            None,
        )
        if not lat_col or not lon_col:
            return [], []

        address_cols = [
            c
            for c in ["실제 주소", "address", "주소", "장소", "filename"]
            if c in df_coords.columns
        ]

        # Vectorized 처리
        # 좌표 변환
        df_coords["lat_num"] = pd.to_numeric(df_coords[lat_col], errors="coerce")
        df_coords["lon_num"] = pd.to_numeric(df_coords[lon_col], errors="coerce")

        # 유효한 좌표만 필터링
        df_valid = df_coords.dropna(subset=["lat_num", "lon_num"])

        if df_valid.empty:
            return [], []

        # 주소 추출 (vectorized)
        def get_address(row):
            for c in address_cols:
                v = row.get(c, None)
                if v is not None:
                    s = str(v).strip()
                    if s:
                        return s
            return ""

        df_valid["address"] = df_valid.apply(get_address, axis=1)
        # 벡터화된 조건 처리
        df_valid["label_text"] = df_valid["address"].fillna("위치 미상").replace("", "위치 미상")

        # 포인트 및 메타 생성
        points = []
        meta = []

        for idx, (lat, lon, label, address) in enumerate(zip(
            df_valid["lat_num"],
            df_valid["lon_num"],
            df_valid["label_text"],
            df_valid["address"]
        )):
            photo = _find_rockfall_photo(address) if address else None
            if photo is None and "filename" in df_valid.columns:
                filename = df_valid.iloc[idx].get("filename", "")
                if filename:
                    photo = _find_rockfall_photo(filename)

            points.append((float(lat), float(lon), f"낙석 발생 위치 : {label}"))

            row_data = df_valid.iloc[idx]
            meta.append({
                "idx": int(idx),
                "lat": float(lat),
                "lon": float(lon),
                "photo": str(photo) if photo else None,
                "name": str(label),
                "date": row_data.get("사고일자", None),
                "damage": row_data.get("피해여부", None),
            })

        return points, meta

    # rockfall_coords_final.csv만 사용(최신 좌표/주소)
    if coords_final_path.exists():
        points, meta = _build_from_coords_df(_read_csv_safely(coords_final_path))
        if points:
            return points, meta
    return [], []


@st.cache_data(show_spinner=False)
def load_bus_stops_csv() -> pd.DataFrame:
    """버스 정류장 CSV 로드."""
    csv_path = Path(__file__).parent / "ullengdo_bus_stops.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

    df.columns = [str(c).strip() for c in df.columns]

    lat_col = next(
        (c for c in ["위도", "latitude", "Latitude"] if c in df.columns), None
    )
    lon_col = next(
        (c for c in ["경도", "longitude", "Longitude"] if c in df.columns), None
    )
    name_col = next(
        (c for c in ["정류장명", "name", "정류장"] if c in df.columns), None
    )
    if not (lat_col and lon_col and name_col):
        return pd.DataFrame()

    df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    df["stop_name"] = df[name_col].astype(str)
    df["stop_norm"] = df["stop_name"].apply(_norm_text)
    df = df.dropna(subset=["lat", "lon"])
    return df[["stop_name", "stop_norm", "lat", "lon"]]


def _match_bus_stop(df: pd.DataFrame, name: str):
    """정류장 이름으로 좌표 매칭 (vectorized 최적화)."""
    if df.empty:
        return None
    target = _norm_text(name)
    if not target:
        return None

    # 정확히 일치하는 정류장 찾기
    exact = df[df["stop_norm"] == target]
    if not exact.empty:
        row = exact.iloc[0]
        return float(row["lat"]), float(row["lon"])

    # 부분 문자열 매칭 (vectorized): target이 stop_norm에 포함되는 경우
    mask1 = df["stop_norm"].str.contains(target, na=False, regex=False)
    matches = df[mask1]
    if not matches.empty:
        row = matches.iloc[0]
        return float(row["lat"]), float(row["lon"])

    # 반대: stop_norm이 target에 포함되는 경우
    for idx in df.index:
        norm = df.at[idx, "stop_norm"]
        if norm and norm in target:
            return float(df.at[idx, "lat"]), float(df.at[idx, "lon"])

    return None


def _bus_route_defs():
    """PDF에서 읽은 노선 요약(수동 정의)."""
    return [
        {
            "id": "1",
            "name": "1노선 (도동→사동 방면 섬일주)",
            "color": "#d94f5c",
            "stops": [
                "울릉군도동정류소",
                "사동항",
                "남양",
                "태하삼거리",
                "현포",
                "천부정류장",
                "관음도",
                "저동여객선터미널",
                "울릉군도동정류소",
            ],
        },
        {
            "id": "2",
            "name": "2노선 (도동→저동 방면 섬일주)",
            "color": "#4f8bd9",
            "stops": [
                "울릉군도동정류소",
                "저동여객선터미널",
                "관음도",
                "천부정류장",
                "현포",
                "태하삼거리",
                "남양",
                "사동항",
                "울릉군도동정류소",
            ],
        },
        {
            "id": "3",
            "name": "3노선 (도동↔저동↔봉래폭포)",
            "color": "#8b5cd9",
            "stops": [
                "울릉군도동정류소",
                "저동",
                "봉래폭포",
                "저동",
                "울릉군도동정류소",
            ],
        },
        {
            "id": "4",
            "name": "4노선 (천부↔나리분지)",
            "color": "#22a979",
            "stops": [
                "천부정류장",
                "나리",
                "천부정류장",
            ],
        },
        {
            "id": "5",
            "name": "5노선 (사동항↔도동↔저동↔관음도↔석포)",
            "color": "#d9a54f",
            "stops": [
                "사동항",
                "울릉군도동정류소",
                "저동여객선터미널",
                "관음도",
                "석포전망대입구",
            ],
        },
        {
            "id": "11",
            "name": "11노선 (천부→관음도→저동약국→도동→사동항→남양→태하→현포→천부)",
            "color": "#ef7fb0",
            "stops": [
                "천부정류장",
                "관음도",
                "저동약국",
                "울릉군도동정류소",
                "사동항",
                "남양",
                "태하",
                "현포",
                "천부정류장",
            ],
        },
        {
            "id": "22",
            "name": "22노선 (천부→현포→태하→남양→사동항→도동→저동여객선터미널→관음도→천부)",
            "color": "#2a9d8f",
            "stops": [
                "천부정류장",
                "현포",
                "태하",
                "남양",
                "사동항",
                "울릉군도동정류소",
                "저동여객선터미널",
                "관음도",
                "천부정류장",
            ],
        },
    ]


def _polyline_segments(points: list[tuple[float, float]]):
    segments = []
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(points, points[1:]):
        seg_len = math.hypot(lat2 - lat1, lon2 - lon1)
        segments.append((seg_len, (lat1, lon1), (lat2, lon2)))
        total += seg_len
    return total, segments


def _point_on_segments(segments, distance: float):
    remaining = distance
    for seg_len, (lat1, lon1), (lat2, lon2) in segments:
        if seg_len <= 0:
            continue
        if remaining <= seg_len:
            t = remaining / seg_len
            return (lat1 + (lat2 - lat1) * t, lon1 + (lon2 - lon1) * t)
        remaining -= seg_len
    if segments:
        return segments[-1][2]
    return None


def _simulate_bus_positions(routes, per_route: int = 2):
    positions = []
    for route in routes:
        points = route.get("points", [])
        if len(points) < 2:
            continue
        total, segments = _polyline_segments(points)
        if total <= 0:
            continue
        route_id = str(route.get("id", "")).strip()
        jitter = (sum(ord(c) for c in route_id) % 7) * 0.01
        for i in range(per_route):
            frac = (i + 1) / (per_route + 1) + jitter
            frac = frac % 1.0
            distance = total * frac
            point = _point_on_segments(segments, distance)
            if point is None:
                continue
            lat, lon = point
            positions.append(
                {
                    "route_id": route_id,
                    "route_name": route.get("name", ""),
                    "lat": lat,
                    "lon": lon,
                    "index": i + 1,
                }
            )
    return positions


@st.cache_data(show_spinner=False)
def build_bus_routes():
    """노선 정의를 좌표와 함께 반환."""
    df = load_bus_stops_csv()
    routes = []

    # 모든 정류장을 기본으로 포함
    stop_map: dict[str, dict] = {}
    if not df.empty:
        # 정규화된 이름 컬럼 추가
        df_temp = df.assign(key=df["stop_name"].apply(_norm_text))

        # to_dict()로 한 번에 변환
        for row in df_temp.to_dict('records'):
            key = row["key"]
            stop_map[key] = {
                "name": row["stop_name"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "routes": [],
            }

    # 노선 정의에 포함된 정류장에 경유 노선 정보 채우기 + 라인 포인트 생성
    for route in _bus_route_defs():
        pts = []
        loop_routes = {"1", "2", "5", "11", "22"}

        if route["id"] in loop_routes and not df.empty:
            # 섬 일주/왕복 노선은 모든 정류장을 각도 기준으로 정렬해 선을 그린다.
            center_lat = df["lat"].mean()
            center_lon = df["lon"].mean()

            # 완전 벡터화된 각도 계산
            df_sorted = df.assign(
                ang=np.arctan2(
                    df["lat"].values - center_lat,
                    df["lon"].values - center_lon
                )
            ).sort_values("ang").reset_index(drop=True)

            # 시작점을 앵커 정류장 근처로 회전
            anchor_match = _match_bus_stop(df, route["stops"][0])
            start_idx = 0
            if anchor_match:
                ax, ay = anchor_match
                # 맨해튼 거리 계산
                df_sorted["distance"] = (df_sorted["lat"] - ax).abs() + (df_sorted["lon"] - ay).abs()
                # 가장 가까운 정류장의 인덱스 찾기
                start_idx = df_sorted["distance"].idxmin()
                # 임시 컬럼 제거
                df_sorted = df_sorted.drop(columns=["distance"])
            rotated = pd.concat(
                [df_sorted.iloc[start_idx:], df_sorted.iloc[:start_idx]]
            )
            pts = [
                (float(r.lat), float(r.lon))
                for r in rotated[["lat", "lon"]].itertuples()
            ]

            # 모든 정류장을 이 노선 경유로 표시
            for key, info in stop_map.items():
                info["routes"].append(route["name"])

        else:
            # 정의된 정류장 순서대로만 연결
            for stop_name in route["stops"]:
                match = _match_bus_stop(df, stop_name)
                if match:
                    lat, lon = match
                    pts.append((lat, lon))
                    key = _norm_text(stop_name)
                    if key not in stop_map:
                        stop_map[key] = {
                            "name": stop_name,
                            "lat": lat,
                            "lon": lon,
                            "routes": [],
                        }
                    stop_map[key]["routes"].append(route["name"])
        routes.append(
            {
                "id": route["id"],
                "name": route["name"],
                "color": route["color"],
                "points": pts,
            }
        )

    stops = list(stop_map.values())
    return routes, stops


@st.cache_data(show_spinner=False)
def load_enforcement_counts_csv() -> pd.DataFrame:
    """여러 해의 교통단속 CSV를 로드."""
    data_dir = Path(__file__).parent / "enforcement_data"
    if not data_dir.exists():
        return pd.DataFrame()

    df_list = []
    for year in range(2019, 2026):
        file_path = data_dir / f"{str(year)[2:]}년 교통단속.csv"
        if not file_path.exists():
            continue
        try:
            temp = pd.read_csv(file_path, encoding="utf-8-sig")
        except Exception:
            temp = pd.read_csv(file_path, encoding="utf-8")
        # 컬럼명 공백/줄바꿈 제거
        temp.columns = temp.columns.astype(str).str.replace(r"\s+", "", regex=True)
        # 불필요한 Unnamed 컬럼 제거
        temp = temp.loc[:, ~temp.columns.str.startswith("Unnamed")]
        # 위반일시 전처리
        if "위반일시" in temp.columns:
            s = temp["위반일시"].astype(str).str.replace(r"\.0$", "", regex=True)
            temp["위반일시"] = pd.to_datetime(
                s,
                format="%Y%m%d%H%M",
                errors="coerce",
            )
            if temp["위반일시"].isna().all():
                temp["위반일시"] = pd.to_datetime(s, errors="coerce")
            temp["연도"] = temp["위반일시"].dt.year
            temp["월"] = temp["위반일시"].dt.month
        # 위반일시가 없거나 파싱이 실패한 경우 연도만 주입
        if "연도" not in temp.columns or temp["연도"].isna().all():
            temp["연도"] = year
        df_list.append(temp)

    if not df_list:
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.astype(str).str.replace(r"\s+", "", regex=True)
    return df


def _ensure_year_month(df: pd.DataFrame) -> pd.DataFrame:
    """연도/월 컬럼이 없으면 발생일시로 생성."""
    if df.empty:
        return df
    # 이미 연도/월이 있으면 원본 반환
    if "연도" in df.columns and "월" in df.columns:
        return df
    # 필요한 경우에만 복사
    work = df.copy()
    if "위반일시" in work.columns:
        work["위반일시"] = pd.to_datetime(work["위반일시"], errors="coerce")
        work["연도"] = work["위반일시"].dt.year
        work["월"] = work["위반일시"].dt.month
    elif "발생일시" in work.columns:
        work["발생일시"] = pd.to_datetime(work["발생일시"], errors="coerce")
        work["연도"] = work["발생일시"].dt.year
        work["월"] = work["발생일시"].dt.month
    return work


@st.cache_data(show_spinner=False)
def load_weather_passenger_monthly() -> pd.DataFrame:
    """강수량/여객 데이터를 월 단위로 집계."""
    data_dir = Path(__file__).parent / "weather_pax"
    if not data_dir.exists():
        return pd.DataFrame()

    rain_path = data_dir / "2018.01.01-2025.10.31 강수량.csv"
    in_path = data_dir / "일별 여객 입항.csv"
    out_path = data_dir / "일별 여객 출항.csv"
    if not rain_path.exists() or not in_path.exists() or not out_path.exists():
        return pd.DataFrame()

    rain_df = pd.read_csv(rain_path, encoding="utf-8")
    rain_df["날짜"] = pd.to_datetime(rain_df["날짜"], errors="coerce")
    rain_df["강수량(mm)"] = pd.to_numeric(
        rain_df["강수량(mm)"], errors="coerce"
    ).fillna(0)
    rain_df["강수량(mm)"] = rain_df["강수량(mm)"].clip(lower=0)
    if "지점" in rain_df.columns:
        rain_df = rain_df.drop(columns=["지점"])
    for col in list(rain_df.columns):
        if "Unnamed" in col or col == "0":
            rain_df = rain_df.drop(columns=[col])

    in_ppl = pd.read_csv(in_path, encoding="utf-8")
    out_ppl = pd.read_csv(out_path, encoding="utf-8")

    for df in (in_ppl, out_ppl):
        df["출항일"] = pd.to_datetime(df["출항일"], errors="coerce").dt.normalize()
        df["합계"] = pd.to_numeric(df["합계"], errors="coerce").fillna(0).astype(int)

    in_p_day = (
        in_ppl.groupby("출항일", as_index=False)["합계"]
        .sum()
        .rename(columns={"출항일": "날짜", "합계": "입항_여객수"})
    )
    out_p_day = (
        out_ppl.groupby("출항일", as_index=False)["합계"]
        .sum()
        .rename(columns={"출항일": "날짜", "합계": "출항_여객수"})
    )

    base_dates = pd.DataFrame(
        pd.Index(
            pd.concat([in_p_day["날짜"], out_p_day["날짜"]]).dropna().unique()
        ).sort_values(),
        columns=["날짜"],
    )

    merged = (
        base_dates.merge(rain_df, on="날짜", how="left")
        .merge(in_p_day, on="날짜", how="left")
        .merge(out_p_day, on="날짜", how="left")
    )

    for col in ["입항_여객수", "출항_여객수"]:
        merged[col] = merged[col].fillna(0).astype(int)
    if "강수량(mm)" in merged.columns:
        merged["강수량(mm)"] = merged["강수량(mm)"].fillna(0)

    monthly = (
        merged.set_index("날짜")
        .resample("MS")
        .agg(
            월강수합=("강수량(mm)", "sum"),
            월입항합=("입항_여객수", "sum"),
            월출항합=("출항_여객수", "sum"),
        )
        .reset_index()
    )
    monthly["연"] = monthly["날짜"].dt.year
    monthly["월"] = monthly["날짜"].dt.month
    return monthly


@st.cache_data(show_spinner=False)
def load_sms_raw() -> pd.DataFrame:
    """원본 울릉알리미 SMS CSV 로드 (날짜 전처리 포함)."""
    path = Path(__file__).parent / "울릉알리미_텍스트.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8")

    # 날짜 컬럼이 있으면 미리 전처리
    if "sms_resDate" in df.columns:
        s = df["sms_resDate"].astype(str).str.strip()
        s = s.str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
        df["sms_resDate"] = pd.to_datetime(s, errors="coerce")
        df = df.dropna(subset=["sms_resDate"])

    return df


@st.cache_data(show_spinner=False)
def load_passenger_daily_avg(year: int = 2025) -> dict:
    """여객 입출항 일평균(연도 기준)."""
    data_dir = Path(__file__).parent / "weather_pax"
    in_path = data_dir / "일별 여객 입항.csv"
    out_path = data_dir / "일별 여객 출항.csv"
    if not in_path.exists() or not out_path.exists():
        return {"입항": 0, "출항": 0}

    def _avg(path: Path) -> int:
        df = pd.read_csv(path, encoding="utf-8")
        s = df["출항일"].astype(str).str.strip()
        s = s.str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
        df["출항일"] = pd.to_datetime(s, errors="coerce")
        df = df[df["출항일"].dt.year == year]
        if "합계" not in df.columns:
            return 0
        df["합계"] = pd.to_numeric(df["합계"], errors="coerce").fillna(0)
        daily = df.groupby(df["출항일"].dt.date)["합계"].sum()
        if daily.empty:
            return 0
        return int(round(float(daily.mean())))

    return {"입항": _avg(in_path), "출항": _avg(out_path)}


@st.cache_data(show_spinner=False)
def load_passenger_daily(kind: str) -> pd.DataFrame:
    """일별 여객 입출항 데이터 로드 (최근 통계용)."""
    data_dir = Path(__file__).parent / "weather_pax"
    if kind == "입항":
        path = data_dir / "일별 여객 입항.csv"
    else:
        path = data_dir / "일별 여객 출항.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]
    if "출항일" not in df.columns:
        return pd.DataFrame()

    s = df["출항일"].astype(str).str.strip()
    s = s.str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
    df["date"] = pd.to_datetime(s, errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    if "합계" not in df.columns:
        return pd.DataFrame()
    df["passengers"] = pd.to_numeric(df["합계"], errors="coerce").fillna(0).astype(int)

    vehicle_file = None
    for f in data_dir.iterdir():
        if not f.is_file() or not f.name.endswith(".csv"):
            continue
        name = unicodedata.normalize("NFC", f.name)
        if "차량" in name and kind in name:
            vehicle_file = f
            break

    if vehicle_file is not None:
        vdf = pd.read_csv(vehicle_file, encoding="utf-8")
        vdf.columns = [str(c).strip() for c in vdf.columns]
        if "출항일" in vdf.columns:
            vs = vdf["출항일"].astype(str).str.strip()
            vs = vs.str.replace(".", "-", regex=False).str.replace(
                "/", "-", regex=False
            )
            vdf["date"] = pd.to_datetime(vs, errors="coerce").dt.normalize()
            vdf = vdf.dropna(subset=["date"])
            if "건수" in vdf.columns:
                vdf["vehicles"] = (
                    vdf["건수"]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .pipe(pd.to_numeric, errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                df = df.merge(vdf[["date", "vehicles"]], on="date", how="left")
                df["vehicles"] = (
                    pd.to_numeric(df["vehicles"], errors="coerce").fillna(0).astype(int)
                )
            else:
                df["vehicles"] = None
        else:
            df["vehicles"] = None
    else:
        df["vehicles"] = None

    return df[["date", "passengers", "vehicles"]]


def _recent_passenger_stats() -> dict:
    """최근 입항/출항 1건 및 최근 3회 평균."""
    arrive_df = load_passenger_daily("입항")
    depart_df = load_passenger_daily("출항")

    def _latest(df: pd.DataFrame):
        if df.empty:
            return {"date": None, "passengers": 0, "vehicles": None}
        row = df.sort_values("date", ascending=False).iloc[0]
        return {
            "date": row["date"],
            "passengers": int(row["passengers"]),
            "vehicles": row.get("vehicles", None),
        }

    def _avg_last3(df: pd.DataFrame):
        if df.empty:
            return {"passengers": 0, "vehicles": None}
        recent = df.sort_values("date", ascending=False).head(3)
        return {
            "passengers": int(round(float(recent["passengers"].mean()))),
            "vehicles": (
                int(round(float(recent["vehicles"].mean())))
                if "vehicles" in recent.columns and recent["vehicles"].notna().any()
                else None
            ),
        }

    return {
        "arrive_latest": _latest(arrive_df),
        "depart_latest": _latest(depart_df),
        "arrive_avg3": _avg_last3(arrive_df),
        "depart_avg3": _avg_last3(depart_df),
    }


def _monthly_passenger_stats(
    days: int = 30, end_dt: pd.Timestamp | None = None
) -> dict:
    """최근 N일 기준 월간 통계 (여객 합계)."""
    arrive_df = load_passenger_daily("입항")
    depart_df = load_passenger_daily("출항")

    if end_dt is None:
        all_dates = pd.concat([arrive_df["date"], depart_df["date"]]).dropna()
        if all_dates.empty:
            end_dt = None
            start_dt = None
        else:
            end_dt = all_dates.max()
            start_dt = end_dt - pd.Timedelta(days=days - 1)
    else:
        start_dt = end_dt - pd.Timedelta(days=days - 1)

    def _sum_window(df: pd.DataFrame):
        if df.empty or start_dt is None or end_dt is None:
            return 0
        window = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
        return int(window["passengers"].sum())

    def _sum_vehicle_window(df: pd.DataFrame):
        if df.empty or start_dt is None or end_dt is None:
            return None
        if "vehicles" not in df.columns or df["vehicles"].isna().all():
            return None
        window = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
        return int(window["vehicles"].sum())

    return {
        "start_dt": start_dt,
        "end_dt": end_dt,
        "arrive_sum": _sum_window(arrive_df),
        "depart_sum": _sum_window(depart_df),
        "arrive_vehicle_sum": _sum_vehicle_window(arrive_df),
        "depart_vehicle_sum": _sum_vehicle_window(depart_df),
    }


def _latest_sea_event(df: pd.DataFrame, year: int, kind: str) -> dict:
    """SMS에서 최신 입항/출항 이벤트 추출."""
    if df.empty or "sms_resDate" not in df.columns or "sms_msg" not in df.columns:
        return {"datetime": None, "name": "정보 없음"}

    # 날짜가 이미 전처리되어 있으므로 필터링만 수행
    work = df[df["sms_resDate"].dt.year == year]
    if work.empty:
        return {"datetime": None, "name": "정보 없음"}

    # Vectorized 메시지 분류
    work = work.copy()
    work["sms_msg_str"] = work["sms_msg"].fillna("").astype(str).str.strip()

    # 셔틀 제외
    work = work[~work["sms_msg_str"].str.contains("셔틀", na=False, regex=False)]
    work = work[work["sms_msg_str"] != ""]

    if work.empty:
        return {"datetime": None, "name": "정보 없음"}

    # 입항/출항 분류
    work["label"] = None

    # 포항→울릉 패턴 체크
    pohang_pattern = r"[\"'""]?\s*포항\s*[\"'""]?\s*(?:→|->|➡|>)\s*울릉"
    work.loc[work["sms_msg_str"].str.contains(pohang_pattern, na=False, regex=True), "label"] = "입항"

    # 입항 키워드
    for keyword in SMS_ARRIVE_KEYWORDS:
        work.loc[work["sms_msg_str"].str.contains(keyword, na=False, regex=False) & work["label"].isna(), "label"] = "입항"

    # 출항 키워드
    for keyword in SMS_DEPART_KEYWORDS:
        work.loc[work["sms_msg_str"].str.contains(keyword, na=False, regex=False) & work["label"].isna(), "label"] = "출항"

    # 필터링
    filtered = work[work["label"] == kind]

    if filtered.empty:
        return {"datetime": None, "name": "정보 없음"}

    # 가장 최근 메시지 찾기
    filtered = filtered.sort_values("sms_resDate", ascending=False)
    latest = filtered.iloc[0]
    dt = latest["sms_resDate"]
    msg = latest["sms_msg_str"]

    names = (
        SMS_SHIP_KEYWORDS + SMS_SHIP_VESSEL_KEYWORDS + SMS_PEOPLE_KEYWORDS + SMS_PEOPLE_VESSEL_KEYWORDS
    )
    names = sorted(names, key=len, reverse=True)
    name = next((n for n in names if n in msg), "선박 정보 없음")

    time_match = re.search(r"(\d{1,2})[:시](\d{2})", msg)
    if time_match:
        time_text = f"{int(time_match.group(1)):02d}:{time_match.group(2)}"
    else:
        time_text = dt.strftime("%H:%M") if dt else ""
    dt_text = dt.strftime("%Y-%m-%d") if dt else "미상"
    return {"datetime": f"{dt_text} {time_text}".strip(), "name": name}


def _summarize_sms_notice_counts_window(
    df: pd.DataFrame, start_dt: pd.Timestamp | None, end_dt: pd.Timestamp | None
) -> tuple[dict, dict]:
    """최근 기간 기준 SMS 통계 요약."""
    counts = {
        "입항": 0,
        "출항": 0,
        "운항통제": 0,
        "결항": 0,
        "시간변경": 0,
    }
    breakdown = {
        "입항": {"선박": 0, "사람": 0},
        "출항": {"선박": 0, "사람": 0},
    }
    if df.empty or "sms_resDate" not in df.columns or "sms_msg" not in df.columns:
        return counts, breakdown
    if start_dt is None or end_dt is None:
        return counts, breakdown

    # 날짜가 이미 전처리되어 있으므로 필터링만 수행
    work = df[(df["sms_resDate"] >= start_dt) & (df["sms_resDate"] <= end_dt)]
    if work.empty:
        return counts, breakdown

    # 완전 벡터화된 처리
    work = work.copy()
    work["sms_msg_str"] = work["sms_msg"].astype(str).str.strip()

    # 셔틀 제외
    work = work[~work["sms_msg_str"].str.contains("셔틀", na=False, regex=False)]

    # 벡터화된 분류
    work["label"] = None

    # 결항
    mask = work["sms_msg_str"].str.contains("|".join(SMS_CANCEL_KEYWORDS), na=False, regex=True)
    work.loc[mask, "label"] = "결항"

    # 운항통제
    mask = work["sms_msg_str"].str.contains("|".join(SMS_CONTROL_KEYWORDS), na=False, regex=True)
    work.loc[mask & work["label"].isna(), "label"] = "운항통제"

    # 시간변경
    mask = work["sms_msg_str"].str.contains("|".join(SMS_CHANGE_KEYWORDS), na=False, regex=True)
    work.loc[mask & work["label"].isna(), "label"] = "시간변경"

    # 입항 (포항→울릉 패턴)
    mask = work["sms_msg_str"].str.contains(r'["\'""]?\s*포항\s*["\'""]?\s*(?:→|->|➡|>)\s*울릉', na=False, regex=True)
    work.loc[mask & work["label"].isna(), "label"] = "입항"

    # 입항 (키워드)
    mask = work["sms_msg_str"].str.contains("|".join(SMS_ARRIVE_KEYWORDS), na=False, regex=True)
    work.loc[mask & work["label"].isna(), "label"] = "입항"

    # 출항
    mask = work["sms_msg_str"].str.contains("|".join(SMS_DEPART_KEYWORDS), na=False, regex=True)
    work.loc[mask & work["label"].isna(), "label"] = "출항"

    work = work[work["label"].notna()]

    # 날짜 추출
    work["day"] = work["sms_resDate"].dt.date
    work = work[work["day"].notna()]

    # 입항/출항은 그룹별로 집계
    arrival_departure = work[work["label"].isin(["입항", "출항"])].copy()
    if not arrival_departure.empty:
        # 벡터화된 그룹 분류
        arrival_departure["group"] = None

        # 선박 그룹
        ship_keywords = list(SMS_SHIP_KEYWORDS) + list(SMS_SHIP_VESSEL_KEYWORDS) + list(SMS_CARGO_KEYWORDS)
        mask = arrival_departure["sms_msg_str"].str.contains("|".join(ship_keywords), na=False, regex=True)
        arrival_departure.loc[mask, "group"] = "선박"

        # 사람 그룹
        people_keywords = list(SMS_PEOPLE_KEYWORDS) + list(SMS_PEOPLE_VESSEL_KEYWORDS) + list(SMS_PASSENGER_KEYWORDS)
        mask = arrival_departure["sms_msg_str"].str.contains("|".join(people_keywords), na=False, regex=True)
        arrival_departure.loc[mask & arrival_departure["group"].isna(), "group"] = "사람"

        arrival_departure = arrival_departure[arrival_departure["group"].notna()]

        # 중복 제거 후 집계
        grouped = arrival_departure.drop_duplicates(subset=["day", "label", "group"])
        for label in ["입항", "출항"]:
            for group in ["선박", "사람"]:
                count = len(grouped[(grouped["label"] == label) & (grouped["group"] == group)])
                breakdown[label][group] = count

    # 기타 카테고리 집계
    other_categories = work[work["label"].isin(["결항", "시간변경", "운항통제"])]
    if not other_categories.empty:
        deduped = other_categories.drop_duplicates(subset=["day", "label"])
        for label in ["결항", "시간변경", "운항통제"]:
            counts[label] = len(deduped[deduped["label"] == label])

    counts["입항"] = breakdown["입항"]["선박"] + breakdown["입항"]["사람"]
    counts["출항"] = breakdown["출항"]["선박"] + breakdown["출항"]["사람"]
    return counts, breakdown


def _summarize_sms_notice_counts(
    df: pd.DataFrame, year: int = 2025
) -> tuple[dict, int, dict]:
    """해상공지 유형별 건수(연도 필터)."""
    counts = {
        "입항": 0,
        "출항": 0,
        "운항통제": 0,
        "결항": 0,
        "시간변경": 0,
    }
    breakdown = {
        "입항": {"선박": 0, "사람": 0},
        "출항": {"선박": 0, "사람": 0},
    }
    if df.empty or "sms_resDate" not in df.columns or "sms_msg" not in df.columns:
        return counts, 0, breakdown

    # 날짜가 이미 전처리되어 있으므로 필터링만 수행
    work = df[df["sms_resDate"].dt.year == year]

    def classify(msg: str) -> str | None:
        if not msg:
            return None
        if any(k in msg for k in SMS_CANCEL_KEYWORDS):
            return "결항"
        if any(k in msg for k in SMS_CONTROL_KEYWORDS):
            return "운항통제"
        if any(k in msg for k in SMS_CHANGE_KEYWORDS):
            return "시간변경"
        if re.search(r"[\"'""]?\s*포항\s*[\"'""]?\s*(?:→|->|➡|>)\s*울릉", msg):
            return "입항"
        arrive_pos = None
        for p in SMS_ARRIVE_ROUTE_PATTERNS:
            m = re.search(p, msg)
            if m:
                arrive_pos = (
                    m.start() if arrive_pos is None else min(arrive_pos, m.start())
                )
        depart_pos = None
        for p in SMS_DEPART_ROUTE_PATTERNS:
            m = re.search(p, msg)
            if m:
                depart_pos = (
                    m.start() if depart_pos is None else min(depart_pos, m.start())
                )
        if arrive_pos is not None and depart_pos is not None:
            return "출항" if depart_pos < arrive_pos else "입항"
        if depart_pos is not None:
            return "출항"
        if arrive_pos is not None:
            return "입항"
        has_arrive = any(k in msg for k in SMS_ARRIVE_KEYWORDS)
        has_depart = any(k in msg for k in SMS_DEPART_KEYWORDS)
        if has_arrive and not has_depart:
            return "입항"
        if has_depart and not has_arrive:
            return "출항"
        return None

    # 완전 벡터화된 처리
    work = work.copy()
    work["sms_msg_str"] = work["sms_msg"].astype(str).str.strip()

    # 셔틀 제외
    work = work[~work["sms_msg_str"].str.contains("셔틀", na=False, regex=False)]

    # classify 함수는 복잡한 로직이므로 apply 유지
    work["label"] = work["sms_msg_str"].apply(classify)
    work = work[work["label"].notna()]

    # 날짜 추출
    work["day"] = work["sms_resDate"].dt.date
    work = work[work["day"].notna()]

    # 입항/출항은 그룹별로 집계
    arrival_departure = work[work["label"].isin(["입항", "출항"])].copy()
    if not arrival_departure.empty:
        # 벡터화된 그룹 분류
        arrival_departure["group"] = None

        # 선박 그룹
        ship_keywords = list(SMS_SHIP_KEYWORDS) + list(SMS_SHIP_VESSEL_KEYWORDS) + list(SMS_CARGO_KEYWORDS)
        mask = arrival_departure["sms_msg_str"].str.contains("|".join(ship_keywords), na=False, regex=True)
        arrival_departure.loc[mask, "group"] = "선박"

        # 사람 그룹
        people_keywords = list(SMS_PEOPLE_KEYWORDS) + list(SMS_PEOPLE_VESSEL_KEYWORDS) + list(SMS_PASSENGER_KEYWORDS)
        mask = arrival_departure["sms_msg_str"].str.contains("|".join(people_keywords), na=False, regex=True)
        arrival_departure.loc[mask & arrival_departure["group"].isna(), "group"] = "사람"
        arrival_departure = arrival_departure[arrival_departure["group"].notna()]

        # 전체 집계
        for label in ["입항", "출항"]:
            for group in ["선박", "사람"]:
                count = len(arrival_departure[(arrival_departure["label"] == label) & (arrival_departure["group"] == group)])
                breakdown[label][group] = count

    # 기타 카테고리
    other_categories = work[work["label"].isin(["결항", "시간변경", "운항통제"])]
    if not other_categories.empty:
        deduped = other_categories.drop_duplicates(subset=["day", "label"])
        for label in ["결항", "시간변경", "운항통제"]:
            counts[label] = len(deduped[deduped["label"] == label])

    counts["입항"] = breakdown["입항"]["선박"] + breakdown["입항"]["사람"]
    counts["출항"] = breakdown["출항"]["선박"] + breakdown["출항"]["사람"]

    total = sum(counts.values())
    return counts, total, breakdown


def _latest_sea_notice(df: pd.DataFrame, year: int = 2025) -> tuple[str, str]:
    """가장 최신 해상 공지 (카테고리, 요약 문자열)."""
    if df.empty or "sms_resDate" not in df.columns or "sms_msg" not in df.columns:
        return "입항", "최신 공지 없음"

    # 날짜가 이미 전처리되어 있으므로 필터링만 수행
    work = df[df["sms_resDate"].dt.year == year]
    if work.empty:
        return "입항", "최신 공지 없음"

    def classify(msg: str) -> str | None:
        if not msg:
            return None
        if any(k in msg for k in SMS_CANCEL_KEYWORDS):
            return "결항"
        if any(k in msg for k in SMS_CONTROL_KEYWORDS):
            return "운항통제"
        if any(k in msg for k in SMS_CHANGE_KEYWORDS):
            return "시간변경"
        if re.search(r"[\"'""]?\s*포항\s*[\"'""]?\s*(?:→|->|➡|>)\s*울릉", msg):
            return "입항"
        arrive_pos = None
        for p in SMS_ARRIVE_ROUTE_PATTERNS:
            m = re.search(p, msg)
            if m:
                arrive_pos = (
                    m.start() if arrive_pos is None else min(arrive_pos, m.start())
                )
        depart_pos = None
        for p in SMS_DEPART_ROUTE_PATTERNS:
            m = re.search(p, msg)
            if m:
                depart_pos = (
                    m.start() if depart_pos is None else min(depart_pos, m.start())
                )
        if arrive_pos is not None and depart_pos is not None:
            return "출항" if depart_pos < arrive_pos else "입항"
        if depart_pos is not None:
            return "출항"
        if arrive_pos is not None:
            return "입항"
        has_arrive = any(k in msg for k in SMS_ARRIVE_KEYWORDS)
        has_depart = any(k in msg for k in SMS_DEPART_KEYWORDS)
        if has_arrive and not has_depart:
            return "입항"
        if has_depart and not has_arrive:
            return "출항"
        return None

    # Vectorized 처리
    work = work.copy()
    work["sms_msg_str"] = work["sms_msg"].astype(str).str.strip()

    # 빈 메시지와 셔틀 제외
    work = work[work["sms_msg_str"] != ""]
    work = work[~work["sms_msg_str"].str.contains("셔틀", na=False)]

    # 분류 적용
    work["label"] = work["sms_msg_str"].apply(classify)
    work = work[work["label"].notna()]

    if work.empty:
        return "입항", "최신 공지 없음"

    # 가장 최근 메시지 찾기
    work = work.sort_values("sms_resDate", ascending=False)
    latest = work.iloc[0]
    dt = latest["sms_resDate"]
    msg = latest["sms_msg_str"]
    label = latest["label"]

    names = (
        SMS_SHIP_KEYWORDS + SMS_SHIP_VESSEL_KEYWORDS + SMS_PEOPLE_KEYWORDS + SMS_PEOPLE_VESSEL_KEYWORDS
    )
    names = sorted(names, key=len, reverse=True)
    name = next((n for n in names if n in msg), "공지")

    time_match = re.search(r"(\d{1,2})[:시](\d{2})", msg)
    if time_match:
        time_text = f"{int(time_match.group(1)):02d}:{time_match.group(2)}"
    else:
        time_text = dt.strftime("%H:%M") if dt else ""
    dt_text = dt.strftime("%Y-%m-%d") if dt else "미상"
    summary_text = f"{dt_text} {time_text} - {name}"
    return label, summary_text.strip()
