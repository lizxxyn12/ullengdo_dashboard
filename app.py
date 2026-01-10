import streamlit as st
import pandas as pd
from pathlib import Path
import re
import time
import math
import unicodedata
from datetime import datetime
import base64
import textwrap
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image
import os

try:
    import folium
except Exception:
    folium = None

try:
    from folium.plugins import MarkerCluster
except Exception:
    MarkerCluster = None
try:
    from folium.features import DivIcon
except Exception:
    DivIcon = None

try:
    from streamlit_folium import st_folium
except Exception:
    st_folium = None

st.set_page_config(
    page_title="울릉 교통/안전 대시보드",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Matplotlib 한글 폰트 설정
_font_candidates = [
    "AppleGothic",  # macOS
    "NanumGothic",  # Linux/Windows
    "Malgun Gothic",  # Windows
    "Noto Sans CJK KR",  # Linux
    "Noto Sans KR",  # Linux
]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _fname in _font_candidates:
    if _fname in _available_fonts:
        plt.rcParams["font.family"] = _fname
        break
plt.rcParams["axes.unicode_minus"] = False

# -----------------------------
# CSS (업데이트됨: 해상공지 카드 디자인 적용)
# -----------------------------
st.markdown(
    """
<style>
/* 전체 폭 여백 조정 */
.block-container {
  padding-top: 2rem;
  padding-bottom: 2.4rem;
  max-width: 100%;
}

.notice-pill {
  width: 100%;
  margin-top: 0.5rem;
  line-height: 1.2;
  background: #f3f3f3;
  border-radius: 999px;
  padding: 14px 18px;
  font-weight: 400;
  color: #333;
  border: 1px solid #e6e6e6;
}

.dashboard-title {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 0.8rem 0 0.6rem 0;
}
.dashboard-title img {
  width: 40px;
  height: 40px;
  object-fit: contain;
}
.dashboard-title .title-text {
  font-size: 1.6rem;
  font-weight: 800;
  color: #1f1f1f;
}

.card-title {
  font-weight: 700;
  margin-bottom: 8px;
}
.card-sub {
  color: #666;
  font-size: 0.9rem;
}

.photo-placeholder {
  background: #e9f2ff;
  color: #0b5cab;
  border-radius: 16px;
  height: 250px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 400;
  font-size: 1.05rem;
}

/* UI 요소 z-index 조정 */
div[data-baseweb="select"] { position: relative; z-index: 3000; }
div[data-baseweb="popover"] { z-index: 4000; }
section.main iframe { position: relative; z-index: 1; }

/* 다이얼로그 스타일 */
div[data-testid="stDialog"] > div { width: min(96vw, 1400px); margin: 0 auto; }
div[data-testid="stDialog"] div[role="dialog"] { max-height: 92vh; padding: 0; }
div[data-testid="stDialog"] img { max-height: 86vh; width: 100%; object-fit: contain; display: block; }

/* --- [NEW] Card & Sea Notice Styles --- */
.r2-card {
  background: #f6f7fb;
  border: 1px solid #ebedf3;
  border-radius: 22px;
  padding: 18px 18px 16px 18px;
  height: 90%;
  box-sizing: border-box;
  overflow-y: auto;
}
.r2-top {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.r2-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}
.r2-title {
  font-size: 1.05rem;
  font-weight: 800;
}
.r2-date {
  color: #777;
  font-size: 0.85rem;
}
.r2-card-body {
  margin-top: 8px;
}

/* Sea Section & Layout */
.sea-section {
  background: #ffffff;
  border: 1px solid #e8ebf2;
  border-radius: 16px;
  padding: 12px;
  margin-bottom: 12px;
}
.sea-section-title {
  font-size: 0.82rem;
  font-weight: 500;
  color: #6b7280;
  margin-bottom: 8px;
  letter-spacing: 0.2px;
}
.sea-latest {
  display: flex;
  align-items: center;
  gap: 10px;
}
.sea-pill {
  background: #e8f0ff;
  color: #2f6bff;
  border-radius: 999px;
  padding: 6px 12px;
  font-weight: 700;
  font-size: 0.88rem;
}
.sea-latest-text {
  font-size: 1.02rem;
  font-weight: 700;
  color: #1d1d1d;
}

/* Bar Charts (Updated) */
.sea-bars {
  display: grid;
  gap: 12px;
  margin-bottom: 2px;
}
.bar-row {
  display: grid;
  grid-template-columns: 120px 1fr; /* 값 표시 영역 제거(바 내부로 이동) */
  gap: 10px;
  align-items: center;
}
.bar-label {
  font-weight: 600;
  font-size: 0.86rem;
}
.bar-label-wrap {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}
.bar-sub {
  font-size: 0.82rem;
  font-weight: 400;
  color: #666;
}
.bar-track {
  background: #ffffff;
  border: 1px solid #edf0f5;
  border-radius: 999px;
  padding: 4px;
  position: relative;
}
.bar-fill {
  height: 14px;
  border-radius: 999px;
  position: relative;
}
.bar-fill-split {
  height: 14px;
  border-radius: 999px;
  overflow: hidden;
  display: flex;
  position: relative;
}
.bar-seg {
  height: 100%;
}
.bar-value-onfill {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  background: #ffffff;
  color: #374151;
  font-size: 0.74rem;
  font-weight: 700;
  padding: 2px 10px;
  border-radius: 999px;
  border: 1px solid rgba(0, 0, 0, 0.08);
  box-shadow: 0 4px 10px rgba(17, 24, 39, 0.1);
  pointer-events: none;
  white-space: nowrap;
}

/* Tooltip (Help Pop) */
.help-pop {
  position: relative;
  display: inline-flex;
  align-items: center;
}
.help-pop-btn {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  border: 1px solid #d1d7e2;
  color: #6b7280;
  font-size: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: default;
  background: #ffffff;
}
.help-pop-body {
  display: none;
  position: absolute;
  top: 22px;
  left: 0;
  min-width: 200px;
  background: #ffffff;
  border: 1px solid #e5e9f2;
  border-radius: 10px;
  padding: 8px 10px;
  font-size: 0.78rem;
  font-weight: 400;
  color: #4b5563;
  line-height: 1.4;
  box-shadow: 0 6px 16px rgba(17, 24, 39, 0.08);
  z-index: 100;
}
.help-pop:hover .help-pop-body {
  display: block;
}

/* Road List Styles */
.road-list { display: grid; gap: 10px; }
.road-item {
  background: #ffffff;
  border: 1px solid #e8ebf2;
  border-radius: 14px;
  padding: 10px 12px;
}
.road-item-title { font-weight: 800; margin-bottom: 4px; }
.road-item-meta { color: #666; font-size: 0.82rem; }
.road-tag {
  display: inline-block;
  margin-right: 6px;
  padding: 2px 8px;
  border-radius: 8px;
  background: #eef2ff;
  color: #2f5dff;
  font-size: 0.72rem;
  font-weight: 800;
}

div[data-testid="stPopover"] > button {
  background: #2f5dff;
  color: #fff;
  font-size: 0.72rem;
  font-weight: 800;
  padding: 4px 10px;
  border-radius: 999px;
  border: none;
}
</style>
""",
    unsafe_allow_html=True,
)


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
        m = re.search(r"(20\\d{2})년도", name)
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
        temp = temp.dropna(subset=["latitude", "longitude"]).copy()

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
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df["year"] = 2025
    return df


@st.cache_data(show_spinner=False)
def load_ev_charger_points() -> list[tuple[float, float, str]]:
    """울릉군 전기차 충전소 좌표 로드."""
    csv_path = Path(__file__).parent / "울릉군 전기차 충전소 2020-07-13.csv"
    if not csv_path.exists():
        return []

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
        return []

    df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()

    points = []
    for _, row in df.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])

        name = _clean_text(row.get("충전소명"))
        detail = _clean_text(row.get("충전소위치상세"))
        road_addr = _clean_text(row.get("소재지도로명주소"))
        lot_addr = _clean_text(row.get("소재지지번주소"))
        address = _first_text(road_addr, lot_addr, detail)

        label_name = name if name else "충전소"
        label_addr = address if address else "주소 미상"
        label = f"충전소 : {label_name}<br/>주소 : {label_addr}"
        points.append((lat, lon, label))

    return points


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

        df_coords = df_coords.copy()
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

        points = []
        meta = []

        idx_counter = 0
        for _, row in df_coords.iterrows():
            lat = pd.to_numeric(row.get(lat_col, None), errors="coerce")
            lon = pd.to_numeric(row.get(lon_col, None), errors="coerce")
            if pd.isna(lat) or pd.isna(lon):
                continue

            address = ""
            for c in address_cols:
                v = row.get(c, None)
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    address = s
                    break

            label_text = address or "위치 미상"

            photo = _find_rockfall_photo(address) if address else None
            if photo is None and "filename" in row:
                photo = _find_rockfall_photo(row.get("filename", ""))

            points.append((float(lat), float(lon), f"낙석 발생 위치 : {label_text}"))
            meta.append(
                {
                    "idx": int(idx_counter),
                    "lat": float(lat),
                    "lon": float(lon),
                    "photo": str(photo) if photo else None,
                    "name": str(label_text),
                    "date": row.get("사고일자", None),
                    "damage": row.get("피해여부", None),
                }
            )
            idx_counter += 1

        return points, meta

    # rockfall_coords_final.csv만 사용(최신 좌표/주소)
    if coords_final_path.exists():
        points, meta = _build_from_coords_df(_read_csv_safely(coords_final_path))
        if points:
            return points, meta
    return [], [] # Fallback empty if file not found


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
    df = df.dropna(subset=["lat", "lon"]).copy()
    return df[["stop_name", "stop_norm", "lat", "lon"]]


def _match_bus_stop(df: pd.DataFrame, name: str):
    """정류장 이름으로 좌표 매칭."""
    if df.empty:
        return None
    target = _norm_text(name)
    if not target:
        return None

    exact = df[df["stop_norm"] == target]
    if not exact.empty:
        row = exact.iloc[0]
        return float(row["lat"]), float(row["lon"])

    for _, row in df.iterrows():
        norm = row["stop_norm"]
        if target in norm or norm in target:
            return float(row["lat"]), float(row["lon"])
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
            "color": "#ff8c42",
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


@st.cache_data(show_spinner=False)
def build_bus_routes():
    """노선 정의를 좌표와 함께 반환."""
    df = load_bus_stops_csv()
    routes = []

    # 모든 정류장을 기본으로 포함(경유 노선은 추후 채움)
    stop_map: dict[str, dict] = {}
    for _, row in df.iterrows():
        key = _norm_text(row["stop_name"])
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

            def angle(row):
                return math.atan2(row["lat"] - center_lat, row["lon"] - center_lon)

            df_sorted = df.copy()
            df_sorted["ang"] = df_sorted.apply(angle, axis=1)
            df_sorted = df_sorted.sort_values("ang").reset_index(drop=True)

            # 시작점을 앵커 정류장 근처로 회전
            anchor_match = _match_bus_stop(df, route["stops"][0])
            start_idx = 0
            if anchor_match:
                ax, ay = anchor_match
                best = None
                best_d = None
                for i, row in df_sorted.iterrows():
                    d = abs(row["lat"] - ax) + abs(row["lon"] - ay)
                    if best_d is None or d < best_d:
                        best_d = d
                        best = i
                if best is not None:
                    start_idx = best
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


def render_ulleung_folium_map(
    kind: str = "base",
    height: int = 420,
    accident_df: pd.DataFrame | None = None,
    highlight_idx: int | None = None,
    center_override: tuple[float, float] | None = None,
):
    """울릉군 Folium 지도 렌더."""

    if folium is None:
        st.error(
            "folium 패키지가 설치되어 있지 않아 지도를 표시할 수 없어. 터미널에서 `pip install folium` 해줘."
        )
        return

    requested_kind = kind

    # 울릉도 중심(대략)
    center = (37.5044, 130.8757)
    if center_override is not None:
        center = center_override

    m = folium.Map(
        location=center, zoom_start=12, tiles="OpenStreetMap", control_scale=True
    )

    # 전기차 충전소 데이터 (모든 지도에 추가 표시)
    ev_points = load_ev_charger_points()

    if kind == "accident":
        df_acc = accident_df if accident_df is not None else load_accidents_csv()

        # CSV가 있으면 실제 좌표로 마커 생성
        if not df_acc.empty:
            # popup/tooltip에 보여줄 사고 유형 컬럼 찾기
            type_col_candidates = [
                c
                for c in ["type", "accident_type", "사고유형", "사고_type"]
                if c in df_acc.columns
            ]
            type_col = type_col_candidates[0] if type_col_candidates else None

            sample_points = []
            acc_points_meta = []  # 클릭 좌표 → 원본 행 인덱스 매칭용
            # 너무 많을 수 있어서 기본은 2000개로 제한
            for i, row in df_acc.head(2000).iterrows():
                lat = float(row["latitude"])
                lon = float(row["longitude"])

                # 사고 유형(type)만 표시 (없으면 미상)
                acc_type = "미상"
                if type_col is not None:
                    v = row.get(type_col, None)
                    if v is not None:
                        s = str(v).strip()
                        if s and s.lower() not in ["nan", "none"]:
                            acc_type = s

                # 주소는 사진 매칭용으로만 내부에서 계산(마커에는 표시하지 않음)
                _ = _row_to_address(df_acc, row)

                # 클릭 처리용 IDX 포함(주소는 포함하지 않음)
                sample_points.append((lat, lon, f"사고 유형 : {acc_type}"))
                acc_points_meta.append({"idx": int(i), "lat": lat, "lon": lon})

            st.session_state["acc_points_meta"] = acc_points_meta

        else:
            if accident_df is not None:
                sample_points = []
                st.session_state["acc_points_meta"] = []
            else:
                # CSV가 없거나 형식이 다르면 샘플로 fallback
                sample_points = [
                    (37.4890, 130.9050, "사고 유형 : 사고(샘플) A"),
                    (37.4770, 130.9130, "사고 유형 : 사고(샘플) B"),
                    (37.4705, 130.8985, "사고 유형 : 사고(샘플) C"),
                ]
                st.session_state["acc_points_meta"] = [
                    {"idx": 0, "lat": 37.4890, "lon": 130.9050},
                    {"idx": 1, "lat": 37.4770, "lon": 130.9130},
                    {"idx": 2, "lat": 37.4705, "lon": 130.8985},
                ]

        color = "red"
    elif kind == "rockfall":
        sample_points, rockfall_meta = load_rockfall_points()
        st.session_state["rockfall_points_meta"] = rockfall_meta
        if not sample_points:
            sample_points = [
                (37.4950, 130.9145, "낙석 발생 위치 : (샘플) A"),
                (37.4680, 130.8920, "낙석 발생 위치 : (샘플) B"),
            ]
        color = "orange"
    elif kind == "bus":
        routes, bus_stops = build_bus_routes()
        if not bus_stops:
            bus_stops = [
                {
                    "name": "버스정류장(샘플)",
                    "lat": 37.4868,
                    "lon": 130.9098,
                    "routes": ["샘플"],
                },
                {
                    "name": "버스정류장(샘플2)",
                    "lat": 37.4758,
                    "lon": 130.9032,
                    "routes": ["샘플"],
                },
            ]
        st.session_state["bus_stops_meta"] = bus_stops

        sample_points = []
        color = "blue"
        for stop in bus_stops:
            name = stop.get("name", "(이름 없음)")
            routes_txt = (
                ", ".join(stop.get("routes", []))
                if stop.get("routes")
                else "경유 노선 정보 없음"
            )
            label = f"정류장 : {name}<br/>경유 노선 : {routes_txt}"
            sample_points.append((stop["lat"], stop["lon"], label))
    else:
        sample_points = []
        color = "green"

    fg = folium.FeatureGroup(name=kind)

    # 마커 클러스터 사용(사고/낙석은 항상 클러스터, 버스는 전체 표시를 위해 클러스터 미사용)
    marker_parent = fg
    marker_points = sample_points
    if kind == "bus":
        # bus는 경유 노선 색상 기반으로 마커 색을 나눔
        routes_defs = {r["name"]: r["color"] for r in _bus_route_defs()}
        marker_points = []
        for stop in st.session_state.get("bus_stops_meta", []):
            routes_txt = (
                ", ".join(stop.get("routes", []))
                if stop.get("routes")
                else "경유 노선 정보 없음"
            )
            label = f"정류장 : {stop['name']}<br/>경유 노선 : {routes_txt}"
            first_route = stop.get("routes", [None])[0] if stop.get("routes") else None
            color_for_stop = routes_defs.get(first_route, "#666666")
            marker_points.append((stop["lat"], stop["lon"], label, color_for_stop))

    if MarkerCluster is not None and kind not in {"bus"}:
        if kind in {"accident", "rockfall"} or len(marker_points) > 50:
            marker_parent = MarkerCluster(name=f"{kind}_cluster").add_to(fg)

    for mp in marker_points:
        if kind == "bus":
            lat, lon, label, m_color = mp
        else:
            lat, lon, label = mp
            m_color = color
        # hover(tooltip)는 사용하지 않고, 클릭(popup)만 사용
        # 클릭 시 뜨는 정보(팝업) 크기/폰트 줄이기
        popup_html = f"""
        <div style='font-size:12px; line-height:1.25; max-width:200px; white-space:normal;'>
            {label}
        </div>
        """
        popup = folium.Popup(popup_html, max_width=220)

        folium.CircleMarker(
            location=(lat, lon),
            radius=5,
            color=m_color,
            fill=True,
            fill_opacity=0.85,
            popup=popup,
        ).add_to(marker_parent)

    if kind in {"accident", "rockfall"} and highlight_idx is not None:
        meta_key = "acc_points_meta" if kind == "accident" else "rockfall_points_meta"
        pulse_color = "#ff0000" if kind == "accident" else "#ff8a00"
        pulse_rgba = "255, 0, 0" if kind == "accident" else "255, 138, 0"
        for p in st.session_state.get(meta_key, []):
            if int(p.get("idx", -1)) == int(highlight_idx):
                lat, lon = float(p["lat"]), float(p["lon"])
                if DivIcon is not None:
                    pulse_css = f"""
                    <div style="
                        width: 20px;
                        height: 20px;
                        background-color: rgba({pulse_rgba}, 0.6);
                        border-radius: 50%;
                        box-shadow: 0 0 0 0 rgba({pulse_rgba}, 0.7);
                        animation: pulse-red 1.5s infinite;
                        "></div>
                    <style>
                        @keyframes pulse-red {{
                            0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba({pulse_rgba}, 0.7); }}
                            70% {{ transform: scale(1); box-shadow: 0 0 0 20px rgba({pulse_rgba}, 0); }}
                            100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba({pulse_rgba}, 0); }}
                        }}
                    </style>
                    """
                    folium.Marker(
                        location=(lat, lon),
                        icon=DivIcon(
                            icon_size=(20, 20),
                            icon_anchor=(10, 10),
                            html=pulse_css,
                        ),
                    ).add_to(fg)
                folium.CircleMarker(
                    location=(lat, lon),
                    radius=6,
                    color="white",
                    weight=2,
                    fill=True,
                    fill_color=pulse_color,
                    fill_opacity=1.0,
                ).add_to(fg)
                break

    # 노선 라인(버스만 해당)
    if kind == "bus":
        routes, _ = build_bus_routes()
        for r in routes:
            pts = r.get("points", [])
            if len(pts) < 2:
                continue
            folium.PolyLine(
                pts,
                color=r.get("color", "blue"),
                weight=6,
                opacity=0.65,
                tooltip=r.get("name", ""),
            ).add_to(fg)

    fg.add_to(m)

    # 전기차 충전소 마커(모든 지도에 오버레이)
    if ev_points:
        ev_fg = folium.FeatureGroup(name="ev_chargers")
        for lat, lon, label in ev_points:
            popup_html = f"""
            <div style='font-size:12px; line-height:1.25; max-width:220px; white-space:normal;'>
                {label}
            </div>
            """
            popup = folium.Popup(popup_html, max_width=240)
            folium.CircleMarker(
                location=(lat, lon),
                radius=2,
                color="#2ca02c",
                fill=True,
                fill_opacity=0.9,
                popup=popup,
            ).add_to(ev_fg)

        ev_fg.add_to(m)

    # 지도 렌더 (가능하면 클릭 이벤트까지 받기)
    if st_folium is not None:
        return st_folium(
            m,
            height=height,
            width=None,
            key=f"folium_{requested_kind}",
        )

    # streamlit-folium이 없으면 이벤트 없이 지도만 표시
    import streamlit.components.v1 as components

    components.html(m.get_root().render(), height=height)
    return None


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
        # 위반일시 전처리 (단속데이터.ipynb 기준)
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


def _summarize_accident_counts(df: pd.DataFrame, mode: str) -> pd.Series:
    """연도별/월별 사고 건수 요약."""
    if df.empty:
        return pd.Series(dtype="int64")

    work = df.copy()
    if "연도" not in work.columns or "월" not in work.columns:
        if "발생일시" in work.columns:
            work["발생일시"] = pd.to_datetime(work["발생일시"], errors="coerce")
            work["연도"] = work["발생일시"].dt.year
            work["월"] = work["발생일시"].dt.month

    if mode == "연도별":
        if "연도" not in work.columns:
            return pd.Series(dtype="int64")
        return work.dropna(subset=["연도"]).groupby("연도").size().sort_index()

    if "월" not in work.columns:
        return pd.Series(dtype="int64")
    return work.dropna(subset=["월"]).groupby("월").size().sort_index()


def _ensure_year_month(df: pd.DataFrame) -> pd.DataFrame:
    """연도/월 컬럼이 없으면 발생일시로 생성."""
    if df.empty:
        return df
    work = df.copy()
    if "연도" not in work.columns or "월" not in work.columns:
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


# -----------------------------
# Vega-Lite Spec Functions
# -----------------------------

def _vega_base_config():
    """Vega-Lite 차트 공통 스타일 설정."""
    return {
        "axis": {
            "titleFontSize": 10,
            "labelFontSize": 10,
            "labelColor": "#1F2D3D",
            "titleColor": "#1F2D3D",
            "gridColor": "#E6EEF5",
        },
        "view": {"stroke": "transparent"},
    }


def _vega_bar_spec(x_field: str, y_field: str, title: str, height: int):
    return {
        "padding": {"top": 6, "right": 8, "bottom": 2, "left": 8},
        "mark": {
            "type": "bar",
            "cornerRadiusTopLeft": 6,
            "cornerRadiusTopRight": 6,
            "color": "#F5B97A",
            "opacity": 0.65,
        },
        "encoding": {
            "x": {"field": x_field, "type": "ordinal", "axis": {"labelAngle": 0}},
            "y": {
                "field": y_field,
                "type": "quantitative",
                "axis": {"title": f"{y_field}(건)"},
            },
            "tooltip": [
                {"field": x_field, "type": "ordinal"},
                {"field": y_field, "type": "quantitative"},
            ],
        },
        "height": height,
        "title": None,
        "config": _vega_base_config(),
    }


def _vega_weather_passenger_spec(x_field: str, title: str, height: int):
    return {
        "padding": {"top": 16, "right": 8, "bottom": 2, "left": 8},
        "layer": [
            {
                "transform": [{"calculate": "'월 강수량 합 (mm)'", "as": "시리즈"}],
                "mark": {"type": "bar", "color": "#B9CFE3", "opacity": 0.45},
                "encoding": {
                    "x": {"field": x_field, "type": "ordinal", "axis": {"labelAngle": 0}},
                    "y": {
                        "field": "강수량",
                        "type": "quantitative",
                        "axis": {"title": "강수량(mm)"},
                    },
                    "color": {
                        "field": "시리즈",
                        "type": "nominal",
                        "scale": {
                            "domain": ["월 강수량 합 (mm)"],
                            "range": ["#B9CFE3"],
                        },
                        "legend": {
                            "orient": "top",
                            "direction": "horizontal",
                            "title": None,
                            "offset": 6,
                            "padding": 0,
                            "labelFontSize": 10,
                            "labelLimit": 120,
                        },
                    },
                    "tooltip": [
                        {"field": x_field, "type": "ordinal"},
                        {"field": "강수량", "type": "quantitative"},
                    ],
                },
            },
            {
                "transform": [{"calculate": "'월 입도객수(명)'", "as": "시리즈"}],
                "mark": {
                    "type": "line",
                    "color": "#2CA02C",
                    "strokeWidth": 2.6,
                    "point": {"filled": True, "size": 70},
                },
                "encoding": {
                    "x": {"field": x_field, "type": "ordinal"},
                    "y": {
                        "field": "입도",
                        "type": "quantitative",
                        "axis": {"title": "여객수(명)", "orient": "right"},
                    },
                    "color": {
                        "field": "시리즈",
                        "type": "nominal",
                        "scale": {
                            "domain": ["월 입도객수(명)", "월 출도객수(명)"],
                            "range": ["#2CA02C", "#D62728"],
                        },
                        "legend": {
                            "orient": "top",
                            "direction": "horizontal",
                            "title": None,
                            "symbolType": "stroke",
                            "offset": 6,
                            "padding": 0,
                            "labelFontSize": 10,
                            "labelLimit": 120,
                        },
                    },
                    "tooltip": [
                        {"field": x_field, "type": "ordinal"},
                        {"field": "입도", "type": "quantitative"},
                    ],
                },
            },
            {
                "transform": [{"calculate": "'월 출도객수(명)'", "as": "시리즈"}],
                "mark": {
                    "type": "line",
                    "color": "#E45756",
                    "strokeWidth": 2.6,
                    "point": {"filled": True, "size": 70},
                },
                "encoding": {
                    "x": {"field": x_field, "type": "ordinal"},
                    "y": {
                        "field": "출도",
                        "type": "quantitative",
                        "axis": None,
                    },
                    "color": {
                        "field": "시리즈",
                        "type": "nominal",
                        "scale": {
                            "domain": ["월 입도객수(명)", "월 출도객수(명)"],
                            "range": ["#2CA02C", "#D62728"],
                        },
                        "legend": None,
                    },
                    "tooltip": [
                        {"field": x_field, "type": "ordinal"},
                        {"field": "출도", "type": "quantitative"},
                    ],
                },
            },
        ],
        "height": height,
        "resolve": {"scale": {"y": "independent", "color": "independent"}},
        "title": None,
        "config": _vega_base_config(),
    }


def _vega_bar_color_spec(
    x_field: str, y_field: str, color_field: str, title: str, height: int
):
    return {
        "padding": {"top": 10, "right": 8, "bottom": 2, "left": 18},
        "mark": {
            "type": "bar",
            "cornerRadiusTopLeft": 6,
            "cornerRadiusTopRight": 6,
            "opacity": 0.85,
        },
        "encoding": {
            "x": {"field": x_field, "type": "ordinal", "axis": {"labelAngle": 0}},
            "y": {
                "field": y_field,
                "type": "quantitative",
                "axis": {"title": "여객수(명)"},
            },
            "color": {
                "field": color_field,
                "type": "nominal",
                "scale": {
                    "domain": ["비수기", "성수기", "비수기(평균↑)"],
                    "range": ["#A9CFAE", "#F1C58B", "#E6D07A"],
                },
                "legend": {
                    "orient": "top-right",
                    "direction": "horizontal",
                    "title": None,
                    "padding": 0,
                    "offset": 6,
                    "labelFontSize": 10,
                },
            },
            "tooltip": [
                {"field": x_field, "type": "ordinal"},
                {"field": y_field, "type": "quantitative"},
                {"field": color_field, "type": "nominal"},
            ],
        },
        "height": height,
        "title": None,
        "config": _vega_base_config(),
    }


def _vega_base_config():
    """Vega-Lite 차트 공통 스타일 설정."""
    return {
        "axis": {
            "titleFontSize": 10,
            "labelFontSize": 10,
            "labelColor": "#1F2D3D",
            "titleColor": "#1F2D3D",
            "gridColor": "#E6EEF5",
        },
        "view": {"stroke": "transparent"},
    }


def _vega_bar_spec(x_field: str, y_field: str, title: str, height: int):
    return {
        "padding": {"top": 6, "right": 8, "bottom": 2, "left": 8},
        "mark": {
            "type": "bar",
            "cornerRadiusTopLeft": 6,
            "cornerRadiusTopRight": 6,
            "color": "#F5B97A",
            "opacity": 0.65,
        },
        "encoding": {
            "x": {"field": x_field, "type": "ordinal", "axis": {"labelAngle": 0}},
            "y": {
                "field": y_field,
                "type": "quantitative",
                "axis": {"title": f"{y_field}(건)"},
            },
            "tooltip": [
                {"field": x_field, "type": "ordinal"},
                {"field": y_field, "type": "quantitative"},
            ],
        },
        "height": height,
        "title": None,
        "config": _vega_base_config(),
    }


def _vega_weather_passenger_spec(x_field: str, title: str, height: int):
    return {
        "padding": {"top": 16, "right": 8, "bottom": 2, "left": 8},
        "layer": [
            {
                "transform": [{"calculate": "'월 강수량 합 (mm)'", "as": "시리즈"}],
                "mark": {"type": "bar", "color": "#B9CFE3", "opacity": 0.45},
                "encoding": {
                    "x": {"field": x_field, "type": "ordinal", "axis": {"labelAngle": 0}},
                    "y": {
                        "field": "강수량",
                        "type": "quantitative",
                        "axis": {"title": "강수량(mm)"},
                    },
                    "color": {
                        "field": "시리즈",
                        "type": "nominal",
                        "scale": {
                            "domain": ["월 강수량 합 (mm)"],
                            "range": ["#B9CFE3"],
                        },
                        "legend": {
                            "orient": "top",
                            "direction": "horizontal",
                            "title": None,
                            "offset": 6,
                            "padding": 0,
                            "labelFontSize": 10,
                            "labelLimit": 120,
                        },
                    },
                    "tooltip": [
                        {"field": x_field, "type": "ordinal"},
                        {"field": "강수량", "type": "quantitative"},
                    ],
                },
            },
            {
                "transform": [{"calculate": "'월 입도객수(명)'", "as": "시리즈"}],
                "mark": {
                    "type": "line",
                    "color": "#2CA02C",
                    "strokeWidth": 2.6,
                    "point": {"filled": True, "size": 70},
                },
                "encoding": {
                    "x": {"field": x_field, "type": "ordinal"},
                    "y": {
                        "field": "입도",
                        "type": "quantitative",
                        "axis": {"title": "여객수(명)", "orient": "right"},
                    },
                    "color": {
                        "field": "시리즈",
                        "type": "nominal",
                        "scale": {
                            "domain": ["월 입도객수(명)", "월 출도객수(명)"],
                            "range": ["#2CA02C", "#D62728"],
                        },
                        "legend": {
                            "orient": "top",
                            "direction": "horizontal",
                            "title": None,
                            "symbolType": "stroke",
                            "offset": 6,
                            "padding": 0,
                            "labelFontSize": 10,
                            "labelLimit": 120,
                        },
                    },
                    "tooltip": [
                        {"field": x_field, "type": "ordinal"},
                        {"field": "입도", "type": "quantitative"},
                    ],
                },
            },
            {
                "transform": [{"calculate": "'월 출도객수(명)'", "as": "시리즈"}],
                "mark": {
                    "type": "line",
                    "color": "#E45756",
                    "strokeWidth": 2.6,
                    "point": {"filled": True, "size": 70},
                },
                "encoding": {
                    "x": {"field": x_field, "type": "ordinal"},
                    "y": {
                        "field": "출도",
                        "type": "quantitative",
                        "axis": None,
                    },
                    "color": {
                        "field": "시리즈",
                        "type": "nominal",
                        "scale": {
                            "domain": ["월 입도객수(명)", "월 출도객수(명)"],
                            "range": ["#2CA02C", "#D62728"],
                        },
                        "legend": None,
                    },
                    "tooltip": [
                        {"field": x_field, "type": "ordinal"},
                        {"field": "출도", "type": "quantitative"},
                    ],
                },
            },
        ],
        "height": height,
        "resolve": {"scale": {"y": "independent", "color": "independent"}},
        "title": None,
        "config": _vega_base_config(),
    }


def _vega_bar_color_spec(
    x_field: str, y_field: str, color_field: str, title: str, height: int
):
    return {
        "padding": {"top": 10, "right": 8, "bottom": 2, "left": 18},
        "mark": {
            "type": "bar",
            "cornerRadiusTopLeft": 6,
            "cornerRadiusTopRight": 6,
            "opacity": 0.85,
        },
        "encoding": {
            "x": {"field": x_field, "type": "ordinal", "axis": {"labelAngle": 0}},
            "y": {
                "field": y_field,
                "type": "quantitative",
                "axis": {"title": "여객수(명)"},
            },
            "color": {
                "field": color_field,
                "type": "nominal",
                "scale": {
                    "domain": ["비수기", "성수기", "비수기(평균↑)"],
                    "range": ["#A9CFAE", "#F1C58B", "#E6D07A"],
                },
                "legend": {
                    "orient": "top-right",
                    "direction": "horizontal",
                    "title": None,
                    "padding": 0,
                    "offset": 6,
                    "labelFontSize": 10,
                },
            },
            "tooltip": [
                {"field": x_field, "type": "ordinal"},
                {"field": y_field, "type": "quantitative"},
                {"field": color_field, "type": "nominal"},
            ],
        },
        "height": height,
        "title": None,
        "config": _vega_base_config(),
    }


def _compute_season_map(monthly_df: pd.DataFrame, value_col: str):
    """월별 성수기/비수기 구분 맵 생성."""
    if monthly_df.empty or value_col not in monthly_df.columns:
        return {}, None

    work = monthly_df[["연", "월", value_col]].dropna()
    if work.empty:
        return {}, None

    month_counts = work.groupby("연")["월"].nunique()
    complete_years = month_counts[month_counts == 12].index.tolist()
    base_years = complete_years if complete_years else work["연"].unique().tolist()

    base_avg = (
        work[work["연"].isin(base_years)]
        .groupby("월")[value_col]
        .mean()
        .rename("Base_Avg")
        .reset_index()
    )

    start = int(work["연"].min())
    end = int(work["연"].max())
    year_month_index = pd.date_range(
        start=f"{start}-01-01",
        end=f"{end}-12-01",
        freq="MS",
    )
    full = pd.DataFrame({"YearMonth": year_month_index})
    full["연"] = full["YearMonth"].dt.year
    full["월"] = full["YearMonth"].dt.month

    full = full.merge(work, on=["연", "월"], how="left")
    full = full.merge(base_avg, on="월", how="left")
    full[value_col] = full[value_col].fillna(full["Base_Avg"])

    monthly_avg = full.groupby("월")[value_col].mean()
    threshold = monthly_avg.mean()

    season_map = {
        int(m): ("성수기" if v > threshold else "비수기")
        for m, v in monthly_avg.items()
    }
    return season_map, threshold


@st.cache_data(show_spinner=False)
def load_sms_classified() -> pd.DataFrame:
    """해상공지 분류 결과 CSV 로드."""
    path = Path(__file__).parent / "sms_msg_classified.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_sms_raw() -> pd.DataFrame:
    """원본 울릉알리미 SMS CSV 로드."""
    path = Path(__file__).parent / "울릉알리미_텍스트.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")


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

    work = df.copy()
    s = work["sms_resDate"].astype(str).str.strip()
    s = s.str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
    work["sms_resDate"] = pd.to_datetime(s, errors="coerce")
    work = work[work["sms_resDate"].dt.year == year]

    ship_keywords = [
        "금광해운",
        "대저해운",
        "대저해운 도착시간",
        "에이치해운",
        "우성해운",
        "주식회사태성해운",
        "태성해운 도착시간",
    ]
    ship_vessel_keywords = [
        "금광11호",
    ]
    people_keywords = [
        "대저페리",
        "썬라이즈 도착시간",
        "씨스포빌",
        "씨스포빌 도착시간",
        "울릉크루즈",
        "제이에이치페리",
        "제이에이치페리 도착시간",
    ]
    people_vessel_keywords = [
        "씨스타11호",
        "씨스타1호",
        "씨스타5호",
        "뉴씨다오펄호",
        "뉴시다오펄호",
        "썬라이즈호",
        "퀸스타2호",
        "미래15호",
        "익스프레스호",
        "엘도라도EX호",
        "울릉썬플라워크루즈호",
    ]
    passenger_keywords = ["탑승인원", "여객", "승객", "승선", "크루즈"]
    cargo_keywords = ["화물", "차량", "선적", "택배", "물류"]
    cancel_keywords = ["결항", "취소", "출항 취소", "운항 취소"]
    control_keywords = ["운항 통제", "운항통제", "운항이 통제", "통제되었습니다"]
    change_keywords = ["시간 변경", "시간변경", "시간 변경된", "시간변경된"]
    arrive_keywords = ["입항", "입항 예정", "입항 예정시간", "입항입니다"]
    depart_keywords = [
        "출항",
        "출발",
        "운항예정",
        "운항 예정",
        "정상운항",
        "운항합니다",
    ]

    def classify(msg: str) -> str | None:
        if not msg:
            return None
        if any(k in msg for k in cancel_keywords):
            return "결항"
        if any(k in msg for k in control_keywords):
            return "운항통제"
        if any(k in msg for k in change_keywords):
            return "시간변경"
        if any(k in msg for k in arrive_keywords):
            return "입항"
        if any(k in msg for k in depart_keywords):
            return "출항"
        return None

    def classify_group(msg: str) -> str | None:
        if (
            any(k in msg for k in ship_keywords)
            or any(k in msg for k in ship_vessel_keywords)
            or any(k in msg for k in cargo_keywords)
        ):
            return "선박"
        if (
            any(k in msg for k in people_keywords)
            or any(k in msg for k in people_vessel_keywords)
            or any(k in msg for k in passenger_keywords)
        ):
            return "사람"
        return None

    seen = set()
    seen_group = set()
    for _, row in work.iterrows():
        msg = str(row.get("sms_msg", "")).strip()
        if "셔틀" in msg:
            continue
        label = classify(msg)
        if not label:
            continue
        day = row["sms_resDate"].date() if pd.notna(row["sms_resDate"]) else None
        if day is None:
            continue
        if label in ("입항", "출항"):
            group = classify_group(msg)
            if group is None:
                continue
            key = (day, label, group)
            if key in seen_group:
                continue
            seen_group.add(key)
            breakdown[label][group] += 1
            continue
        key = (day, label)
        if key in seen:
            continue
        seen.add(key)
        counts[label] += 1

    counts["입항"] = breakdown["입항"]["선박"] + breakdown["입항"]["사람"]
    counts["출항"] = breakdown["출항"]["선박"] + breakdown["출항"]["사람"]

    total = sum(counts.values())
    return counts, total, breakdown


def _latest_sea_notice(df: pd.DataFrame, year: int = 2025) -> tuple[str, str]:
    """가장 최신 해상 공지 (카테고리, 요약 문자열)."""
    if df.empty or "sms_resDate" not in df.columns or "sms_msg" not in df.columns:
        return "입항", "최신 공지 없음"

    work = df.copy()
    s = work["sms_resDate"].astype(str).str.strip()
    s = s.str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
    work["sms_resDate"] = pd.to_datetime(s, errors="coerce")
    work = work[work["sms_resDate"].dt.year == year]
    work = work.dropna(subset=["sms_resDate"])
    if work.empty:
        return "입항", "최신 공지 없음"

    ship_keywords = [
        "금광해운",
        "대저해운",
        "대저해운 도착시간",
        "에이치해운",
        "우성해운",
        "주식회사태성해운",
        "태성해운 도착시간",
    ]
    ship_vessel_keywords = [
        "금광11호",
    ]
    people_keywords = [
        "대저페리",
        "썬라이즈 도착시간",
        "씨스포빌",
        "씨스포빌 도착시간",
        "울릉크루즈",
        "제이에이치페리",
        "제이에이치페리 도착시간",
    ]
    people_vessel_keywords = [
        "씨스타11호",
        "씨스타1호",
        "씨스타5호",
        "뉴씨다오펄호",
        "뉴시다오펄호",
        "썬라이즈호",
        "퀸스타2호",
        "미래15호",
        "익스프레스호",
        "엘도라도EX호",
        "울릉썬플라워크루즈호",
    ]
    cancel_keywords = ["결항", "취소", "출항 취소", "운항 취소"]
    control_keywords = ["운항 통제", "운항통제", "운항이 통제", "통제되었습니다"]
    change_keywords = ["시간 변경", "시간변경", "시간 변경된", "시간변경된"]
    arrive_keywords = ["입항", "입항 예정", "입항 예정시간", "입항입니다"]
    depart_keywords = [
        "출항",
        "출발",
        "운항예정",
        "운항 예정",
        "정상운항",
        "운항합니다",
    ]

    def classify(msg: str) -> str | None:
        if not msg:
            return None
        if any(k in msg for k in cancel_keywords):
            return "결항"
        if any(k in msg for k in control_keywords):
            return "운항통제"
        if any(k in msg for k in change_keywords):
            return "시간변경"
        if any(k in msg for k in arrive_keywords):
            return "입항"
        if any(k in msg for k in depart_keywords):
            return "출항"
        return None

    candidates = []
    for _, row in work.iterrows():
        msg = str(row.get("sms_msg", "")).strip()
        if not msg or "셔틀" in msg:
            continue
        label = classify(msg)
        if not label:
            continue
        candidates.append((row["sms_resDate"], msg, label))

    if not candidates:
        return "입항", "최신 공지 없음"

    candidates.sort(key=lambda x: x[0], reverse=True)
    dt, msg, label = candidates[0]

    names = (
        ship_keywords + ship_vessel_keywords + people_keywords + people_vessel_keywords
    )
    names = sorted(names, key=len, reverse=True)
    name = next((n for n in names if n in msg), "공지")

    time_match = re.search(r"(\d{1,2})[:시](\d{2})", msg)
    if time_match:
        time_text = f"{int(time_match.group(1)):02d}:{time_match.group(2)}"
    else:
        time_text = dt.strftime("%H:%M") if dt else ""

    day_text = f"{dt.day}일" if dt else ""
    parts = [name]
    if day_text:
        parts.append(f"({day_text})")
    if time_text and time_text != "00:00":
        parts.append(time_text)
    return label, " ".join(parts).strip()


# -----------------------------
# Defaults (상단 설정 UI 제거)
# -----------------------------
# 필요하면 나중에 다시 UI로 바꿀 수 있게 값만 변수로 유지

date_range = []
region = "울릉도 전체"
show_graphs = True
show_sea_notice = True
show_road_control = True

# -----------------------------
# Session state init (첫 로드 시 선택값 비우기)
# -----------------------------
if "selected_acc_meta" not in st.session_state:
    st.session_state["selected_acc_meta"] = None
if "selected_acc_photo_path" not in st.session_state:
    st.session_state["selected_acc_photo_path"] = None
if "selected_acc_year" not in st.session_state:
    st.session_state["selected_acc_year"] = None
if "selected_acc_idx" not in st.session_state:
    st.session_state["selected_acc_idx"] = None
if "selected_rockfall_meta" not in st.session_state:
    st.session_state["selected_rockfall_meta"] = None
if "selected_rockfall_photo_path" not in st.session_state:
    st.session_state["selected_rockfall_photo_path"] = None
if "selected_bus_meta" not in st.session_state:
    st.session_state["selected_bus_meta"] = None
if "selected_rock_idx" not in st.session_state:
    st.session_state["selected_rock_idx"] = None
if "rock_view_mode" not in st.session_state:
    st.session_state["rock_view_mode"] = "list"

# -----------------------------
# Helper functions for accident photo lookup
# -----------------------------


def _norm_text(s: str) -> str:
    """주소/파일명 매칭용 간단 정규화 (공백/특수문자 제거)."""
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFC", s)
    s = s.strip().lower()
    return re.sub(r"[^0-9a-z가-힣]+", "", s)


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


@st.cache_data(show_spinner=False)
def _build_accident_photo_index() -> dict[str, str]:
    acc_dir = Path(__file__).parent / "acc_pic"
    if not acc_dir.exists() or not acc_dir.is_dir():
        return {}
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    out = {}
    for p in acc_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        key = _norm_text(p.stem)
        if key and key not in out:
            out[key] = str(p)
    return out


def _find_accident_photo_fast(address: str) -> str | None:
    targets = _address_candidates(address)
    if not targets:
        return None
    idx = _build_accident_photo_index()
    for t in targets:
        if t in idx:
            return idx[t]
    # fallback to slower fuzzy match
    p = find_accident_photo_by_address(address)
    return str(p) if p else None


def _format_accident_datetime(df: pd.DataFrame, row: pd.Series) -> str:
    candidates = [
        "발생일시",
        "발생일자",
        "사고일시",
        "일시",
        "date",
        "datetime",
        "발생일",
    ]
    for c in candidates:
        if c not in df.columns:
            continue
        v = row.get(c, None)
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() in ["nan", "none"]:
            continue
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            continue
        if dt.hour == 0 and dt.minute == 0:
            return dt.strftime("%Y-%m-%d")
        return dt.strftime("%Y-%m-%d %H:%M")
    return "미상"


@st.cache_data(show_spinner=False)
def find_accident_photo_by_address(address: str):
    """acc_pic 폴더에서 '주소.JPG' 규칙으로 저장된 사진을 찾음."""
    acc_dir = Path(__file__).parent / "acc_pic"
    if not acc_dir.exists() or not acc_dir.is_dir():
        return None

    targets = _address_candidates(address)
    if not targets:
        return None

    exts = {".jpg", ".jpeg", ".png", ".webp"}

    for p in acc_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        stem_key = _norm_text(p.stem)
        if any(stem_key == t for t in targets):
            return p

    # 완전 일치가 없으면 포함 매칭(옵션)
    for p in acc_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        stem_key = _norm_text(p.stem)
        if any(t and (t in stem_key or stem_key in t) for t in targets):
            return p

    return None


def _find_rockfall_photo(address: str | Path | None):
    """rockfall 폴더에서 주소/파일명 기반으로 사진 찾기."""
    rock_dir = Path(__file__).parent / "rockfall"
    if not rock_dir.exists() or not rock_dir.is_dir():
        return None

    target = _norm_text(str(address)) if address is not None else ""
    if not target:
        return None

    exts = {".jpg", ".jpeg", ".png", ".webp"}

    for p in rock_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        candidate = _norm_text(p.stem)
        if candidate == target or target in candidate or candidate in target:
            return p

    return None


def _set_selected_accident(df_acc: pd.DataFrame, idx: int):
    if df_acc.empty or idx not in df_acc.index:
        return

    row = df_acc.loc[idx]

    # 1. 연도 추출
    year_val = 2025
    if "year" in df_acc.columns:
        try:
            year_val = int(row.get("year"))
        except:
            year_val = 2025

    # 2. 주소 추출
    addr = _row_to_address(df_acc, row)

    # 3. 상세 정보 및 타입 추출
    detail_txt = str(row.get("detail", "")).strip()
    if detail_txt.lower() in ["nan", "none"]:
        detail_txt = ""

    acc_type = "미상"
    for c in ["type", "accident_type", "사고유형", "사고_type"]:
        if c in df_acc.columns:
            val = str(row.get(c, "")).strip()
            if val and val.lower() not in ["nan", "none"]:
                acc_type = val
                break

    # 4. 사진 찾기 (연도 제한 없이 무조건 시도)
    photo = find_accident_photo_by_address(addr)

    # 5. 텍스트 구성
    detail_label = detail_txt if detail_txt else "(없음)"
    addr_label = addr if addr else "(없음)"
    summary = f"{detail_label} 인근, {acc_type} 발생. 주의 요망."

    # 6. 세션 상태 업데이트 (교통사고 정보 입력)
    st.session_state["selected_acc_meta"] = (
        f"연도: {year_val}\n위치: {detail_label}\n유형: {acc_type}\n주소: {addr_label}\n{summary}"
    )
    st.session_state["selected_acc_photo_path"] = str(photo) if photo else None
    st.session_state["selected_acc_year"] = year_val

    # [핵심] 낙석 및 버스 정보는 '반드시' 지워야 화면이 전환됨
    st.session_state["selected_rockfall_meta"] = None
    st.session_state["selected_rockfall_photo_path"] = None
    st.session_state["selected_bus_meta"] = None


# -----------------------------
# Top Notice Bar (공지 자동 순환)
# -----------------------------
NOTICES = [
    "전체 공지 : [보건의료원] 금일 오전 내과 진료가 마감되었습니다. 진료를 원하시는 분들은 오후에 내원해 주시기 바랍니다.",
    "전체 공지 : [재무과] <2025년 12월 자동차세 납부 안내> ○납부기한: 12월31일(수)까지 ○문의: 790-6123,6127 ※납부일정을 확인하시어 납기내 납부 부탁드립니다. *자동이체 신청자는 31일 계좌 잔액 확인*",
    "전체 공지 : [상하수도사업소] 금일(월) 09시30부터~10시30분까지 상수도 관로복구공사로 인하여 [남양 일대] 단수 예정이오니 주민 여러분의 양해 부탁드립니다.",
    "전체 공지 : [문화체육과] 울쓰마스EDM party행사 구조물 철거작업으로 인하여 2025년 12월 28일(일) 체육시설은 배드민턴, 탁구만 이용이 가능합니다.",
]
NOTICE_INTERVAL_SEC = 5  # 몇 초마다 바꿀지

try:
    # 권장: pip install streamlit-autorefresh
    from streamlit_autorefresh import st_autorefresh

    _notice_count = st_autorefresh(
        interval=NOTICE_INTERVAL_SEC * 1000,
        limit=None,
        key="notice_autorefresh",
    )
except Exception:
    # autorefresh가 없으면, 현재 시간 기반으로 인덱스만 계산(사용자 인터랙션/새로고침 시 변경)
    _notice_count = int(time.time() // NOTICE_INTERVAL_SEC)

_notice_idx = int(_notice_count) % len(NOTICES)

_notice_text = NOTICES[_notice_idx]
_prefix = "전체 공지 :"
if isinstance(_notice_text, str) and _notice_text.startswith(_prefix):
    _rest = _notice_text[len(_prefix) :].lstrip()
    _notice_html = f"<span style='font-weight:800;'>{_prefix}</span> {_rest}"
else:
    _notice_html = _notice_text

logo_path = Path(__file__).parent / "logo.svg"
logo_html = ""
if logo_path.exists():
    try:
        svg_bytes = logo_path.read_bytes()
        svg_b64 = base64.b64encode(svg_bytes).decode("ascii")
        logo_html = (
            f'<img src="data:image/svg+xml;base64,{svg_b64}" alt="울릉군 마크" />'
        )
    except Exception:
        logo_html = ""
st.markdown(
    f"""
    <div class="dashboard-title">
        {logo_html}
        <div class="title-text">울릉도 데이터 대시보드</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")
st.markdown(
    f'<div class="notice-pill">{_notice_html}</div>',
    unsafe_allow_html=True,
)
st.write("")  # 약간의 여백

# =============================
# Row 2: Layer 2개 (해상공지 / 도로통제)
# =============================
sms_counts, sms_total, sms_breakdown = _summarize_sms_notice_counts(
    load_sms_raw(),
    year=2025,
)
sea_latest_label, sea_latest_text = _latest_sea_notice(load_sms_raw(), year=2025)


# [수정] 백분율 계산 로직 개선
def _pct(count: int, total: int) -> int:
    if total <= 0:
        return 0
    return int(round(count / total * 100))


def _bar_pct(count: int, total: int, min_pct: int = 6) -> int:
    if total <= 0 or count <= 0:
        return 0
    pct = int(round(count / total * 100))
    return max(pct, min_pct)

# 1. 각 항목의 건수 가져오기
sea_arrive = sms_counts["입항"]
sea_depart = sms_counts["출항"]
sea_control = sms_counts["운항통제"]
sea_cancel = sms_counts["결항"]
sea_change = sms_counts["시간변경"]

# [수정] 막대 그래프의 '시각적 스케일'을 위해 전체 합(Total)이 아닌 최댓값(Max)을 기준으로 100%를 잡음
sea_max_val = max(sea_arrive, sea_depart, sea_control, sea_cancel, sea_change)
if sea_max_val == 0:
    sea_max_val = 1

sea_arrive_pct = _bar_pct(sea_arrive, sea_max_val)
sea_depart_pct = _bar_pct(sea_depart, sea_max_val)
sea_control_pct = _bar_pct(sea_control, sea_max_val)
sea_cancel_pct = _bar_pct(sea_cancel, sea_max_val)
sea_change_pct = _bar_pct(sea_change, sea_max_val)

# 2. 내부 분할(선박/사람) 비율은 해당 항목의 합계를 기준으로 계산 (이건 기존 유지)
sea_arrive_ship = sms_breakdown["입항"]["선박"]
sea_arrive_people = sms_breakdown["입항"]["사람"]
sea_depart_ship = sms_breakdown["출항"]["선박"]
sea_depart_people = sms_breakdown["출항"]["사람"]

# 내부 세그먼트 비율 계산
sea_arrive_ship_pct = _pct(sea_arrive_ship, sea_arrive)
sea_arrive_people_pct = 100 - sea_arrive_ship_pct if sea_arrive > 0 else 0
sea_depart_ship_pct = _pct(sea_depart_ship, sea_depart)
sea_depart_people_pct = 100 - sea_depart_ship_pct if sea_depart > 0 else 0

st.write("")
c1, c2 = st.columns(2, gap="large")

# [수정] 카드 높이 조절
ROW2_CARD_H = 420

with c1:
    with st.container(border=True, height=ROW2_CARD_H):
        if show_sea_notice:
            html = "\n".join(
                line.lstrip()
                for line in textwrap.dedent(
                    f"""
<div class="r2-card">
  <div class="r2-head">
    <div class="r2-title">해상 공지</div>
    <div class="r2-date">2025년 기준</div>
  </div>

  <div class="sea-section">
    <div class="sea-section-title">최신 공지</div>
    <div class="sea-latest">
      <div class="sea-pill">{sea_latest_label}</div>
      <div class="sea-latest-text">{sea_latest_text}</div>
    </div>
  </div>

  <div class="sea-section">
    <div class="sea-section-title">통계 요약 (건수)</div>
    <div class="sea-bars">
      <div class="bar-row">
        <div class="bar-label">
          <div class="bar-label-wrap">
            <span>입항</span>
            <span class="bar-sub">(선박/사람)</span>
            <span class="help-pop">
              <span class="help-pop-btn">?</span>
              <span class="help-pop-body">
                입항 알림 합계: <b>{sea_arrive:,}건</b><br/>
                선박: {sea_arrive_ship}건, 사람: {sea_arrive_people}명
              </span>
            </span>
          </div>
        </div>
        <div class="bar-track">
          <div class="bar-fill-split" style="width:{sea_arrive_pct}%;">
            <div class="bar-seg" style="width:{sea_arrive_ship_pct}%; background:#ff8a3d;"></div>
            <div class="bar-seg" style="width:{sea_arrive_people_pct}%; background:#ffd3a8;"></div>
            <div class="bar-value-onfill">{sea_arrive:,}</div>
          </div>
        </div>
      </div>

      <div class="bar-row">
        <div class="bar-label">
          <div class="bar-label-wrap">
            <span>출항</span>
            <span class="bar-sub">(선박/사람)</span>
            <span class="help-pop">
              <span class="help-pop-btn">?</span>
              <span class="help-pop-body">
                출항 알림 합계: <b>{sea_depart:,}건</b><br/>
                선박: {sea_depart_ship}건, 사람: {sea_depart_people}명
              </span>
            </span>
          </div>
        </div>
        <div class="bar-track">
          <div class="bar-fill-split" style="width:{sea_depart_pct}%;">
            <div class="bar-seg" style="width:{sea_depart_ship_pct}%; background:#00b3a4;"></div>
            <div class="bar-seg" style="width:{sea_depart_people_pct}%; background:#8fe3da;"></div>
            <div class="bar-value-onfill">{sea_depart:,}</div>
          </div>
        </div>
      </div>

      <div class="bar-row">
        <div class="bar-label">
          <div class="bar-label-wrap">
            <span>운항통제</span>
            <span class="help-pop">
              <span class="help-pop-btn">?</span>
              <span class="help-pop-body">
                기상 악화 등으로 통제된 알림 수입니다.<br/>
                배 운항통제 건수: {sea_control:,}건
              </span>
            </span>
          </div>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:{sea_control_pct}%; background:#5b2bff;">
            <div class="bar-value-onfill">{sea_control:,}</div>
          </div>
        </div>
      </div>

      <div class="bar-row">
        <div class="bar-label">
          <div class="bar-label-wrap">
            <span>결항</span>
            <span class="help-pop">
              <span class="help-pop-btn">?</span>
              <span class="help-pop-body">
                기상 또는 점검 사유로 취소된 알림 수입니다.<br/>
                배 결항 건수: {sea_cancel:,}건
              </span>
            </span>
          </div>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:{sea_cancel_pct}%; background:#e24a4a;">
            <div class="bar-value-onfill">{sea_cancel:,}</div>
          </div>
        </div>
      </div>

      <div class="bar-row">
        <div class="bar-label">
          <div class="bar-label-wrap">
            <span>시간변경</span>
            <span class="help-pop">
              <span class="help-pop-btn">?</span>
              <span class="help-pop-body">
                출항/입항 시간이 변경된 알림 수입니다.<br/>
                배 시간변경 건수: {sea_change:,}건
              </span>
            </span>
          </div>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:{sea_change_pct}%; background:#7b61ff;">
            <div class="bar-value-onfill">{sea_change:,}</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
                    """
                ).splitlines()
            )
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.caption("사이드바에서 해상공지 레이어가 꺼져있음")

with c2:
    with st.container(border=True, height=ROW2_CARD_H):
        if show_road_control:
            head_left, head_right = st.columns([1, 0.35])
            with head_left:
                st.markdown(
                    """
<div class="r2-top">
  <div class="r2-title">도로 통제 공지</div>
  <div class="r2-date">최신 기준</div>
</div>
                    """,
                    unsafe_allow_html=True,
                )
            with head_right:
                with st.popover("안전 안내"):
                    st.write(
                        "- 통제 구간 진입 전 우회 경로를 확인해 주세요.\n"
                        "- 현장 안내 요원의 지시에 따라 서행/정차해 주세요.\n"
                        "- 야간에는 전조등을 켜고 낙석 구간은 주의 운행 바랍니다.\n"
                        "- 긴급 상황 시 112 또는 119로 즉시 연락해 주세요."
                    )

            st.markdown(
                """
<div class="r2-card r2-card-body">
  <div class="road-list">
    <div class="road-item">
      <div class="road-item-title"><span class="road-tag">주차장 정비</span>사동항 주차장 전면 통제</div>
      <div class="road-item-meta">차량을 다른 곳으로 이동 주차 바랍니다.</div>
    </div>
    <div class="road-item">
      <div class="road-item-title"><span class="road-tag">도로공사</span>나리 도로구간 공사</div>
      <div class="road-item-meta">도로열선 관련 공사 중 · 통행 주의</div>
    </div>
    <div class="road-item">
      <div class="road-item-title"><span class="road-tag">이동요청</span>도동약수공원 주차장 도색작업</div>
      <div class="road-item-meta">11.11.(화)~11.14.(금) 차량 이동 요청</div>
    </div>
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.caption("사이드바에서 도로통제 레이어가 꺼져있음")

st.write("")

# =============================
# Row 1: (탭 내 전환형) 목록 보기 vs 지도 보기
# =============================

# 상태 초기화: 기본은 'list' 모드
if "view_mode" not in st.session_state:
    st.session_state["view_mode"] = "list"

# 상단 2개 카드(좌/우) 영역 높이 고정
TOP_CARD_H = 600  # 전체 카드 높이(px)
PHOTO_H = 280  # 사진 영역 높이(px)
MAP_H = 360  # 지도 영역 높이(px)

with st.container(border=True, height=TOP_CARD_H):
    st.markdown('<div class="card-title">울릉군 지도</div>', unsafe_allow_html=True)
    st.caption("2019-2025년 울릉군 위치 데이터 기반")

    # 지도 상단 탭
    t1, t2, t3 = st.tabs(["버스 실시간 상황", "교통사고 지점", "낙석 발생 지점"])

    def _render_photo_detail_panel(key_suffix: str):
        with st.container(border=True, height=TOP_CARD_H):
            st.markdown('<div class="card-title">사고 장소 사진</div>', unsafe_allow_html=True)

            sel_rock_photo = st.session_state.get("selected_rockfall_photo_path")
            sel_acc_photo = st.session_state.get("selected_acc_photo_path")
            sel_acc_meta = st.session_state.get("selected_acc_meta")
            sel_rock_meta = st.session_state.get("selected_rockfall_meta")
            sel_bus_meta = st.session_state.get("selected_bus_meta")

            with st.container(height=PHOTO_H):
                image_loaded = False
                if sel_rock_photo:
                    try:
                        path_str = str(sel_rock_photo)
                        if os.path.isfile(path_str):
                            img = Image.open(path_str)
                            st.image(img, width="stretch")
                            image_loaded = True
                    except Exception:
                        pass
                elif sel_acc_photo and not image_loaded:
                    try:
                        path_str = str(sel_acc_photo)
                        if os.path.isfile(path_str):
                            img = Image.open(path_str)
                            st.image(img, width="stretch")
                            image_loaded = True
                    except Exception:
                        pass

                if not image_loaded and (sel_acc_meta or sel_rock_meta):
                    st.markdown(
                        """
                        <div class="photo-placeholder">등록된 현장 사진이 없습니다.<br/><span style="font-size:0.8rem;">(지도상의 위치를 참고해주세요)</span></div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif not image_loaded and not sel_acc_meta and not sel_rock_meta:
                    st.info(
                        "- 우측 지도에서 사고 지점을 클릭하면, 선택된 사고의 정보가 갱신됩니다.\n"
                        "- 관련 사진이 등록된 사고의 경우, 본 영역에 사고 장소 사진이 표시됩니다."
                    )

            st.write("")
            if image_loaded and (sel_rock_photo or sel_acc_photo):
                selected_photo_path = sel_rock_photo or sel_acc_photo

                @st.dialog("사진 확대")
                def _show_photo_dialog(photo_path: str):
                    try:
                        st.image(str(photo_path), width="content")
                    except Exception:
                        st.warning("이미지를 불러오지 못했어.")

                if st.button("사진 확대 보기", key=f"photo_zoom_{key_suffix}"):
                    _show_photo_dialog(selected_photo_path)

            st.write("")
            st.markdown('<div class="card-title">자세히 보기</div>', unsafe_allow_html=True)
            if sel_rock_meta:
                st.markdown(str(sel_rock_meta).replace("\n", "  \n"))
            elif sel_bus_meta:
                st.markdown(str(sel_bus_meta).replace("\n", "  \n"))
            elif sel_acc_meta:
                st.markdown(str(sel_acc_meta).replace("\n", "  \n"))
            else:
                st.markdown("- 지도에서 마커를 클릭하면 상세 정보가 이곳에 표시됩니다.")

    # [탭 1] 버스
    with t1:
        left_main, right_detail = st.columns([2.2, 1], gap="large")
        with left_main:
            st.caption("울릉군 버스 노선/정류장")
            bus_map_state = render_ulleung_folium_map(kind="bus", height=MAP_H)
            if isinstance(bus_map_state, dict):
                last = bus_map_state.get("last_object_clicked")
                bus_meta = st.session_state.get("bus_stops_meta", [])
                if isinstance(last, dict) and "lat" in last and "lng" in last and bus_meta:
                    lat0 = float(last["lat"])
                    lon0 = float(last["lng"])
                    best = None
                    best_d = None
                    for p in bus_meta:
                        d = abs(float(p["lat"]) - lat0) + abs(float(p["lon"]) - lon0)
                        if best_d is None or d < best_d:
                            best_d = d
                            best = p
                    if best is not None and best_d is not None and best_d < 0.002:
                        st.session_state["selected_acc_meta"] = None
                        st.session_state["selected_acc_photo_path"] = None
                        st.session_state["selected_rockfall_meta"] = None
                        st.session_state["selected_rockfall_photo_path"] = None
                        name = best.get("name", "")
                        routes_txt = (
                            ", ".join(best.get("routes", []))
                            if best.get("routes")
                            else "노선 정보 없음"
                        )
                        st.session_state["selected_bus_meta"] = (
                            f"정류장 : {name}\n경유 노선 : {routes_txt}"
                        )
            st.caption(f"조회기준: {datetime.now():%Y-%m-%d %H:%M}")

        with right_detail:
            routes_defs = {r["id"]: r for r in _bus_route_defs()}
            route_22 = routes_defs.get("22")
            route_3 = routes_defs.get("3")

            def _route_dir_label(route):
                if not route or not route.get("stops"):
                    return "상행 -> (정보 없음), 하행 -> (정보 없음)"
                up = route["stops"][0]
                down = route["stops"][-1]
                return f"상행 -> {up}, 하행 -> {down}"

            with st.container(border=True, height=TOP_CARD_H):
                st.markdown('<div class="card-title">버스 실시간 정보</div>', unsafe_allow_html=True)
                st.markdown(
                    f"""
<div style="padding:10px 12px; border:1px solid #e8ebf2; border-radius:12px; margin-bottom:10px; background:#f8f9fc;">
  <div style="font-weight:700;">22노선</div>
  <div style="color:#444; font-size:0.9rem;">{_route_dir_label(route_22)}</div>
</div>
<div style="padding:10px 12px; border:1px solid #e8ebf2; border-radius:12px; margin-bottom:10px; background:#f8f9fc;">
  <div style="font-weight:700;">3노선</div>
  <div style="color:#444; font-size:0.9rem;">{_route_dir_label(route_3)}</div>
</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="card-title">정류장 상세</div>', unsafe_allow_html=True)
                sel_bus_meta = st.session_state.get("selected_bus_meta")
                if sel_bus_meta:
                    st.markdown(str(sel_bus_meta).replace("\n", "  \n"))
                else:
                    st.markdown("- 지도에서 정류장을 클릭하면 상세 정보가 표시됩니다.")

    # [탭 2] 교통사고 (탭 안에서 목록/지도 전환)
    with t2:
        left_main, right_detail = st.columns([2.2, 1], gap="large")
        with left_main:
            top_left, top_right = st.columns([4, 1])
            with top_left:
                if st.session_state["view_mode"] == "list":
                    st.caption("발생한 사고 목록입니다. 위치 확인 버튼을 누르면 지도로 이동합니다.")
                else:
                    st.caption("울릉군 교통사고 지점")
            with top_right:
                if st.session_state["view_mode"] == "list":
                    if st.button(
                        "🗺️ 지도에서 보기",
                        use_container_width=True,
                        type="primary",
                        key="acc_view_map",
                    ):
                        st.session_state["view_mode"] = "map"
                        st.rerun()
                else:
                    if st.button(
                        "⬅ 목록으로",
                        use_container_width=True,
                        key="acc_view_list",
                    ):
                        st.session_state["view_mode"] = "list"
                        st.rerun()

            df_acc_list = load_accidents_csv(_accident_files_signature())
            if df_acc_list.empty:
                st.info("표시할 사고 데이터가 없습니다.")
            else:
                if st.session_state["view_mode"] == "list":
                    seen_keys = set()
                    with st.container(height=TOP_CARD_H - 120, border=True):
                        df_list_view = df_acc_list.copy()
                        if "year" in df_list_view.columns:
                            df_list_view["_year_sort"] = df_list_view["year"].fillna(0).astype(int)
                            df_list_view = df_list_view.sort_values(
                                by="_year_sort", ascending=False
                            )
                        for idx, row in df_list_view.head(10).iterrows():
                            year_val = row.get("year", 2025)
                            acc_type = row.get("type", "미상")
                            if pd.isna(acc_type):
                                acc_type = "미상"

                            addr = _row_to_address(df_acc_list, row)
                            addr_key = _norm_text(addr) if addr else ""
                            if addr_key and addr_key in seen_keys:
                                continue
                            if addr_key:
                                seen_keys.add(addr_key)
                            detail = str(row.get("detail", "")).strip()
                            if detail == "nan":
                                detail = ""

                            display_title = detail if detail else addr
                            if not display_title:
                                display_title = "위치 정보 없음"

                            lat = row.get("latitude", None)
                            lon = row.get("longitude", None)
                            lat_lon = (
                                f"{float(lat):.5f}, {float(lon):.5f}"
                                if pd.notna(lat) and pd.notna(lon)
                                else "미상"
                            )
                            photo_path = _find_accident_photo_fast(addr) if addr else None
                            is_selected = st.session_state.get("selected_acc_idx") == idx

                            with st.container(border=True):
                                c_img, c_info, c_btn = st.columns([1.2, 3, 1])
                                with c_img:
                                    if photo_path and os.path.isfile(str(photo_path)):
                                        try:
                                            st.image(str(photo_path), width="stretch")
                                        except Exception:
                                            st.markdown(
                                                """
                                                <div style="background:#f0f2f6; height:86px; display:flex; align-items:center; justify-content:center; border-radius:8px; color:#999; font-size:0.8rem;">
                                                    사진 불러오는 중
                                                </div>
                                                """,
                                                unsafe_allow_html=True,
                                            )
                                    else:
                                        st.markdown(
                                            """
                                            <div style="background:#f0f2f6; height:86px; display:flex; align-items:center; justify-content:center; border-radius:8px; color:#999; font-size:0.8rem;">
                                                사진 준비중
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                                with c_info:
                                    sel_tag = " <span style='color:#d12c2c;'>● 선택</span>" if is_selected else ""
                                    st.markdown(f"**{display_title}**{sel_tag}", unsafe_allow_html=True)
                                    st.caption(f"발생연도: {year_val} | 유형: {acc_type}")
                                    st.markdown(
                                        f"<div style='color:#666; font-size:0.85rem;'>위치: {addr if addr else '미상'}<br/>좌표: {lat_lon}</div>",
                                        unsafe_allow_html=True,
                                    )
                                with c_btn:
                                    st.write("")
                                    if st.button(
                                        "위치 확인 >",
                                        key=f"btn_go_map_{idx}",
                                        use_container_width=True,
                                    ):
                                        _set_selected_accident(df_acc_list, idx)
                                        st.session_state["selected_acc_idx"] = int(idx)
                                        st.session_state["view_mode"] = "map"
                                        st.rerun()
                else:
                    df_acc = df_acc_list
                    year_filter = None
                    if "year" in df_acc.columns and not df_acc["year"].dropna().empty:
                        years = sorted({int(y) for y in df_acc["year"].dropna().unique()})
                        idx_2025 = years.index(2025) + 1 if 2025 in years else 0
                        options = ["전체"] + [str(y) for y in years]
                        selected_year_label = st.selectbox("연도 선택", options, index=idx_2025)
                        if selected_year_label != "전체":
                            year_filter = int(selected_year_label)
                    df_view = df_acc
                    if year_filter is not None:
                        df_view = df_acc[df_acc["year"] == year_filter]

                    highlight_idx = st.session_state.get("selected_acc_idx")
                    center_override = None
                    if highlight_idx is not None:
                        for p in st.session_state.get("acc_points_meta", []):
                            if int(p.get("idx", -1)) == int(highlight_idx):
                                center_override = (float(p["lat"]), float(p["lon"]))
                                break

                    map_state = render_ulleung_folium_map(
                        kind="accident",
                        height=MAP_H,
                        accident_df=df_view,
                        highlight_idx=highlight_idx,
                        center_override=center_override,
                    )

                    if isinstance(map_state, dict):
                        last = map_state.get("last_object_clicked")
                        if isinstance(last, dict) and "lat" in last and "lng" in last:
                            lat0 = float(last["lat"])
                            lon0 = float(last["lng"])
                            best_idx = None
                            best_d = None
                            for i in df_view.index:
                                row_lat = df_view.at[i, "latitude"]
                                row_lon = df_view.at[i, "longitude"]
                                d = abs(row_lat - lat0) + abs(row_lon - lon0)
                                if best_d is None or d < best_d:
                                    best_d = d
                                    best_idx = i
                            if best_d is not None and best_d < 0.002:
                                st.session_state["selected_rockfall_meta"] = None
                                st.session_state["selected_rockfall_photo_path"] = None
                                st.session_state["selected_bus_meta"] = None
                                _set_selected_accident(df_acc, best_idx)
                                st.session_state["selected_acc_idx"] = int(best_idx)

        with right_detail:
            _render_photo_detail_panel("accident")

    # [탭 3] 낙석
    with t3:
        left_main, right_detail = st.columns([2.2, 1], gap="large")
        with left_main:
            top_left, top_right = st.columns([4, 1])
            with top_left:
                if st.session_state["rock_view_mode"] == "list":
                    st.caption("낙석 발생 목록입니다. 위치 확인 버튼을 누르면 지도로 이동합니다.")
                else:
                    st.caption("울릉군 낙석 발생 지점")
            with top_right:
                if st.session_state["rock_view_mode"] == "list":
                    if st.button(
                        "🗺️ 지도에서 보기",
                        use_container_width=True,
                        type="primary",
                        key="rock_view_map",
                    ):
                        st.session_state["rock_view_mode"] = "map"
                        st.rerun()
                else:
                    if st.button(
                        "⬅ 목록으로",
                        use_container_width=True,
                        key="rock_view_list",
                    ):
                        st.session_state["rock_view_mode"] = "list"
                        st.rerun()

            def _rockfall_meta_text(item: dict):
                location_label = item.get("name") or "(없음)"
                date_val = item.get("date", None)
                damage_val = item.get("damage", None)
                date_label = (
                    "미상"
                    if date_val in (None, "") or pd.isna(date_val)
                    else str(date_val).strip()
                )
                damage_label = (
                    "미상"
                    if damage_val in (None, "") or pd.isna(damage_val)
                    else str(damage_val).strip()
                )
                return "\n".join(
                    [
                        f"발견일: {date_label}",
                        f"위치: {location_label}",
                        f"피해여부: {damage_label}",
                        "조치상태: 완료",
                    ]
                )

            if st.session_state["rock_view_mode"] == "list":
                _, rock_meta = load_rockfall_points()
                if not rock_meta:
                    st.info("표시할 낙석 데이터가 없습니다.")
                else:
                    with st.container(height=TOP_CARD_H - 120, border=True):
                        for item in rock_meta[:10]:
                            item_idx = int(item.get("idx", 0))
                            name = item.get("name", "위치 미상")
                            photo = item.get("photo", None)
                            lat = item.get("lat", None)
                            lon = item.get("lon", None)
                            lat_lon = (
                                f"{float(lat):.5f}, {float(lon):.5f}"
                                if pd.notna(lat) and pd.notna(lon)
                                else "미상"
                            )
                            is_selected = st.session_state.get("selected_rock_idx") == item_idx
                            date_val = item.get("date", None)
                            damage_val = item.get("damage", None)
                            date_label = (
                                "미상"
                                if date_val in (None, "") or pd.isna(date_val)
                                else str(date_val).strip()
                            )
                            damage_label = (
                                "미상"
                                if damage_val in (None, "") or pd.isna(damage_val)
                                else str(damage_val).strip()
                            )

                            with st.container(border=True):
                                c_img, c_info, c_btn = st.columns([1.2, 3, 1])
                                with c_img:
                                    if photo and os.path.isfile(str(photo)):
                                        try:
                                            st.image(str(photo), width="stretch")
                                        except Exception:
                                            st.markdown(
                                                """
                                                <div style="background:#f0f2f6; height:86px; display:flex; align-items:center; justify-content:center; border-radius:8px; color:#999; font-size:0.8rem;">
                                                    사진 불러오는 중
                                                </div>
                                                """,
                                                unsafe_allow_html=True,
                                            )
                                    else:
                                        st.markdown(
                                            """
                                            <div style="background:#f0f2f6; height:86px; display:flex; align-items:center; justify-content:center; border-radius:8px; color:#999; font-size:0.8rem;">
                                                사진 준비중
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                                with c_info:
                                    sel_tag = (
                                        " <span style='color:#d12c2c;'>● 선택</span>"
                                        if is_selected
                                        else ""
                                    )
                                    st.markdown(f"**{name}**{sel_tag}", unsafe_allow_html=True)
                                    st.caption(
                                        f"발견일: {date_label} | 피해여부: {damage_label}"
                                    )
                                    st.markdown(
                                        f"<div style='color:#666; font-size:0.85rem;'>조치상태: 완료<br/>좌표: {lat_lon}</div>",
                                        unsafe_allow_html=True,
                                    )
                                with c_btn:
                                    st.write("")
                                    if st.button(
                                        "위치 확인 >",
                                        key=f"btn_rock_map_{item_idx}",
                                        use_container_width=True,
                                    ):
                                        st.session_state["selected_acc_meta"] = None
                                        st.session_state["selected_acc_photo_path"] = None
                                        st.session_state["selected_bus_meta"] = None
                                        st.session_state["selected_rock_idx"] = item_idx
                                        st.session_state["selected_rockfall_meta"] = _rockfall_meta_text(
                                            item
                                        )
                                        st.session_state["selected_rockfall_photo_path"] = (
                                            str(photo) if photo else None
                                        )
                                        st.session_state["rock_view_mode"] = "map"
                                        st.rerun()
            else:
                highlight_idx = st.session_state.get("selected_rock_idx")
                center_override = None
                _, rock_meta = load_rockfall_points()
                if highlight_idx is not None:
                    for p in rock_meta:
                        if int(p.get("idx", -1)) == int(highlight_idx):
                            center_override = (float(p["lat"]), float(p["lon"]))
                            break
                rock_map_state = render_ulleung_folium_map(
                    kind="rockfall",
                    height=MAP_H,
                    highlight_idx=highlight_idx,
                    center_override=center_override,
                )
                if isinstance(rock_map_state, dict):
                    last = rock_map_state.get("last_object_clicked")
                    rock_meta = st.session_state.get("rockfall_points_meta", [])
                    if (
                        isinstance(last, dict)
                        and "lat" in last
                        and "lng" in last
                        and rock_meta
                    ):
                        lat0 = float(last["lat"])
                        lon0 = float(last["lng"])
                        best = None
                        best_d = None
                        for p in rock_meta:
                            d = abs(float(p["lat"]) - lat0) + abs(float(p["lon"]) - lon0)
                            if best_d is None or d < best_d:
                                best_d = d
                                best = p
                        if best is not None and best_d is not None and best_d < 0.002:
                            st.session_state["selected_acc_meta"] = None
                            st.session_state["selected_acc_photo_path"] = None
                            st.session_state["selected_acc_year"] = None
                            st.session_state["selected_bus_meta"] = None
                            name = best.get("name", "")
                            photo = best.get("photo", None)
                            best_idx = int(best.get("idx", 0))
                            st.session_state["selected_rock_idx"] = best_idx
                            st.session_state["selected_rockfall_meta"] = _rockfall_meta_text(
                                best
                            )
                            st.session_state["selected_rockfall_photo_path"] = (
                                str(photo) if photo else None
                            )

        with right_detail:
            _render_photo_detail_panel("rockfall")
# =============================
# Row 3: 그래프 3개 (Vega-Lite + 상세 분석 텍스트)
# =============================
if show_graphs:

    st.write("")
    g1, g2, g3 = st.columns(3, gap="large")
    GRAPH_CARD_H = 680
    GRAPH_CHART_H = 360
    with g1:
        with st.container(border=True, height=GRAPH_CARD_H):
            st.markdown(
                '<div class="card-title">교통위반 단속건수 통계</div>',
                unsafe_allow_html=True,
            )
            df_counts = load_enforcement_counts_csv()
            if df_counts.empty:
                st.info("enforcement_data 폴더의 교통단속 CSV 파일을 찾지 못했어.")
            else:
                mode = st.selectbox(
                    "집계 기준",
                    ["연도별", "월별"],
                    index=0,
                    key="acc_count_mode",
                )
                df_counts = _ensure_year_month(df_counts)
                if "연도" not in df_counts.columns or "월" not in df_counts.columns:
                    st.info("집계에 필요한 컬럼이 없어.")
                else:
                    years = list(range(2019, 2026))
                    if mode == "연도별":
                        year = st.selectbox(
                            "연도 선택",
                            years,
                            index=years.index(2025) if 2025 in years else 0,
                            key="acc_count_year",
                        )
                        summary = (
                            df_counts[df_counts["연도"] == year]
                            .dropna(subset=["월"])
                            .groupby("월")
                            .size()
                            .reindex(range(1, 13), fill_value=0)
                        )
                        plot_df = pd.DataFrame(
                            {"월": summary.index.tolist(), "건수": summary.tolist()}
                        )
                        spec = _vega_bar_spec(
                            "월",
                            "건수",
                            f"{year}년 월별 교통단속 건수",
                            GRAPH_CHART_H,
                        )
                        st.vega_lite_chart(plot_df, spec, use_container_width=True)
                    else:
                        month = st.selectbox(
                            "월 선택",
                            list(range(1, 13)),
                            index=0,
                            key="acc_count_month",
                        )
                        summary = (
                            df_counts[df_counts["월"] == month]
                            .dropna(subset=["연도"])
                            .groupby("연도")
                            .size()
                            .reindex(years, fill_value=0)
                        )
                        plot_df = pd.DataFrame(
                            {"연도": summary.index.tolist(), "건수": summary.tolist()}
                        )
                        spec = _vega_bar_spec(
                            "연도",
                            "건수",
                            f"{month}월 연도별 교통단속 건수",
                            GRAPH_CHART_H,
                        )
                        st.vega_lite_chart(plot_df, spec, use_container_width=True)
            st.write("")
            st.write(
                "교통단속 통계 결과\n\n"
                "- 연도·월별 교통 단속 발생 특성\n"
                "연도별 교통 단속 건수는 2023년이 가장 많고, 그다음이 2021년, 2024년 순으로 나타났다.\n"
                "월별로는 8월, 5월, 7월 순으로 단속 건수가 많아, 성수기 기간에 단속이 집중되는 경향이 확인된다.\n"
                "- 가장 많이 단속된 법 조항: 이륜차 안전모 착용 의무\n"
                "전체 단속 중 도로교통법 제50조 제3항(이륜차 안전모 착용 의무)이 65건으로 가장 높은 비중을 차지하였다.\n"
                "안전모 미착용, 턱끈 미고정, 동승자 미착용 등 이륜차 이용 과정에서 반복적으로 발생하는 위반 유형이 주요 단속 대상이었다.\n"
                "- 차량 이동 관련 주요 단속 유형\n"
                "제54조 제1항(사고 발생 시 조치의무 위반)과 제48조 제1항(안전운전의무 위반)이 각각 41건, 39건으로 나타나,\n"
                "차량 이동이 많아지는 시기에 운전자 준수 의무 위반에 대한 단속 비중이 높아지는 구조가 확인된다.\n"
                "- 성수기 단속 집중 현상\n"
                "평균 대비 단속 건수가 높은 성수기 달은 4~8월과 10월로 나타났으며, 특히 5월과 10월에 단속 건수가 집중되었다.\n"
                "여객 유입이 많은 5월에는 이륜차 관련 단속, 차량 유입이 많은 8월에는 차량 관련 단속이 상대적으로 많았다.\n"
                "- 비수기(2월) 주정차 단속의 특징\n"
                "2월은 전반적으로 여객·차량 이동이 적은 시기임에도 불구하고, 제73조 제2항(불법 주정차) 단속이 상대적으로 많이 발생하였다.\n"
                "이는 겨울철 도로 여건 변화로 인해 정차·주차 질서 위반 단속 비중이 높아지는 월별 특성으로 나타난다."
            )

    with g2:
        with st.container(border=True, height=GRAPH_CARD_H):
            st.markdown(
                '<div class="card-title">강수량 및 여객수 통계</div>',
                unsafe_allow_html=True,
            )
            monthly = load_weather_passenger_monthly()
            if monthly.empty:
                st.info("weather_pax 폴더의 강수량/여객 CSV 파일을 찾지 못했어.")
            else:
                mode = st.selectbox(
                    "집계 기준",
                    ["연도별", "월별"],
                    index=0,
                    key="weather_passenger_mode",
                )
                years = sorted(monthly["연"].dropna().unique().astype(int).tolist())
                if not years:
                    st.info("집계에 필요한 데이터가 없어.")
                else:
                    if mode == "연도별":
                        year = st.selectbox(
                            "연도 선택",
                            years,
                            index=len(years) - 1,
                            key="weather_passenger_year",
                        )
                        sub = (
                            monthly[monthly["연"] == year]
                            .set_index("월")
                            .reindex(range(1, 13), fill_value=0)
                        )
                        plot_df = pd.DataFrame(
                            {
                                "월": sub.index.tolist(),
                                "강수량": sub["월강수합"].tolist(),
                                "입도": sub["월입항합"].tolist(),
                                "출도": sub["월출항합"].tolist(),
                            }
                        )
                        spec = _vega_weather_passenger_spec(
                            "월", f"{year}년 월별 강수량/여객수", GRAPH_CHART_H
                        )
                        st.vega_lite_chart(plot_df, spec, use_container_width=True)
                    else:
                        month = st.selectbox(
                            "월 선택",
                            list(range(1, 13)),
                            index=0,
                            key="weather_passenger_month",
                        )
                        sub = (
                            monthly[monthly["월"] == month]
                            .set_index("연")
                            .reindex(years, fill_value=0)
                        )
                        plot_df = pd.DataFrame(
                            {
                                "연도": sub.index.tolist(),
                                "강수량": sub["월강수합"].tolist(),
                                "입도": sub["월입항합"].tolist(),
                                "출도": sub["월출항합"].tolist(),
                            }
                        )
                        spec = _vega_weather_passenger_spec(
                            "연도", f"{month}월 연도별 강수량/여객수", GRAPH_CHART_H
                        )
                        st.vega_lite_chart(plot_df, spec, use_container_width=True)
            st.write("")
            st.write(
                "강수량 및 입도객 수 통계 결과\n\n"
                "- 입·출도 여객수는 2021년 데이터 시작 시점을 기준으로 월별 흐름을 정렬하여 비교하였다.\n"
                "- 봄철 수요 증가 패턴\n"
                "3~5월 구간에서는 입·출도 여객수가 월 단위로 연속 증가하는 흐름이 확인된다. "
                "해당 기간은 강수량이 연중 최저 수준에 해당하여, 기상 변수의 간섭이 상대적으로 적은 상태에서 "
                "교통 수요 증가가 뚜렷하게 나타난 구간이다.\n"
                "- 강수량 피크 구간의 방향성 변화\n"
                "강수량이 높은 구간에서는 입도 대비 출도 여객이 상대적으로 커지며, "
                "출도 우세(교통 흐름 역전) 패턴이 관측된다.\n"
                "- 입도·출도 최고치 시점의 비대칭\n"
                "입도 여객수는 8월에 정점을 기록한 뒤 감소하는 흐름이 나타나는 반면, "
                "출도 여객수는 10월에 재상승(증가)이 뚜렷하게 나타나 정점 시점이 서로 다르게 형성된다."
            )

    with g3:
        with st.container(border=True, height=GRAPH_CARD_H):
            st.markdown(
                '<div class="card-title">입/출도 성수기 · 비수기</div>',
                unsafe_allow_html=True,
            )
            monthly = load_weather_passenger_monthly()
            if monthly.empty:
                st.info("weather_pax 폴더의 여객 데이터가 없어.")
            else:
                years = sorted(monthly["연"].dropna().unique().astype(int).tolist())
                direction = st.selectbox(
                    "구분 선택",
                    ["입도", "출도"],
                    index=0,
                    key="peak_dir",
                )
                year = st.selectbox(
                    "연도 선택",
                    years,
                    index=len(years) - 1 if years else 0,
                    key="peak_year",
                )
                value_col = "월입항합" if direction == "입도" else "월출항합"
                sub = (
                    monthly[monthly["연"] == year]
                    .set_index("월")
                    .reindex(range(1, 13), fill_value=0)
                )
                months = list(range(1, 13))
                values = sub[value_col].tolist()
                threshold = sum(values) / len(values) if values else None

                peak_months = {6, 7, 8}
                plot_df = pd.DataFrame(
                    {
                        "월": months,
                        "여객수": values,
                        "구분": [
                            (
                                "성수기"
                                if m in peak_months
                                else (
                                    "비수기(평균↑)"
                                    if (threshold is not None and v > threshold)
                                    else "비수기"
                                )
                            )
                            for m, v in zip(months, values)
                        ],
                    }
                )
                spec = _vega_bar_color_spec(
                    "월",
                    "여객수",
                    "구분",
                    f"{year}년 월별 여객 수 ({direction} 기준)",
                    GRAPH_CHART_H,
                )
                if threshold is not None:
                    spec = {
                        "layer": [
                            spec,
                            {
                                "data": {"values": [{"label": "연평균", "value": float(threshold)}]},
                                "mark": {
                                    "type": "rule",
                                    "color": "#000000",
                                    "strokeWidth": 1.2,
                                    "strokeDash": [6, 4],
                                },
                                "encoding": {
                                    "y": {"field": "value", "type": "quantitative"},
                                    "strokeDash": {
                                        "field": "label",
                                        "type": "nominal",
                                        "scale": {"range": [[6, 4]]},
                                        "legend": {
                                            "orient": "top-right",
                                            "direction": "horizontal",
                                            "title": None,
                                            "symbolType": "stroke",
                                            "symbolStrokeDash": [6, 4],
                                            "symbolStrokeWidth": 2,
                                            "offset": 6,
                                            "padding": 0,
                                            "legendY": 0,
                                            "labelFontSize": 10,
                                        },
                                    },
                                    "tooltip": [
                                        {
                                            "field": "value",
                                            "type": "quantitative",
                                            "title": "연평균",
                                            "format": ",.0f",
                                        },
                                    ],
                                    "axis": None,
                                },
                            },
                        ],
                        "config": _vega_base_config(),
                    }
                st.vega_lite_chart(plot_df, spec, use_container_width=True)
            st.write("")
            st.write(
                "입출도객 수 통계 결과\n\n"
                "- 평균 산출 기준 및 보정 방식\n"
                "완전한 연도인 2022~2024년 자료만을 사용해 월별 평균을 계산하였으며, "
                "2021년과 2025년의 누락된 월은 해당 평균값으로 보정하였다. 이를 통해 출도 평균 여객 수는 "
                "17,341명으로 산출되었다.\n"
                "- 출도 여객 수의 계절적 분포\n"
                "출도 여객 수는 4~8월과 10월에 평균보다 높게 나타났으며, "
                "이 중 5월이 연중 가장 많은 출도 여객 수를 기록하였다. 평균보다 높은 달은 성수기, "
                "낮은 달은 비수기로 구분하였다.\n"
                "- 입도 여객 수의 분포 특징\n"
                "입도 여객 수 역시 4~8월과 10월에 집중되었고, 출도와 동일하게 5월에 가장 많은 입도 여객 수가 발생하였다. "
                "다만, 입도 평균 여객 수는 약 552명으로 출도 평균에 비해 현저히 낮은 수준이다.\n"
                "- 입도·출도 규모 차이에 대한 해석\n"
                "출도 평균 여객 수(17,341명)에 비해 입도 평균 여객 수가 크게 적은 것은, "
                "체류 후 외부로 이동하는 수요가 상대적으로 크거나 일시적 방문 성격의 이동이 많음을 시사한다.\n"
                "- 기상 및 관광 요인에 따른 종합 분석\n"
                "4~10월은 겨울철 대비 해상 기상이 안정되고 파도가 낮아 선박 운항이 원활한 시기로, "
                "여객 수 증가에 직접적인 영향을 미친 것으로 보인다. 또한 이 시기는 자연 경관과 야외 활동 여건이 좋아 "
                "관광객 중심의 여객 수요가 집중되는 계절적 특성을 보인다."
            )
else:
    st.write("")
    st.caption("하단 그래프는 사이드바에서 꺼져있음")
# =============================
st.write("")
st.markdown(
    """
---
본 페이지는 울릉군청에서 제공하는 공개 데이터를 활용하여 제작되었습니다."""
)
