import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import re
import copy
import time
import math
import unicodedata
from datetime import datetime
import base64
# textwrap은 templates.py로 이동됨
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image
import os
from functools import lru_cache

# 분리된 모듈에서 import
from utils import (
    haversine_distance,
    _norm_text,
    _tokenize_address,
    _address_candidates,
    _row_to_address,
    _load_and_cache_image,
    _build_accident_photo_index,
    _find_accident_photo_fast,
    find_accident_photo_by_address,
    _build_rockfall_photo_index,
    _find_rockfall_photo,
)

from data_loaders import (
    SMS_SHIP_KEYWORDS,
    SMS_SHIP_VESSEL_KEYWORDS,
    SMS_PEOPLE_KEYWORDS,
    SMS_PEOPLE_VESSEL_KEYWORDS,
    SMS_PASSENGER_KEYWORDS,
    SMS_CARGO_KEYWORDS,
    SMS_CANCEL_KEYWORDS,
    SMS_CONTROL_KEYWORDS,
    SMS_CHANGE_KEYWORDS,
    SMS_ARRIVE_KEYWORDS,
    SMS_DEPART_KEYWORDS,
    SMS_ARRIVE_ROUTE_PATTERNS,
    SMS_DEPART_ROUTE_PATTERNS,
    _accident_files_signature,
    load_accidents_csv,
    load_ev_charger_points,
    load_ev_charger_data,
    load_rockfall_points,
    load_bus_stops_csv,
    _match_bus_stop,
    _bus_route_defs,
    build_bus_routes,
    _simulate_bus_positions,
    _polyline_segments,
    _point_on_segments,
    load_enforcement_counts_csv,
    _ensure_year_month,
    load_weather_passenger_monthly,
    load_sms_raw,
    load_passenger_daily_avg,
    load_passenger_daily,
    _recent_passenger_stats,
    _monthly_passenger_stats,
    _latest_sea_event,
    _summarize_sms_notice_counts_window,
    _summarize_sms_notice_counts,
    _latest_sea_notice,
)

from visualizations import (
    _build_accident_points,
    _build_folium_base_map,
    _cached_folium_base_map,
    render_ulleung_folium_map,
    _vega_base_config,
    _vega_bar_spec,
    _vega_weather_passenger_spec,
    _vega_bar_color_spec,
)

from styles import GLOBAL_CSS, get_map_height_css
import templates as tpl

try:
    import folium
except Exception:
    folium = None

try:
    from folium.plugins import MarkerCluster
except Exception:
    MarkerCluster = None
try:
    from folium.plugins import FastMarkerCluster
except Exception:
    FastMarkerCluster = None
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
# UI 레이아웃 상수
# -----------------------------
# 시각적 간격 표준화를 위한 상수
SPACING_SMALL = "0.5rem"
SPACING_MEDIUM = "1rem"
SPACING_LARGE = "2rem"

# -----------------------------
# CSS (styles.py에서 import)
# -----------------------------
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


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
if "selected_acc_label" not in st.session_state:
    st.session_state["selected_acc_label"] = None
if "selected_acc_center" not in st.session_state:
    st.session_state["selected_acc_center"] = None
if "selected_rockfall_meta" not in st.session_state:
    st.session_state["selected_rockfall_meta"] = None
if "selected_rockfall_photo_path" not in st.session_state:
    st.session_state["selected_rockfall_photo_path"] = None
if "selected_rock_label" not in st.session_state:
    st.session_state["selected_rock_label"] = None
if "selected_rock_center" not in st.session_state:
    st.session_state["selected_rock_center"] = None
if "selected_bus_meta" not in st.session_state:
    st.session_state["selected_bus_meta"] = None
if "rock_view_mode" not in st.session_state:
    st.session_state["rock_view_mode"] = "list"


@st.cache_data(show_spinner=False)
def _filter_accidents_by_year(df_acc: pd.DataFrame, year_filter: int | None):
    """연도별 사고 데이터 필터링"""
    if year_filter is None:
        return df_acc
    return df_acc[df_acc["year"] == year_filter]


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
    display_label = detail_txt if detail_txt else (addr if addr else "위치 정보 없음")

    # 6. 세션 상태 업데이트 (교통사고 정보 입력)
    st.session_state["selected_acc_meta"] = (
        f"연도: {year_val}\n위치: {detail_label}\n유형: {acc_type}\n주소: {addr_label}\n{summary}"
    )
    st.session_state["selected_acc_photo_path"] = str(photo) if photo else None
    st.session_state["selected_acc_year"] = year_val
    st.session_state["selected_acc_label"] = display_label

    # [핵심] 낙석 및 버스 정보는 '반드시' 지워야 화면이 전환됨
    st.session_state["selected_rockfall_meta"] = None
    st.session_state["selected_rockfall_photo_path"] = None
    st.session_state["selected_rock_label"] = None
    st.session_state["selected_rock_center"] = None
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
st.markdown(tpl.dashboard_title(logo_html), unsafe_allow_html=True)
st.markdown(tpl.notice_pill(_notice_html), unsafe_allow_html=True)

st.divider()

# =============================
# Row 2: Layer 2개 (해상공지 / 도로통제)
# =============================
sns_raw = load_sms_raw()
sms_counts, sms_total, sms_breakdown = _summarize_sms_notice_counts(
    sns_raw,
    year=2025,
)
sea_latest_label, sea_latest_text = _latest_sea_notice(sns_raw, year=2025)
pax_avgs = load_passenger_daily_avg(2025)
recent_stats = _recent_passenger_stats()
latest_arrive_sms = _latest_sea_event(sns_raw, 2025, "입항")
latest_depart_sms = _latest_sea_event(sns_raw, 2025, "출항")
sms_dates = (
    pd.to_datetime(
        sns_raw["sms_resDate"]
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)
        .str.replace("/", "-", regex=False),
        errors="coerce",
    )
    if not sns_raw.empty and "sms_resDate" in sns_raw.columns
    else pd.Series(dtype="datetime64[ns]")
)
sms_end_dt = sms_dates.dropna().max() if not sms_dates.empty else None
pax_dates = pd.concat(
    [
        load_passenger_daily("입항")["date"],
        load_passenger_daily("출항")["date"],
    ],
    ignore_index=True,
).dropna()
pax_end_dt = pax_dates.max() if not pax_dates.empty else None

monthly_ship_window = _monthly_passenger_stats(30, end_dt=sms_end_dt)
monthly_pax_window = _monthly_passenger_stats(30, end_dt=pax_end_dt)
monthly_counts, monthly_breakdown = _summarize_sms_notice_counts_window(
    sns_raw, monthly_ship_window.get("start_dt"), monthly_ship_window.get("end_dt")
)

monthly_arrive_ship = monthly_breakdown["입항"]["선박"]
monthly_depart_ship = monthly_breakdown["출항"]["선박"]
monthly_control = monthly_counts["운항통제"]
monthly_cancel = monthly_counts["결항"]
monthly_change = monthly_counts["시간변경"]


# 백분율 계산 로직
def _pct(count: int, total: int) -> int:
    if total <= 0:
        return 0
    return int(round(count / total * 100))


def _bar_pct(count: int, total: int, min_pct: int = 6) -> int:
    if total <= 0 or count <= 0:
        return 0
    pct = int(round(count / total * 100))
    return max(pct, min_pct)


# 1. 각 항목의 건수/합계 가져오기
sea_arrive_ship_total = sms_breakdown["입항"]["선박"]
sea_depart_ship_total = sms_breakdown["출항"]["선박"]
sea_arrive_people = pax_avgs.get("입항", 0)
sea_depart_people = pax_avgs.get("출항", 0)
sea_arrive = sea_arrive_people
sea_depart = sea_depart_people
sea_control = sms_counts["운항통제"]
sea_cancel = sms_counts["결항"]
sea_change = sms_counts["시간변경"]

# 막대 그래프는 최댓값 기준으로 100% 스케일링
sea_max_val = max(
    sea_arrive,
    sea_depart,
    sea_arrive_ship_total,
    sea_depart_ship_total,
    sea_control,
    sea_cancel,
    sea_change,
)
if sea_max_val == 0:
    sea_max_val = 1

sea_arrive_pct = _bar_pct(sea_arrive, sea_max_val)
sea_depart_pct = _bar_pct(sea_depart, sea_max_val)
sea_arrive_ship_pct = _bar_pct(sea_arrive_ship_total, sea_max_val)
sea_depart_ship_pct = _bar_pct(sea_depart_ship_total, sea_max_val)
sea_control_pct = _bar_pct(sea_control, sea_max_val)
sea_cancel_pct = _bar_pct(sea_cancel, sea_max_val)
sea_change_pct = _bar_pct(sea_change, sea_max_val)

# 2. 내부 분할(선박/사람) 비율은 해당 항목의 합계를 기준으로 계산 (이건 기존 유지)
sea_arrive_people_pct = 100
sea_depart_people_pct = 100

c1, c2 = st.columns(2, gap="large")

with c1:
    with st.container(border=True):
        if show_sea_notice:
            st.markdown(
                """
<div class="r2-head">
  <div class="r2-title">입출항 정보/통계</div>
  <div class="r2-date">2025년 기준</div>
</div>
                """,
                unsafe_allow_html=True,
            )
            sea_tab_recent, sea_tab_month, sea_tab_year = st.tabs(
                ["최근통계", "월간통계", "연간통계(2025)"]
            )

            arrive_latest = recent_stats["arrive_latest"]
            depart_latest = recent_stats["depart_latest"]
            arrive_avg3 = recent_stats["arrive_avg3"]
            depart_avg3 = recent_stats["depart_avg3"]

            def _fmt_date_label(primary: str | None, fallback_dt: datetime | None):
                if primary:
                    return primary
                if fallback_dt:
                    return fallback_dt.strftime("%Y-%m-%d")
                return "미상"

            def _fmt_vehicle(val):
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return "-"
                return f"{int(val):,}대"

            arrive_dt_label = _fmt_date_label(
                latest_arrive_sms.get("datetime"), arrive_latest.get("date")
            )
            depart_dt_label = _fmt_date_label(
                latest_depart_sms.get("datetime"), depart_latest.get("date")
            )

            with sea_tab_recent:
                recent_html = tpl.sea_recent_events(
                    arrive_passengers=arrive_latest.get("passengers", 0),
                    arrive_dt_label=arrive_dt_label,
                    arrive_ship_name=latest_arrive_sms.get("name"),
                    arrive_vehicles=_fmt_vehicle(arrive_latest.get("vehicles")),
                    depart_passengers=depart_latest.get("passengers", 0),
                    depart_dt_label=depart_dt_label,
                    depart_ship_name=latest_depart_sms.get("name"),
                    depart_vehicles=_fmt_vehicle(depart_latest.get("vehicles")),
                    arrive_avg3_passengers=arrive_avg3.get("passengers", 0),
                    arrive_avg3_vehicles=_fmt_vehicle(arrive_avg3.get("vehicles")),
                    depart_avg3_passengers=depart_avg3.get("passengers", 0),
                    depart_avg3_vehicles=_fmt_vehicle(depart_avg3.get("vehicles")),
                )
                st.markdown(recent_html, unsafe_allow_html=True)

            with sea_tab_month:
                ship_start = monthly_ship_window.get("start_dt")
                ship_end = monthly_ship_window.get("end_dt")
                if ship_start and ship_end:
                    period_label = f"{ship_start:%Y-%m-%d} ~ {ship_end:%Y-%m-%d}"
                else:
                    period_label = "데이터 없음"

                badges = []
                if monthly_cancel > 0:
                    badges.append(f"⚠️ 결항 {monthly_cancel}건")
                if monthly_control > 0:
                    badges.append(f"⚠️ 운항통제 {monthly_control}건")
                if monthly_change > 0:
                    badges.append(f"⚠️ 시간변경 {monthly_change}건")

                badge_html = tpl.sea_badges(badges)
                month_html = tpl.sea_monthly_stats(
                    period_label=period_label,
                    monthly_arrive_ship=monthly_arrive_ship,
                    monthly_depart_ship=monthly_depart_ship,
                    arrive_sum=monthly_pax_window.get("arrive_sum", 0),
                    arrive_vehicle_sum=_fmt_vehicle(monthly_pax_window.get("arrive_vehicle_sum")),
                    depart_sum=monthly_pax_window.get("depart_sum", 0),
                    depart_vehicle_sum=_fmt_vehicle(monthly_pax_window.get("depart_vehicle_sum")),
                    badge_html=badge_html,
                )
                st.markdown(month_html, unsafe_allow_html=True)

            with sea_tab_year:
                year_html = tpl.sea_yearly_stats(
                    sea_arrive=sea_arrive,
                    sea_arrive_pct=sea_arrive_pct,
                    sea_arrive_people=sea_arrive_people,
                    sea_depart=sea_depart,
                    sea_depart_pct=sea_depart_pct,
                    sea_depart_people=sea_depart_people,
                    sea_arrive_ship_total=sea_arrive_ship_total,
                    sea_arrive_ship_pct=sea_arrive_ship_pct,
                    sea_depart_ship_total=sea_depart_ship_total,
                    sea_depart_ship_pct=sea_depart_ship_pct,
                    sea_control=sea_control,
                    sea_control_pct=sea_control_pct,
                    sea_cancel=sea_cancel,
                    sea_cancel_pct=sea_cancel_pct,
                    sea_change=sea_change,
                    sea_change_pct=sea_change_pct,
                )
                st.markdown(year_html, unsafe_allow_html=True)
        else:
            st.caption("사이드바에서 해상공지 레이어가 꺼져있음")

with c2:
    with st.container(border=True):
        if show_road_control:
            head_left, head_right = st.columns([1, 0.35])
            with head_left:
                st.markdown(tpl.road_control_header(), unsafe_allow_html=True)
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

# =============================
# Row 1: (탭 내 전환형) 목록 보기 vs 지도 보기
# =============================

# 상태 초기화: 기본은 'list' 모드
if "view_mode" not in st.session_state:
    st.session_state["view_mode"] = "list"

# 상단 영역 높이 설정
MAP_H = 360  # 지도 영역 높이(px)
st.markdown(
    f"""
    <style>
    .stFolium, .stFolium iframe {{
      width: 100% !important;
      height: {MAP_H}px !important;
      min-height: {MAP_H}px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown(tpl.card_title("울릉군 지도"), unsafe_allow_html=True)
    st.caption("2019-2025년 울릉군 위치 데이터 기반")

    # 지도 상단 탭
    t1, t2, t3, t4 = st.tabs(["버스 실시간 상황", "교통사고 지점", "낙석 발생 지점", "전기차 충전소"])

    def _render_photo_detail_panel(key_suffix: str):
        with st.container(border=True):
            st.markdown(
                tpl.card_title("사고 장소 사진"), unsafe_allow_html=True
            )

            sel_rock_photo = st.session_state.get("selected_rockfall_photo_path")
            sel_acc_photo = st.session_state.get("selected_acc_photo_path")
            sel_acc_meta = st.session_state.get("selected_acc_meta")
            sel_rock_meta = st.session_state.get("selected_rockfall_meta")
            sel_bus_meta = st.session_state.get("selected_bus_meta")

            with st.container():
                image_loaded = False
                if sel_rock_photo:
                    try:
                        path_str = str(sel_rock_photo)
                        if os.path.isfile(path_str):
                            # 캐싱된 이미지 로드 (메모리 최적화)
                            img = _load_and_cache_image(path_str)
                            if img:
                                st.image(img, use_container_width=True)
                                image_loaded = True
                    except Exception:
                        pass
                elif sel_acc_photo and not image_loaded:
                    try:
                        path_str = str(sel_acc_photo)
                        if os.path.isfile(path_str):
                            # 캐싱된 이미지 로드 (메모리 최적화)
                            img = _load_and_cache_image(path_str)
                            if img:
                                st.image(img, use_container_width=True)
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

            if image_loaded and (sel_rock_photo or sel_acc_photo):
                selected_photo_path = sel_rock_photo or sel_acc_photo

                @st.dialog("사진 확대")
                def _show_photo_dialog(photo_path: str):
                    try:
                        # 캐싱된 이미지 로드
                        img = _load_and_cache_image(str(photo_path))
                        if img:
                            st.image(img, use_container_width=True)
                        else:
                            st.warning("이미지를 불러오지 못했습니다.")
                    except Exception:
                        st.warning("이미지를 불러오지 못했습니다.")

                if st.button("사진 확대 보기", key=f"photo_zoom_{key_suffix}"):
                    _show_photo_dialog(selected_photo_path)

            st.markdown(
                tpl.card_title("자세히 보기"), unsafe_allow_html=True
            )
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
        left_main, right_detail = st.columns([2, 1.3], gap="large")
        with left_main:
            st.caption("울릉군 버스 노선/정류장")
            bus_map_state = render_ulleung_folium_map(
                kind="bus",
                height=MAP_H,
                selected_route_id=st.session_state.get("selected_bus_route_id"),
            )
            if isinstance(bus_map_state, dict):
                last = bus_map_state.get("last_object_clicked")
                bus_meta = st.session_state.get("bus_stops_meta", [])
                if (
                    isinstance(last, dict)
                    and "lat" in last
                    and "lng" in last
                    and bus_meta
                ):
                    lat0 = float(last["lat"])
                    lon0 = float(last["lng"])
                    # 가장 가까운 버스 정류장 찾기
                    best = None
                    best_d = None
                    for p in bus_meta:
                        d = haversine_distance(lat0, lon0, float(p["lat"]), float(p["lon"]))
                        if best_d is None or d < best_d:
                            best_d = d
                            best = p
                    if best is not None and best_d is not None and best_d < 100:
                        new_meta = {
                            "name": best.get("name", ""),
                            "routes": best.get("routes", []) or [],
                        }
                        # 세션 상태 업데이트 (st.rerun 불필요 - 같은 사이클에서 패널이 새 값 표시)
                        st.session_state["selected_acc_meta"] = None
                        st.session_state["selected_acc_photo_path"] = None
                        st.session_state["selected_rockfall_meta"] = None
                        st.session_state["selected_rockfall_photo_path"] = None
                        st.session_state["selected_bus_meta"] = new_meta
            st.caption(f"조회기준: {datetime.now():%Y-%m-%d %H:%M}")

        with right_detail:
            routes_defs = {r["id"]: r for r in _bus_route_defs()}
            route_22 = routes_defs.get("22")
            route_3 = routes_defs.get("3")
            route_options = list(routes_defs.keys())
            if route_options:
                # 초기값 설정
                if "selected_bus_route_id" not in st.session_state or st.session_state.get("selected_bus_route_id") not in route_options:
                    st.session_state["selected_bus_route_id"] = route_options[0]

                # key 매개변수를 사용하여 Streamlit이 자동으로 session_state를 관리하도록 함
                current_index = route_options.index(st.session_state["selected_bus_route_id"])
                st.selectbox(
                    "현재 노선 선택",
                    route_options,
                    index=current_index,
                    format_func=lambda rid: routes_defs[rid]["name"],
                    key="selected_bus_route_id",
                )

            def _route_dir_label(route):
                if not route or not route.get("stops"):
                    return "상행 -> (정보 없음), 하행 -> (정보 없음)"
                up = route["stops"][0]
                down = route["stops"][-1]
                return f"상행 -> {up}, 하행 -> {down}"

            with st.container(border=True):
                st.markdown(
                    tpl.card_title("버스 실시간 정보"),
                    unsafe_allow_html=True,
                )
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
                st.markdown(
                    tpl.card_title("정류장 상세"), unsafe_allow_html=True
                )
                sel_bus_meta = st.session_state.get("selected_bus_meta")
                if sel_bus_meta:
                    route_defs = {r["id"]: r for r in _bus_route_defs()}
                    if isinstance(sel_bus_meta, dict):
                        stop_name = sel_bus_meta.get("name", "")
                        routes = sel_bus_meta.get("routes", []) or []
                    else:
                        stop_name = ""
                        routes = []
                        for line in str(sel_bus_meta).splitlines():
                            if "정류장" in line:
                                stop_name = line.split(":", 1)[-1].strip()
                            if "경유 노선" in line:
                                raw = line.split(":", 1)[-1].strip()
                                if "없음" not in raw:
                                    routes = [
                                        r.strip() for r in raw.split(",") if r.strip()
                                    ]

                    st.markdown(
                        f"""
<div class="bus-detail">
  <div class="bus-detail-title">{stop_name or "정류장 정보 없음"}</div>
  <div class="bus-detail-sub">경유 노선 {len(routes)}개</div>
</div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if routes:
                        cards_html = []
                        for route_name in routes:
                            match = re.match(
                                r"^(\d+)\s*노선\s*(?:\((.*)\))?$", route_name
                            )
                            route_id = match.group(1) if match else ""
                            route_desc = (
                                match.group(2).strip()
                                if match and match.group(2)
                                else ""
                            )
                            if (
                                not route_desc
                                and route_id
                                and route_name != f"{route_id}노선"
                            ):
                                route_desc = route_name
                            color = (
                                route_defs.get(route_id, {}).get("color", "#9aa3b2")
                                if route_id
                                else "#9aa3b2"
                            )
                            cards_html.append(
                                f"""
<div class="bus-route-card" style="border-left-color: {color};">
  <div class="bus-route-id">{route_id + "노선" if route_id else route_name}</div>
  <div class="bus-route-desc">{route_desc or route_name}</div>
</div>
                                """
                            )
                        st.markdown(
                            f'<div class="bus-route-grid">{"".join(cards_html)}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="bus-route-empty">경유 노선 정보가 없습니다.</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown("- 지도에서 정류장을 클릭하면 상세 정보가 표시됩니다.")

    # [탭 2] 교통사고 (탭 안에서 목록/지도 전환)
    with t2:
        left_main, right_detail = st.columns([2, 1.3], gap="large")
        with left_main:
            top_left, top_right = st.columns([4, 1])
            with top_left:
                if st.session_state["view_mode"] == "list":
                    st.caption(
                        "발생한 사고 목록입니다. 위치 확인 버튼을 누르면 지도로 이동합니다."
                    )
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
                    # 정렬이 필요한 경우 assign으로 새 DataFrame 생성
                    if "year" in df_acc_list.columns:
                        df_list_view = df_acc_list.assign(
                            _year_sort=df_acc_list["year"].fillna(0).astype(int)
                        ).sort_values(
                            by="_year_sort", ascending=False
                        )
                    else:
                        df_list_view = df_acc_list
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
                        photo_path = (
                            _find_accident_photo_fast(addr) if addr else None
                        )
                        is_selected = (
                            st.session_state.get("selected_acc_idx") == idx
                        )

                        with st.container(border=True):
                            c_img, c_info, c_btn = st.columns([1.5, 3.5, 1])
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
                                sel_tag = (
                                    " <span style='color:#d12c2c;'>● 선택</span>"
                                    if is_selected
                                    else ""
                                )
                                st.markdown(
                                    f"**{display_title}**{sel_tag}",
                                    unsafe_allow_html=True,
                                )
                                st.caption(
                                    f"발생연도: {year_val} | 유형: {acc_type}"
                                )
                                st.markdown(
                                    f"<div style='color:#666; font-size:0.85rem;'>위치: {addr if addr else '미상'}<br/>좌표: {lat_lon}</div>",
                                    unsafe_allow_html=True,
                                )
                            with c_btn:
                                if st.button(
                                    "위치 확인",
                                    key=f"btn_go_map_{idx}",
                                    use_container_width=True,
                                ):
                                    _set_selected_accident(df_acc_list, idx)
                                    if pd.notna(lat) and pd.notna(lon):
                                        st.session_state["selected_acc_center"] = (
                                            float(lat),
                                            float(lon),
                                        )
                                    st.session_state["selected_acc_idx"] = int(idx)
                                    st.session_state["view_mode"] = "map"
                                    st.rerun()
                else:
                    df_acc = df_acc_list
                    # selectbox 변경마다 지도 rerun 방지: form + 적용 버튼
                    year_filter = None
                    df_view = df_acc

                    if "year" in df_acc.columns and not df_acc["year"].dropna().empty:
                        years = sorted(
                            {int(y) for y in df_acc["year"].dropna().unique()}
                        )
                        options = ["전체"] + [str(y) for y in years]

                        if "acc_year_label" not in st.session_state:
                            st.session_state["acc_year_label"] = (
                                "전체" if 2025 not in years else "2025"
                            )

                        with st.form(
                            "acc_year_form", clear_on_submit=False
                        ):
                            default_label = st.session_state["acc_year_label"]
                            if default_label not in options:
                                default_label = options[0]
                            default_idx = options.index(default_label)

                            selected_year_label = st.selectbox(
                                "연도 선택",
                                options,
                                index=default_idx,
                            )
                            apply_year = st.form_submit_button("적용")

                        if apply_year:
                            st.session_state["acc_year_label"] = selected_year_label

                        selected_year_label = st.session_state["acc_year_label"]
                        if selected_year_label != "전체":
                            year_filter = int(selected_year_label)

                    if year_filter is not None:
                        df_view = _filter_accidents_by_year(
                            df_acc,
                            year_filter,
                        )

                    selected_acc_idx = st.session_state.get("selected_acc_idx")
                    selected_acc_center = st.session_state.get("selected_acc_center")
                    if selected_acc_idx is not None:
                        st.caption(
                            f"선택된 사고 위치: {st.session_state.get('selected_acc_label') or '정보 없음'}"
                        )
                    map_state = render_ulleung_folium_map(
                        kind="accident",
                        height=MAP_H,
                        accident_year_filter=year_filter,
                        highlight_idx=selected_acc_idx,
                        center_override=selected_acc_center,
                    )
                    if isinstance(map_state, dict):
                        last = map_state.get("last_object_clicked")
                        if isinstance(last, dict) and "lat" in last and "lng" in last:
                            lat0 = float(last["lat"])
                            lon0 = float(last["lng"])
                            # 가장 가까운 사고 지점 찾기
                            best_idx = None
                            best_d = None
                            for i in df_view.index:
                                row_lat = df_view.at[i, "latitude"]
                                row_lon = df_view.at[i, "longitude"]
                                d = haversine_distance(lat0, lon0, row_lat, row_lon)
                                if best_d is None or d < best_d:
                                    best_d = d
                                    best_idx = i
                            if best_d is not None and best_d < 100:
                                st.session_state["selected_rockfall_meta"] = None
                                st.session_state["selected_rockfall_photo_path"] = None
                                st.session_state["selected_bus_meta"] = None
                                _set_selected_accident(df_acc, best_idx)
                                st.session_state["selected_acc_idx"] = int(best_idx)
                                st.session_state["selected_acc_center"] = (
                                    float(df_acc.at[best_idx, "latitude"]),
                                    float(df_acc.at[best_idx, "longitude"]),
                                )

        with right_detail:
            _render_photo_detail_panel("accident")

    # [탭 3] 낙석
    with t3:
        left_main, right_detail = st.columns([2, 1.3], gap="large")
        with left_main:
            top_left, top_right = st.columns([4, 1])
            with top_left:
                if st.session_state["rock_view_mode"] == "list":
                    st.caption(
                        "낙석 발생 목록입니다. 위치 확인 버튼을 누르면 지도로 이동합니다."
                    )
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
                        is_selected = (
                            st.session_state.get("selected_rock_idx") == item_idx
                        )
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
                            c_img, c_info, c_btn = st.columns([1.5, 3.5, 1])
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
                                st.markdown(
                                    f"**{name}**{sel_tag}", unsafe_allow_html=True
                                )
                                st.caption(
                                    f"발견일: {date_label} | 피해여부: {damage_label}"
                                )
                                st.markdown(
                                    f"<div style='color:#666; font-size:0.85rem;'>조치상태: 완료<br/>좌표: {lat_lon}</div>",
                                    unsafe_allow_html=True,
                                )
                            with c_btn:
                                if st.button(
                                    "위치 확인",
                                    key=f"btn_rock_map_{item_idx}",
                                    use_container_width=True,
                                ):
                                    st.session_state["selected_acc_meta"] = None
                                    st.session_state["selected_acc_photo_path"] = (
                                        None
                                    )
                                    st.session_state["selected_acc_label"] = None
                                    st.session_state["selected_acc_center"] = None
                                    st.session_state["selected_bus_meta"] = None
                                    st.session_state["selected_rock_idx"] = item_idx
                                    st.session_state["selected_rockfall_meta"] = (
                                        _rockfall_meta_text(item)
                                    )
                                    st.session_state[
                                        "selected_rockfall_photo_path"
                                    ] = (str(photo) if photo else None)
                                    st.session_state["selected_rock_label"] = name
                                    if pd.notna(lat) and pd.notna(lon):
                                        st.session_state["selected_rock_center"] = (
                                            float(lat),
                                            float(lon),
                                        )
                                    st.session_state["rock_view_mode"] = "map"
                                    st.rerun()
            else:
                selected_rock_idx = st.session_state.get("selected_rock_idx")
                selected_rock_center = st.session_state.get("selected_rock_center")
                if selected_rock_idx is not None:
                    st.caption(
                        f"선택된 낙석 위치: {st.session_state.get('selected_rock_label') or '정보 없음'}"
                    )
                rock_map_state = render_ulleung_folium_map(
                    kind="rockfall",
                    height=MAP_H,
                    highlight_idx=selected_rock_idx,
                    center_override=selected_rock_center,
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
                        # 가장 가까운 낙석 지점 찾기
                        best = None
                        best_d = None
                        for p in rock_meta:
                            d = haversine_distance(lat0, lon0, float(p["lat"]), float(p["lon"]))
                            if best_d is None or d < best_d:
                                best_d = d
                                best = p
                        if best is not None and best_d is not None and best_d < 100:
                            best_idx = int(best.get("idx", 0))
                            st.session_state["selected_acc_meta"] = None
                            st.session_state["selected_acc_photo_path"] = None
                            st.session_state["selected_acc_year"] = None
                            st.session_state["selected_acc_label"] = None
                            st.session_state["selected_acc_center"] = None
                            st.session_state["selected_bus_meta"] = None
                            name = best.get("name", "")
                            photo = best.get("photo", None)
                            st.session_state["selected_rock_idx"] = best_idx
                            st.session_state["selected_rockfall_meta"] = (
                                _rockfall_meta_text(best)
                            )
                            st.session_state["selected_rockfall_photo_path"] = (
                                str(photo) if photo else None
                            )
                            st.session_state["selected_rock_label"] = name
                            st.session_state["selected_rock_center"] = (
                                float(best.get("lat")),
                                float(best.get("lon")),
                            )

        with right_detail:
            _render_photo_detail_panel("rockfall")

    # [탭 4] 전기차 충전소
    with t4:
        ev_left_map, ev_right_detail = st.columns([2, 1], gap="medium")

        with ev_left_map:
            st.caption("울릉군 전기차 충전소 위치")
            ev_map_state = render_ulleung_folium_map(
                kind="ev",
                height=MAP_H,
            )
            ev_points, ev_meta_list = load_ev_charger_data()
            if ev_points:
                st.info(f"총 {len(ev_points)}개의 전기차 충전소가 표시되어 있습니다.")
            else:
                st.warning("전기차 충전소 데이터가 없습니다.")

            # 마커 클릭 처리
            if isinstance(ev_map_state, dict):
                last = ev_map_state.get("last_object_clicked")
                ev_meta = st.session_state.get("ev_charger_meta", [])
                if (
                    isinstance(last, dict)
                    and "lat" in last
                    and "lng" in last
                    and ev_meta
                ):
                    lat0 = float(last["lat"])
                    lon0 = float(last["lng"])
                    # 가장 가까운 충전소 찾기
                    best = None
                    best_d = None
                    for p in ev_meta:
                        d = haversine_distance(lat0, lon0, float(p["lat"]), float(p["lon"]))
                        if best_d is None or d < best_d:
                            best_d = d
                            best = p
                    if best is not None and best_d is not None and best_d < 100:
                        st.session_state["selected_ev_meta"] = best

        with ev_right_detail:
            st.markdown(
                tpl.card_title("전기차 충전소 정보"),
                unsafe_allow_html=True,
            )
            selected_ev = st.session_state.get("selected_ev_meta")
            if selected_ev:
                # 충전소명
                st.markdown(f"### {selected_ev.get('name', '충전소')}")

                # 주소
                st.markdown(f"**주소:** {selected_ev.get('address', '주소 미상')}")
                if selected_ev.get("detail"):
                    st.markdown(f"**상세위치:** {selected_ev.get('detail')}")

                st.divider()

                # 충전기 정보
                col1, col2 = st.columns(2)
                with col1:
                    fast_cnt = selected_ev.get("fast_charger", "0")
                    fast_avail = selected_ev.get("fast_available", "")
                    st.metric("급속 충전기", f"{fast_cnt}대")
                    if fast_avail:
                        st.caption(f"가용: {fast_avail}")
                    fast_type = selected_ev.get("fast_type", "")
                    if fast_type:
                        st.caption(f"타입: {fast_type}")

                with col2:
                    slow_cnt = selected_ev.get("slow_charger", "0")
                    slow_avail = selected_ev.get("slow_available", "")
                    st.metric("완속 충전기", f"{slow_cnt}대")
                    if slow_avail:
                        st.caption(f"가용: {slow_avail}")

                st.divider()

                # 운영 정보
                open_time = selected_ev.get("open_time", "")
                close_time = selected_ev.get("close_time", "")
                if open_time or close_time:
                    time_str = f"{open_time or '00:00'} ~ {close_time or '24:00'}"
                    st.markdown(f"**운영시간:** {time_str}")

                parking_fee = selected_ev.get("parking_fee", "")
                if parking_fee:
                    st.markdown(f"**주차료:** {parking_fee}")

                # 관리업체 정보
                operator = selected_ev.get("operator", "")
                phone = selected_ev.get("phone", "")
                if operator or phone:
                    st.divider()
                    if operator:
                        st.markdown(f"**관리업체:** {operator}")
                    if phone:
                        st.markdown(f"**연락처:** {phone}")

                # 좌표 정보
                st.divider()
                lat = selected_ev.get("lat", 0)
                lon = selected_ev.get("lon", 0)
                st.caption(f"좌표: {lat:.6f}, {lon:.6f}")
            else:
                st.info("지도에서 충전소 마커를 클릭하면 상세 정보가 표시됩니다.")

# =============================
# Row 3: 그래프 3개 (Vega-Lite + 상세 분석 텍스트)
# =============================
if show_graphs:

    g1, g2, g3 = st.columns(3, gap="large")
    GRAPH_CHART_H = 360
    with g1:
        with st.container(border=True):
            st.markdown(
                tpl.card_title("교통위반 단속건수 통계"),
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

                        # 필터링 및 집계 (불린 인덱싱은 이미 새 DataFrame 반환)
                        filtered_df = df_counts[df_counts["연도"] == year]

                        summary = (
                            filtered_df
                            .dropna(subset=["월"])
                            .groupby("월")
                            .size()
                            .reindex(range(1, 13), fill_value=0)
                        )
                        plot_df = pd.DataFrame(
                            {"월": summary.index.tolist(), "건수": summary.tolist()}
                        )

                        st.caption(f"총 단속 건수: {summary.sum()}건")

                        spec = _vega_bar_spec(
                            "월",
                            "건수",
                            f"{year}년 월별 교통단속 건수",
                            GRAPH_CHART_H,
                        )
                        # 차트 강제 재렌더링: DataFrame을 깨끗한 복사본으로 변환
                        st.vega_lite_chart(plot_df, spec, use_container_width=True)  # 읽기 전용이므로 copy 불필요
                    else:
                        month = st.selectbox(
                            "월 선택",
                            list(range(1, 13)),
                            index=0,
                            key="acc_count_month",
                        )
                        # 필터링 및 집계
                        filtered_df = df_counts[df_counts["월"] == month]  # 불린 인덱싱은 이미 새 DataFrame 반환
                        summary = (
                            filtered_df
                            .dropna(subset=["연도"])
                            .groupby("연도")
                            .size()
                            .reindex(years, fill_value=0)
                        )
                        plot_df = pd.DataFrame(
                            {"연도": summary.index.tolist(), "건수": summary.tolist()}
                        )

                        st.caption(f"총 단속 건수: {summary.sum()}건")

                        spec = _vega_bar_spec(
                            "연도",
                            "건수",
                            f"{month}월 연도별 교통단속 건수",
                            GRAPH_CHART_H,
                        )
                        # 차트 강제 재렌더링: DataFrame을 깨끗한 복사본으로 변환
                        st.vega_lite_chart(plot_df, spec, use_container_width=True)  # 읽기 전용이므로 copy 불필요
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
        with st.container(border=True):
            st.markdown(
                tpl.card_title("강수량 및 여객수 통계"),
                unsafe_allow_html=True,
            )
            monthly = load_weather_passenger_monthly()
            if monthly.empty:
                st.info("weather_pax 폴더의 강수량/여객 CSV 파일을 찾지 못했어요.")
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
                        # 필터링 및 집계
                        filtered_monthly = monthly[monthly["연"] == year]  # 불린 인덱싱은 이미 새 DataFrame 반환
                        sub = (
                            filtered_monthly
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
                        # 차트 강제 재렌더링: DataFrame을 깨끗한 복사본으로 변환
                        st.vega_lite_chart(plot_df, spec, use_container_width=True)  # 읽기 전용이므로 copy 불필요
                    else:
                        month = st.selectbox(
                            "월 선택",
                            list(range(1, 13)),
                            index=0,
                            key="weather_passenger_month",
                        )
                        # 필터링 및 집계
                        filtered_monthly = monthly[monthly["월"] == month]  # 불린 인덱싱은 이미 새 DataFrame 반환
                        sub = (
                            filtered_monthly
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
                        # 차트 강제 재렌더링: DataFrame을 깨끗한 복사본으로 변환
                        st.vega_lite_chart(plot_df, spec, use_container_width=True)  # 읽기 전용이므로 copy 불필요
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
        with st.container(border=True):
            st.markdown(
                tpl.card_title("입/출도 성수기 · 비수기"),
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
                # 필터링 및 집계
                filtered_monthly = monthly[monthly["연"] == year].copy()
                sub = (
                    filtered_monthly
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
                                "data": {
                                    "values": [
                                        {"label": "연평균", "value": float(threshold)}
                                    ]
                                },
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
                # 차트 강제 재렌더링: DataFrame을 깨끗한 복사본으로 변환
                st.vega_lite_chart(plot_df, spec, use_container_width=True)  # 읽기 전용이므로 copy 불필요
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
    st.caption("하단 그래프는 사이드바에서 꺼져있음")
# =============================
st.markdown(
    """
---
본 페이지는 울릉군청에서 제공하는 공개 데이터를 기반으로 제작되었습니다. \n\n 현재는 파일럿 단계로, 실시간 데이터는 반영되지 않았으며 사용성 검증을 위해 일부 가상 데이터를 활용하여 구성되었습니다."""
)
