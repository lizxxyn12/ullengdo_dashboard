import streamlit as st
import pandas as pd
from pathlib import Path
import re
import time

try:
    import folium
except Exception:
    folium = None

try:
    from folium.plugins import MarkerCluster
except Exception:
    MarkerCluster = None

try:
    from streamlit_folium import st_folium
except Exception:
    st_folium = None

st.set_page_config(
    page_title="울릉 교통/안전 대시보드",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# CSS (카드/여백/폰트 약간 정리)
# -----------------------------
st.markdown(
    """
<style>
/* 전체 폭 여백 조금 줄이기 */
.block-container { padding-top: 2.4rem; padding-bottom: 2rem; }

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

.card-title {
  font-weight: 700;
  margin-bottom: 8px;
}
.card-sub {
  color: #666;
  font-size: 0.9rem;
}
.small-muted {
  color: #777;
  font-size: 0.85rem;
}

</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_accidents_csv() -> pd.DataFrame:
    """프로젝트 루트의 사고 CSV를 로드.

    기대 컬럼:
      - latitude, longitude (필수)
      - 나머지는 있으면 popup에 같이 보여줄 수 있음
    """

    csv_path = Path(__file__).parent / "ulleung_accidents_with_coords.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # 컬럼명 표준화(혹시 대소문자/공백이 섞여있을 경우 대비)
    df.columns = [str(c).strip() for c in df.columns]

    # latitude/longitude 없으면 빈 DF
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return pd.DataFrame()

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).copy()

    return df


def render_ulleung_folium_map(kind: str = "base", height: int = 420):
    """울릉군 Folium 지도 렌더.

    kind:
      - base: 기본 지도
      - accident: 교통사고 지점(샘플 마커)
      - rockfall: 낙석 발생 지점(샘플 마커)
      - bus: 버스 실시간(샘플 마커)

    실제 데이터 붙일 때는 아래 sample_points만 교체하면 됨.
    """

    if folium is None:
        st.error(
            "folium 패키지가 설치되어 있지 않아 지도를 표시할 수 없어. 터미널에서 `pip install folium` 해줘."
        )
        return

    # 울릉도 중심(대략)
    center = (37.4844, 130.9057)

    m = folium.Map(
        location=center, zoom_start=12, tiles="OpenStreetMap", control_scale=True
    )

    # 샘플 포인트(나중에 실제 데이터로 교체)
    if kind == "accident":
        df_acc = load_accidents_csv()

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
            # 너무 많을 수 있어서 기본은 2000개로 제한(원하면 늘리면 됨)
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
        sample_points = [
            (37.4950, 130.9145, "낙석(샘플) A"),
            (37.4680, 130.8920, "낙석(샘플) B"),
        ]
        color = "orange"
    elif kind == "bus":
        sample_points = [
            (37.4868, 130.9098, "버스(샘플) 101"),
            (37.4758, 130.9032, "버스(샘플) 202"),
        ]
        color = "blue"
    else:
        sample_points = []
        color = "green"

    fg = folium.FeatureGroup(name=kind)

    # 마커가 많을 때를 대비해 클러스터 사용(가능한 경우)
    marker_parent = fg
    if MarkerCluster is not None and len(sample_points) > 50:
        marker_parent = MarkerCluster(name=f"{kind}_cluster").add_to(fg)

    for lat, lon, label in sample_points:
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
            color=color,
            fill=True,
            fill_opacity=0.85,
            popup=popup,
        ).add_to(marker_parent)

    fg.add_to(m)

    # 지도 렌더 (가능하면 클릭 이벤트까지 받기)
    if st_folium is not None:
        return st_folium(m, height=height, width=None)

    # streamlit-folium이 없으면 이벤트 없이 지도만 표시
    import streamlit.components.v1 as components

    components.html(m.get_root().render(), height=height)
    return None


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

# -----------------------------
# Helper functions for accident photo lookup
# -----------------------------


def _norm_text(s: str) -> str:
    """주소/파일명 매칭용 간단 정규화 (공백/특수문자 제거)."""
    s = "" if s is None else str(s)
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


@st.cache_data(show_spinner=False)
def find_accident_photo_by_address(address: str):
    """acc_pic 폴더에서 '주소.JPG' 규칙으로 저장된 사진을 찾음.

    - 파일명 비교 시 공백/특수문자는 제거하고 비교
    - 확장자는 .JPG/.jpg/.jpeg/.png/.webp 모두 허용
    """
    acc_dir = Path(__file__).parent / "acc_pic"
    if not acc_dir.exists() or not acc_dir.is_dir():
        return None

    target = _norm_text(address)
    if not target:
        return None

    exts = {".jpg", ".jpeg", ".png", ".webp"}

    for p in acc_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if _norm_text(p.stem) == target:
            return p

    # 완전 일치가 없으면 포함 매칭(옵션)
    for p in acc_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if target and target in _norm_text(p.stem):
            return p

    return None


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
    _rest = _notice_text[len(_prefix):].lstrip()
    _notice_html = f"<span style='font-weight:800;'>{_prefix}</span> {_rest}"
else:
    _notice_html = _notice_text

st.markdown(
    f'<div class="notice-pill">{_notice_html}</div>',
    unsafe_allow_html=True,
)
st.write("")  # 약간의 여백

# =============================
# Row 1: (좌) 사진+설명 / (우) 지도+탭
# =============================

# 상단 2개 카드(좌/우) 영역 높이 고정
TOP_CARD_H = 600   # 전체 카드 높이(px)
PHOTO_H = 280     # 사진 영역 높이(px)
MAP_H = 360        # 지도 영역 높이(px)

left, right = st.columns([1, 2.2], gap="large")

with left:
    with st.container(border=True, height=TOP_CARD_H):
        st.markdown(
            '<div class="card-title">사고 장소 사진</div>', unsafe_allow_html=True
        )

        selected_photo = st.session_state.get("selected_acc_photo_path")
        selected_meta = st.session_state.get("selected_acc_meta")

        # 사진 영역 높이 고정(사진이 크거나 없을 때도 레이아웃 유지)
        with st.container(height=PHOTO_H):
            if selected_photo:
                st.image(selected_photo, width="stretch")
            else:
                st.info(
                    """
- 우측 지도에서 사고 지점을 클릭하면, 선택된 사고의 정보가 갱신됩니다.
- 관련 사진이 등록된 사고의 경우, 본 영역에 사고 장소 사진이 표시됩니다.
- 사진이 등록되지 않은 사고는 사진이 표시되지 않을 수 있습니다.
- 지도를 확대/축소하여 표시된 지점을 확인하시기 바랍니다.
                    """.strip()
                )

        st.write("")
        st.markdown('<div class="card-title">자세히 보기</div>', unsafe_allow_html=True)

        # 선택된 사고 정보는 여기(설명 텍스트)에 표시
        if selected_meta:
            st.write(selected_meta)
        else:
            st.markdown(
                """
- 우측 지도에서 사고 지점을 클릭하면 본 영역에 사고 유형 및 주소가 표시됩니다.
- 관련 사진이 등록된 사고의 경우, 상단에 사고 장소 사진이 함께 표시됩니다.
- 지도를 확대/축소하여 표시된 지점을 확인하시기 바랍니다.
                """.strip()
            )

with right:
    with st.container(border=True, height=TOP_CARD_H):
        st.markdown(
            '<div class="card-title">울릉군 지도</div>', unsafe_allow_html=True
        )
        st.caption("2025년 울릉군 위치 데이터 기반")

        # 지도 상단 탭(맵 카드 안에서만 탭)
        t1, t2, t3 = st.tabs(["교통사고 지점", "낙석 발생 지점", "버스 실시간 상황"])

        with t1:
            st.caption("울릉군 교통사고 지점")

            map_state = render_ulleung_folium_map(kind="accident", height=MAP_H)

            # 클릭 이벤트 처리(streamlit-folium 설치된 경우에만 동작)
            if isinstance(map_state, dict):
                # 클릭 좌표로 가장 가까운 마커(=CSV 행)를 찾음
                last = map_state.get("last_object_clicked")
                idx = None

                acc_points_meta = st.session_state.get("acc_points_meta", [])
                if (
                    isinstance(last, dict)
                    and "lat" in last
                    and "lng" in last
                    and acc_points_meta
                ):
                    lat0 = float(last["lat"])
                    lon0 = float(last["lng"])

                    best_idx = None
                    best_d = None
                    for p in acc_points_meta:
                        d = abs(float(p["lat"]) - lat0) + abs(float(p["lon"]) - lon0)
                        if best_d is None or d < best_d:
                            best_d = d
                            best_idx = int(p["idx"])

                    # 완전 근접한 경우만 채택(원하면 임계값 조절)
                    if best_d is not None and best_d < 1e-5:
                        idx = best_idx

                df_acc = load_accidents_csv()

                acc_type = "미상"
                addr = ""

                if idx is not None and (not df_acc.empty) and (idx in df_acc.index):
                    row = df_acc.loc[idx]
                    addr = _row_to_address(df_acc, row)

                    # 사고 유형 컬럼 후보
                    type_col_candidates = [
                        c
                        for c in ["type", "accident_type", "사고유형", "사고_type"]
                        if c in df_acc.columns
                    ]
                    type_col = type_col_candidates[0] if type_col_candidates else None
                    if type_col is not None:
                        v = row.get(type_col, None)
                        if v is not None:
                            s = str(v).strip()
                            if s and s.lower() not in ["nan", "none"]:
                                acc_type = s

                # 실제 클릭이 있었고, 클릭 좌표로 유효한 idx를 찾았을 때만 상태 업데이트
                if last is not None and idx is not None:
                    # 주소로 사진 찾기
                    photo = find_accident_photo_by_address(addr)

                    # 설명 영역에 표시될 텍스트
                    if addr:
                        st.session_state["selected_acc_meta"] = (
                            f"사고 유형 : {acc_type} / 주소 : {addr}"
                        )
                    else:
                        st.session_state["selected_acc_meta"] = (
                            f"사고 유형 : {acc_type} / 주소 : (없음)"
                        )

                    st.session_state["selected_acc_photo_path"] = (
                        str(photo) if photo else None
                    )

        with t2:
            st.caption("울릉군 낙석 발생 지점")
            render_ulleung_folium_map(kind="rockfall", height=MAP_H)

        with t3:
            st.caption("울릉군 버스 실시간 상황(샘플)")
            render_ulleung_folium_map(kind="bus", height=MAP_H)

        st.caption("※ 확대해서 확인해보세요")

# =============================
# Row 2: Layer 2개 (해상공지 / 도로통제)
# =============================
st.write("")
c1, c2 = st.columns(2, gap="large")

with c1:
    with st.container(border=True):
        st.markdown(
            '<div class="card-title">해상공지 (Layer)</div>', unsafe_allow_html=True
        )
        if show_sea_notice:
            st.write("여기에 해상공지 레이어/리스트/요약 들어갈 자리")
        else:
            st.caption("사이드바에서 해상공지 레이어가 꺼져있음")

with c2:
    with st.container(border=True):
        st.markdown(
            '<div class="card-title">도로 통제 공지 (Layer)</div>',
            unsafe_allow_html=True,
        )
        if show_road_control:
            st.write("여기에 도로 통제 공지 레이어/리스트/요약 들어갈 자리")
        else:
            st.caption("사이드바에서 도로통제 레이어가 꺼져있음")

# =============================
# Row 3: 그래프 3개
# =============================
if show_graphs:
    st.write("")
    g1, g2, g3 = st.columns(3, gap="large")

    def graph_card(col, title):
        with col:
            with st.container(border=True):
                st.markdown(
                    f'<div class="card-title">{title}</div>', unsafe_allow_html=True
                )
                st.info("그래프 자리 (placeholder)")
                st.write("")
                st.markdown(
                    '<div class="card-sub">설명 영역</div>', unsafe_allow_html=True
                )
                st.write("그래프 해석/요약/주의사항 등 들어갈 자리")

    graph_card(g1, "그래프 1")
    graph_card(g2, "그래프 2")
    graph_card(g3, "그래프 3")
else:
    st.write("")
    st.caption("하단 그래프는 사이드바에서 꺼져있음")

# -----------------------------
# (선택) 디버그
# -----------------------------
with st.expander("디버그(필터 확인)", expanded=False):
    st.json(
        {
            "date_range": str(date_range),
            "region": region,
            "show_sea_notice": show_sea_notice,
            "show_road_control": show_road_control,
            "show_graphs": show_graphs,
        }
    )
