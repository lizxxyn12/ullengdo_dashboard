import streamlit as st
import pandas as pd
from pathlib import Path
import re
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

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

# Matplotlib 한글 폰트 설정 (환경별 자동 선택)
_font_candidates = [
    "AppleGothic",  # macOS
    "NanumGothic",  # Linux/Windows (설치 시)
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
# CSS (카드/여백/폰트 약간 정리)
# -----------------------------
st.markdown(
    """
<style>
/* 전체 폭 여백 조금 줄이기 */
.block-container { padding-top: 2.4rem; padding-bottom: 2.4rem; }

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

.r2-card {
  background: #f6f7fb;
  border: 1px solid #ebedf3;
  border-radius: 22px;
  padding: 18px 18px 16px 18px;
  height: 90%;
  box-sizing: border-box;
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
.sea-bars {
  display: grid;
  gap: 12px;
  margin-bottom: 14px;
}
.bar-row {
  display: grid;
  grid-template-columns: 120px 1fr 80px;
  gap: 10px;
  align-items: center;
}
.bar-label {
  font-weight: 700;
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
}
.bar-fill {
  height: 14px;
  border-radius: 999px;
  position: relative;
}
.bar-pill {
  position: absolute;
  left: 8px;
  top: -8px;
  background: #ffffff;
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 0.72rem;
  font-weight: 700;
  border: 1px solid rgba(0,0,0,0.06);
}
.bar-value {
  color: #666;
  font-size: 0.85rem;
  text-align: right;
}
.sea-latest {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
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
.bar-fill-split {
  height: 14px;
  border-radius: 999px;
  overflow: hidden;
  display: flex;
}
.bar-seg {
  height: 100%;
}
.sea-table {
  background: #ffffff;
  border: 1px solid #e8ebf2;
  border-radius: 14px;
  padding: 10px 12px;
}
.sea-table-row {
  display: grid;
  grid-template-columns: 90px 1fr 1fr 1fr 1fr 1fr;
  gap: 8px;
  padding: 8px 0;
  border-top: 1px solid #f0f2f6;
}
.sea-table-row:first-child {
  border-top: none;
}
.sea-table-head {
  font-weight: 800;
  background: #f0f2f7;
  border-radius: 10px;
  padding: 8px 10px;
}
.road-list {
  display: grid;
  gap: 10px;
}
.road-item {
  background: #ffffff;
  border: 1px solid #e8ebf2;
  border-radius: 14px;
  padding: 10px 12px;
}
.road-item-title {
  font-weight: 800;
  margin-bottom: 4px;
}
.road-item-meta {
  color: #666;
  font-size: 0.82rem;
}
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


def load_rockfall_points() -> tuple[list[tuple[float, float, str]], list[str]]:
    """rockfall 폴더 사진명(주소) 기반으로 좌표 매칭."""
    rock_dir = Path(__file__).parent / "rockfall"
    if not rock_dir.exists():
        return [], []

    coords_path = Path(__file__).parent / "rockfall_coords.csv"
    if coords_path.exists():
        df_coords = pd.read_csv(coords_path, encoding="utf-8")
        needed = {"filename", "latitude", "longitude"}
        if needed.issubset(df_coords.columns):
            points = []
            meta = []
            for _, row in df_coords.iterrows():
                lat = pd.to_numeric(row.get("latitude", None), errors="coerce")
                lon = pd.to_numeric(row.get("longitude", None), errors="coerce")
                if pd.isna(lat) or pd.isna(lon):
                    continue
                name = row.get("filename", "")
                if not name:
                    continue
                address = row.get("address", "") or Path(str(name)).stem
                photo_path = rock_dir / str(name)
                points.append((float(lat), float(lon), f"낙석: {address}"))
                meta.append(
                    {
                        "lat": float(lat),
                        "lon": float(lon),
                        "photo": str(photo_path) if photo_path.exists() else None,
                        "name": str(address),
                    }
                )
            if points:
                return points, meta

    df = load_accidents_csv()
    if df.empty:
        return [], []

    addr_cols = [
        c for c in ["clean_normalized", "raw", "detail", "주소"] if c in df.columns
    ]
    if not addr_cols:
        return [], []

    norm_lookup = {}
    for _, row in df.iterrows():
        lat = row.get("latitude", None)
        lon = row.get("longitude", None)
        if pd.isna(lat) or pd.isna(lon):
            continue
        for col in addr_cols:
            val = row.get(col, None)
            if val is None:
                continue
            key = _norm_text(val)
            if key and key not in norm_lookup:
                norm_lookup[key] = (float(lat), float(lon))

    exts = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"}
    points = []
    meta = []
    for p in rock_dir.iterdir():
        if not p.is_file() or p.suffix not in exts:
            continue
        name = p.stem
        key = _norm_text(name)
        loc = norm_lookup.get(key)
        if loc is None:
            continue
        lat, lon = loc
        points.append((lat, lon, f"낙석: {name}"))
        meta.append(
            {
                "lat": float(lat),
                "lon": float(lon),
                "photo": str(p),
                "name": str(name),
            }
        )

    return points, meta


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
        sample_points, rockfall_meta = load_rockfall_points()
        st.session_state["rockfall_points_meta"] = rockfall_meta
        if not sample_points:
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


@st.cache_data(show_spinner=False)
def load_enforcement_counts_csv() -> pd.DataFrame:
    """여러 해의 교통단속 CSV를 로드.

    기대 컬럼:
      - 위반일시 또는 연도/월
    """
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
            # 일부가 포맷이 다를 경우 fallback 파싱
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


def _plot_weather_passenger(
    x_vals, rain_vals, in_vals, out_vals, x_label, title, fig_size=(5.0, 3.0)
):
    """강수량(막대) + 입/출항 여객수(라인) 그래프."""
    fig, ax = plt.subplots(figsize=fig_size)
    ax2 = ax.twinx()

    ax.bar(x_vals, rain_vals, color="#6BAED6", alpha=0.45, label="강수량 합 (mm)")
    ax2.plot(
        x_vals,
        in_vals,
        marker="o",
        linewidth=2,
        color="#2CA02C",
        label="입도객수",
    )
    ax2.plot(
        x_vals,
        out_vals,
        marker="o",
        linewidth=2,
        linestyle="--",
        color="#E45756",
        label="출도객수",
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel("강수량 합 (mm)", color="#6BAED6")
    ax2.set_ylabel("여객수")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)

    handles = [
        plt.Line2D([0], [0], color="#6BAED6", lw=8, alpha=0.45),
        plt.Line2D([0], [0], color="#2CA02C", marker="o"),
        plt.Line2D([0], [0], color="#E45756", marker="o", linestyle="--"),
    ]
    labels = ["강수량 합 (mm)", "입도객수", "출도객수"]
    ax.legend(handles, labels, loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


def _plot_bar_matplotlib(x_vals, y_vals, x_label, title, fig_size=(5.0, 3.0)):
    """막대 그래프 (matplotlib)"""
    fig, ax = plt.subplots(figsize=fig_size)
    ax.bar(x_vals, y_vals, color="#6BAED6", alpha=0.75)
    ax.set_xlabel(x_label)
    ax.set_ylabel("건수")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


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
    path = Path(__file__).parent / "울릉알리미_텍스트.csv"
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

    time_match = re.search(r"(\\d{1,2})[:시](\\d{2})", msg)
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
if "selected_rockfall_meta" not in st.session_state:
    st.session_state["selected_rockfall_meta"] = None
if "selected_rockfall_photo_path" not in st.session_state:
    st.session_state["selected_rockfall_photo_path"] = None

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
    _rest = _notice_text[len(_prefix) :].lstrip()
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
TOP_CARD_H = 600  # 전체 카드 높이(px)
PHOTO_H = 280  # 사진 영역 높이(px)
MAP_H = 360  # 지도 영역 높이(px)

left, right = st.columns([1, 2.2], gap="large")

with left:
    with st.container(border=True, height=TOP_CARD_H):
        st.markdown(
            '<div class="card-title">사고 장소 사진</div>', unsafe_allow_html=True
        )

        selected_rockfall_photo = st.session_state.get("selected_rockfall_photo_path")
        selected_rockfall_meta = st.session_state.get("selected_rockfall_meta")
        selected_acc_photo = st.session_state.get("selected_acc_photo_path")
        selected_acc_meta = st.session_state.get("selected_acc_meta")

        # 사진 영역 높이 고정(사진이 크거나 없을 때도 레이아웃 유지)
        with st.container(height=PHOTO_H):
            if selected_rockfall_photo:
                st.image(selected_rockfall_photo, width="stretch")
            elif selected_acc_photo:
                st.image(selected_acc_photo, width="stretch")
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
        if selected_rockfall_meta:
            st.write(selected_rockfall_meta)
        elif selected_acc_meta:
            st.write(selected_acc_meta)
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
        st.markdown('<div class="card-title">울릉군 지도</div>', unsafe_allow_html=True)
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
            rock_map_state = render_ulleung_folium_map(kind="rockfall", height=MAP_H)
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
                    if best is not None and best_d is not None and best_d < 1e-5:
                        name = best.get("name", "")
                        photo = best.get("photo", None)
                        if name:
                            st.session_state["selected_rockfall_meta"] = (
                                f"낙석 위치 : {name}"
                            )
                        else:
                            st.session_state["selected_rockfall_meta"] = (
                                "낙석 위치 : (없음)"
                            )
                        st.session_state["selected_rockfall_photo_path"] = (
                            str(photo) if photo else None
                        )

        with t3:
            st.caption("울릉군 버스 실시간 상황(샘플)")
            render_ulleung_folium_map(kind="bus", height=MAP_H)

        st.caption("※ 확대해서 확인해보세요")


# =============================
# Row 2: Layer 2개 (해상공지 / 도로통제)
# =============================
sms_counts, sms_total, sms_breakdown = _summarize_sms_notice_counts(
    load_sms_raw(),
    year=2025,
)
sea_latest_label, sea_latest_text = _latest_sea_notice(load_sms_raw(), year=2025)


def _pct(count: int, total: int) -> int:
    if total <= 0:
        return 0
    return int(round(count / total * 100))


sea_arrive = sms_counts["입항"]
sea_depart = sms_counts["출항"]
sea_control = sms_counts["운항통제"]
sea_cancel = sms_counts["결항"]
sea_change = sms_counts["시간변경"]
sea_total = sms_total
sea_arrive_pct = _pct(sea_arrive, sea_total)
sea_depart_pct = _pct(sea_depart, sea_total)
sea_control_pct = _pct(sea_control, sea_total)
sea_cancel_pct = _pct(sea_cancel, sea_total)
sea_change_pct = _pct(sea_change, sea_total)
sea_arrive_ship = sms_breakdown["입항"]["선박"]
sea_arrive_people = sms_breakdown["입항"]["사람"]
sea_depart_ship = sms_breakdown["출항"]["선박"]
sea_depart_people = sms_breakdown["출항"]["사람"]
sea_arrive_ship_pct = _pct(sea_arrive_ship, sea_arrive)
sea_arrive_people_pct = 100 - sea_arrive_ship_pct if sea_arrive > 0 else 0
sea_depart_ship_pct = _pct(sea_depart_ship, sea_depart)
sea_depart_people_pct = 100 - sea_depart_ship_pct if sea_depart > 0 else 0

st.write("")
c1, c2 = st.columns(2, gap="large")
ROW2_CARD_H = 350

with c1:
    with st.container(border=True, height=ROW2_CARD_H):
        if show_sea_notice:
            st.markdown(
                f"""
<div class="r2-card">
  <div class="r2-head">
    <div class="r2-title">해상 공지</div>
    <div class="r2-date">2025년 기준</div>
  </div>
  <div class="sea-latest">
    <div class="sea-pill">{sea_latest_label}</div>
    <div class="sea-latest-text">{sea_latest_text}</div>
  </div>
  <div class="sea-bars">
    <div class="bar-row">
      <div class="bar-label">입항 <span class="bar-sub">(선박/사람)</span></div>
      <div class="bar-track">
        <div class="bar-fill-split" style="width:{sea_arrive_pct}%;">
          <div class="bar-seg" style="width:{sea_arrive_ship_pct}%; background:#ff8a3d;"></div>
          <div class="bar-seg" style="width:{sea_arrive_people_pct}%; background:#ffd3a8;"></div>
        </div>
      </div>
      <div class="bar-value">{sea_arrive:,}</div>
    </div>
    <div class="bar-row">
      <div class="bar-label">출항 <span class="bar-sub">(선박/사람)</span></div>
      <div class="bar-track">
        <div class="bar-fill-split" style="width:{sea_depart_pct}%;">
          <div class="bar-seg" style="width:{sea_depart_ship_pct}%; background:#00b3a4;"></div>
          <div class="bar-seg" style="width:{sea_depart_people_pct}%; background:#8fe3da;"></div>
        </div>
      </div>
      <div class="bar-value">{sea_depart:,}</div>
    </div>
    <div class="bar-row">
      <div class="bar-label">운항통제</div>
      <div class="bar-track">
        <div class="bar-fill" style="width:{sea_control_pct}%; background:#5b2bff;"></div>
      </div>
      <div class="bar-value">{sea_control:,}</div>
    </div>
    <div class="bar-row">
      <div class="bar-label">결항</div>
      <div class="bar-track">
        <div class="bar-fill" style="width:{sea_cancel_pct}%; background:#e24a4a;"></div>
      </div>
      <div class="bar-value">{sea_cancel:,}</div>
    </div>
    <div class="bar-row">
      <div class="bar-label">시간변경</div>
      <div class="bar-track">
        <div class="bar-fill" style="width:{sea_change_pct}%; background:#7b61ff;"></div>
      </div>
      <div class="bar-value">{sea_change:,}</div>
    </div>
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )
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

# =============================
# Row 3: 그래프 3개
# =============================
if show_graphs:

    st.write("")
    g1, g2, g3 = st.columns(3, gap="large")
    GRAPH_CARD_H = 600
    GRAPH_CHART_H = 320
    GRAPH_FIG_W = 5.0
    GRAPH_FIG_H = 3.2
    GRAPH_FIG_W_G2 = 7.0
    GRAPH_FIG_H_G2 = 4.2

    def graph_card(col, title):
        with col:
            with st.container(border=True, height=GRAPH_CARD_H):
                st.markdown(
                    f'<div class="card-title">{title}</div>', unsafe_allow_html=True
                )
                st.info("그래프 자리 (placeholder)")
                st.write("")
                st.markdown(
                    '<div class="card-sub">설명 영역</div>', unsafe_allow_html=True
                )
                st.write("그래프 해석/요약/주의사항 등 들어갈 자리")

    with g1:
        with st.container(border=True, height=GRAPH_CARD_H):
            st.markdown(
                '<div class="card-title">교통단속 건수</div>',
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
                        fig = _plot_bar_matplotlib(
                            summary.index.tolist(),
                            summary.tolist(),
                            "월",
                            f"{year}년 월별 교통단속 건수",
                            fig_size=(GRAPH_FIG_W, GRAPH_FIG_H),
                        )
                        st.pyplot(fig, width="stretch")
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
                        fig = _plot_bar_matplotlib(
                            summary.index.tolist(),
                            summary.tolist(),
                            "연도",
                            f"{month}월 연도별 교통단속 건수",
                            fig_size=(GRAPH_FIG_W, GRAPH_FIG_H),
                        )
                        st.pyplot(fig, width="stretch")
            st.write("")
            st.markdown('<div class="card-sub">설명 영역</div>', unsafe_allow_html=True)
            st.write(
                "- 연도별 선택 시: 해당 연도의 월별 단속 건수가 막대로 표시됩니다.\n"
                "- 월별 선택 시: 선택한 월의 연도별 단속 건수 비교가 가능합니다.\n"
                "- 막대 높이 차이를 통해 성수/비수기의 변화 폭을 직관적으로 봅니다.\n"
                "- 기준선이 없으므로, 변화 추이는 막대 간 상대 비교로 해석합니다."
            )

    with g2:
        with st.container(border=True, height=GRAPH_CARD_H):
            st.markdown(
                '<div class="card-title">강수량 · 여객수</div>',
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
                        fig = _plot_weather_passenger(
                            sub.index.tolist(),
                            sub["월강수합"].tolist(),
                            sub["월입항합"].tolist(),
                            sub["월출항합"].tolist(),
                            "월",
                            f"{year}년 월별 강수량/여객수",
                            fig_size=(GRAPH_FIG_W_G2, GRAPH_FIG_H_G2),
                        )
                        st.pyplot(fig, width="stretch")
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
                        fig = _plot_weather_passenger(
                            sub.index.tolist(),
                            sub["월강수합"].tolist(),
                            sub["월입항합"].tolist(),
                            sub["월출항합"].tolist(),
                            "연도",
                            f"{month}월 연도별 강수량/여객수",
                            fig_size=(GRAPH_FIG_W_G2, GRAPH_FIG_H_G2),
                        )
                        st.pyplot(fig, width="stretch")
            st.write("")
            st.markdown('<div class="card-sub">설명 영역</div>', unsafe_allow_html=True)
            st.write(
                "- 막대는 강수량 합, 선은 입도/출도 여객수 추이를 함께 보여줍니다.\n"
                "- 연도별 선택 시: 한 해의 월별 패턴을 한 화면에서 비교합니다.\n"
                "- 월별 선택 시: 같은 달의 연도별 변화 방향을 확인합니다.\n"
                "- 막대와 선의 동조/역행 여부가 핵심 해석 포인트입니다."
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
                season_map, threshold = _compute_season_map(monthly, value_col)
                sub = (
                    monthly[monthly["연"] == year]
                    .set_index("월")
                    .reindex(range(1, 13), fill_value=0)
                )
                months = list(range(1, 13))
                values = sub[value_col].tolist()
                colors = [
                    "#B22222" if season_map.get(m) == "성수기" else "#87CEEB"
                    for m in months
                ]

                fig, ax = plt.subplots(figsize=(GRAPH_FIG_W, GRAPH_FIG_H))
                ax.bar(months, values, color=colors, alpha=0.8)
                if threshold is not None:
                    ax.axhline(
                        threshold,
                        color="gray",
                        linestyle="--",
                        linewidth=1.2,
                        label=f"전체 평균 ({threshold:,.0f})",
                    )
                ax.set_title(
                    f"{year}년 월별 여객 수 ({direction} 기준)",
                    fontsize=12,
                )
                ax.set_xlabel("월")
                ax.set_ylabel("여객 수")
                ax.set_xticks(months)
                ax.grid(axis="y", linestyle="--", alpha=0.6)
                if threshold is not None:
                    ax.legend(loc="upper right")
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
            st.write("")
            st.markdown('<div class="card-sub">설명 영역</div>', unsafe_allow_html=True)
            st.write(
                "- 월별 막대 색으로 성수기/비수기를 구분해 시각화합니다.\n"
                "- 기준선은 전체 평균이며, 평균을 넘는 달이 성수기로 표시됩니다.\n"
                "- 입도/출도 선택에 따라 성수기 판단이 달라질 수 있습니다.\n"
                "- 월별 변동이 큰 구간을 색 대비로 빠르게 파악합니다."
            )
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
