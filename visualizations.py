"""
시각화 관련 함수 모음

이 모듈은 다음을 포함합니다:
- Folium 지도 렌더링 함수들
- Vega-Lite 차트 스펙 생성 함수들
- 사고/낙석 마커 빌드 함수들
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import textwrap
import copy

# Folium 관련 import (try/except로 안전하게)
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

# data_loaders.py에서 필요한 함수들 import
from data_loaders import (
    _accident_files_signature,
    load_accidents_csv,
    load_rockfall_points,
    build_bus_routes,
    _bus_route_defs,
    load_ev_charger_points,
    load_ev_charger_data,
    _simulate_bus_positions,
    _polyline_segments,
    _point_on_segments,
)


# -----------------------------
# Folium 지도 관련 함수들
# -----------------------------


def _build_accident_points(
    df_acc: pd.DataFrame, max_points: int = 2000
) -> tuple[list[tuple[float, float, str]], list[dict]]:
    """사고 마커와 메타 생성 (필터된 DF용) - vectorized 최적화."""
    if df_acc.empty:
        return [], []

    type_col_candidates = [
        c for c in ["type", "accident_type", "사고유형", "사고_type"] if c in df_acc.columns
    ]
    type_col = type_col_candidates[0] if type_col_candidates else None

    # 상위 max_points만 선택 (head()는 이미 새 DataFrame 반환)
    df_sample = df_acc.head(max_points)

    # 사고 유형 처리 (vectorized)
    if type_col is not None and type_col in df_sample.columns:
        # 결측값 처리
        df_sample["acc_type"] = df_sample[type_col].fillna("미상").astype(str).str.strip()
        # "nan", "none" 같은 문자열을 "미상"으로 변환
        df_sample.loc[df_sample["acc_type"].str.lower().isin(["nan", "none", ""]), "acc_type"] = "미상"
    else:
        df_sample["acc_type"] = "미상"

    # 좌표 추출 (vectorized)
    df_sample["lat_float"] = df_sample["latitude"].astype(float)
    df_sample["lon_float"] = df_sample["longitude"].astype(float)
    df_sample["popup_text"] = "사고 유형 : " + df_sample["acc_type"]

    # 리스트 생성 (zip은 iterrows보다 훨씬 빠름)
    sample_points = list(zip(
        df_sample["lat_float"],
        df_sample["lon_float"],
        df_sample["popup_text"]
    ))

    # 메타데이터 생성 (list comprehension with zip)
    acc_points_meta = [
        {"idx": int(idx), "lat": lat, "lon": lon}
        for idx, lat, lon in zip(df_sample.index, df_sample["lat_float"], df_sample["lon_float"])
    ]

    return sample_points, acc_points_meta


def _build_folium_base_map(
    kind: str,
    accident_signature: tuple | None = None,
    accident_year_filter: int | None = None,
    selected_route_id: str | None = None,
) -> tuple[folium.Map, list[dict], list[dict], list[dict], list[dict]]:
    """마커/라인 포함 기본 Folium 지도 생성 (하이라이트 제외)."""
    center = (37.5044, 130.8757)
    m = folium.Map(
        location=center, zoom_start=12, tiles="OpenStreetMap", control_scale=True
    )

    acc_points_meta: list[dict] = []
    rockfall_meta: list[dict] = []
    bus_stops_meta: list[dict] = []
    ev_meta: list[dict] = []

    if kind == "accident":
        if accident_signature is None:
            accident_signature = _accident_files_signature()
        df_acc = load_accidents_csv(accident_signature)
        if accident_year_filter is not None and "year" in df_acc.columns:
            df_acc = df_acc[df_acc["year"] == accident_year_filter]
        sample_points, acc_points_meta = _build_accident_points(df_acc)
        if not sample_points:
            sample_points = [
                (37.4890, 130.9050, "사고 유형 : 사고(샘플) A"),
                (37.4770, 130.9130, "사고 유형 : 사고(샘플) B"),
                (37.4705, 130.8985, "사고 유형 : 사고(샘플) C"),
            ]
            acc_points_meta = [
                {"idx": 0, "lat": 37.4890, "lon": 130.9050},
                {"idx": 1, "lat": 37.4770, "lon": 130.9130},
                {"idx": 2, "lat": 37.4705, "lon": 130.8985},
            ]
        color = "red"
    elif kind == "rockfall":
        sample_points, rockfall_meta = load_rockfall_points()
        if not sample_points:
            sample_points = [
                (37.4950, 130.9145, "낙석 발생 위치 : (샘플) A"),
                (37.4680, 130.8920, "낙석 발생 위치 : (샘플) B"),
            ]
        color = "orange"
    elif kind == "bus":
        routes, bus_stops_meta = build_bus_routes()
        if not bus_stops_meta:
            bus_stops_meta = [
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
        if selected_route_id:
            route_name_map = {r["id"]: r["name"] for r in _bus_route_defs()}
            selected_route_name = route_name_map.get(selected_route_id)
            if selected_route_name:
                bus_stops_meta = [
                    s
                    for s in bus_stops_meta
                    if selected_route_name in (s.get("routes") or [])
                ]

        sample_points = []
        color = "blue"
        for stop in bus_stops_meta:
            name = stop.get("name", "(이름 없음)")
            routes_txt = (
                ", ".join(stop.get("routes", []))
                if stop.get("routes")
                else "경유 노선 정보 없음"
            )
            label = f"정류장 : {name}<br/>경유 노선 : {routes_txt}"
            sample_points.append((stop["lat"], stop["lon"], label))
    elif kind == "ev":
        # 전기차 충전소 전용 탭
        ev_points, ev_meta = load_ev_charger_data()
        sample_points = ev_points if ev_points else []
        color = "#2ca02c"
    else:
        sample_points = []
        color = "green"

    fg = folium.FeatureGroup(name=kind)
    marker_parent = fg
    bus_marker_parent = fg
    marker_points = sample_points

    if kind == "bus":
        routes_defs = {r["name"]: r["color"] for r in _bus_route_defs()}
        marker_points = []
        for stop in bus_stops_meta:
            routes_txt = (
                ", ".join(stop.get("routes", []))
                if stop.get("routes")
                else "경유 노선 정보 없음"
            )
            label = f"정류장 : {stop['name']}<br/>경유 노선 : {routes_txt}"
            first_route = stop.get("routes", [None])[0] if stop.get("routes") else None
            color_for_stop = routes_defs.get(first_route, "#666666")
            marker_points.append((stop["lat"], stop["lon"], label, color_for_stop))

    # MarkerCluster 최적화 사용
    # 참고: FastMarkerCluster는 팝업 지원이 제한적이므로 MarkerCluster 사용
    if MarkerCluster is not None:
        # 마커가 많을 때만 클러스터링 (성능 향상)
        use_clustering = len(marker_points) > 20
        if use_clustering:
            if kind in {"accident", "rockfall", "ev"}:
                marker_parent = MarkerCluster(
                    name=f"{kind}_cluster",
                    options={
                        'disableClusteringAtZoom': 15,  # 줌 15 이상에서는 클러스터 해제
                        'maxClusterRadius': 60,  # 클러스터 반경 최적화
                    }
                ).add_to(fg)
            if kind == "bus":
                marker_parent = MarkerCluster(
                    name="bus_stops_cluster",
                    options={'disableClusteringAtZoom': 14, 'maxClusterRadius': 50}
                ).add_to(fg)
                bus_marker_parent = MarkerCluster(
                    name="bus_cluster",
                    options={'disableClusteringAtZoom': 14, 'maxClusterRadius': 40}
                ).add_to(fg)

    # 마커 일괄 생성 (루프 최적화)
    for mp in marker_points:
        if kind == "bus":
            lat, lon, label, m_color = mp
        else:
            lat, lon, label = mp
            m_color = color

        # 최적화: 간단한 팝업 HTML (f-string 최소화)
        folium.CircleMarker(
            location=(lat, lon),
            radius=5,
            color=m_color,
            fill=True,
            fill_opacity=0.85,
            popup=folium.Popup(
                f"<div style='font-size:12px;line-height:1.25;max-width:200px;white-space:normal;'>{label}</div>",
                max_width=220
            ),
        ).add_to(marker_parent)

    if kind == "bus":
        routes, _ = build_bus_routes()
        if selected_route_id:
            routes = [r for r in routes if r.get("id") == selected_route_id]
        for r in routes:
            pts = r.get("points", [])
            if len(pts) < 2:
                continue
            is_selected = selected_route_id and r.get("id") == selected_route_id
            if is_selected:
                folium.PolyLine(
                    pts,
                    color="#ffffff",
                    weight=10,
                    opacity=0.9,
                ).add_to(fg)
            folium.PolyLine(
                pts,
                color=r.get("color", "blue"),
                weight=8 if is_selected else 3,
                opacity=0.95 if is_selected else 0.25,
                tooltip=r.get("name", ""),
            ).add_to(fg)

        bus_positions = _simulate_bus_positions(
            routes, per_route=2 if selected_route_id else 1
        )
        selected_bus_pos = None
        if selected_route_id:
            for bus in bus_positions:
                if bus.get("route_id") == selected_route_id:
                    selected_bus_pos = bus
                    break
        if selected_bus_pos is None and selected_route_id:
            for route in routes:
                if route.get("id") == selected_route_id:
                    pts = route.get("points", [])
                    total, segments = _polyline_segments(pts)
                    if total > 0:
                        midpoint = _point_on_segments(segments, total * 0.5)
                        if midpoint:
                            selected_bus_pos = {
                                "route_id": selected_route_id,
                                "lat": midpoint[0],
                                "lon": midpoint[1],
                            }
                    break
        if selected_bus_pos and DivIcon is not None:
            pulse_css = """
            <div style="
                width: 18px;
                height: 18px;
                background-color: rgba(229, 57, 53, 0.7);
                border-radius: 50%;
                box-shadow: 0 0 0 0 rgba(229, 57, 53, 0.8);
                animation: pulse-red 1.4s infinite;
                "></div>
            <style>
                @keyframes pulse-red {
                    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(229, 57, 53, 0.8); }
                    70% { transform: scale(1); box-shadow: 0 0 0 18px rgba(229, 57, 53, 0); }
                    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(229, 57, 53, 0); }
                }
            </style>
            """
            folium.Marker(
                location=(selected_bus_pos["lat"], selected_bus_pos["lon"]),
                icon=DivIcon(
                    icon_size=(18, 18),
                    icon_anchor=(9, 9),
                    html=pulse_css,
                ),
                tooltip="현재 위치",
            ).add_to(fg)
        for bus in bus_positions:
            tooltip = f"가상 버스 {bus['route_id']}노선"
            if DivIcon is not None:
                bus_svg = """
                <svg width="30" height="20" viewBox="0 0 30 20" xmlns="http://www.w3.org/2000/svg">
                  <rect x="1.5" y="2.5" width="27" height="13" rx="4" fill="#ffca3a" stroke="#1f1f1f" stroke-width="1.5"/>
                  <rect x="4" y="5" width="9" height="5" rx="1.5" fill="#f6f7fb" stroke="#1f1f1f" stroke-width="0.8"/>
                  <rect x="14.5" y="5" width="11" height="5" rx="1.5" fill="#f6f7fb" stroke="#1f1f1f" stroke-width="0.8"/>
                  <rect x="3.5" y="10.5" width="22" height="2" rx="1" fill="#1f1f1f" opacity="0.12"/>
                  <circle cx="9" cy="16.5" r="2" fill="#1f1f1f"/>
                  <circle cx="21.5" cy="16.5" r="2" fill="#1f1f1f"/>
                  <circle cx="9" cy="16.5" r="0.8" fill="#f6f7fb"/>
                  <circle cx="21.5" cy="16.5" r="0.8" fill="#f6f7fb"/>
                </svg>
                """
                folium.Marker(
                    location=(bus["lat"], bus["lon"]),
                    icon=DivIcon(
                        icon_size=(28, 18),
                        icon_anchor=(14, 9),
                        html=bus_svg,
                    ),
                    tooltip=tooltip,
                ).add_to(bus_marker_parent)
            else:
                folium.CircleMarker(
                    location=(bus["lat"], bus["lon"]),
                    radius=6,
                    color="#222222",
                    weight=2,
                    fill=True,
                    fill_color="#ffd54a",
                    fill_opacity=0.95,
                    tooltip=tooltip,
                ).add_to(bus_marker_parent)

    fg.add_to(m)

    # 전기차 충전소 마커 - kind="ev"일 때만 별도로 추가 (다른 탭에서는 미표시)
    # kind="ev"인 경우 sample_points로 이미 처리됨

    return m, acc_points_meta, rockfall_meta, bus_stops_meta, ev_meta


@st.cache_data(show_spinner=False, max_entries=10)
def _cached_folium_base_map(
    kind: str,
    accident_signature: tuple | None = None,
    accident_year_filter: int | None = None,
    selected_route_id: str | None = None,
) -> tuple[folium.Map, list[dict], list[dict], list[dict], list[dict]]:
    # 캐시됨: selected_route_id가 변경되면 새로운 캐시 엔트리 생성
    # max_entries로 메모리 사용량 제한
    return _build_folium_base_map(
        kind, accident_signature, accident_year_filter, selected_route_id
    )


def render_ulleung_folium_map(
    kind: str = "base",
    height: int = 420,
    accident_df: pd.DataFrame | None = None,
    accident_year_filter: int | None = None,
    highlight_idx: int | None = None,
    center_override: tuple[float, float] | None = None,
    selected_route_id: str | None = None,
    show_ev: bool = True,
):
    """
    울릉군 Folium 지도 렌더 (최적화됨).

    최적화:
    - copy.deepcopy() 제거하여 수백 ms 절약
    - highlight가 없으면 캐시된 지도 재사용
    """

    if folium is None:
        st.error(
            "folium 패키지가 설치되어 있지 않아 지도를 표시할 수 없어. 터미널에서 `pip install folium` 해줘."
        )
        return

    requested_kind = kind

    accident_signature = None
    if kind == "accident" and accident_df is None:
        accident_signature = _accident_files_signature()

    base_map, acc_points_meta, rockfall_meta, bus_meta, ev_meta = _cached_folium_base_map(
        kind,
        accident_signature=accident_signature,
        accident_year_filter=accident_year_filter,
        selected_route_id=selected_route_id,
    )

    if acc_points_meta:
        st.session_state["acc_points_meta"] = acc_points_meta
    if rockfall_meta:
        st.session_state["rockfall_points_meta"] = rockfall_meta
    if bus_meta:
        st.session_state["bus_stops_meta"] = bus_meta
    if ev_meta:
        st.session_state["ev_charger_meta"] = ev_meta

    # 최적화: highlight나 center_override가 없으면 deepcopy 불필요
    # deepcopy는 매우 무거운 연산 (수백ms 소요)
    need_modification = (highlight_idx is not None) or (center_override is not None)

    if need_modification:
        # 수정이 필요한 경우에만 deepcopy
        m = copy.deepcopy(base_map)
    else:
        # 수정 불필요 시 원본 재사용 (10-100배 빠름)
        m = base_map
    if center_override is not None:
        m.location = center_override

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
                    ).add_to(m)
                folium.CircleMarker(
                    location=(lat, lon),
                    radius=6,
                    color="white",
                    weight=2,
                    fill=True,
                    fill_color=pulse_color,
                    fill_opacity=1.0,
                ).add_to(m)
                break

    # 지도 렌더 (가능하면 클릭 이벤트까지 받기)
    if st_folium is not None:
        return st_folium(
            m,
            height=height,
            width=None,
            key=f"folium_{requested_kind}",
            returned_objects=["last_object_clicked", "zoom", "center"],
        )

    import streamlit.components.v1 as components

    components.html(m.get_root().render(), height=height)
    return None


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
        "title": {"text": title, "fontSize": 0},  # title을 설정하되 크기 0으로 숨김
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
                    "x": {
                        "field": x_field,
                        "type": "ordinal",
                        "axis": {"labelAngle": 0},
                    },
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
        "title": {"text": title, "fontSize": 0},  # title을 설정하되 크기 0으로 숨김
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
        "title": {"text": title, "fontSize": 0},  # title을 설정하되 크기 0으로 숨김
        "config": _vega_base_config(),
    }
