import streamlit as st
from typing import Dict, Optional, Any, List


# layer 라벨은 src/layer.py에서 관리(없으면 fallback)
try:
    from src.layer import LAYER_LABELS  # type: ignore
except Exception:
    LAYER_LABELS = {
        "alert": "울릉 알리미",
        "EDA": "입출도&날씨",
        "acc_caught": "사고 & 단속",
        "hazard": "낙석 위험 지역",
        "etc": "기타",
    }


def section_title(title: str, desc: str = "") -> None:
    st.markdown(f"## {title}")
    if desc:
        st.caption(desc)


def placeholder_chart(title: str = "시각화") -> None:
    st.markdown(f"### {title}")
    st.info("여기에 차트/표 들어갈 자리")


def placeholder_kpis(n: int = 4) -> None:
    cols = st.columns(n, gap="large")
    for i, c in enumerate(cols, start=1):
        with c:
            st.metric(f"KPI {i}", "—")


def _enabled_layer_keys(layers: Dict[str, bool]) -> List[str]:
    return [k for k, v in layers.items() if bool(v)]


def render_enabled_layers(layers: Dict[str, bool]) -> None:
    """사이드바 레이어 토글 결과를 화면에 보여주는 최소 UI."""
    enabled = _enabled_layer_keys(layers)
    if not enabled:
        st.info("현재 켜진 레이어가 없음. 사이드바에서 레이어를 켜야함")
        return

    st.caption("현재 켜진 레이어")
    # 라벨을 한 줄로 보여주기
    labels = [LAYER_LABELS.get(k, k) for k in enabled]
    st.write(" • ".join(labels))


def placeholder_map(title: str = "지도", layers: Optional[Dict[str, bool]] = None) -> None:
    """지도 영역 placeholder.

    - layers가 들어오면(=sidebar 연결) 켜진 레이어만 아래에 표시.
    - 나중에 실제 지도(pydeck/folium 등) 붙일 때 이 함수만 교체/확장하면 됨.
    """
    st.markdown(f"### {title}")
    st.warning("여기에 지도(레이어) 들어갈 자리")
    st.caption("예: 사용자 위치 기반 중심 이동, 레이어 토글, 팝업 등")

    if layers is None:
        return

    # 실제 지도 오버레이는 추후 연결


def grid_2x2() -> None:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        placeholder_chart("시각화 1")
        st.caption("시각화 내용에 대한 설명")
    with c2:
        placeholder_chart("시각화 2")
        st.caption("시각화 내용에 대한 설명")

    c3, c4 = st.columns(2, gap="large")
    with c3:
        placeholder_chart("시각화 3")
        st.caption("시각화 내용에 대한 설명")
    with c4:
        placeholder_chart("시각화 4")
        st.caption("시각화 내용에 대한 설명")


def bottom_controls() -> None:
    st.markdown("### 세부 조정바(자리만)")
    col1, col2, col3, col4 = st.columns(4, gap="large")
    with col1:
        st.selectbox("필터 1", ["—"], index=0)
    with col2:
        st.selectbox("필터 2", ["—"], index=0)
    with col3:
        st.selectbox("필터 3", ["—"], index=0)
    with col4:
        st.button("적용", use_container_width=True)


def filters_debug(filters: Dict[str, Any]) -> None:
    """개발 중에만 쓰는 공통 필터 디버그 표시."""
    with st.expander("공통필터(디버그)", expanded=False):
        st.json(filters)
