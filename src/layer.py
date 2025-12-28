import streamlit as st

LAYER_LABELS = {
    "alert": "울릉 알리미",
    "EDA": "입출도&날씨",
    "acc_caught": "사고 & 단속",
    "hazard": "낙석 위험 지역",
    "etc": "기타",
}


def get_enabled_layers(layers: dict) -> list[str]:
    return [k for k, v in layers.items() if v]


def render_layers_on_map(layers: dict):
    """
    실제 지도 구현 전 단계: 켜진 레이어만 placeholder로 렌더.
    나중에 여기만 교체해서 pydeck/folium/kepler.gl 붙이면 됨.
    """
    enabled = get_enabled_layers(layers)

    if not enabled:
        st.info("켜진 레이어 없음. 사이드바에서 레이어 켜야함.")
        return

    st.caption("현재 지도에 표시되는 레이어")
    for key in enabled:
        label = LAYER_LABELS.get(key, key)
        with st.container(border=True):
            st.markdown(f"**{label} 레이어**")
            st.write("→ 여기서 실제 지도 오버레이(점/라인/폴리곤/마커) 렌더 예정")