import streamlit as st
from src.sidebar import render_sidebar
from src.components import section_title, placeholder_map
from src.nav import go

st.set_page_config(page_title="홈", layout="wide")

filters = render_sidebar()

section_title("홈", "요약 카드 + 지도(전체 레이어 미리보기) 구조 (레이어 토글 연결됨)")

left, right = st.columns([1.2, 1.0], gap="large")

with left:
    st.markdown("### 메인 시각화 보드 (카드 클릭 → 탭 이동)")

    r1c1, r1c2 = st.columns(2, gap="large")
    with r1c1:
        st.markdown("#### 울릉 알리미")
        st.caption("요약/핵심 알림 카드")
        if st.button("울릉 알리미로 이동", use_container_width=True):
            go("alert")

    with r1c2:
        st.markdown("#### 입출도&날씨")
        st.caption("운항/기상 요약 카드")
        if st.button("입출도&날씨로 이동", use_container_width=True):
            go("EDA")

    r2c1, r2c2 = st.columns(2, gap="large")
    with r2c1:
        st.markdown("#### 사고 & 단속")
        st.caption("사고/단속 요약 카드")
        if st.button("사고 & 단속으로 이동", use_container_width=True):
            go("acc_caught")

    with r2c2:
        st.markdown("#### 낙석 위험 지역")
        st.caption("낙석 위험 요약 카드")
        if st.button("낙석 위험 지역으로 이동", use_container_width=True):
            go("hazard")

with right:
    # ✅ 여기서 layers 연결
    placeholder_map("맵 (전체 레이어 확인)", layers=filters["layers"])
    st.caption("레이어: 전체 표시(항상 켜짐)")

st.divider()
st.markdown("### 최근 알림 타임라인(자리만)")
st.info("여기에 최근 알림 리스트 / 로그 / 타임라인 들어갈 자리")
