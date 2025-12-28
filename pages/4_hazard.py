import streamlit as st

from src.sidebar import render_sidebar
from src.components import (
    section_title,
    placeholder_map,
    bottom_controls,
)


st.set_page_config(page_title="낙석 위험 지역", layout="wide")

# 공통 사이드바(필터/레이어 토글)
filters = render_sidebar()

section_title("낙석 위험 지역", "지도 중심 페이지 + 세부 조정바 (내용은 추후 채움)")

# 지도 자리 + 사이드바 레이어 토글 연결
placeholder_map("지도에 표시", layers=filters["layers"])

st.divider()

# 하단 조정바 자리
bottom_controls()
