

import streamlit as st

from src.sidebar import render_sidebar
from src.components import (
    section_title,
    placeholder_kpis,
    grid_2x2,
)


st.set_page_config(page_title="울릉알리미", layout="wide")

# 공통 사이드바(필터/레이어 토글)
filters = render_sidebar()

section_title("울릉 알리미", "상단 KPI + 2x2 시각화 그리드 (내용은 추후 채움)")

# KPI 자리
placeholder_kpis(4)

st.divider()

# 2x2 콘텐츠 자리
grid_2x2()
