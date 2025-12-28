import streamlit as st

from src.sidebar import render_sidebar
from src.components import (
    section_title,
    placeholder_chart,
)


st.set_page_config(page_title="기타", layout="wide")

# 공통 사이드바(필터/레이어 토글)
filters = render_sidebar()

section_title("기타", "문서/참고자료/데이터 출처/문의/버전정보 등을 넣는 페이지")

st.info("여기는 안내/문서 페이지로 써도 좋고, 앞으로 추가 탭(기능) 넣는 공간으로 써도 된다.")

st.divider()

placeholder_chart("데이터 출처/참고자료 (자리만)")
placeholder_chart("로드맵/할 일 (자리만)")
placeholder_chart("연락처/문의 (자리만)")
