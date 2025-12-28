import streamlit as st

from src.nav import go


NAV_ITEMS = [
    ("home", "홈"),
    ("alert", "울릉알리미"),
    ("EDA", "선박 데이터"),
    ("acc_caught", "사고 및 단속"),
    ("hazard", "낙석 위험 지역"),
    ("etc", "기타"),
]


def render_sidebar():
    """
    모든 페이지에서 동일하게 쓰는 공통 사이드바.
    - 레이어는 항상 켜진 상태로 고정.
    - 기본 페이지 목록은 숨기고, 커스텀 탭 버튼만 노출.
    """
    st.markdown(
        """
        <style>
          [data-testid="stSidebarNav"] { display: none; }
          [data-testid="stSidebar"] {
            background: #f4f5f7;
          }
          [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            background: transparent;
            border: 1px solid transparent;
            color: #2b2f36;
            padding: 10px 12px;
            font-size: 15px;
            font-weight: 600;
            text-align: left;
            border-radius: 10px;
            display: flex;
            justify-content: flex-start;
          }
          [data-testid="stSidebar"] .stButton > button:hover {
            background: #e7eaf0;
            border-color: #e0e3e8;
          }
          [data-testid="stSidebar"] .stButton > button:focus {
            box-shadow: none;
            outline: none;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("## ")
    for key, label in NAV_ITEMS:
        if st.sidebar.button(label, use_container_width=True):
            go(key)

    filters = {
        "layers": {
            "alert": True,
            "EDA": True,
            "acc_caught": True,
            "hazard": True,
            "etc": True,
        }
    }
    return filters
