import streamlit as st

# 프로젝트 pages/ 폴더에 실제로 있는 파일명 기준
PAGE_PATHS = {
    "home": "app.py",
    "alert": "pages/1_alert.py",
    "EDA": "pages/2_EDA.py",
    "acc_caught": "pages/3_acc_caught.py",
    "hazard": "pages/4_hazard.py",
    "etc": "pages/5_etc.py",
}


def go(page: str):
    """페이지 이동 헬퍼.

    사용 예:
      - go("alert")
      - go("EDA")
      - go("pages/3_acc_caught.py")

    Streamlit 버전에 따라 st.switch_page 지원 여부가 달라서
    없으면 안내만 띄우게 해둠.
    """

    # page가 키면 경로로 변환
    target = PAGE_PATHS.get(page, page)

    # Streamlit 멀티페이지 이동(지원되는 경우)
    if hasattr(st, "switch_page"):
        st.switch_page(target)
        return

    # fallback: switch_page가 없는 버전
    st.info(
        "현재 Streamlit 버전에서는 버튼으로 페이지 이동이 안 될 수 있음. "
        "왼쪽 사이드바의 페이지 목록에서 직접 이동.\n\n"
        f"(이동하려던 대상: {target})"
    )
