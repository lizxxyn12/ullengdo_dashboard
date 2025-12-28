"""Shared app modules for the Streamlit dashboard.

- sidebar.py: 공통 사이드바(필터/레이어 토글)
- components.py: 공통 UI 컴포넌트(placeholder_map 등)
- nav.py: 페이지 이동 헬퍼(go)
- layer.py: 레이어 키 ↔ 라벨 매핑(LAYER_LABELS)

이 파일은 `src`를 패키지로 인식시키고, 모듈 구조를 명확히 하기 위한 용도.
"""

__all__ = [
    "sidebar",
    "components",
    "nav",
    "layer",
]
