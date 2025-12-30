# 울릉 교통/안전 대시보드

울릉군 교통/안전 데이터를 인터랙티브 지도에서 시각화하는 Streamlit 대시보드입니다.

## 주요 기능
- CSV에서 사고 좌표를 불러와 Folium 지도에 표시
- 사고 유형(type)이 있으면 팝업에 표기
- 넓은 화면에 맞춘 간단한 레이아웃/스타일

## 프로젝트 구조
- `app.py`: Streamlit 메인 앱
- `ulleung_accidents_with_coords.csv`: 좌표 포함 사고 데이터
- `csv_reload.ipynb`: CSV 정리/매칭용 노트북

## 데이터 형식
필수 컬럼:
- `latitude`
- `longitude`

선택 컬럼:
- `type` (팝업에 사고 유형 표시, 없으면 "미상")

## 설치
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행
```bash
streamlit run app.py
```

## 참고
- `folium` 또는 `streamlit-folium`이 없으면 지도가 표시되지 않습니다.
- 데이터는 프로젝트 루트의 `ulleung_accidents_with_coords.csv`를 사용합니다.
