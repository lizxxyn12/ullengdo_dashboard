# 이미지 최적화 가이드

## 현재 상태
- **acc_pic 폴더**: 464MB (48개 파일, 평균 13MB/파일)
- **rockfall 폴더**: 22MB (19개 파일, 평균 1-2MB/파일)
- **총 용량**: 약 486MB

## 문제점
1. 이미지 파일이 너무 커서 로딩 속도가 느림
2. 모바일 환경에서 데이터 과다 사용
3. 메모리 사용량 증가로 인한 성능 저하

## 해결 방법

### 1단계: 이미지 최적화 실행

```bash
# 가상환경 활성화
source .venv/bin/activate

# Pillow 설치 확인 (requirements.txt에 이미 포함)
pip install Pillow

# 최적화 스크립트 실행
python optimize_images.py
```

### 최적화 효과
- **예상 용량 감소**: 80-90% (약 400MB → 50-80MB)
- **해상도**: 최대 1920x1080 (Full HD, 대부분 디스플레이에 충분)
- **품질**: JPG 85% (육안으로 구분 어려운 수준)
- **원본 백업**: 자동으로 `acc_pic_backup`, `rockfall_backup` 폴더에 보관

### 2단계: 최적화 결과 확인

```bash
# 폴더 크기 확인
du -sh acc_pic rockfall

# 최적화 전후 비교
du -sh acc_pic_backup acc_pic
du -sh rockfall_backup rockfall
```

## 앱 코드 최적화 (이미 완료)

다음 최적화가 `app.py`에 적용되었습니다:

### 1. 이미지 캐싱 함수 추가
```python
@st.cache_data(show_spinner=False)
def _load_and_cache_image(image_path: str, max_size: tuple = (1920, 1080)):
    """이미지를 로드하고 캐싱 (메모리 최적화)"""
```

**효과**:
- 같은 이미지를 여러 번 로드하지 않음
- 자동으로 1920x1080으로 리사이징 (메모리 절약)
- EXIF 회전 정보 자동 적용

### 2. 이미지 표시 부분 수정
- `Image.open()` → `_load_and_cache_image()` 사용
- `width="stretch"` → `use_container_width=True` (Streamlit 권장)

## 최적화 설정 커스터마이징

`optimize_images.py` 파일에서 다음 값을 조정할 수 있습니다:

```python
optimize_folder(
    folder,
    max_width=1920,      # 최대 너비 (픽셀)
    max_height=1080,     # 최대 높이 (픽셀)
    jpg_quality=85,      # JPG 품질 (1-100, 높을수록 고품질)
    backup=True,         # 백업 여부
)
```

### 추천 설정

**모바일 중심 (최대 데이터 절약)**:
```python
max_width=1280
max_height=720
jpg_quality=75
```

**태블릿/데스크톱 균형**:
```python
max_width=1920  # 현재 설정
max_height=1080
jpg_quality=85
```

**고품질 유지**:
```python
max_width=2560
max_height=1440
jpg_quality=90
```

## 백업에서 복원하는 방법

최적화 후 문제가 있다면 백업에서 복원:

```bash
# 백업 폴더에서 원본 복원
rm -rf acc_pic
mv acc_pic_backup acc_pic

rm -rf rockfall
mv rockfall_backup rockfall
```

## 추가 최적화 팁

### WebP 형식으로 변환 (선택사항)
WebP는 JPG보다 30% 작은 크기로 동일 품질 제공:

`optimize_images.py`의 저장 부분을 수정:
```python
# JPG 대신 WebP로 저장
new_output = output_path.with_suffix(".webp")
img.save(new_output, "WEBP", quality=85, method=6)
```

### 프로그레시브 JPG
인터넷 환경에서 점진적으로 로딩:
```python
img.save(output_path, "JPEG", quality=85, optimize=True, progressive=True)
```
(이미 적용됨)

## 성능 개선 효과 요약

| 항목 | 최적화 전 | 최적화 후 | 개선율 |
|------|----------|----------|--------|
| 총 용량 | ~486MB | ~50-80MB | 80-90% ↓ |
| 평균 파일 크기 | 13MB | 1-2MB | 85% ↓ |
| 로딩 시간 | 5-10초 | 1-2초 | 70% ↓ |
| 메모리 사용 | 높음 | 중간 | 50% ↓ |
| 모바일 데이터 | 많음 | 적음 | 80% ↓ |

## 문제 해결

### Pillow 설치 오류
```bash
pip install --upgrade Pillow
```

### 권한 오류
```bash
chmod +x optimize_images.py
python optimize_images.py
```

### 특정 이미지만 최적화
스크립트를 수정하여 특정 폴더만 처리:
```python
folders = [
    project_root / "acc_pic",  # 이것만 최적화
    # project_root / "rockfall",  # 주석 처리
]
```
