#!/usr/bin/env python3
"""
이미지 최적화 스크립트
- 원본 이미지를 백업 폴더로 이동
- 웹/모바일에 적합한 크기로 리사이징 및 압축
- JPG: 최대 1920x1080, 품질 85%
- PNG: 최대 1920x1080, 최적화 적용
"""

import os
from pathlib import Path
from PIL import Image
import shutil


def optimize_image(
    input_path: Path,
    output_path: Path,
    max_width: int = 1920,
    max_height: int = 1080,
    jpg_quality: int = 85,
) -> tuple[int, int]:
    """
    이미지를 최적화하여 저장

    Args:
        input_path: 원본 이미지 경로
        output_path: 저장할 경로
        max_width: 최대 너비
        max_height: 최대 높이
        jpg_quality: JPG 품질 (1-100)

    Returns:
        (원본 크기, 최적화된 크기) in bytes
    """
    try:
        # 원본 파일 크기
        original_size = input_path.stat().st_size

        # 이미지 열기
        with Image.open(input_path) as img:
            # EXIF 회전 정보 적용
            try:
                from PIL import ImageOps
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass

            # 원본 크기
            orig_width, orig_height = img.size

            # 리사이징 비율 계산
            ratio = min(max_width / orig_width, max_height / orig_height, 1.0)

            if ratio < 1.0:
                new_width = int(orig_width * ratio)
                new_height = int(orig_height * ratio)
                # 고품질 리샘플링
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"  리사이징: {orig_width}x{orig_height} → {new_width}x{new_height}")
            else:
                print(f"  크기 유지: {orig_width}x{orig_height}")

            # RGB로 변환 (RGBA, P 모드 처리)
            if img.mode in ("RGBA", "LA", "P"):
                if img.mode == "P" and "transparency" in img.info:
                    img = img.convert("RGBA")
                if img.mode in ("RGBA", "LA"):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "RGBA":
                        background.paste(img, mask=img.split()[3])
                    else:
                        background.paste(img, mask=img.split()[1])
                    img = background
                else:
                    img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # 파일 형식별 저장
            ext = output_path.suffix.lower()
            if ext in [".jpg", ".jpeg"]:
                img.save(
                    output_path,
                    "JPEG",
                    quality=jpg_quality,
                    optimize=True,
                    progressive=True,
                )
            elif ext == ".png":
                img.save(output_path, "PNG", optimize=True)
            elif ext == ".webp":
                img.save(output_path, "WEBP", quality=jpg_quality, method=6)
            else:
                # 기본은 JPG로 저장
                new_output = output_path.with_suffix(".jpg")
                img.save(
                    new_output,
                    "JPEG",
                    quality=jpg_quality,
                    optimize=True,
                    progressive=True,
                )
                output_path = new_output

        # 최적화된 파일 크기
        optimized_size = output_path.stat().st_size

        return original_size, optimized_size

    except Exception as e:
        print(f"  ❌ 오류: {e}")
        return 0, 0


def optimize_folder(
    folder_path: Path,
    max_width: int = 1920,
    max_height: int = 1080,
    jpg_quality: int = 85,
    backup: bool = True,
):
    """
    폴더 내 모든 이미지 최적화

    Args:
        folder_path: 이미지 폴더 경로
        max_width: 최대 너비
        max_height: 최대 높이
        jpg_quality: JPG 품질
        backup: 백업 여부
    """
    if not folder_path.exists():
        print(f"❌ 폴더를 찾을 수 없습니다: {folder_path}")
        return

    # 백업 폴더 생성
    if backup:
        backup_folder = folder_path.parent / f"{folder_path.name}_backup"
        if not backup_folder.exists():
            backup_folder.mkdir()
            print(f"📁 백업 폴더 생성: {backup_folder}")
        else:
            print(f"📁 기존 백업 폴더 사용: {backup_folder}")

    # 이미지 파일 확장자
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

    # 이미지 파일 목록
    image_files = [
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"⚠️  이미지 파일이 없습니다: {folder_path}")
        return

    print(f"\n{'='*60}")
    print(f"📂 폴더: {folder_path.name}")
    print(f"🖼️  이미지 파일 개수: {len(image_files)}")
    print(f"{'='*60}\n")

    total_original = 0
    total_optimized = 0
    success_count = 0

    for i, img_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {img_file.name}")

        # 백업
        if backup:
            backup_path = backup_folder / img_file.name
            if not backup_path.exists():
                shutil.copy2(img_file, backup_path)
                print(f"  백업 완료: {backup_folder.name}/{img_file.name}")

        # 임시 파일로 최적화
        temp_output = folder_path / f"_temp_{img_file.name}"
        if temp_output.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            temp_output = temp_output.with_suffix(".jpg")

        original_size, optimized_size = optimize_image(
            img_file, temp_output, max_width, max_height, jpg_quality
        )

        if optimized_size > 0:
            # 원본 삭제 후 임시 파일을 원본 이름으로 변경
            final_output = img_file
            if temp_output.suffix != img_file.suffix:
                final_output = img_file.with_suffix(temp_output.suffix)

            img_file.unlink()
            temp_output.rename(final_output)

            total_original += original_size
            total_optimized += optimized_size
            success_count += 1

            reduction = (1 - optimized_size / original_size) * 100 if original_size > 0 else 0
            print(f"  ✅ {original_size / 1024 / 1024:.1f}MB → {optimized_size / 1024 / 1024:.1f}MB ({reduction:.1f}% 감소)\n")
        else:
            # 실패 시 임시 파일 삭제
            if temp_output.exists():
                temp_output.unlink()

    # 요약
    print(f"\n{'='*60}")
    print(f"✨ 최적화 완료!")
    print(f"{'='*60}")
    print(f"성공: {success_count}/{len(image_files)} 파일")
    print(f"원본 총 크기: {total_original / 1024 / 1024:.1f}MB")
    print(f"최적화 후 크기: {total_optimized / 1024 / 1024:.1f}MB")
    if total_original > 0:
        total_reduction = (1 - total_optimized / total_original) * 100
        saved_mb = (total_original - total_optimized) / 1024 / 1024
        print(f"절약된 용량: {saved_mb:.1f}MB ({total_reduction:.1f}% 감소)")
    print(f"{'='*60}\n")


def main():
    """메인 실행 함수"""
    # 프로젝트 루트 경로
    project_root = Path(__file__).parent

    print("\n" + "="*60)
    print("🖼️  이미지 최적화 스크립트")
    print("="*60)
    print("\n설정:")
    print("  - 최대 해상도: 1920x1080 (Full HD)")
    print("  - JPG 품질: 85% (웹/모바일 최적)")
    print("  - 원본 백업: 활성화")
    print("\n목표:")
    print("  - 인터넷 환경: 빠른 로딩 속도")
    print("  - 모바일 환경: 데이터 절약")
    print("  - 품질: 육안으로 구분 어려운 수준 유지")
    print("="*60 + "\n")

    # 최적화할 폴더들
    folders = [
        project_root / "acc_pic",
        project_root / "rockfall",
    ]

    for folder in folders:
        if folder.exists():
            optimize_folder(
                folder,
                max_width=1920,
                max_height=1080,
                jpg_quality=85,
                backup=True,
            )
        else:
            print(f"⚠️  폴더가 존재하지 않습니다: {folder}")

    print("\n" + "="*60)
    print("🎉 모든 작업 완료!")
    print("="*60)
    print("\n참고:")
    print("  - 원본 이미지는 '*_backup' 폴더에 보관됩니다")
    print("  - 문제 발생 시 백업 폴더에서 복원 가능합니다")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
