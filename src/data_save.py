"""
FoundationPose 입력 준비를 위한
RGB / Depth / Camera intrinsic 저장
"""

from pathlib import Path

import cv2
import numpy as np

from config import (
    RGB_OUTPUT_DIR,
    DEPTH_OUTPUT_DIR,
    INTRINSIC_OUTPUT_DIR,
)


def create_output_directories() -> None:
    """
    RGB / Depth / Intrinsic 데이터 저장 디렉터리를 생성한다.

    이미 디렉터리가 존재하는 경우에는 그대로 사용한다.
    """

    Path(RGB_OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(DEPTH_OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(INTRINSIC_OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )


def save_frame(
    color_image: np.ndarray,
    depth_image: np.ndarray,
    K: np.ndarray,
    frame_id: int,
) -> None:
    """
    현재 RGB-D 프레임과 카메라 내부행렬 K를 파일로 저장한다.

    저장 형식
    ----------
    RGB:
        PNG 이미지

    Depth:
        uint16 원본 Depth PNG

    Intrinsic:
        3x3 카메라 내부행렬 K를 TXT 파일로 저장
    """

    # ---------------------------------------------------------
    # 데이터 저장 디렉터리 확인 및 생성
    # ---------------------------------------------------------
    create_output_directories()

    # ---------------------------------------------------------
    # RGB 이미지 저장
    #
    # RealSense에서 받은 영상은 RGB 순서이고,
    # OpenCV의 imwrite는 BGR 순서를 기준으로 사용하므로 변환한다.
    # ---------------------------------------------------------
    color_bgr = cv2.cvtColor(
        color_image,
        cv2.COLOR_RGB2BGR,
    )

    rgb_path = (
        Path(RGB_OUTPUT_DIR)
        / f"{frame_id:06d}.png"
    )

    cv2.imwrite(
        str(rgb_path),
        color_bgr,
    )

    # ---------------------------------------------------------
    # Depth 이미지 저장
    #
    # 시각화용 colormap이 아니라
    # 실제 센서에서 얻은 uint16 Depth 원본을 저장한다.
    # ---------------------------------------------------------
    depth_path = (
        Path(DEPTH_OUTPUT_DIR)
        / f"{frame_id:06d}.png"
    )

    cv2.imwrite(
        str(depth_path),
        depth_image,
    )

    # ---------------------------------------------------------
    # Camera intrinsic K 저장
    #
    # 3x3 카메라 내부행렬을 txt 형식으로 저장한다.
    # ---------------------------------------------------------
    intrinsic_path = (
        Path(INTRINSIC_OUTPUT_DIR)
        / f"K_{frame_id:06d}.txt"
    )

    np.savetxt(
        intrinsic_path,
        K,
        fmt="%.8f",
    )

    # ---------------------------------------------------------
    # 저장 결과 출력
    # ---------------------------------------------------------
    print()
    print("Frame saved:")
    print(rgb_path)
    print(depth_path)
    print(intrinsic_path)