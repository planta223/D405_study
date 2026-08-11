"""
FoundationPose 입력 준비를 위한
RGB / Depth / Camera intrinsic 저장
"""

from pathlib import Path

import cv2
import numpy as np

from .config import (
    RGB_OUTPUT_DIR,
    DEPTH_OUTPUT_DIR,
    INTRINSIC_OUTPUT_DIR,
)


def create_output_directories() -> None:
    """
    RGB / Depth / Intrinsic 데이터 저장
    디렉터리를 생성한다.

    이미 존재하면 그대로 사용한다.
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
    frame_id: int,
) -> None:
    """
    현재 RGB / Depth 프레임을 저장한다.

    RGB:
        PNG

    Depth:
        uint16 PNG
    """

    create_output_directories()

    # ---------------------------------------------------------
    # RGB 저장
    #
    # RealSense:
    # RGB
    #
    # OpenCV imwrite:
    # BGR
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
    # Depth 저장
    #
    # 시각화용 colormap이 아니라
    # 실제 uint16 Depth 값을 저장한다.
    # ---------------------------------------------------------
    depth_path = (
        Path(DEPTH_OUTPUT_DIR)
        / f"{frame_id:06d}.png"
    )

    cv2.imwrite(
        str(depth_path),
        depth_image,
    )

    print()
    print("Frame saved:")
    print(rgb_path)
    print(depth_path)


def save_intrinsic(
    K: np.ndarray,
) -> None:
    """
    Camera intrinsic matrix K를 저장한다.

    같은 카메라 / 해상도 / 스트림 설정에서는
    K가 고정이므로 프로그램 시작 시 1회 저장한다.
    """

    create_output_directories()

    intrinsic_path = (
        Path(INTRINSIC_OUTPUT_DIR)
        / "K.txt"
    )

    np.savetxt(
        intrinsic_path,
        K,
        fmt="%.8f",
    )

    print()
    print("Intrinsic saved:")
    print(intrinsic_path)