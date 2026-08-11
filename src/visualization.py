"""
RGB / Depth 영상 시각화
"""

import cv2
import numpy as np

from .config import DEPTH_COLORMAP_ALPHA


def make_depth_colormap(
    depth_image: np.ndarray,
) -> np.ndarray:
    """
    uint16 Depth 이미지를 사람이 보기 쉬운
    컬러 이미지로 변환한다.

    주의:
    반환 영상은 시각화용이며
    실제 Depth 데이터가 아니다.
    """

    depth_8bit = cv2.convertScaleAbs(
        depth_image,
        alpha=DEPTH_COLORMAP_ALPHA,
    )

    depth_colormap = cv2.applyColorMap(
        depth_8bit,
        cv2.COLORMAP_JET,
    )

    return depth_colormap


def show_frames(
    color_image: np.ndarray,
    depth_image: np.ndarray,
    u: int,
    v: int,
) -> None:
    """
    RGB / Depth 영상을 화면에 표시한다.

    (u, v):
        현재 확인할 pixel 위치
    """

    # ---------------------------------------------------------
    # RealSense RGB → OpenCV BGR
    # ---------------------------------------------------------
    color_bgr = cv2.cvtColor(
        color_image,
        cv2.COLOR_RGB2BGR,
    )

    # ---------------------------------------------------------
    # Depth 시각화 영상 생성
    # ---------------------------------------------------------
    depth_colormap = make_depth_colormap(
        depth_image
    )

    # ---------------------------------------------------------
    # 현재 확인 pixel 표시
    # ---------------------------------------------------------
    cv2.circle(
        color_bgr,
        (u, v),
        5,
        (0, 0, 255),
        -1,
    )

    cv2.circle(
        depth_colormap,
        (u, v),
        5,
        (255, 255, 255),
        -1,
    )

    # ---------------------------------------------------------
    # 화면 출력
    # ---------------------------------------------------------
    cv2.imshow(
        "D405 Color",
        color_bgr,
    )

    cv2.imshow(
        "D405 Aligned Depth",
        depth_colormap,
    )