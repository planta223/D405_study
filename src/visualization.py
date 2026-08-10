"""
RGB / Depth 영상 시각화
"""

import cv2
import numpy as np

from config import DEPTH_COLORMAP_ALPHA


def make_depth_colormap(
    depth_image: np.ndarray,
) -> np.ndarray:
    """
    uint16 depth image를
    사람이 보기 쉬운 컬러 이미지로 변환

    주의:
    이 영상은 시각화용이며,
    실제 depth 데이터가 아님.
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
    Color / Depth 화면 표시
    """

    # ---------------------------------------------------------
    # RealSense RGB → OpenCV BGR
    # ---------------------------------------------------------
    color_bgr = cv2.cvtColor(
        color_image,
        cv2.COLOR_RGB2BGR,
    )

    depth_colormap = make_depth_colormap(
        depth_image
    )

    # ---------------------------------------------------------
    # 현재 측정 pixel 표시
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

    cv2.imshow(
        "D405 Color",
        color_bgr,
    )

    cv2.imshow(
        "D405 Aligned Depth",
        depth_colormap,
    )