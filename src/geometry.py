"""
Camera geometry 관련 함수

역할:
1. Camera intrinsic matrix K 생성
2. 2D pixel + Depth → Camera coordinate 3D point 변환
"""

import numpy as np
import pyrealsense2 as rs


def make_camera_matrix(
    intrinsics: rs.intrinsics,
) -> np.ndarray:
    """
    RealSense intrinsic 객체로부터
    3×3 camera intrinsic matrix K를 생성한다.
    """

    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return K


def deproject_manual(
    u: int,
    v: int,
    z: float,
    intrinsics: rs.intrinsics,
) -> np.ndarray:
    """
    핀홀 카메라 모델을 이용하여

    pixel (u, v)
    + depth Z

    → Camera coordinate (X, Y, Z)

    로 직접 변환한다.
    """

    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.array(
        [x, y, z],
        dtype=np.float64,
    )


def deproject_sdk(
    u: int,
    v: int,
    z: float,
    intrinsics: rs.intrinsics,
) -> np.ndarray:
    """
    RealSense SDK를 사용하여

    pixel (u, v)
    + depth Z

    → Camera coordinate (X, Y, Z)

    로 변환한다.
    """

    point = rs.rs2_deproject_pixel_to_point(
        intrinsics,
        [u, v],
        z,
    )

    return np.array(
        point,
        dtype=np.float64,
    )