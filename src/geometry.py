"""
Camera geometry 관련 함수

역할:
2D pixel + Depth
→ Camera coordinate의 3D point 변환
"""

import numpy as np
import pyrealsense2 as rs


def make_camera_matrix(
    intrinsics: rs.intrinsics,
) -> np.ndarray:
    """
    RealSense intrinsic 객체로부터
    3x3 camera intrinsic matrix K 생성
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


# 직접 구현한 역투영함수
def deproject_manual(
    u: int,
    v: int,
    z: float,
    intrinsics: rs.intrinsics,
) -> np.ndarray:
    """
    핀홀 카메라 모델을 직접 이용해

    pixel (u, v)
    + depth Z

    → camera coordinate (X, Y, Z)

    로 변환
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


# RealSense SDK의 역투영함수
def deproject_sdk(
    u: int,
    v: int,
    z: float,
    intrinsics: rs.intrinsics,
) -> np.ndarray:
    """
    RealSense SDK의 deprojection 함수 사용
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