"""
Intel RealSense D405 인터페이스

역할:
1. D405 스트리밍 시작/종료
2. RGB / Depth frame 취득
3. Depth를 Color 좌표계에 정렬
4. Camera intrinsic 취득
5. Depth scale 취득
"""

import numpy as np
import pyrealsense2 as rs

from .config import (
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    FPS,
)


class D405Camera:
    def __init__(self) -> None:
        # -----------------------------------------------------
        # RealSense pipeline
        # -----------------------------------------------------
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # -----------------------------------------------------
        # Depth stream 설정
        #
        # Z16:
        # 픽셀 하나가 uint16 depth raw value를 가짐
        # -----------------------------------------------------
        self.config.enable_stream(
            rs.stream.depth,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            rs.format.z16,
            FPS,
        )

        # -----------------------------------------------------
        # Color stream 설정
        # -----------------------------------------------------
        self.config.enable_stream(
            rs.stream.color,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            rs.format.rgb8,
            FPS,
        )

        # -----------------------------------------------------
        # Depth → Color alignment
        #
        # RGB와 Depth가 서로 다른 영상 좌표계를 가지므로
        # Depth를 Color pixel coordinate에 맞춘다.
        # -----------------------------------------------------
        self.align = rs.align(
            rs.stream.color
        )

        self.profile = None
        self.depth_scale = None

    def start(self) -> None:
        """
        D405 스트리밍 시작
        """

        self.profile = self.pipeline.start(
            self.config
        )

        # -----------------------------------------------------
        # Depth scale 취득
        #
        # raw depth × depth_scale = meter
        # -----------------------------------------------------
        depth_sensor = (
            self.profile
            .get_device()
            .first_depth_sensor()
        )

        self.depth_scale = (
            depth_sensor.get_depth_scale()
        )

        print(
            f"Depth scale = "
            f"{self.depth_scale} m/unit"
        )

    def stop(self) -> None:
        """
        D405 스트리밍 종료
        """

        self.pipeline.stop()

    def get_aligned_frames(self):
        """
        RGB와 Depth 프레임을 취득하고,
        Depth를 Color 좌표계에 정렬하여 반환한다.

        Returns
        -------
        color_image : np.ndarray
            RGB 이미지

        depth_image : np.ndarray
            Color 좌표계에 정렬된 uint16 Depth 이미지

        depth_frame : rs.depth_frame
            Color 좌표계에 정렬된 Depth frame

        intrinsics : rs.intrinsics
            정렬된 Depth 이미지에 대응하는
            camera intrinsic
        """

        # -----------------------------------------------------
        # 새로운 frameset 대기
        # -----------------------------------------------------
        frames = self.pipeline.wait_for_frames()

        # -----------------------------------------------------
        # Depth → Color alignment
        # -----------------------------------------------------
        aligned_frames = self.align.process(
            frames
        )

        aligned_depth_frame = (
            aligned_frames.get_depth_frame()
        )

        color_frame = (
            aligned_frames.get_color_frame()
        )

        if (
            not aligned_depth_frame
            or not color_frame
        ):
            return None

        # -----------------------------------------------------
        # RealSense frame → NumPy array
        # -----------------------------------------------------
        depth_image = np.asanyarray(
            aligned_depth_frame.get_data()
        )

        color_image = np.asanyarray(
            color_frame.get_data()
        )

        # -----------------------------------------------------
        # Alignment 이후 영상에 대응하는 intrinsic
        # -----------------------------------------------------
        intrinsics = (
            aligned_depth_frame
            .profile
            .as_video_stream_profile()
            .get_intrinsics()
        )

        return (
            color_image,
            depth_image,
            aligned_depth_frame,
            intrinsics,
        )