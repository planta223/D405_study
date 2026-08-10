import cv2
import numpy as np
import pyrealsense2 as rs


def main() -> None:
    # ---------------------------------------------------------
    # 1. RealSense 파이프라인 및 설정 생성
    # ---------------------------------------------------------
    pipeline = rs.pipeline()
    config = rs.config()

    # Depth / Color 스트림 활성화
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)

    # ---------------------------------------------------------
    # 2. 카메라 스트리밍 시작
    # ---------------------------------------------------------
    profile = pipeline.start(config)

    # ---------------------------------------------------------
    # 3. Depth 카메라 내부 파라미터 확인
    # ---------------------------------------------------------
    depth_stream = profile.get_stream(rs.stream.depth)
    depth_profile = depth_stream.as_video_stream_profile()
    intr = depth_profile.get_intrinsics()

    fx = intr.fx
    fy = intr.fy
    cx = intr.ppx
    cy = intr.ppy

    print("=== Depth Camera Intrinsic ===")
    print(f"fx = {fx}")
    print(f"fy = {fy}")
    print(f"cx = {cx}")
    print(f"cy = {cy}")

    # 카메라 내부행렬 K
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    print("\nK =")
    print(K)

    # ---------------------------------------------------------
    # 4. Depth scale 확인
    # ---------------------------------------------------------
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    print(f"\nDepth scale = {depth_scale} m/unit")
    print()

    try:
        while True:
            # -------------------------------------------------
            # 5. 새로운 프레임 묶음 수신
            # -------------------------------------------------
            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # -------------------------------------------------
            # 6. RealSense frame -> NumPy 배열
            # -------------------------------------------------
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # -------------------------------------------------
            # 7. 중앙 픽셀 위치 계산
            # -------------------------------------------------
            height, width = depth_image.shape

            u = width // 2
            v = height // 2

            # -------------------------------------------------
            # 8. 중앙 픽셀의 raw depth 값
            # -------------------------------------------------
            raw_depth = depth_image[v, u]

            # -------------------------------------------------
            # 9. 중앙 픽셀의 실제 거리 [m]
            # -------------------------------------------------
            z = depth_frame.get_distance(u, v)

            # -------------------------------------------------
            # 10. 직접 수식으로 3D 좌표 계산
            # -------------------------------------------------
            if z > 0:
                # 핀홀 카메라 모델 기반 deprojection
                x_manual = (u - cx) * z / fx
                y_manual = (v - cy) * z / fy
                z_manual = z

                # -------------------------------------------------
                # 11. RealSense SDK 함수로 3D 좌표 계산
                # -------------------------------------------------
                point_sdk = rs.rs2_deproject_pixel_to_point(
                    intr,
                    [u, v],
                    z,
                )

                x_sdk = point_sdk[0]
                y_sdk = point_sdk[1]
                z_sdk = point_sdk[2]

                # -------------------------------------------------
                # 12. 결과 출력
                # -------------------------------------------------
                print(
                    f"raw={raw_depth:5d} | "
                    f"distance={z:.3f} m | "
                    f"manual=({x_manual:.3f}, "
                    f"{y_manual:.3f}, "
                    f"{z_manual:.3f}) | "
                    f"sdk=({x_sdk:.3f}, "
                    f"{y_sdk:.3f}, "
                    f"{z_sdk:.3f})",
                    end="\r",
                )

            else:
                print(
                    f"raw={raw_depth:5d} | invalid depth",
                    end="\r",
                )

            # -------------------------------------------------
            # 13. Depth 영상 시각화
            # -------------------------------------------------
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(
                    depth_image,
                    alpha=0.03,
                ),
                cv2.COLORMAP_JET,
            )

            # -------------------------------------------------
            # 14. 중앙 픽셀 표시
            # -------------------------------------------------
            cv2.circle(
                color_image,
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

            # -------------------------------------------------
            # 15. 화면 출력
            # -------------------------------------------------
            cv2.imshow("D405 Color", color_image)
            cv2.imshow("D405 Depth", depth_colormap)

            # q를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # -----------------------------------------------------
        # 16. 카메라 및 OpenCV 창 종료
        # -----------------------------------------------------
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()