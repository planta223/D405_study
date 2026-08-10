"""
D405 Study Main

역할:
1. D405 스트리밍 시작
2. RGB / 정렬된 Depth 프레임 취득
3. Camera intrinsic K 생성
4. 중앙 픽셀의 Depth 및 3D 좌표 확인
5. RGB / Depth 영상 시각화
6. 현재 프레임 저장
"""

import cv2

from camera import D405Camera
from geometry import (
    make_camera_matrix,
    deproject_manual,
    deproject_sdk,
)
from visualization import show_frames
from data_save import save_frame


def main() -> None:
    # ---------------------------------------------------------
    # 1. D405 카메라 객체 생성 및 스트리밍 시작
    # ---------------------------------------------------------
    camera = D405Camera()
    camera.start()

    # 저장할 프레임 번호
    frame_id = 0

    # Camera intrinsic은 처음 한 번만 출력하기 위한 플래그
    intrinsic_printed = False

    try:
        while True:
            # -------------------------------------------------
            # 2. RGB / 정렬된 Depth 프레임 취득
            #
            # 반환값:
            # color_image : RGB NumPy 배열
            # depth_image : uint16 Depth NumPy 배열
            # depth_frame : RealSense Depth frame 객체
            # intrinsics  : 현재 정렬된 Depth의 camera intrinsic
            # -------------------------------------------------
            result = camera.get_aligned_frames()

            if result is None:
                continue

            (
                color_image,
                depth_image,
                depth_frame,
                intrinsics,
            ) = result

            # -------------------------------------------------
            # 3. Camera intrinsic matrix K 생성
            # -------------------------------------------------
            K = make_camera_matrix(intrinsics)

            # K는 매 프레임 출력할 필요가 없으므로
            # 프로그램 시작 후 한 번만 출력
            if not intrinsic_printed:
                print()
                print("=== Camera Intrinsic K ===")
                print(K)
                print()

                intrinsic_printed = True

            # -------------------------------------------------
            # 4. 현재 Depth 영상의 중앙 픽셀 좌표 계산
            # -------------------------------------------------
            height, width = depth_image.shape

            u = width // 2
            v = height // 2

            # -------------------------------------------------
            # 5. 중앙 픽셀의 raw Depth 값 취득
            #
            # depth_image에는 uint16 raw Depth 값이 저장됨
            # -------------------------------------------------
            raw_depth = depth_image[v, u]

            # -------------------------------------------------
            # 6. 중앙 픽셀의 실제 거리 [m] 취득
            #
            # RealSense SDK가 raw Depth와 depth scale을 이용해
            # meter 단위 거리로 변환
            # -------------------------------------------------
            z = depth_frame.get_distance(u, v)

            # -------------------------------------------------
            # 7. 2D Pixel → 3D Point 변환
            # -------------------------------------------------
            if z > 0:
                # 직접 핀홀 카메라 모델을 이용한 계산
                point_manual = deproject_manual(
                    u,
                    v,
                    z,
                    intrinsics,
                )

                # RealSense SDK를 이용한 계산
                point_sdk = deproject_sdk(
                    u,
                    v,
                    z,
                    intrinsics,
                )

                # -------------------------------------------------
                # 현재 측정값 출력
                # -------------------------------------------------
                print(
                    f"raw={raw_depth:5d} | "
                    f"Z={z:.3f} m | "
                    f"manual="
                    f"({point_manual[0]:.3f}, "
                    f"{point_manual[1]:.3f}, "
                    f"{point_manual[2]:.3f}) | "
                    f"sdk="
                    f"({point_sdk[0]:.3f}, "
                    f"{point_sdk[1]:.3f}, "
                    f"{point_sdk[2]:.3f})",
                    end="\r",
                )

            # -------------------------------------------------
            # 8. RGB / Depth 영상 시각화
            # -------------------------------------------------
            show_frames(
                color_image,
                depth_image,
                u,
                v,
            )

            # -------------------------------------------------
            # 9. 키보드 입력 처리
            #
            # q : 프로그램 종료
            # s : 현재 RGB / Depth / K 저장
            # -------------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                save_frame(
                    color_image,
                    depth_image,
                    K,
                    frame_id,
                )

                frame_id += 1

    finally:
        # -----------------------------------------------------
        # 10. 프로그램 종료 처리
        # -----------------------------------------------------
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()