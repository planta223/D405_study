"""
D405 RGB-D Capture

역할:
1. D405 스트리밍 시작
2. RGB / 정렬된 Depth 프레임 취득
3. Camera intrinsic K 최초 1회 생성 및 저장
4. RGB / Depth 영상 시각화
5. 현재 RGB / Depth 프레임 저장
"""

import cv2

from src.camera import D405Camera
from src.geometry import make_camera_matrix
from src.visualization import show_frames
from src.data_save import save_frame, save_intrinsic


def main() -> None:
    # ---------------------------------------------------------
    # 1. D405 카메라 객체 생성 및 스트리밍 시작
    # ---------------------------------------------------------
    camera = D405Camera()
    camera.start()

    # 저장할 프레임 번호
    frame_id = 0

    # Camera intrinsic matrix
    # 최초 프레임에서 한 번만 생성한다.
    K = None

    try:
        while True:
            # -------------------------------------------------
            # 2. RGB / 정렬된 Depth 프레임 취득
            #
            # 반환값:
            # color_image : RGB NumPy 배열
            # depth_image : Color 좌표계에 정렬된 uint16 Depth 배열
            # depth_frame : 정렬된 RealSense Depth frame 객체
            # intrinsics  : 정렬된 Depth에 대응하는 camera intrinsic
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
            # 3. Camera intrinsic matrix K 최초 1회 생성 및 저장
            #
            # 같은 카메라 / 해상도 / 스트림 설정에서는
            # intrinsic이 고정이므로 매 프레임 계산하지 않는다.
            # -------------------------------------------------
            if K is None:
                K = make_camera_matrix(intrinsics)

                print()
                print("=== Camera Intrinsic K ===")
                print(K)
                print()

                save_intrinsic(K)

            # -------------------------------------------------
            # 4. 영상 중앙 픽셀 좌표 계산
            #
            # 현재는 화면 중앙 위치 표시용이다.
            # -------------------------------------------------
            height, width = depth_image.shape

            u = width // 2
            v = height // 2

            # -------------------------------------------------
            # 5. RGB / Depth 영상 시각화
            # -------------------------------------------------
            show_frames(
                color_image,
                depth_image,
                u,
                v,
            )

            # -------------------------------------------------
            # 6. 키보드 입력 처리
            #
            # q : 프로그램 종료
            # s : 현재 RGB / Depth 프레임 저장
            # -------------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                save_frame(
                    color_image,
                    depth_image,
                    frame_id,
                )

                frame_id += 1

    finally:
        # -----------------------------------------------------
        # 7. 프로그램 종료 처리
        # -----------------------------------------------------
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()