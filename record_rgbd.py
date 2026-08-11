"""
D405 RGB-D Sequence Recorder

목적:
FoundationPose Tracking 테스트를 위한
연속 RGB / aligned Depth frame 저장

조작:
- r : Recording 시작 / 종료
- q : 프로그램 종료

저장:
data/
├── rgb/
│   ├── seq01_000000.png
│   ├── seq01_000001.png
│   └── ...
├── depth/
│   ├── seq01_000000.png
│   ├── seq01_000001.png
│   └── ...
├── intrinsic/
│   └── K.txt
└── mask/
    └── seq01_000000.png
"""

from pathlib import Path
import time

import cv2
import numpy as np

from src.camera import D405Camera
from src.geometry import make_camera_matrix
from src.visualization import make_depth_colormap
from src.config import (
    RGB_OUTPUT_DIR,
    DEPTH_OUTPUT_DIR,
    INTRINSIC_OUTPUT_DIR,
    MASK_OUTPUT_DIR,
)


# ---------------------------------------------------------
# Recording 설정
# ---------------------------------------------------------

# 저장 프레임률
RECORD_FPS = 10

# 저장 주기 [s]
RECORD_PERIOD = 1.0 / RECORD_FPS

# 현재 녹화 세션 이름
SEQUENCE_NAME = "seq01"


def create_output_directories() -> None:
    """
    기존 data 하위 출력 디렉터리를 생성한다.
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

    Path(MASK_OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )


def save_intrinsic(
    K: np.ndarray,
) -> None:
    """
    Camera intrinsic matrix K를 저장한다.
    """

    K_path = (
        Path(INTRINSIC_OUTPUT_DIR)
        / "K.txt"
    )

    np.savetxt(
        K_path,
        K,
        fmt="%.8f",
    )

    print(
        f"K saved: {K_path}"
    )


def save_rgbd_frame(
    color_image: np.ndarray,
    depth_image: np.ndarray,
    frame_id: int,
) -> None:
    """
    현재 RGB / aligned Depth frame을
    기존 data/rgb, data/depth에 저장한다.
    """

    # ---------------------------------------------------------
    # RGB
    # ---------------------------------------------------------
    color_bgr = cv2.cvtColor(
        color_image,
        cv2.COLOR_RGB2BGR,
    )

    rgb_path = (
        Path(RGB_OUTPUT_DIR)
        / f"{SEQUENCE_NAME}_{frame_id:06d}.png"
    )

    # ---------------------------------------------------------
    # Depth
    # ---------------------------------------------------------
    depth_path = (
        Path(DEPTH_OUTPUT_DIR)
        / f"{SEQUENCE_NAME}_{frame_id:06d}.png"
    )

    rgb_success = cv2.imwrite(
        str(rgb_path),
        color_bgr,
    )

    depth_success = cv2.imwrite(
        str(depth_path),
        depth_image,
    )

    if not rgb_success:
        raise RuntimeError(
            f"Failed to save RGB: {rgb_path}"
        )

    if not depth_success:
        raise RuntimeError(
            f"Failed to save Depth: {depth_path}"
        )


def draw_status(
    color_image: np.ndarray,
    recording: bool,
    frame_id: int,
) -> np.ndarray:
    """
    RGB preview 위에 현재 녹화 상태를 표시한다.
    """

    preview = cv2.cvtColor(
        color_image,
        cv2.COLOR_RGB2BGR,
    )

    if recording:
        status = (
            f"REC | {SEQUENCE_NAME} | "
            f"Frame: {frame_id:06d}"
        )

    else:
        status = (
            "READY | Press R to record"
        )

    cv2.putText(
        preview,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return preview


def main() -> None:
    # ---------------------------------------------------------
    # 1. 출력 디렉터리 확인
    # ---------------------------------------------------------
    create_output_directories()

    # ---------------------------------------------------------
    # 2. D405 시작
    # ---------------------------------------------------------
    camera = D405Camera()
    camera.start()

    recording = False

    frame_id = 0

    # 최초 프레임에서 한 번만 생성
    K = None

    # 마지막 저장 시각
    last_record_time = 0.0

    print()
    print("=== D405 RGB-D Recorder ===")
    print(f"Sequence   : {SEQUENCE_NAME}")
    print(f"Record FPS : {RECORD_FPS}")
    print("r          : Start / Stop recording")
    print("q          : Quit")
    print()

    try:
        while True:
            # -------------------------------------------------
            # 3. RGB / aligned Depth 취득
            # -------------------------------------------------
            result = (
                camera.get_aligned_frames()
            )

            if result is None:
                continue

            (
                color_image,
                depth_image,
                depth_frame,
                intrinsics,
            ) = result

            # -------------------------------------------------
            # 4. K 최초 1회 생성 및 저장
            # -------------------------------------------------
            if K is None:
                K = make_camera_matrix(
                    intrinsics
                )

                print()
                print(
                    "=== Camera Intrinsic K ==="
                )
                print(K)
                print()

                save_intrinsic(K)

            # -------------------------------------------------
            # 5. 화면 표시
            # -------------------------------------------------
            color_preview = draw_status(
                color_image,
                recording,
                frame_id,
            )

            depth_preview = (
                make_depth_colormap(
                    depth_image
                )
            )

            cv2.imshow(
                "D405 RGB",
                color_preview,
            )

            cv2.imshow(
                "D405 Aligned Depth",
                depth_preview,
            )

            # -------------------------------------------------
            # 6. Recording 중 지정 FPS로 저장
            # -------------------------------------------------
            if recording:
                current_time = (
                    time.monotonic()
                )

                if (
                    current_time
                    - last_record_time
                    >= RECORD_PERIOD
                ):
                    save_rgbd_frame(
                        color_image,
                        depth_image,
                        frame_id,
                    )

                    frame_id += 1

                    last_record_time = (
                        current_time
                    )

            # -------------------------------------------------
            # 7. 키 입력
            # -------------------------------------------------
            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # q : 종료
            if key == ord("q"):
                break

            # r : 녹화 시작 / 종료
            if key == ord("r"):
                if not recording:
                    frame_id = 0

                    last_record_time = (
                        time.monotonic()
                        - RECORD_PERIOD
                    )

                    recording = True

                    print()
                    print(
                        "=== Recording started ==="
                    )

                else:
                    recording = False

                    print()
                    print(
                        "=== Recording stopped ==="
                    )
                    print(
                        f"Saved frames: {frame_id}"
                    )

    finally:
        # -----------------------------------------------------
        # 8. 종료
        # -----------------------------------------------------
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()