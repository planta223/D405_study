"""
FoundationPose용 Binary Mask 생성

사용 방법:
- 마우스 좌클릭 : polygon 점 추가
- z             : 마지막 점 되돌리기
- r             : 전체 점 초기화
- 마우스 휠     : 확대 / 축소
- s             : mask 저장
- q             : 종료
"""

from pathlib import Path

import cv2
import numpy as np

from src.config import (
    RGB_OUTPUT_DIR,
    MASK_OUTPUT_DIR,
)


# ---------------------------------------------------------
# 생성할 Mask의 RGB frame 번호
# ---------------------------------------------------------
FRAME_ID = 0


# ---------------------------------------------------------
# Polygon 꼭짓점
# 원본 이미지 좌표계 기준으로 저장한다.
# ---------------------------------------------------------
points: list[tuple[int, int]] = []


# ---------------------------------------------------------
# 확대/축소 상태
# ---------------------------------------------------------
zoom_scale = 1.0

MIN_ZOOM = 0.5
MAX_ZOOM = 5.0
ZOOM_STEP = 1.2


def mouse_callback(
    event: int,
    x: int,
    y: int,
    flags: int,
    param,
) -> None:
    """
    마우스 입력 처리

    좌클릭:
        현재 화면 좌표를 원본 이미지 좌표로 변환하여
        polygon 꼭짓점으로 저장

    마우스 휠:
        확대 / 축소
    """

    global zoom_scale

    # ---------------------------------------------------------
    # 좌클릭
    # ---------------------------------------------------------
    if event == cv2.EVENT_LBUTTONDOWN:
        # 확대된 화면 좌표 → 원본 이미지 좌표
        original_x = int(x / zoom_scale)
        original_y = int(y / zoom_scale)

        points.append(
            (original_x, original_y)
        )

        print(
            f"Point added: "
            f"({original_x}, {original_y})"
        )

    # ---------------------------------------------------------
    # 마우스 휠
    # ---------------------------------------------------------
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            # 확대
            zoom_scale *= ZOOM_STEP

        else:
            # 축소
            zoom_scale /= ZOOM_STEP

        zoom_scale = max(
            MIN_ZOOM,
            min(MAX_ZOOM, zoom_scale),
        )

        print(
            f"Zoom: {zoom_scale:.2f}x"
        )


def create_mask(
    image_shape: tuple[int, ...],
) -> np.ndarray:
    """
    선택된 polygon을 이용하여 binary mask 생성

    배경 = 0
    물체 = 255
    """

    height, width = image_shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    if len(points) < 3:
        return mask

    polygon = np.array(
        points,
        dtype=np.int32,
    )

    cv2.fillPoly(
        mask,
        [polygon],
        255,
    )

    return mask


def draw_preview(
    image: np.ndarray,
) -> np.ndarray:
    """
    원본 영상 위에 현재 polygon을 표시하고
    확대/축소된 preview를 반환한다.
    """

    preview = image.copy()

    # ---------------------------------------------------------
    # 선택 점 표시
    # ---------------------------------------------------------
    for point in points:
        cv2.circle(
            preview,
            point,
            4,
            (0, 0, 255),
            -1,
        )

    # ---------------------------------------------------------
    # Polygon 선 표시
    # ---------------------------------------------------------
    if len(points) >= 2:
        polygon = np.array(
            points,
            dtype=np.int32,
        )

        cv2.polylines(
            preview,
            [polygon],
            False,
            (0, 255, 0),
            2,
        )

    # ---------------------------------------------------------
    # 3점 이상이면 마지막 점과 첫 점 연결
    # ---------------------------------------------------------
    if len(points) >= 3:
        cv2.line(
            preview,
            points[-1],
            points[0],
            (0, 255, 0),
            2,
        )

    # ---------------------------------------------------------
    # 확대 / 축소
    # ---------------------------------------------------------
    preview = cv2.resize(
        preview,
        None,
        fx=zoom_scale,
        fy=zoom_scale,
        interpolation=cv2.INTER_NEAREST,
    )

    return preview


def main() -> None:
    # ---------------------------------------------------------
    # 1. RGB 이미지 경로
    # ---------------------------------------------------------
    rgb_path = (
        Path(RGB_OUTPUT_DIR)
        / f"{FRAME_ID:06d}.png"
    )

    if not rgb_path.exists():
        raise FileNotFoundError(
            f"RGB image not found: {rgb_path}"
        )

    # ---------------------------------------------------------
    # 2. RGB 이미지 로드
    # ---------------------------------------------------------
    image = cv2.imread(
        str(rgb_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Failed to load image: {rgb_path}"
        )

    # ---------------------------------------------------------
    # 3. Mask 출력 디렉터리 생성
    # ---------------------------------------------------------
    Path(MASK_OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_path = (
        Path(MASK_OUTPUT_DIR)
        / f"{FRAME_ID:06d}.png"
    )

    # ---------------------------------------------------------
    # 4. Window 설정
    # ---------------------------------------------------------
    window_name = "Generate Mask"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback,
    )

    print()
    print("=== Mask Generator ===")
    print("Left click  : Add point")
    print("Mouse wheel : Zoom")
    print("z           : Undo last point")
    print("r           : Reset all points")
    print("s           : Save mask")
    print("q           : Quit")
    print()

    while True:
        # -----------------------------------------------------
        # Preview 생성
        # -----------------------------------------------------
        preview = draw_preview(
            image
        )

        cv2.imshow(
            window_name,
            preview,
        )

        # -----------------------------------------------------
        # Binary Mask 표시
        # -----------------------------------------------------
        mask = create_mask(
            image.shape
        )

        cv2.imshow(
            "Binary Mask",
            mask,
        )

        key = cv2.waitKey(20) & 0xFF

        # -----------------------------------------------------
        # 종료
        # -----------------------------------------------------
        if key == ord("q"):
            break

        # -----------------------------------------------------
        # 마지막 점 되돌리기
        # -----------------------------------------------------
        if key == ord("z"):
            if points:
                removed = points.pop()

                print(
                    f"Undo: {removed}"
                )

        # -----------------------------------------------------
        # 전체 초기화
        # -----------------------------------------------------
        if key == ord("r"):
            points.clear()

            print("Points reset.")

        # -----------------------------------------------------
        # Mask 저장
        # -----------------------------------------------------
        if key == ord("s"):
            if len(points) < 3:
                print(
                    "Mask requires at least "
                    "3 polygon points."
                )
                continue

            mask = create_mask(
                image.shape
            )

            cv2.imwrite(
                str(mask_path),
                mask,
            )

            print()
            print("Mask saved:")
            print(mask_path)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()