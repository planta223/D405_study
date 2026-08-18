"""
D405 30 FPS RGB-D Sequence Recorder

목적
----
FoundationPose tracking refinement 비교 실험을 위해
동일한 RGB-D sequence를 가능한 한 D405 원래 30 FPS로 저장한다.

핵심 원칙
---------
- D405 stream 설정은 src/config.py의 FPS=30을 그대로 사용한다.
- 기존처럼 10 FPS 시간 게이트를 두지 않는다.
- 매 camera frame을 수집한다.
- PNG 저장은 별도 worker thread에서 수행하여 acquisition loop의 I/O 대기를 줄인다.
- RGB와 aligned Depth는 동일한 frame_id로 저장한다.
- intrinsic K, depth scale, frame timestamp/frame number를 함께 저장한다.

조작
----
R : 녹화 시작
R : 녹화 종료 후 프로그램 종료
Q : 녹화하지 않고 종료 / 녹화 중이면 현재까지 저장 후 종료

예시
----
python record_rgbd.py --sequence refine01
python record_rgbd.py --sequence refine01 --duration 20
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse
import csv
import queue
import threading
import time

import cv2
import numpy as np

from src.camera import D405Camera
from src.geometry import make_camera_matrix
from src.config import (
    FPS,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    RGB_OUTPUT_DIR,
    DEPTH_OUTPUT_DIR,
    INTRINSIC_OUTPUT_DIR,
    MASK_OUTPUT_DIR,
)


# ---------------------------------------------------------
# Writer 종료 신호
# ---------------------------------------------------------
STOP_TOKEN = object()


def create_output_directories() -> None:
    """기존 data 하위 출력 디렉터리를 생성한다."""

    for directory in (
        RGB_OUTPUT_DIR,
        DEPTH_OUTPUT_DIR,
        INTRINSIC_OUTPUT_DIR,
        MASK_OUTPUT_DIR,
    ):
        Path(directory).mkdir(
            parents=True,
            exist_ok=True,
        )


def get_sequence_paths(sequence_name: str) -> Dict[str, Path]:
    """현재 sequence에 대응하는 metadata 파일 경로를 만든다."""

    intrinsic_dir = Path(INTRINSIC_OUTPUT_DIR)

    return {
        "K": intrinsic_dir / f"{sequence_name}_K.txt",
        "depth_scale": intrinsic_dir / f"{sequence_name}_depth_scale.txt",
        "frames_csv": intrinsic_dir / f"{sequence_name}_frames.csv",
        "summary": intrinsic_dir / f"{sequence_name}_summary.txt",
    }


def find_existing_sequence_files(sequence_name: str) -> List[Path]:
    """같은 sequence 이름으로 이미 존재하는 파일들을 찾는다."""

    existing: List[Path] = []

    existing.extend(
        sorted(
            Path(RGB_OUTPUT_DIR).glob(
                f"{sequence_name}_*.png"
            )
        )
    )

    existing.extend(
        sorted(
            Path(DEPTH_OUTPUT_DIR).glob(
                f"{sequence_name}_*.png"
            )
        )
    )

    existing.extend(
        sorted(
            Path(MASK_OUTPUT_DIR).glob(
                f"{sequence_name}_*.png"
            )
        )
    )

    for path in get_sequence_paths(sequence_name).values():
        if path.exists():
            existing.append(path)

    return existing


def prepare_sequence(
    sequence_name: str,
    overwrite: bool,
) -> None:
    """
    기존 동일 sequence 파일이 있으면 실수로 덮어쓰지 않도록 막는다.
    --overwrite가 지정된 경우에만 기존 sequence 파일을 삭제한다.
    """

    existing = find_existing_sequence_files(
        sequence_name
    )

    if not existing:
        return

    if not overwrite:
        sample = "\n".join(
            f"  - {path}"
            for path in existing[:10]
        )

        raise FileExistsError(
            "Same sequence already exists.\n"
            f"Sequence: {sequence_name}\n"
            f"Examples:\n{sample}\n"
            "Use another --sequence name or add --overwrite."
        )

    print()
    print(
        f"Removing existing sequence: {sequence_name}"
    )

    for path in existing:
        path.unlink()


def save_intrinsic_files(
    sequence_name: str,
    K: np.ndarray,
    depth_scale: float,
) -> None:
    """
    K와 depth scale을 저장한다.

    호환성을 위해 기존 data/intrinsic/K.txt도 함께 갱신하고,
    비교 실험 재현을 위해 sequence별 파일도 별도로 남긴다.
    """

    intrinsic_dir = Path(
        INTRINSIC_OUTPUT_DIR
    )

    paths = get_sequence_paths(
        sequence_name
    )

    # 기존 프로젝트 호환용 K
    np.savetxt(
        intrinsic_dir / "K.txt",
        K,
        fmt="%.8f",
    )

    # sequence 전용 K
    np.savetxt(
        paths["K"],
        K,
        fmt="%.8f",
    )

    # D405 raw depth -> meter 변환에 필요한 값
    with open(
        paths["depth_scale"],
        "w",
    ) as file:
        file.write(
            f"{depth_scale:.12f}\n"
        )

    print()
    print("Camera calibration saved:")
    print(f"  K           : {paths['K']}")
    print(
        f"  Depth scale : {paths['depth_scale']}"
    )


def writer_worker(
    save_queue: "queue.Queue[Any]",
    stats: Dict[str, Any],
    sequence_name: str,
    png_compression: int,
) -> None:
    """
    Acquisition loop와 분리된 PNG writer.

    queue item:
        (frame_id, RGB ndarray, aligned Depth ndarray)
    """

    rgb_dir = Path(RGB_OUTPUT_DIR)
    depth_dir = Path(DEPTH_OUTPUT_DIR)

    write_params = [
        cv2.IMWRITE_PNG_COMPRESSION,
        png_compression,
    ]

    while True:
        item = save_queue.get()

        try:
            if item is STOP_TOKEN:
                return

            frame_id, color_rgb, depth_raw = item

            filename = (
                f"{sequence_name}_{frame_id:06d}.png"
            )

            rgb_path = (
                rgb_dir / filename
            )

            depth_path = (
                depth_dir / filename
            )

            # OpenCV PNG 저장은 BGR 기준
            color_bgr = cv2.cvtColor(
                color_rgb,
                cv2.COLOR_RGB2BGR,
            )

            rgb_ok = cv2.imwrite(
                str(rgb_path),
                color_bgr,
                write_params,
            )

            depth_ok = cv2.imwrite(
                str(depth_path),
                depth_raw,
                write_params,
            )

            if not rgb_ok:
                raise RuntimeError(
                    f"Failed to save RGB: {rgb_path}"
                )

            if not depth_ok:
                raise RuntimeError(
                    f"Failed to save Depth: {depth_path}"
                )

            stats["saved_frames"] += 1

        except Exception as exc:
            stats["errors"].append(
                str(exc)
            )

        finally:
            save_queue.task_done()


def save_frames_csv(
    sequence_name: str,
    rows: List[Dict[str, Any]],
) -> None:
    """각 저장 frame의 host/device timestamp와 frame number를 저장한다."""

    if not rows:
        return

    csv_path = get_sequence_paths(
        sequence_name
    )["frames_csv"]

    fieldnames = [
        "frame_id",
        "host_elapsed_s",
        "device_timestamp_ms",
        "device_frame_number",
        "rgb_file",
        "depth_file",
    ]

    with open(
        csv_path,
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Frame metadata: {csv_path}"
    )


def calculate_capture_stats(
    rows: List[Dict[str, Any]],
) -> Dict[str, float]:
    """녹화 sequence의 실제 frame interval과 frame gap을 계산한다."""

    if len(rows) < 2:
        return {
            "host_fps": 0.0,
            "device_fps": 0.0,
            "missing_device_frames": 0.0,
            "max_device_gap": 0.0,
        }

    host_duration = (
        float(rows[-1]["host_elapsed_s"])
        - float(rows[0]["host_elapsed_s"])
    )

    device_duration_s = (
        float(rows[-1]["device_timestamp_ms"])
        - float(rows[0]["device_timestamp_ms"])
    ) / 1000.0

    # N개의 frame 사이에는 N-1개의 interval이 존재한다.
    interval_count = len(rows) - 1

    host_fps = (
        interval_count / host_duration
        if host_duration > 0.0
        else 0.0
    )

    device_fps = (
        interval_count / device_duration_s
        if device_duration_s > 0.0
        else 0.0
    )

    missing_frames = 0
    max_gap = 0

    previous_number: Optional[int] = None

    for row in rows:
        current_number = int(
            row["device_frame_number"]
        )

        if previous_number is not None:
            gap = (
                current_number
                - previous_number
            )

            if gap > max_gap:
                max_gap = gap

            if gap > 1:
                missing_frames += (
                    gap - 1
                )

        previous_number = current_number

    return {
        "host_fps": host_fps,
        "device_fps": device_fps,
        "missing_device_frames": float(
            missing_frames
        ),
        "max_device_gap": float(max_gap),
    }


def save_summary(
    sequence_name: str,
    captured_frames: int,
    saved_frames: int,
    rows: List[Dict[str, Any]],
    max_queue_size: int,
    png_compression: int,
) -> None:
    """녹화 품질 확인용 요약을 터미널과 txt에 남긴다."""

    capture_stats = calculate_capture_stats(
        rows
    )

    duration = (
        float(rows[-1]["host_elapsed_s"])
        if rows
        else 0.0
    )

    lines = [
        "=== D405 30 FPS Sequence Summary ===",
        f"Sequence              : {sequence_name}",
        f"Configured FPS        : {FPS}",
        f"Resolution            : {IMAGE_WIDTH}x{IMAGE_HEIGHT}",
        f"Captured frames       : {captured_frames}",
        f"Saved RGB-D pairs     : {saved_frames}",
        f"Host duration         : {duration:.3f} s",
        f"Measured host FPS     : {capture_stats['host_fps']:.2f}",
        f"Measured device FPS   : {capture_stats['device_fps']:.2f}",
        f"Missing device frames : {int(capture_stats['missing_device_frames'])}",
        f"Max device frame gap  : {int(capture_stats['max_device_gap'])}",
        f"Max writer queue      : {max_queue_size}",
        f"PNG compression       : {png_compression}",
    ]

    print()
    print("\n".join(lines))

    summary_path = get_sequence_paths(
        sequence_name
    )["summary"]

    with open(
        summary_path,
        "w",
    ) as file:
        file.write(
            "\n".join(lines) + "\n"
        )

    print()
    print(
        f"Summary saved: {summary_path}"
    )


def draw_preview(
    color_rgb: np.ndarray,
    recording: bool,
    captured_frames: int,
    saved_frames: int,
    queue_size: int,
    sequence_name: str,
) -> np.ndarray:
    """원본 저장 영상과 별개인 상태 확인용 RGB preview를 만든다."""

    preview = cv2.cvtColor(
        color_rgb,
        cv2.COLOR_RGB2BGR,
    )

    if recording:
        status = (
            f"REC  {sequence_name} | "
            f"captured={captured_frames} | "
            f"saved={saved_frames} | "
            f"queue={queue_size}"
        )
        color = (0, 0, 255)
    else:
        status = (
            f"READY  {sequence_name} | "
            "Press R to start"
        )
        color = (0, 255, 255)

    cv2.putText(
        preview,
        status,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )

    return preview


def main() -> None:
    # -----------------------------------------------------
    # Arguments
    # -----------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence",
        type=str,
        default="refine01",
        help="저장 sequence 이름",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help=(
            "R을 누른 뒤 자동 종료할 녹화 시간 [s]. "
            "0이면 두 번째 R 또는 Q까지 계속 녹화"
        ),
    )

    parser.add_argument(
        "--png_compression",
        type=int,
        default=1,
        choices=range(0, 10),
        metavar="0-9",
        help=(
            "PNG compression level. "
            "낮을수록 저장 속도가 빠름. 기본=1"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="동일 sequence의 기존 파일 삭제 후 다시 녹화",
    )

    args = parser.parse_args()

    sequence_name = args.sequence.strip()

    if not sequence_name:
        raise ValueError(
            "--sequence must not be empty."
        )

    # -----------------------------------------------------
    # 출력 준비
    # -----------------------------------------------------
    create_output_directories()

    prepare_sequence(
        sequence_name=sequence_name,
        overwrite=args.overwrite,
    )

    # -----------------------------------------------------
    # 비동기 PNG writer 시작
    # -----------------------------------------------------
    save_queue: "queue.Queue[Any]" = queue.Queue()

    writer_stats: Dict[str, Any] = {
        "saved_frames": 0,
        "errors": [],
    }

    writer_thread = threading.Thread(
        target=writer_worker,
        args=(
            save_queue,
            writer_stats,
            sequence_name,
            args.png_compression,
        ),
        daemon=True,
    )

    writer_thread.start()

    # -----------------------------------------------------
    # D405 시작
    # -----------------------------------------------------
    camera = D405Camera()
    camera_started = False

    recording = False
    recording_finished = False

    captured_frames = 0
    max_queue_size = 0

    metadata_rows: List[Dict[str, Any]] = []

    K: Optional[np.ndarray] = None
    record_start_host: Optional[float] = None

    print()
    print("=== D405 30 FPS RGB-D Recorder ===")
    print(f"Sequence       : {sequence_name}")
    print(f"Camera FPS     : {FPS}")
    print(
        f"Resolution     : {IMAGE_WIDTH}x{IMAGE_HEIGHT}"
    )
    print(
        f"PNG compression: {args.png_compression}"
    )

    if args.duration > 0.0:
        print(
            f"Duration       : {args.duration:.1f} s"
        )
    else:
        print("Duration       : manual")

    print("R              : Start / Stop")
    print("Q              : Quit")
    print()

    try:
        camera.start()
        camera_started = True

        while True:
            # -------------------------------------------------
            # D405 30 FPS RGB + aligned Depth 획득
            # -------------------------------------------------
            result = camera.get_aligned_frames()

            if result is None:
                continue

            (
                color_rgb,
                depth_raw,
                depth_frame,
                intrinsics,
            ) = result

            # -------------------------------------------------
            # 최초 frame에서 K 저장
            # -------------------------------------------------
            if K is None:
                K = make_camera_matrix(
                    intrinsics
                )

                print()
                print("=== Camera Intrinsic K ===")
                print(K)

                save_intrinsic_files(
                    sequence_name=sequence_name,
                    K=K,
                    depth_scale=float(
                        camera.depth_scale
                    ),
                )

            # -------------------------------------------------
            # Recording 중에는 '받은 camera frame마다' 저장 queue에 넣는다.
            # 10 FPS용 시간 게이트는 사용하지 않는다.
            # -------------------------------------------------
            if recording:
                capture_host_time = (
                    time.perf_counter()
                )

                if record_start_host is None:
                    record_start_host = (
                        capture_host_time
                    )

                host_elapsed_s = (
                    capture_host_time
                    - record_start_host
                )

                device_timestamp_ms = float(
                    depth_frame.get_timestamp()
                )

                device_frame_number = int(
                    depth_frame.get_frame_number()
                )

                filename = (
                    f"{sequence_name}_"
                    f"{captured_frames:06d}.png"
                )

                # RealSense frame buffer와 분리하기 위해 copy해서 queue에 넣는다.
                save_queue.put(
                    (
                        captured_frames,
                        color_rgb.copy(),
                        depth_raw.copy(),
                    )
                )

                metadata_rows.append(
                    {
                        "frame_id": captured_frames,
                        "host_elapsed_s": (
                            f"{host_elapsed_s:.9f}"
                        ),
                        "device_timestamp_ms": (
                            f"{device_timestamp_ms:.6f}"
                        ),
                        "device_frame_number": (
                            device_frame_number
                        ),
                        "rgb_file": filename,
                        "depth_file": filename,
                    }
                )

                captured_frames += 1

                current_queue_size = (
                    save_queue.qsize()
                )

                max_queue_size = max(
                    max_queue_size,
                    current_queue_size,
                )

                # 2초마다 저장 상태를 짧게 출력
                if captured_frames % max(1, FPS * 2) == 0:
                    print(
                        f"[REC] captured={captured_frames} | "
                        f"saved={writer_stats['saved_frames']} | "
                        f"queue={current_queue_size}"
                    )

                # 지정 시간 녹화 시 자동 종료
                if (
                    args.duration > 0.0
                    and host_elapsed_s
                    >= args.duration
                ):
                    print()
                    print(
                        "Requested recording duration reached."
                    )
                    recording = False
                    recording_finished = True

            # -------------------------------------------------
            # 상태 확인용 RGB preview
            # -------------------------------------------------
            preview = draw_preview(
                color_rgb=color_rgb,
                recording=recording,
                captured_frames=captured_frames,
                saved_frames=int(
                    writer_stats["saved_frames"]
                ),
                queue_size=save_queue.qsize(),
                sequence_name=sequence_name,
            )

            cv2.imshow(
                "D405 30 FPS Recorder",
                preview,
            )

            key = cv2.waitKey(1) & 0xFF

            # -------------------------------------------------
            # R : 첫 번째는 시작, 두 번째는 종료
            # -------------------------------------------------
            if key == ord("r"):
                if not recording and captured_frames == 0:
                    recording = True
                    record_start_host = None

                    print()
                    print(
                        "=== Recording started ==="
                    )

                elif recording:
                    recording = False
                    recording_finished = True

                    print()
                    print(
                        "=== Recording stopped ==="
                    )

            # -------------------------------------------------
            # Q : 현재까지 저장하고 종료
            # -------------------------------------------------
            if key == ord("q"):
                if recording:
                    recording = False
                    recording_finished = True

                    print()
                    print(
                        "=== Recording stopped by Q ==="
                    )

                break

            # 녹화 종료 후에는 acquisition loop도 종료한다.
            if recording_finished:
                break

    finally:
        # -----------------------------------------------------
        # 카메라 종료
        # -----------------------------------------------------
        if camera_started:
            camera.stop()

        cv2.destroyAllWindows()

        # -----------------------------------------------------
        # Writer queue에 들어간 모든 RGB-D가 disk에 저장될 때까지 대기
        # -----------------------------------------------------
        print()
        print(
            "Flushing RGB-D writer queue..."
        )

        save_queue.put(
            STOP_TOKEN
        )

        save_queue.join()
        writer_thread.join()

        saved_frames = int(
            writer_stats["saved_frames"]
        )

        if writer_stats["errors"]:
            print()
            print("Writer errors:")

            for error in writer_stats["errors"]:
                print(f"  - {error}")

        # -----------------------------------------------------
        # Metadata / summary 저장
        # -----------------------------------------------------
        if metadata_rows:
            save_frames_csv(
                sequence_name,
                metadata_rows,
            )

            save_summary(
                sequence_name=sequence_name,
                captured_frames=captured_frames,
                saved_frames=saved_frames,
                rows=metadata_rows,
                max_queue_size=max_queue_size,
                png_compression=args.png_compression,
            )

        if captured_frames != saved_frames:
            raise RuntimeError(
                "Captured/Saved frame count mismatch: "
                f"captured={captured_frames}, "
                f"saved={saved_frames}"
            )


if __name__ == "__main__":
    main()
