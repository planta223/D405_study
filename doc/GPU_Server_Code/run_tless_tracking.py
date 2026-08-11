"""
T-LESS Object #6 RGB-D Sequence Tracking

처리 흐름:
1. 첫 프레임:
   RGB + Depth + Mask + K + CAD
   → FoundationPose register()

2. 이후 프레임:
   RGB + Depth + K
   → FoundationPose track_one()

3. 각 프레임:
   - 4x4 Pose 저장
   - 3D Bounding Box / XYZ Axis 시각화 저장

4. 최종:
   - 시각화 영상을 MP4로 저장
"""

from pathlib import Path
import argparse
import os
import time

import cv2
import imageio
import numpy as np
import trimesh

from estimater import *
from datareader import *


def load_rgb(
    path: Path,
) -> np.ndarray:
    """
    RGB 이미지를 읽는다.

    OpenCV:
        BGR

    FoundationPose:
        RGB
    """

    image_bgr = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise RuntimeError(
            f"Failed to load RGB: {path}"
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return image_rgb


def load_depth(
    path: Path,
    depth_scale: float,
) -> np.ndarray:
    """
    D405 raw uint16 depth를 읽어
    meter 단위 float32 depth로 변환한다.

    raw × depth_scale = meter
    """

    depth_raw = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if depth_raw is None:
        raise RuntimeError(
            f"Failed to load Depth: {path}"
        )

    depth = (
        depth_raw.astype(np.float32)
        * depth_scale
    )

    # invalid depth
    depth[depth < 0.001] = 0.0

    return depth


def load_mask(
    path: Path,
) -> np.ndarray:
    """
    Binary object mask를 읽는다.

    반환:
        bool mask
    """

    mask_raw = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask_raw is None:
        raise RuntimeError(
            f"Failed to load Mask: {path}"
        )

    return mask_raw > 0


def main() -> None:
    # ---------------------------------------------------------
    # 1. Argument
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--depth_scale",
        type=float,
        required=True,
        help="D405 raw depth 1 unit의 meter 값",
    )

    parser.add_argument(
        "--est_refine_iter",
        type=int,
        default=5,
        help="첫 프레임 registration refinement 횟수",
    )

    parser.add_argument(
        "--track_refine_iter",
        type=int,
        default=2,
        help="이후 tracking refinement 횟수",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="결과 MP4 FPS",
    )

    parser.add_argument(
        "--debug",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 2. Path
    # ---------------------------------------------------------
    project_root = (
        Path(__file__).resolve().parent
    )

    tless_dir = (
        project_root
        / "custom_data"
        / "tless06"
    )

    sequence_dir = (
        tless_dir
        / "sequence01"
    )

    rgb_dir = (
        sequence_dir
        / "rgb"
    )

    depth_dir = (
        sequence_dir
        / "depth"
    )

    K_path = (
        sequence_dir
        / "K.txt"
    )

    mask_path = (
        sequence_dir
        / "mask_000000.png"
    )

    mesh_path = (
        tless_dir
        / "obj_06.ply"
    )

    output_dir = (
        sequence_dir
        / "output"
    )

    pose_dir = (
        output_dir
        / "pose"
    )

    vis_dir = (
        output_dir
        / "vis"
    )

    debug_dir = (
        output_dir
        / "debug"
    )

    pose_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    vis_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 3. 필수 파일 확인
    # ---------------------------------------------------------
    for path in [
        K_path,
        mask_path,
        mesh_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}"
            )

    # ---------------------------------------------------------
    # 4. RGB frame 목록
    # ---------------------------------------------------------
    rgb_files = sorted(
        rgb_dir.glob("*.png")
    )

    if not rgb_files:
        raise RuntimeError(
            f"No RGB frames found: {rgb_dir}"
        )

    print()
    print(
        f"Number of frames: {len(rgb_files)}"
    )

    # ---------------------------------------------------------
    # 5. Camera intrinsic
    # ---------------------------------------------------------
    K = np.loadtxt(
        K_path,
        dtype=np.float64,
    ).reshape(3, 3)

    print()
    print("=== Camera Intrinsic K ===")
    print(K)

    # ---------------------------------------------------------
    # 6. T-LESS #6 mesh
    #
    # T-LESS:
    # mm
    #
    # FoundationPose:
    # meter
    # ---------------------------------------------------------
    mesh = trimesh.load(
        str(mesh_path)
    )

    mesh.vertices *= 1e-3

    # ---------------------------------------------------------
    # 7. Bounding box
    # ---------------------------------------------------------
    to_origin, extents = (
        trimesh.bounds.oriented_bounds(
            mesh
        )
    )

    bbox = np.stack(
        [
            -extents / 2,
            extents / 2,
        ],
        axis=0,
    ).reshape(2, 3)

    # ---------------------------------------------------------
    # 8. FoundationPose 초기화
    # ---------------------------------------------------------
    set_logging_format()
    set_seed(0)

    os.system(
        f"rm -rf {debug_dir}/*"
    )

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()

    glctx = dr.RasterizeCudaContext()

    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(debug_dir),
        debug=args.debug,
        glctx=glctx,
    )

    print()
    print(
        "FoundationPose initialized."
    )

    # ---------------------------------------------------------
    # 9. 첫 프레임 Mask
    # ---------------------------------------------------------
    first_mask = load_mask(
        mask_path
    )

    # ---------------------------------------------------------
    # 10. Tracking loop
    # ---------------------------------------------------------
    processing_times = []

    for index, rgb_path in enumerate(
        rgb_files
    ):
        frame_id = (
            rgb_path.stem
        )

        depth_path = (
            depth_dir
            / f"{frame_id}.png"
        )

        if not depth_path.exists():
            raise FileNotFoundError(
                f"Depth not found: {depth_path}"
            )

        # -----------------------------------------------------
        # RGB / Depth load
        # -----------------------------------------------------
        rgb = load_rgb(
            rgb_path
        )

        depth = load_depth(
            depth_path,
            args.depth_scale,
        )

        # -----------------------------------------------------
        # Shape 확인
        # -----------------------------------------------------
        if depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"RGB / Depth mismatch "
                f"at frame {frame_id}"
            )

        # -----------------------------------------------------
        # 처리시간 측정
        # -----------------------------------------------------
        start_time = (
            time.perf_counter()
        )

        # -----------------------------------------------------
        # 첫 프레임:
        # Registration
        # -----------------------------------------------------
        if index == 0:
            print()
            print(
                "========================================"
            )
            print(
                f"Frame {frame_id} : REGISTER"
            )
            print(
                "========================================"
            )

            if (
                first_mask.shape
                != rgb.shape[:2]
            ):
                raise ValueError(
                    "RGB / Mask resolution mismatch"
                )

            pose = estimator.register(
                K=K,
                rgb=rgb,
                depth=depth,
                ob_mask=first_mask,
                iteration=args.est_refine_iter,
            )

        # -----------------------------------------------------
        # 이후:
        # Tracking
        # -----------------------------------------------------
        else:
            print()
            print(
                f"Frame {frame_id} : TRACK"
            )

            pose = estimator.track_one(
                rgb=rgb,
                depth=depth,
                K=K,
                iteration=args.track_refine_iter,
            )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        processing_times.append(
            elapsed
        )

        # -----------------------------------------------------
        # Pose 저장
        # -----------------------------------------------------
        pose_path = (
            pose_dir
            / f"{frame_id}.txt"
        )

        np.savetxt(
            pose_path,
            pose.reshape(4, 4),
            fmt="%.8f",
        )

        # -----------------------------------------------------
        # Pose visualization
        # -----------------------------------------------------
        center_pose = (
            pose
            @ np.linalg.inv(to_origin)
        )

        vis = draw_posed_3d_box(
            K,
            img=rgb,
            ob_in_cam=center_pose,
            bbox=bbox,
        )

        vis = draw_xyz_axis(
            vis,
            ob_in_cam=center_pose,
            scale=0.05,
            K=K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )

        # -----------------------------------------------------
        # 처리시간 표시
        # -----------------------------------------------------
        fps_now = (
            1.0 / elapsed
            if elapsed > 0.0
            else 0.0
        )

        vis_bgr = cv2.cvtColor(
            vis,
            cv2.COLOR_RGB2BGR,
        )

        cv2.putText(
            vis_bgr,
            f"{frame_id} | {fps_now:.1f} FPS",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # 다시 RGB
        vis = cv2.cvtColor(
            vis_bgr,
            cv2.COLOR_BGR2RGB,
        )

        # -----------------------------------------------------
        # Visualization 저장
        # -----------------------------------------------------
        vis_path = (
            vis_dir
            / f"{frame_id}.png"
        )

        imageio.imwrite(
            vis_path,
            vis,
        )

        print(
            f"Time: {elapsed:.4f} s "
            f"({fps_now:.2f} FPS)"
        )

    # ---------------------------------------------------------
    # 11. 처리속도 통계
    #
    # 첫 프레임 registration은 tracking보다 훨씬 느릴 수 있으므로
    # tracking FPS는 2번째 프레임부터 따로 계산한다.
    # ---------------------------------------------------------
    if len(processing_times) > 1:
        tracking_times = (
            processing_times[1:]
        )

        mean_tracking_time = (
            float(
                np.mean(
                    tracking_times
                )
            )
        )

        mean_tracking_fps = (
            1.0 / mean_tracking_time
        )

        print()
        print(
            "=== Tracking Performance ==="
        )

        print(
            f"Mean tracking time : "
            f"{mean_tracking_time:.4f} s"
        )

        print(
            f"Mean tracking FPS  : "
            f"{mean_tracking_fps:.2f}"
        )

    # ---------------------------------------------------------
    # 12. MP4 생성
    # ---------------------------------------------------------
    video_path = (
        output_dir
        / "tracking_result.mp4"
    )

    vis_files = sorted(
        vis_dir.glob("*.png")
    )

    if vis_files:
        first_frame = cv2.imread(
            str(vis_files[0])
        )

        height, width = (
            first_frame.shape[:2]
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            args.fps,
            (width, height),
        )

        for vis_path in vis_files:
            frame = cv2.imread(
                str(vis_path)
            )

            writer.write(
                frame
            )

        writer.release()

        print()
        print(
            f"Video saved: {video_path}"
        )

    # ---------------------------------------------------------
    # 13. 완료
    # ---------------------------------------------------------
    print()
    print(
        "========================================"
    )
    print(
        "Tracking complete"
    )
    print(
        "========================================"
    )

    print(
        f"Pose directory: {pose_dir}"
    )

    print(
        f"Visualization directory: {vis_dir}"
    )


if __name__ == "__main__":
    main()