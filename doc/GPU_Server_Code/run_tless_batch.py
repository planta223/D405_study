"""
T-LESS Object #6 Multi-frame Independent Registration Test

목적:
- additional_sample에 저장된 여러 RGB-D 프레임에 대해
  FoundationPose register()를 각각 독립적으로 수행한다.

입력:
custom_data/tless06/
├── obj_06.ply
└── additional_sample/
    ├── K.txt
    ├── rgb_000000.png
    ├── depth_000000.png
    ├── mask_000000.png
    └── ...

출력:
custom_data/tless06/additional_sample/output/
├── pose_000000.txt
├── vis_000000.png
├── pose_000001.txt
├── vis_000001.png
└── ...

주의:
- Tracking 테스트가 아니다.
- 각 프레임마다 FoundationPose register()를 새로 수행한다.
"""

from pathlib import Path
import argparse
import os

import cv2
import imageio
import numpy as np
import trimesh

from estimater import *
from datareader import *


def load_sample(
    sample_dir: Path,
    frame_id: int,
    depth_scale: float,
):
    """
    특정 frame의 RGB / Depth / Mask를 읽는다.

    Parameters
    ----------
    sample_dir:
        additional_sample 디렉터리

    frame_id:
        읽을 프레임 번호

    depth_scale:
        D405 raw depth 1 unit에 해당하는 거리 [m]

    Returns
    -------
    rgb:
        RGB uint8 image

    depth:
        meter 단위 float32 depth image

    mask:
        binary boolean mask
    """

    # ---------------------------------------------------------
    # 파일 경로
    # ---------------------------------------------------------
    rgb_path = (
        sample_dir
        / f"rgb_{frame_id:06d}.png"
    )

    depth_path = (
        sample_dir
        / f"depth_{frame_id:06d}.png"
    )

    mask_path = (
        sample_dir
        / f"mask_{frame_id:06d}.png"
    )

    # ---------------------------------------------------------
    # 파일 존재 확인
    # ---------------------------------------------------------
    for path in [
        rgb_path,
        depth_path,
        mask_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}"
            )

    # ---------------------------------------------------------
    # RGB
    #
    # OpenCV는 BGR로 읽으므로
    # FoundationPose 입력을 위해 RGB로 변환
    # ---------------------------------------------------------
    rgb_bgr = cv2.imread(
        str(rgb_path),
        cv2.IMREAD_COLOR,
    )

    if rgb_bgr is None:
        raise RuntimeError(
            f"Failed to load RGB: {rgb_path}"
        )

    rgb = cv2.cvtColor(
        rgb_bgr,
        cv2.COLOR_BGR2RGB,
    )

    # ---------------------------------------------------------
    # Depth
    #
    # D405 raw uint16
    #       ↓
    # raw × depth_scale
    #       ↓
    # meter
    # ---------------------------------------------------------
    depth_raw = cv2.imread(
        str(depth_path),
        cv2.IMREAD_UNCHANGED,
    )

    if depth_raw is None:
        raise RuntimeError(
            f"Failed to load Depth: {depth_path}"
        )

    depth = (
        depth_raw.astype(np.float32)
        * depth_scale
    )

    # invalid depth는 0으로 유지
    depth[depth < 0.001] = 0.0

    # ---------------------------------------------------------
    # Binary Mask
    # ---------------------------------------------------------
    mask_raw = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask_raw is None:
        raise RuntimeError(
            f"Failed to load Mask: {mask_path}"
        )

    mask = mask_raw > 0

    # ---------------------------------------------------------
    # 입력 해상도 정합성 확인
    # ---------------------------------------------------------
    height, width = rgb.shape[:2]

    if depth.shape != (height, width):
        raise ValueError(
            f"RGB / Depth mismatch: "
            f"{rgb.shape[:2]} vs {depth.shape}"
        )

    if mask.shape != (height, width):
        raise ValueError(
            f"RGB / Mask mismatch: "
            f"{rgb.shape[:2]} vs {mask.shape}"
        )

    return rgb, depth, mask


def main() -> None:
    # ---------------------------------------------------------
    # 1. 실행 옵션
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
    )

    parser.add_argument(
        "--debug",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 2. 경로
    # ---------------------------------------------------------
    project_root = (
        Path(__file__).resolve().parent
    )

    tless_dir = (
        project_root
        / "custom_data"
        / "tless06"
    )

    sample_dir = (
        tless_dir
        / "additional_sample"
    )

    mesh_path = (
        tless_dir
        / "obj_06.ply"
    )

    K_path = (
        sample_dir
        / "K.txt"
    )

    output_dir = (
        sample_dir
        / "output"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 3. 필수 파일 확인
    # ---------------------------------------------------------
    if not mesh_path.exists():
        raise FileNotFoundError(
            f"Mesh not found: {mesh_path}"
        )

    if not K_path.exists():
        raise FileNotFoundError(
            f"K not found: {K_path}"
        )

    # ---------------------------------------------------------
    # 4. Camera intrinsic K
    # ---------------------------------------------------------
    K = np.loadtxt(
        K_path,
        dtype=np.float64,
    ).reshape(3, 3)

    print()
    print("=== Camera Intrinsic K ===")
    print(K)

    # ---------------------------------------------------------
    # 5. T-LESS #6 CAD model
    #
    # T-LESS mesh:
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
    # 6. Bounding box
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
    # 7. FoundationPose 초기화
    #
    # 네트워크와 mesh는 모든 sample에서 동일하므로
    # 한 번만 초기화한다.
    # ---------------------------------------------------------
    set_logging_format()
    set_seed(0)

    debug_dir = (
        output_dir
        / "debug"
    )

    os.system(
        f"rm -rf {debug_dir}"
    )

    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
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

    # ---------------------------------------------------------
    # 8. RGB 파일에서 frame ID 자동 탐색
    #
    # 예:
    # rgb_000000.png
    #       ↓
    # frame_id = 0
    # ---------------------------------------------------------
    rgb_files = sorted(
        sample_dir.glob(
            "rgb_*.png"
        )
    )

    if not rgb_files:
        raise RuntimeError(
            f"No RGB samples found: {sample_dir}"
        )

    frame_ids = []

    for rgb_path in rgb_files:
        frame_id = int(
            rgb_path.stem.split("_")[-1]
        )

        frame_ids.append(
            frame_id
        )

    print()
    print(
        f"Found {len(frame_ids)} samples:"
    )
    print(frame_ids)

    # ---------------------------------------------------------
    # 9. 각 frame 독립 Registration
    # ---------------------------------------------------------
    for frame_id in frame_ids:
        print()
        print(
            "========================================"
        )
        print(
            f"Frame {frame_id:06d}"
        )
        print(
            "========================================"
        )

        # -----------------------------------------------------
        # 입력 load
        # -----------------------------------------------------
        rgb, depth, mask = load_sample(
            sample_dir=sample_dir,
            frame_id=frame_id,
            depth_scale=args.depth_scale,
        )

        # -----------------------------------------------------
        # FoundationPose Registration
        #
        # 이전 frame pose는 사용하지 않는다.
        # 각각 새로운 6D pose를 추정한다.
        # -----------------------------------------------------
        pose = estimator.register(
            K=K,
            rgb=rgb,
            depth=depth,
            ob_mask=mask,
            iteration=args.est_refine_iter,
        )

        # -----------------------------------------------------
        # Pose 저장
        # -----------------------------------------------------
        pose_path = (
            output_dir
            / f"pose_{frame_id:06d}.txt"
        )

        np.savetxt(
            pose_path,
            pose.reshape(4, 4),
            fmt="%.8f",
        )

        print()
        print(
            "Estimated Pose:"
        )
        print(pose)

        # -----------------------------------------------------
        # Visualization
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

        vis_path = (
            output_dir
            / f"vis_{frame_id:06d}.png"
        )

        imageio.imwrite(
            vis_path,
            vis,
        )

        print(
            f"Pose: {pose_path}"
        )

        print(
            f"Visualization: {vis_path}"
        )

    # ---------------------------------------------------------
    # 10. 완료
    # ---------------------------------------------------------
    print()
    print("========================================")
    print("Batch registration complete")
    print("========================================")

    print(
        f"Output directory: {output_dir}"
    )


if __name__ == "__main__":
    main()