"""
T-LESS Object #6 Single-frame FoundationPose Test

입력:
- custom_data/tless06/rgb.png
- custom_data/tless06/depth.png
- custom_data/tless06/mask.png
- custom_data/tless06/K.txt
- custom_data/tless06/obj_06.ply

출력:
- custom_data/tless06/output/pose.txt
- custom_data/tless06/output/pose_vis.png
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
    )

    parser.add_argument(
        "--debug",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 2. 경로 설정
    # ---------------------------------------------------------
    project_root = Path(__file__).resolve().parent

    data_dir = (
        project_root
        / "custom_data"
        / "tless06"
    )

    rgb_path = data_dir / "rgb.png"
    depth_path = data_dir / "depth.png"
    mask_path = data_dir / "mask.png"
    K_path = data_dir / "K.txt"
    mesh_path = data_dir / "obj_06.ply"

    output_dir = data_dir / "output"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 3. 입력 파일 확인
    # ---------------------------------------------------------
    for path in [
        rgb_path,
        depth_path,
        mask_path,
        K_path,
        mesh_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}"
            )

    # ---------------------------------------------------------
    # 4. RGB load
    #
    # OpenCV는 BGR로 읽으므로 RGB로 변환
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
    # 5. Depth load
    #
    # 저장된 값:
    # uint16 raw depth
    #
    # 실제 거리:
    # raw × depth_scale = meter
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
        * args.depth_scale
    )

    # invalid depth는 0 유지
    depth[depth < 0.001] = 0.0

    # ---------------------------------------------------------
    # 6. Mask load
    #
    # FoundationPose에는 binary mask 전달
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
    # 7. Camera intrinsic K
    # ---------------------------------------------------------
    K = np.loadtxt(
        K_path,
        dtype=np.float64,
    ).reshape(3, 3)

    # ---------------------------------------------------------
    # 8. 입력 shape 확인
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

    # ---------------------------------------------------------
    # 9. Mesh load
    #
    # T-LESS model은 mm 기준이므로
    # FoundationPose에서 사용할 meter 단위로 변환
    # ---------------------------------------------------------
    mesh = trimesh.load(
        str(mesh_path)
    )

    mesh.vertices *= 1e-3

    # ---------------------------------------------------------
    # 10. Bounding box 계산
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
    # 11. FoundationPose 초기화
    # ---------------------------------------------------------
    set_logging_format()
    set_seed(0)

    debug_dir = output_dir / "debug"

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
    # 12. Initial pose registration
    # ---------------------------------------------------------
    pose = estimator.register(
        K=K,
        rgb=rgb,
        depth=depth,
        ob_mask=mask,
        iteration=args.est_refine_iter,
    )

    # ---------------------------------------------------------
    # 13. Pose 저장
    # ---------------------------------------------------------
    pose_path = (
        output_dir / "pose.txt"
    )

    np.savetxt(
        pose_path,
        pose.reshape(4, 4),
        fmt="%.8f",
    )

    print()
    print("=== Estimated Pose ===")
    print(pose)
    print()
    print(
        f"Pose saved: {pose_path}"
    )

    # ---------------------------------------------------------
    # 14. Pose visualization
    # ---------------------------------------------------------
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
        output_dir / "pose_vis.png"
    )

    imageio.imwrite(
        vis_path,
        vis,
    )

    print(
        f"Visualization saved: {vis_path}"
    )


if __name__ == "__main__":
    main()