"""
D405 카메라 및 프로젝트 공통 설정
"""

from pathlib import Path


# ---------------------------------------------------------
# 프로젝트 경로
# ---------------------------------------------------------

# 현재 파일:
# D405_STUDY/src/config.py
#
# parent       -> src/
# parent.parent -> D405_STUDY/
PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)


# ---------------------------------------------------------
# RealSense 스트림 설정
# ---------------------------------------------------------

# RGB / Depth 가로 해상도 [pixel]
IMAGE_WIDTH = 848

# RGB / Depth 세로 해상도 [pixel]
IMAGE_HEIGHT = 480

# 카메라 프레임률 [frame/s]
FPS = 30


# ---------------------------------------------------------
# 데이터 저장 경로
# ---------------------------------------------------------

RGB_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "rgb"
)

DEPTH_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "depth"
)

INTRINSIC_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "intrinsic"
)

MASK_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "mask"
)


# ---------------------------------------------------------
# 모델 경로
# ---------------------------------------------------------

MODEL_DIR = (
    PROJECT_ROOT / "models"
)

TLESS_MODEL_PATH = (
    MODEL_DIR / "obj_000006.ply"
)


# ---------------------------------------------------------
# Depth 시각화 설정
# ---------------------------------------------------------

# uint16 Depth를 OpenCV 표시용 8-bit 영상으로
# 변환할 때 사용하는 시각화 전용 계수
DEPTH_COLORMAP_ALPHA = 0.03