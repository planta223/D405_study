"""
D405 카메라 및 프로젝트 공통 설정
"""

# ---------------------------------------------------------
# RealSense 스트림 설정
# ---------------------------------------------------------

# RGB / Depth 스트림의 가로 해상도 [pixel]
IMAGE_WIDTH = 848

# RGB / Depth 스트림의 세로 해상도 [pixel]
IMAGE_HEIGHT = 480

# 카메라 스트림 프레임률 [frame/s]
FPS = 30


# ---------------------------------------------------------
# 데이터 저장 경로
# ---------------------------------------------------------

# 프로젝트 루트 경로
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# RGB 이미지 저장 디렉터리
RGB_OUTPUT_DIR = PROJECT_ROOT / "data" / "rgb"

# 원본 Depth 이미지(uint16) 저장 디렉터리
DEPTH_OUTPUT_DIR = PROJECT_ROOT / "data" / "depth"

# 카메라 내부 파라미터 K 저장 디렉터리
INTRINSIC_OUTPUT_DIR = PROJECT_ROOT / "data" / "intrinsic"


# ---------------------------------------------------------
# Depth 시각화 설정
# ---------------------------------------------------------

# uint16 Depth 값을 OpenCV에서 보기 위한 8-bit 영상으로 변환할 때
# 적용하는 시각화용 스케일 계수
# 실제 Depth 거리 계산에는 사용되지 않음
DEPTH_COLORMAP_ALPHA = 0.03