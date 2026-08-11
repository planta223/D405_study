# D405 Study

Intel RealSense D405 기반 RGB-D 데이터 취득 및 FoundationPose 실습 프로젝트 입니다.

## Environment

- Ubuntu 22.04
- Python 3.10
- Intel RealSense D405
- RealSense SDK 2.0
- NVIDIA GPU Server included FoundationPose(.deb recommended)

## Python Packages

- pyrealsense2
- opencv-python
- numpy

## Structure

```text
d405_study/
├── src/
│   └── python modules
├── data/
│   ├── rgb/
│   ├── depth/
│   ├── intrinsic/
│   └── mask/
├── models/
│   ├── sample.ply
│   └── sample_print.stl
├── doc/
│   ├── requirements.txt
│   └── memo.txt
├── capture_rgbd.py
├── record_rgbd.py
├── generate_mask.py
├── run_foundationpose.py
├── .gitignore
└── README.md
```

## 사전 작업 (Completed)

- Camera intrinsic 및 Depth scale 확인
- RGB / Depth stream 확인 및 Depth → RGB alignment
- T-LESS CAD model(#6) 준비 및 샘플 rgb/depth 이미지 촬영

## Goals 1 : 단일 이미지에 대한 6D 자세추정 (Completed)

- 샘플 포함 5개 자세에 대해 rgb, depth 이미지 촬영 (capture_rgbd.py)
- 5개 이미지 각각에 대해 수동으로 masking 작업 (generate_mask.py)
- rgb, depth, mask, intrinsic 파일을 GPU 서버로 복사 (Remmina 사용)
- GPU 서버에서 FoundationPose 실행 (doc/GPU_Server_Code/*)
- pose.txt 및 pose_vis.png 출력 정상 확인

## Goals 2 : 영상에 대한 6D 자세추정 (Ongoing)

- 
- 

## Goals 3 : 실시간 영상에 대한 6D 자세추정 (Planned)

- 
- 