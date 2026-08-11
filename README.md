# D405 Study

Intel RealSense D405 기반 RGB-D 데이터 취득 및 FoundationPose 입력 준비 프로젝트.

## Environment

- Ubuntu 22.04
- Python 3.10
- Intel RealSense D405
- RealSense SDK 2.0

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
│   ├── obj_06.ply
│   └── obj_06_print.stl
├── doc/
│   ├── requirements.txt
│   └── memo.txt
├── capture_rgbd.py
├── generate_mask.py
├── run_foundationpose.py
├── .gitignore
└── README.md
```

## Goals

- RGB / Depth stream 확인
- Depth → RGB alignment
- Camera intrinsic 및 Depth scale 확인
- RGB / Depth / K 저장
- Binary object mask 생성
- T-LESS CAD model 준비
- FoundationPose 입력 데이터 구성
- T-LESS object 6D pose 추정