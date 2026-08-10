# D405 Study

Intel RealSense D405를 이용한 RGB-D 데이터 취득 및 기초 실습 프로젝트.

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
│   └── main.py
├── data/
│   ├── rgb/
│   └── depth/
├── requirements.txt
└── README.md
```

## Goals

- RGB stream 확인
- Depth stream 확인
- Camera intrinsic 확인
- Depth scale 확인
- RGB / Depth frame 저장
- FoundationPose 입력 데이터 구성