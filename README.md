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
│   ├── GPU_Server_Code/
│   ├── requirements.txt
│   └── memo.txt
├── capture_rgbd.py
├── record_rgbd.py
├── generate_mask.py
├── .gitignore
└── README.md
```


## 사전 작업 (Completed)

- Camera intrinsic 및 Depth scale 확인
- RGB / Depth stream 확인 및 Depth → RGB alignment
- T-LESS CAD model(#6) 준비 및 샘플 rgb/depth 이미지 촬영


## Goals 1 : 단일 이미지에 대한 6D 자세추정 (Completed)

- 샘플 포함 5개 자세에 대해 rgb, depth 이미지 촬영 (capture_rgbd.py)
- 5개 rgb 이미지 각각에 대해 수동으로 masking 작업 (generate_mask.py)
- rgb, depth, mask, intrinsic 파일을 GPU 서버로 복사 (Remmina 사용)
- GPU 서버에서 FoundationPose 실행 (doc/GPU_Server_Code/*)
- pose.txt 및 pose_vis.png 출력 정상 확인


## Goals 2 : 영상에 대한 6D 자세추정 (Completed)

- 10FPS로 11.8초, 총 118 프레임 촬영 (record_rgbd.py)
- 첫 프레임 rgb 이미지에 대해 수동으로 masking 작업 (generate_mask.py)
- GPU 서버에서 FoundationPose 실행 (doc/GPU_Server_Code/*)
- pos.txt, vis.png, mp4 출력 정상 확인. 평균처리속도는 25.4FPS.


## Goals 3 : 실시간 영상에 대한 6D 자세추정 (Ongoing)

- 
- 


## 그외 Planned Works

위의 사항들은 모두 3D CAD 기반의 model-based 자세추정 입니다.
만약 model-free 자세추정을 진행한다면, FoundationPose는 아래 두 단계를 거칩니다.

- register() : 초기 자세를 모르는 상태에서 다수의 pose hypothesis를 만들고, refine + score까지 수행합니다. 상대적으로 느립니다.
- track_one() : 이전 프레임의 pose를 초기값으로 사용해서 refinement만 수행합니다. 훨씬 빠릅니다. NVIDIA 쪽에서도 tracking은 이전 pose를 기반으로 업데이트하는 구조라고 설명합니다.

또한 실시간 환경에서 실제 시스템은 아래와 같이 구성됨.
```text
D405 획득
+ RGB/Depth 전송
+ 네트워크 지연
+ 디코딩
+ track_one()
+ pose 반환
+ 화면 렌더링
```
Goal2의 결과인 25.4FPS는 track_one()만의 처리시간이라고 볼 수 있음.
따라서 이 수치가 아닌 end-to-end 실시간 추적 (특히 model-free 에서) 의 처리속도가 유의미한 수치가 될 것임.