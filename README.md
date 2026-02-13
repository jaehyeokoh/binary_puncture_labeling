# 설치법
폴더 내에서 터미널 열고 uv sync 입력하면 venv 생성됨

# 사전 준비
- **check_rgb.py** : 바늘의 배경부분의 rgb 값 확인용(마우스 클릭시 print됨)
- **Generate_needlepos.py** : FK를 통해 data share/traj_/에 ee_needle.csv 생성 (바늘 끝단 위치)

# 캘리브레이션 및 마킹
- **Calibrate_and_mark.py** : ee_needle.csv에서 pos와 이미지에서 클릭한 바늘 끝단의 위치로 카메라-로봇 변환행렬(camera_extrinsic.json)생성
- **Check_puncture.py** : 바늘 끝단에 5픽셀 원 내부 rgb값이 특정 값 이상이면 뚫린 것으로 간주, 디버그 코드
