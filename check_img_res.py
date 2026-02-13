import cv2
import os

base_dir = "data share"
traj_folder = "traj_050" # 확인하고 싶은 폴더
img_dir = os.path.join(base_dir, traj_folder, "images")

if os.path.exists(img_dir):
    files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    if files:
        img_path = os.path.join(img_dir, files[0])
        img = cv2.imread(img_path)
        if img is not None:
            h, w, c = img.shape
            print(f"이미지 파일: {files[0]}")
            print(f"해상도 (가로 x 세로): {w} x {h}")
            print(f"채널 수: {c}")
        else:
            print("이미지를 불러올 수 없습니다.")
    else:
        print("해당 폴더에 PNG 파일이 없습니다.")
else:
    print(f"경로를 찾을 수 없습니다: {img_dir}")