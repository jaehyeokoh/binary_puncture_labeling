import cv2
import os
import numpy as np

# --- [설정] ---
BASE_DIR = "data share"
TRAJ_IDX = 1 # 확인하고 싶은 트래젝토리 번호

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img = param
        # 1. BGR 값 추출 (OpenCV 기본)
        bgr_pixel = img[y, x]
        # 2. Grayscale 값 추출 (밝기 정보)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_val = gray_img[y, x]
        
        print(f"\n[클릭 정보]")
        print(f"- 좌표: (u:{x}, v:{y})")
        print(f"- BGR 값: {bgr_pixel}")
        print(f"- Grayscale 값 (Intensity): {gray_val}")
        
        # 화면에 잠깐 표시
        temp_img = img.copy()
        cv2.circle(temp_img, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(temp_img, f"Gray: {gray_val}", (x + 10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Pixel Inspector", temp_img)

def run_inspector():
    traj_path = os.path.join(BASE_DIR, f"traj_{TRAJ_IDX:03d}", "images")
    if not os.path.exists(traj_path):
        print(f"경로를 찾을 수 없습니다: {traj_path}")
        return

    img_files = sorted([f for f in os.listdir(traj_path) if f.endswith('.png')])
    if not img_files:
        print("이미지 파일이 없습니다.")
        return

    current_idx = 0
    print("[조작법] 마우스 클릭: 픽셀 값 확인 | d: 다음 이미지 | a: 이전 이미지 | q: 종료")

    while True:
        img_path = os.path.join(traj_path, img_files[current_idx])
        img = cv2.imread(img_path)
        
        cv2.imshow("Pixel Inspector", img)
        cv2.setMouseCallback("Pixel Inspector", mouse_callback, param=img)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            current_idx = (current_idx + 1) % len(img_files)
        elif key == ord('a'):
            current_idx = (current_idx - 1) % len(img_files)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inspector()