import cv2
import numpy as np
import pandas as pd
import os
import json

# --- [설정 및 파라미터] ---
BASE_DIR = "data share"  # 사용자 폴더명 유지
CALIB_DATA_FILE = "global_collected_points.json"  # 수집 데이터 저장용
FINAL_CALIB_FILE = "camera_extrinsic_global.json" # 최종 결과 저장용

# 초기 내부 파라미터 (1280x720)
INTRINSIC_K = np.array([[645.0, 0, 640.0], [0, 645.0, 360.0], [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.zeros(5, dtype=np.float32) # 왜곡 계수 초기값

# 수집 데이터 리스트
all_img_pts = [] # 2D 픽셀 좌표
all_obj_pts = [] # 3D 로봇 좌표

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        p_3d, display_container = param
        all_img_pts.append([x, y])
        all_obj_pts.append(p_3d)
        print(f"포인트 추가: {len(all_img_pts)}개 수집됨")
        cv2.circle(display_container['img'], (x, y), 2, (0, 0, 255), -1)
        cv2.imshow("Collection Mode", display_container['img'])

def run_global_optimization():
    """수집된 모든 데이터를 사용하여 왜곡 계수와 카메라 위치를 동시 계산"""
    if len(all_img_pts) < 15: # 왜곡까지 잡으려면 점이 많을수록 좋습니다.
        print(f"데이터 부족: 현재 {len(all_img_pts)}개. 최소 15개 이상 수집하세요.")
        return

    print("\n[전역 최적화 계산 중...]")
    # cv2.calibrateCamera 형식에 맞게 데이터 변환
    obj_p = np.array([all_obj_pts], dtype=np.float32)
    img_p = np.array([all_img_pts], dtype=np.float32)
    
    # K와 왜곡 계수를 함께 최적화
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_p, img_p, (1280, 720), INTRINSIC_K, None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )

    if ret:
        R, _ = cv2.Rodrigues(rvecs[0])
        T_cam_base = np.eye(4)
        T_cam_base[:3, :3] = R
        T_cam_base[:3, 3] = tvecs[0].squeeze()
        T_base_cam = np.linalg.inv(T_cam_base)
        
        result = {
            "T_base_cam": T_base_cam.tolist(),
            "K": mtx.tolist(),
            "dist": dist.tolist(),
            "reprojection_error": ret
        }
        with open(FINAL_CALIB_FILE, 'w') as f:
            json.dump(result, f)
        
        print(f"최적화 완료! 오차: {ret:.4f}")
        print(f"왜곡 계수(dist): {dist.flatten()}")
        print(f"결과가 {FINAL_CALIB_FILE}에 저장되었습니다.")
        return result
    return None

def main_collector():
    # 기존 데이터가 있으면 불러오기
    global all_img_pts, all_obj_pts
    if os.path.exists(CALIB_DATA_FILE):
        with open(CALIB_DATA_FILE, 'r') as f:
            data = json.load(f)
            all_img_pts = data['img_pts']
            all_obj_pts = data['obj_pts']
        print(f"기존 데이터 로드됨: {len(all_img_pts)}개")

    idx = 1
    while 1 <= idx <= 50:
        traj_path = os.path.join(BASE_DIR, f"traj_{idx:03d}")
        csv_path = os.path.join(traj_path, "ee_needle.csv")
        if not os.path.exists(csv_path):
            idx += 1; continue

        df = pd.read_csv(csv_path)
        print(f"\n[수집: TRAJ_{idx:03d}] 클릭:추가, Space:다음이미지, d:다음폴더, g:최적화실행, q:저장후종료")

        # 40프레임 간격으로 역순 탐색하여 다양한 위치 확보
        for i in range(len(df)-1, -1, -40):
            row = df.iloc[i]
            p_3d = [row['x'], row['y'], row['z']]
            img_path = os.path.join(traj_path, "images", f"{row['timestamp']}.png")
            if not os.path.exists(img_path): continue
            
            img = cv2.imread(img_path)
            display_container = {'img': img.copy()}
            
            # UI 텍스트 (빨간색)
            text = f"Points: {len(all_img_pts)} | TRAJ_{idx:03d} | Space:Next g:Calib q:Save&Quit"
            cv2.putText(display_container['img'], text, (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Collection Mode", display_container['img'])
            cv2.setMouseCallback("Collection Mode", mouse_callback, param=(p_3d, display_container))
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord(' '): continue # 다음 이미지로
            elif key == ord('g'): run_global_optimization() # 최적화 실행
            elif key == ord('d'): break # 다음 폴더로
            elif key == ord('q'): # 데이터 저장 후 종료
                with open(CALIB_DATA_FILE, 'w') as f:
                    json.dump({"img_pts": all_img_pts, "obj_pts": all_obj_pts}, f)
                print("수집 데이터 저장 완료.")
                return

        idx += 1

if __name__ == "__main__":
    main_collector()
    cv2.destroyAllWindows()