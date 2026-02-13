import cv2
import numpy as np
import pandas as pd
import os
import json

# --- [1. 설정 및 파라미터] ---
BASE_DIR = "data share"
# 전역 최적화 결과 파일 (T_base_cam, K, dist 포함)
GLOBAL_CALIB_FILE = "camera_extrinsic_global.json"

# 사용자가 직접 확인한 수치 반영 (Intensity 30~55, 비율 0.35)
GRAY_MIN, GRAY_MAX = 0, 55
RATIO_THRESH = 0.35
RADIUS = 5

def get_projection_params(T_base_cam):
    """카메라 외부 파라미터에서 투영에 필요한 벡터 추출"""
    T_cam_base = np.linalg.inv(T_base_cam)
    rvec, _ = cv2.Rodrigues(T_cam_base[:3, :3])
    tvec = T_cam_base[:3, 3]
    return rvec, tvec

def get_puncture_status(img, u, v):
    """원형 영역 내 회색 비율을 계산하여 판별"""
    h, w = img.shape[:2]
    u, v = int(u), int(v)
    if not (0 <= u < w and 0 <= v < h): return "NO", 0.0
    
    y1, y2 = max(0, v-RADIUS), min(h, v+RADIUS+1)
    x1, x2 = max(0, u-RADIUS), min(w, u+RADIUS+1)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return "NO", 0.0
    
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    yy, xx = np.ogrid[:gray_roi.shape[0], :gray_roi.shape[1]]
    center_y, center_x = v - y1, u - x1
    mask = (xx - center_x)**2 + (yy - center_y)**2 <= RADIUS**2
    
    pixels = gray_roi[mask]
    if len(pixels) == 0: return "NO", 0.0
    
    gray_count = np.sum((pixels >= GRAY_MIN) & (pixels <= GRAY_MAX))
    ratio = gray_count / len(pixels)
    status = "YES" if ratio >= RATIO_THRESH else "NO"
    return status, ratio

def main_manager():
    # 1. 전역 캘리브레이션 데이터 로드
    if not os.path.exists(GLOBAL_CALIB_FILE):
        print(f"오류: {GLOBAL_CALIB_FILE}이 없습니다. 먼저 전역 수집 코드를 실행해주세요.")
        return
    
    with open(GLOBAL_CALIB_FILE, 'r') as f:
        calib_data = json.load(f)
        T_base_cam = np.array(calib_data['T_base_cam'])
        K = np.array(calib_data['K'])
        dist = np.array(calib_data['dist'])
        error = calib_data.get('reprojection_error', 'N/A')

    rvec, tvec = get_projection_params(T_base_cam)
    print(f"설정 로드 완료 (전역 오차: {error})")

    idx = 1
    while 1 <= idx <= 50:
        traj_path = os.path.join(BASE_DIR, f"traj_{idx:03d}")
        csv_path = os.path.join(traj_path, "ee_needle.csv")
        if not os.path.exists(csv_path):
            idx += 1; continue

        df = pd.read_csv(csv_path)
        print(f"\n[최종 검수: TRAJ_{idx:03d}] Space:저장, d:다음, a:이전, Esc:종료")
        
        results = []
        save_flag = False
        skip_traj = False

        for i, row in df.iterrows():
            # 2. 최적화된 K와 dist를 사용하여 3D를 2D로 투영
            p_3d = np.array([[row['x'], row['y'], row['z']]], dtype=np.float32)
            pts_2d, _ = cv2.projectPoints(p_3d, rvec, tvec, K, dist)
            u, v = pts_2d.ravel().astype(int)
            
            img_path = os.path.join(traj_path, "images", f"{row['timestamp']}.png")
            if not os.path.exists(img_path): continue
            
            img = cv2.imread(img_path)
            
            # 3. Puncture 판별 실행
            status, ratio = get_puncture_status(img, u, v)
            results.append([row['timestamp'], u, v, status, round(ratio, 3)])
            
            # 4. 실시간 시각화
            color = (0, 255, 0) if status == "YES" else (0, 0, 255)
            # 바늘 끝 마킹 (십자가 + 원)
            cv2.circle(img, (u, v), RADIUS, color, 1)
            
            # 안내 텍스트 (빨간색)
            text_status = f"TRAJ_{idx:03d} | PUNCTURE: {status} ({ratio*100:.1f}%)"
            text_ctrl = "d:Next a:Prev Space:Save | Esc:Exit"
            cv2.putText(img, text_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(img, text_ctrl, (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            cv2.imshow("Final Puncture Analysis", img)
            
            key = cv2.waitKey(15) & 0xFF
            if key == ord('d'): idx += 1; skip_traj = True; break
            elif key == ord('a'): idx -= 1; skip_traj = True; break
            elif key == ord(' '): save_flag = True; print(f"traj_{idx:03d} 저장 예약됨")
            elif key == 27: return

        if save_flag:
            out_df = pd.DataFrame(results, columns=['timestamp', 'u', 'v', 'puncture', 'ratio'])
            out_df.to_csv(os.path.join(traj_path, "puncture_label.csv"), index=False)
            print(f"저장 성공: traj_{idx:03d}/puncture_label.csv")

        if not skip_traj: idx += 1

if __name__ == "__main__":
    main_manager()
    cv2.destroyAllWindows()