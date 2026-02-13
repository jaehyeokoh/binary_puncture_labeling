import os
import pandas as pd
import numpy as np
from pandaKinematics import pandaKinematics
from scipy.spatial.transform import Rotation

def calculate_needle_pose(joint_angles):
    """
    7개 관절 각도를 입력받아 바늘 끝단의 4x4 변환 행렬을 반환합니다.
    """
    # 1. FK 계산 (브래킷까지의 포즈)
    Tbs, _ = pandaKinematics.fk(joint_angles) #
    T_bracket = Tbs[-1] # 마지막 행렬이 브래킷의 포즈
    
    # 2. 바늘 오프셋 적용 (브래킷 z축 방향으로 +115mm)
    # 박성준 님 전달 사항: 바늘 중심은 브래킷 좌표계 기준 z축 +115 mm 위치
    z_offset = 0.115 # mm 단위를 m로 변환
    
    # 브래킷 좌표계 기준의 오프셋 행렬
    T_offset = np.eye(4)
    T_offset[2, 3] = z_offset
    
    # 최종 바늘 포즈 = T_bracket * T_offset
    T_needle = T_bracket @ T_offset
    return T_needle

def main():
    base_path = "data share"
    # traj_001부터 traj_050까지 폴더 순회
    for i in range(1, 51):
        folder_name = f"traj_{i:03d}"
        folder_path = os.path.join(base_path, folder_name)
        input_file = os.path.join(folder_path, "joint_states.csv")
        output_file = os.path.join(folder_path, "ee_needle.csv")
        
        if not os.path.exists(input_file):
            print(f"파일을 찾을 수 없음: {input_file}")
            continue
            
        print(f"처리 중: {folder_name}...")
        
        # CSV 읽기
        df = pd.read_csv(input_file)
        
        results = []
        for index, row in df.iterrows():
            # pos_0 ~ pos_6 추출
            joints = row[['pos_0', 'pos_1', 'pos_2', 'pos_3', 'pos_4', 'pos_5', 'pos_6']].values
            
            # 바늘 포즈 계산
            T_needle = calculate_needle_pose(joints)
            
            # 위치(x, y, z) 및 자세(Quaternion) 추출
            pos = T_needle[:3, 3]
            quat = Rotation.from_matrix(T_needle[:3, :3]).as_quat() # [x, y, z, w]
            
            results.append([
                row['timestamp'], 
                pos[0], pos[1], pos[2], 
                quat[0], quat[1], quat[2], quat[3]
            ])
            
        # 결과 저장
        output_df = pd.DataFrame(results, columns=['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        output_df.to_csv(output_file, index=False)
        
    print("모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()