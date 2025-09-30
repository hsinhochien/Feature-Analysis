import numpy as np
from tqdm import tqdm
import pandas as pd

def calculate_angle(point1, point2, point3):
    """
    計算目標點與相鄰兩點形成的角度
    :param point1: 相鄰點1 (x1, y1)
    :param point2: 目標點 (x2, y2)
    :param point3: 相鄰點2 (x3, y3)
    :return: 角度（以度數為單位）
    """
    # 計算向量 AB 和 BC
    vector_ab = np.array([point2[0] - point1[0], point2[1] - point1[1]])
    vector_bc = np.array([point3[0] - point2[0], point3[1] - point2[1]])
    
    # 計算向量的點積
    dot_product = np.dot(vector_ab, vector_bc)
    
    # 計算向量的長度
    magnitude_ab = np.linalg.norm(vector_ab)
    magnitude_bc = np.linalg.norm(vector_bc)

    # 檢查模長是否為零，避免除以零的錯誤
    if magnitude_ab == 0 or magnitude_bc == 0:
        return 0 # 若其中一條邊長為零，則角度為零
    
    # 計算 cos(theta)
    cos_theta = dot_product / (magnitude_ab * magnitude_bc)
    
    # 防止浮點數誤差導致 cos_theta 超出 [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # 計算角度（弧度）
    angle_radians = np.arccos(cos_theta)
    
    # 將弧度轉換為度數
    angle_degrees = np.degrees(angle_radians)
    
    return round(angle_degrees, 2)

joint_dict = {"right_elbow": [2, 3, 4], 
              "right_shoulder": [1, 2, 3], 
              "right_hip": [8, 9, 10], 
              "right_knee": [9, 10, 11], 
              "left_knee": [12, 13, 14], 
              "left_hip": [8, 12, 13], 
              "left_shoulder": [1, 5, 6], 
              "left_elbow": [5, 6, 7]}

def get_angles_and_velocities(grouped_positions_dict, all_dir, start_idx=0):
    """
    計算每個關節的角度和角速度
    :param grouped_positions_dict: 包含每個目錄下的分組位置數據的字典
    :param all_dir: 所有子目錄的列表
    :return: 包含每個關節角度和角速度的字典
    """
    all_angles_dict = {}
    all_velocities_dict = {}

    for idx, children_dir in tqdm(enumerate(all_dir), total=len(all_dir)):
        if idx < start_idx:
            continue  # 跳過前面資料

        # print(f"處理第 {idx} 筆：{children_dir}")  # debug 用

        angles_dict = {}
        velocities_dict = {}
        grouped_positions_lst = grouped_positions_dict[children_dir]

        for joint_name, indices in joint_dict.items():
            angles = []
            frame_num = len(grouped_positions_lst)
            for i in range(frame_num):
                grouped_positions = grouped_positions_lst[i]

                # 取得該關節對應的三個點
                point1 = (grouped_positions[indices[0]][0], grouped_positions[indices[0]][1]) # 相鄰點1
                point2 = (grouped_positions[indices[1]][0], grouped_positions[indices[1]][1]) # 目標點
                point3 = (grouped_positions[indices[2]][0], grouped_positions[indices[2]][1]) # 相鄰點2

                # 計算角度
                angle = calculate_angle(point1, point2, point3)
                # if angle != 0:
                angles.append(angle)

            angles = np.array(angles)
            velocities = np.array([])

            if len(angles) >= 2:
                velocities = np.diff(angles)  
            else:
                velocities = np.array([])

            angles_dict[f"{joint_name}_angles"] = angles
            velocities_dict[f"{joint_name}_velocities"] = velocities

            if len(angles) < 4:
                print(f"{joint_name} angles: {angles}, velocities: {velocities}")  # debug 用
                print("frame_num:", frame_num)  # debug 用

        all_angles_dict[children_dir] = angles_dict
        all_velocities_dict[children_dir] = velocities_dict

    return all_angles_dict, all_velocities_dict

def analyze_time_series_numpy(time_series, m=2, r=None):
    time_series = np.array(time_series)  # 確保是 NumPy 陣列
    std_value = np.std(time_series)
    N = len(time_series)
    if N < m + 2:
        # 資料太短，無法計算樣本熵，回傳 NaN 或 inf 或其他預設值
        return np.nan

    if r is None:
        r = 0.2 * std_value  # 設定相似度閾值

    # 建立 m 維的嵌入矩陣 (shape: (N-m+1, m))
    X_m = np.array([time_series[i:i + m] for i in range(N - m + 1)])
    X_m1 = np.array([time_series[i:i + m + 1] for i in range(N - m)])

    # 使用 broadcasting 計算最大絕對差值
    dist_m = np.max(np.abs(X_m[:, None, :] - X_m[None, :, :]), axis=2)
    dist_m1 = np.max(np.abs(X_m1[:, None, :] - X_m1[None, :, :]), axis=2)

    # 計算符合 r 條件的個數（排除自身比較）
    count_m = np.sum(dist_m < r, axis=1) - 1
    count_m1 = np.sum(dist_m1 < r, axis=1) - 1

    # 計算 phi 值
    B = np.sum(count_m) / (N - m)
    A = np.sum(count_m1) / (N - m - 1)

    # 計算樣本熵
    sampen = -np.log(A / B) if A > 0 and B > 0 else np.inf

    return round(sampen, 4)

def get_final_df(all_angles_dict, all_velocities_dict, all_dir, start_idx=0):
    all_babies_df = []

    for idx, children_dir in tqdm(enumerate(all_dir), total=len(all_dir)):
        if idx < start_idx:
            continue  # 跳過前面資料

        # print(f"處理第 {idx} 筆：{children_dir}")  # debug 用
        baby_angles = all_angles_dict[children_dir]
        baby_velocities = all_velocities_dict[children_dir]
        baby_results = {"video_id": children_dir}

        for joint_name, indices in joint_dict.items():
            if baby_angles[f"{joint_name}_angles"].size == 0:
                sampen_angles = None
            elif baby_velocities[f"{joint_name}_velocities"].size == 0:
                sampen_velocities = None
            else:
                sampen_angles = analyze_time_series_numpy(baby_angles[joint_name + "_angles"])
                sampen_velocities = analyze_time_series_numpy(baby_velocities[joint_name + "_velocities"])

            baby_results[f"{joint_name}_angles"] = sampen_angles
            baby_results[f"{joint_name}_velocities"] = sampen_velocities

        baby_df = pd.DataFrame([baby_results])
        all_babies_df.append(baby_df)

    final_df = pd.concat(all_babies_df, ignore_index=True)
    return final_df
