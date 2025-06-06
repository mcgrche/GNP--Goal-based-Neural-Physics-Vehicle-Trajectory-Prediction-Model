'''
coding:utf-8
@Software:PyCharm
@Time:2024/3/11 11:31
@Author:RUI
'''
import numpy as np
import pandas as pd
import torch
import random
import time
import os

import numpy as np

import config.HighD
import config.Ngsim

# （废弃，目前保证邻居的数据在时间长度内是完整的）V4，只要每个邻居在seq_len长度下存在过（即不全是0），则表示这个位置上有邻居。生成nei_existence的shape是[num_seq, max_nei + 1]
def traj_sup_modified(TrajWithNbr):
    num_seq, seq_len, feat_sum = TrajWithNbr.shape
    max_nei = 8  # 最大邻居数

    # 初始化数组
    traj = TrajWithNbr[:, :, :5]  # 提取当前车辆数据
    supplement = np.zeros((num_seq, seq_len, max_nei, 4))  # 存储邻居特征
    nei_existence = np.zeros((num_seq, max_nei + 1))  # 存储邻居存在情况及其计数
    nei_existence_time = np.zeros((num_seq, seq_len, max_nei))

    # 遍历每个序列以提取邻居特征和存在情况
    for seq_idx in range(num_seq):
        for nei_idx in range(max_nei):
            nei_data = TrajWithNbr[seq_idx, :, 5 + nei_idx * 4:5 + (nei_idx + 1) * 4]
            # 检查邻居是否至少在一个时间帧中存在（不全为0）
            if np.any(nei_data != 0):
                nei_existence[seq_idx, nei_idx] = 1  # 标记这个邻居至少存在过一次
                supplement[seq_idx, :, nei_idx, :] = nei_data  # 将邻居数据存储在supplement中
            # 如果全是0，则在nei_existence中保持为0，不需要额外操作
            # 记录每个step的邻居存在情况，得到nei_existence_time
            for time_idx in range(seq_len):
                if np.any(nei_data[time_idx, :] != 0):
                    nei_existence_time[seq_idx, time_idx, nei_idx] = 1

        # 更新每个序列的邻居计数
        nei_existence[seq_idx, -1] = np.sum(nei_existence[seq_idx, :-1])

    return traj, supplement, nei_existence, nei_existence_time

# （废弃，目前使用将上面从右往左的轨迹对称的方法）将上面从右往左的轨迹，即laneID==2，3的删除掉，简化
def delete_direction(input_TOTAL, supplement, nbr_existence):
    """
    Deletes the trajectory data for sequences where laneID is 2 or 3.

    Parameters:
    input_TOTAL (np.ndarray): The main trajectory data with shape (N, 40, 5).
    supplement (np.ndarray): The neighboring vehicle trajectory data with shape (N, 40, 8, 4).

    Returns:
    np.ndarray: Filtered main trajectory data.
    np.ndarray: Filtered neighboring vehicle trajectory data.
    """

    # Determine the indices of sequences where laneID is not 2 or 3
    valid_indices = [i for i in range(len(input_TOTAL)) if input_TOTAL[i, -1, -1] not in [2, 3]]

    # Filter the arrays based on valid_indices
    input_TOTAL_filtered = input_TOTAL[valid_indices]
    supplement_filtered = supplement[valid_indices]
    nbr_existence = nbr_existence[valid_indices]

    return input_TOTAL_filtered, supplement_filtered, nbr_existence

def symmetrize_trajectories(input_TOTAL, supplement):
    """
    Symmetrizes the trajectory data based on the laneID in input_TOTAL. If laneID=2 or 3, reverse the sequence of 40 timesteps.

    Parameters:
    input_TOTAL (np.ndarray): The main trajectory data with shape (71032, 40, 5).
    supplement (np.ndarray): The neighboring vehicle trajectory data with shape (71032, 40, 8, 4).

    Returns:
    np.ndarray: Symmetrized main trajectory data.
    np.ndarray: Symmetrized neighboring vehicle trajectory data.
    """

    # Iterate through the data and reverse the sequence if laneID=2 or 3
    for i in range(len(input_TOTAL)):
        if input_TOTAL[i, -1, -1] in [2, 3]:
            input_TOTAL[i] = input_TOTAL[i][::-1]
            # Take the absolute value of VX for input_TOTAL
            input_TOTAL[i, :, 2] = np.abs(input_TOTAL[i, :, 2])
            for nei in range(supplement.shape[2]):
                supplement[i, :, nei, :] = supplement[i, ::-1, nei, :]
                # Take the absolute value of VX for supplement
                supplement[i, :, nei, 2] = np.abs(supplement[i, :, nei, 2])

    return input_TOTAL, supplement

def preprocess3(current_trajectories, neighbor_trajectories, nbr_existence, norm_std):

    num_seq, seq_len, num_nei, num_feat = neighbor_trajectories.shape
    # 获取主体车辆在第一个时间点的坐标，并对当前车辆轨迹进行相对位置转换
    # first_point_current = current_trajectories[:, 0, :2]
    # *以current step为相对点，而非采用first step
    first_point_current = current_trajectories[:, config.Ngsim.OB_HORIZON, :2] ################################ 切换数据集修改 ###########################################
    # print(first_point_current[0:10])
    adjusted_current_trajectories = np.copy(current_trajectories)
    adjusted_current_trajectories[:, :, :2] -= first_point_current[:, None, :]

    # 保留一个未标准化的agent数据
    rele_current_trajectories = np.copy(adjusted_current_trajectories)

    # 初始化一个用于存储调整后的邻居轨迹的数组
    adjusted_neighbor_trajectories = np.copy(neighbor_trajectories)

    # 遍历每个序列和每个邻居
    for seq_idx in range(num_seq):
        for nei_idx in range(num_nei):
            if nbr_existence[seq_idx, nei_idx]:
                # 对存在的邻居，将其轨迹转换为相对于主体车辆第一个时间点的坐标
                # print(adjusted_neighbor_trajectories[seq_idx, :, nei_idx, :2])
                adjusted_neighbor_trajectories[seq_idx, :, nei_idx, :2] -= first_point_current[seq_idx]
                # print(adjusted_neighbor_trajectories[seq_idx, :, nei_idx, :2])

    # 以下为检查和调试输出，实际使用时可以去掉
    # print("检查最小值")
    # print(np.min(adjusted_current_trajectories[:, :, 0]))
    # for i in range(num_nei):
    #     min_value = np.min(adjusted_neighbor_trajectories[:, :, i, 0])
    #     negative_count = np.sum(adjusted_neighbor_trajectories[:, :, i, 0] < 0)
    #     print(f"邻居 {i} 的最小X值: {min_value}, 负数数量: {negative_count}")
    # print(adjusted_neighbor_trajectories[0:10, :1, :, 0:2])

    # Assuming the same number of features in both datasets and excluding laneid if present
    feature_count = adjusted_current_trajectories.shape[2] - 1
    # Initialize an array to store min and max values for each feature
    features_min_max = np.zeros((feature_count, 2))
    # 初始化一个数组来存储每个特征的均值和标准差
    features_mean_std = np.zeros((num_feat, 2))

    # 去除外围点，并进行归一化
    # norm_std == 1则选择std标准化，==0则选择minmax归一化
    if norm_std == 1:
        for i in range(num_feat):
            # 组合当前轨迹和邻居轨迹的特征值
            current_features = adjusted_current_trajectories[:, :, i].flatten()
            neighbor_features = adjusted_neighbor_trajectories[:, :, :, i].flatten()
            combined_features = np.concatenate([current_features, neighbor_features])

            # 计算均值和标准差
            mean = np.mean(combined_features)
            std = np.std(combined_features)

            # 存储均值和标准差用于后续的反标准化
            features_mean_std[i] = [mean, std]

            # 如果标准差为0，则意味着所有值都相同，无法进行标准化
            if std != 0:
                # 应用标准化
                adjusted_current_trajectories[:, :, i] = (adjusted_current_trajectories[:, :, i] - mean) / std
                adjusted_neighbor_trajectories[:, :, :, i] = (adjusted_neighbor_trajectories[:, :, :, i] - mean) / std
            else:
                # 如果标准差为0，意味着该特征对于所有实例都是常数
                # 标准化后的值将设为0（因为该特征不提供任何区分性信息）
                adjusted_current_trajectories[:, :, i] = 0
                adjusted_neighbor_trajectories[:, :, :, i] = 0

    elif norm_std == 0:
        adjusted_current_trajectories = adjusted_current_trajectories
        adjusted_neighbor_trajectories = adjusted_neighbor_trajectories

    return adjusted_current_trajectories, adjusted_neighbor_trajectories, features_mean_std, rele_current_trajectories

# V3，只保留seq_len邻居完整的数据
def traj_sup_v3(TrajWithNbr):
    num_seq, seq_len, feat_sum = TrajWithNbr.shape
    max_nei = 8  # 最大邻居数

    valid_seq_indices = []  # 用于存储有效序列的索引
    ignored_seq_count = 0  # 被忽略的序列数

    for seq_idx in range(num_seq):
        all_nei_consistent = True  # 假设所有邻居最初都是一致的
        for nei_idx in range(max_nei):
            nei_data = TrajWithNbr[seq_idx, :, 5 + nei_idx * 4:5 + (nei_idx + 1) * 4]
            # 检查邻居是否在所有帧中一致存在或一致不存在
            if not (np.all(np.any(nei_data != 0, axis=1)) or np.all(np.all(nei_data == 0, axis=1))):
                all_nei_consistent = False
                break  # 如果发现不一致的邻居，则停止检查当前序列

        if all_nei_consistent:
            valid_seq_indices.append(seq_idx)
        else:
            ignored_seq_count += 1

    # 计算被忽略的序列比例
    ignored_seq_ratio = ignored_seq_count / num_seq * 100

    # 使用有效序列索引过滤TrajWithNbr
    valid_TrajWithNbr = TrajWithNbr[valid_seq_indices, :, :]

    # 重新初始化输出数组，仅包含有效序列
    traj = valid_TrajWithNbr[:, :, :5]
    supplement = np.zeros((len(valid_seq_indices), seq_len, max_nei, 4))
    nei_existence = np.ones((len(valid_seq_indices), max_nei + 1))  # 所有选中的序列都满足邻居一致性要求

    # 由于所有选中的序列的邻居都是一致存在的，直接填充supplement和nei_existence
    for i, seq_idx in enumerate(valid_seq_indices):
        for nei_idx in range(max_nei):
            nei_data = valid_TrajWithNbr[i, :, 5 + nei_idx * 4:5 + (nei_idx + 1) * 4]
            if np.any(nei_data != 0):
                supplement[i, :, nei_idx, :] = nei_data
            else:
                nei_existence[i, nei_idx] = 0  # 标记不存在的邻居
        # 更新每个序列的邻居计数
        nei_existence[i, -1] = np.sum(nei_existence[i, :-1])

    return traj, supplement, nei_existence, ignored_seq_ratio

def load_data(sample_rate = 1):
    # 加载TrajWithNbr数据，[num_seq, seq_len, feat_sum](71032, 40, 37)
    # TrajWithNbr是200，2，5；TrajWithNbr_2是40，2，1
    dataset_dir = "../highD-dataset_trackNneighbors/"
    # files = "TrajWithNbrs_from_ngsim_i80 0400-0415.npy" # 30+50,ngsim, window_size=80, stride=5, step=1
    files = "TrajWithNbrs_v4.npy" # 75+125, highd01, window_size=200, stride=1, step=1
    TrajWithNbrs = np.load(dataset_dir + files)
    print("TrajWithNbrs的形状是：",TrajWithNbrs.shape)

    # 检测当前车和所有邻居的X平均值
    # print(np.mean(TrajWithNbrs[:, :, 0]))
    # for i in range(5,TrajWithNbrs.shape[2],4):
    #     print(np.mean(TrajWithNbrs[:, :, i]))

    # 将agent与nei分离出来，input_TOTAL [num_seq, seq_len==40, feat_sum==5]，supplement[num_seq, seq_len==40, num_nei==8, feat_sum==5]，nbr_existence[num_seq, num_nei+1==9]
    start = time.time()
    # input_TOTAL, supplement, nbr_existence, ignored_seq_ratio = traj_sup_v3(TrajWithNbrs) # seq_len中邻居一直存在或者不存在
    input_TOTAL, supplement, nbr_existence, nbr_existence_time = traj_sup_modified(TrajWithNbrs) # seq_len中邻居
    print(supplement[0][1])
    end = time.time()
    traj_sup_duration = end - start
    print("# traj和supplement切分完毕")
    print('# traj和supplement切分时间：', traj_sup_duration)
    # print("# 每个邻居要不然为0要不然填满的比例", ignored_seq_ratio, "%")

    # 将车辆轨迹确定同一个前进方向，对相反的轨迹进行对称处理
    # 检查文件名是否包含'ngsim'
    if 'ngsim' in files.lower():
        print("跳过 symmetrize_trajectories 函数，因为文件名包含 'ngsim'")
        input_TOTAL_sy, supplement_sy = input_TOTAL, supplement  # 直接使用原始数据
    else:
        print("调用 symmetrize_trajectories 函数")
        input_TOTAL_sy, supplement_sy = symmetrize_trajectories(input_TOTAL, supplement)
    print("检查preprocess前的数据")
    print("", input_TOTAL_sy[0, 0, :])
    print("", supplement_sy[0, 0, 0, :])
    # input_TOTAL_sy, supplement_sy, nbr_existence = delete_direction(input_TOTAL, supplement, nbr_existence)
    # print(supplement_sy[0][38])
    # print("打印对称后的X平均值情况")
    # print(np.min(input_TOTAL_sy[:, :, 0]))
    # for i in range(supplement_sy.shape[2]):
    #     print(np.min(supplement_sy[:, :, i, 0]))
    #     print(np.sum(supplement_sy[:, :, i, 0] < 0))

    # 对每个feature进行归一化，并且对X Y坐标数据进行相对位置化处理
    # norm_std == 1则选择std标准化，==0则选择minmax归一化
    norm_std = 1
    input_TOTAL_f, supplement_f, features_min_max, rele_current_trajectories = preprocess3(input_TOTAL_sy, supplement_sy, nbr_existence, norm_std)

    print("检查preprocess后的数据")
    print("", input_TOTAL_f[0, 0, :])
    print("", supplement_f[0, 0, 0, :])

    # 拆分为train和test两个部分
    seed = 1120  # 设置随机种子
    random.seed(seed)
    # 创建索引数组，然后打乱索引
    indices = np.arange(input_TOTAL_f.shape[0])
    np.random.shuffle(indices)

    # random.shuffle(input_TOTAL_f)
    # 使用打乱的索引来重排所有数组
    input_TOTAL_f = input_TOTAL_f[indices]
    supplement_f = supplement_f[indices]
    nbr_existence = nbr_existence[indices]
    nbr_existence_time = nbr_existence_time[indices]
    rele_current_trajectories = rele_current_trajectories[indices]

    # 取少量样本数据做实验
    if sample_rate:
        input_TOTAL_f = input_TOTAL_f[:int(input_TOTAL_f.shape[0] * sample_rate), ]
        supplement_f = supplement_f[:int(supplement_f.shape[0] * sample_rate), ]
        nbr_existence = nbr_existence[:int(nbr_existence.shape[0] * sample_rate), ]
        nbr_existence_time = nbr_existence_time[:int(nbr_existence_time.shape[0] * sample_rate), ]

    split_line1 = int(input_TOTAL_f.shape[0] * 0.8)
    split_line2 = int(input_TOTAL_f.shape[0] * 0.9)

    training_input = input_TOTAL_f[:split_line1, ]
    training_supplement = supplement_f[:split_line1, ]
    training_nbr_existence = nbr_existence[:split_line1, ]
    training_nbr_existence_time = nbr_existence_time[:split_line1, ]

    test_input = input_TOTAL_f[split_line1:split_line2, ]
    testing_supplement = supplement_f[split_line1:split_line2, ]
    testing_nbr_existence = nbr_existence[split_line1:split_line2, ]
    testing_nbr_existence_time = nbr_existence_time[split_line1:split_line2, ]

    inference_input = input_TOTAL_f[split_line2:]
    inference_supplement = supplement_f[split_line2:]
    inference_nbr_existence = nbr_existence[split_line2:]
    inference_nbr_existence_time = nbr_existence_time[split_line2:]

    # 保留一个原本真实的inference data，用作最后的效果比较
    inference_rele_current_trajectories = rele_current_trajectories[split_line2:]

    print('\n', "#######数据划分结束，打印划分结果#########")
    print("training_input.shape", training_input.shape)
    print("training_supplement.shape", training_supplement.shape)
    print("training_nbr_existence.shape", training_nbr_existence.shape)
    print("training_nbr_existence_time.shape", training_nbr_existence_time.shape)

    print("test_input.shape", test_input.shape)
    print("testing_supplement.shape", testing_supplement.shape)
    print("testing_nbr_existence.shape", testing_nbr_existence.shape)
    print("testing_nbr_existence_time.shape", testing_nbr_existence_time.shape)

    print("inference_input.shape", inference_input.shape)
    print("inference_supplement.shape", inference_supplement.shape)
    print("inference_nbr_existence.shape", inference_nbr_existence.shape)
    print("inference_nbr_existence_time.shape", inference_nbr_existence_time.shape)

    print("inference_rele_current_trajectories.shape", inference_rele_current_trajectories.shape)

    ########################## 保存切割好的数据，可切换数据集 ##########################
    folder_path = 'data/Ngsim/rep_sample/'
    # 检查目录是否存在
    if not os.path.exists(folder_path):
        # 如果目录不存在，则创建它
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' was created.")
    else:
        print(f"Directory '{folder_path}' already exists.")
    np.save(folder_path + 'training_input.npy', training_input)
    np.save(folder_path + 'training_supplement.npy', training_supplement)
    np.save(folder_path + 'training_nbr_existence.npy', training_nbr_existence)
    np.save(folder_path + 'training_nbr_existence_time.npy', training_nbr_existence_time)
    np.save(folder_path + 'test_input.npy', test_input)
    np.save(folder_path + 'testing_supplement.npy', testing_supplement)
    np.save(folder_path + 'testing_nbr_existence.npy', testing_nbr_existence)
    np.save(folder_path + 'testing_nbr_existence_time.npy', testing_nbr_existence_time)
    np.save(folder_path + 'inference_input.npy', inference_input)
    np.save(folder_path + 'inference_supplement.npy', inference_supplement)
    np.save(folder_path + 'inference_nbr_existence.npy', inference_nbr_existence)
    np.save(folder_path + 'inference_nbr_existence_time.npy', inference_nbr_existence_time)

    np.save(folder_path + 'features_min_max.npy', features_min_max) # [features, 2] 0:min,1:max，在反归一化时应该只取XY（第一二个）

    np.save(folder_path + 'inference_rele_current_trajectories.npy', inference_rele_current_trajectories) # 真实inference轨迹，可以直接用于绘制对比图

def dir_load_data():

    folder_path = 'data/HighD/rep_sample/' #  Ngsim or HighD
    training_input = np.load(folder_path + 'training_input.npy')
    training_supplement = np.load(folder_path + 'training_supplement.npy')
    training_nbr_existence = np.load(folder_path + 'training_nbr_existence.npy')
    training_nbr_existence_time = np.load(folder_path + 'training_nbr_existence_time.npy')
    test_input = np.load(folder_path + 'test_input.npy')
    testing_supplement = np.load(folder_path + 'testing_supplement.npy')
    testing_nbr_existence = np.load(folder_path + 'testing_nbr_existence.npy')
    testing_nbr_existence_time = np.load(folder_path + 'testing_nbr_existence_time.npy')
    inference_input = np.load(folder_path + 'inference_input.npy')
    inference_supplement = np.load(folder_path + 'inference_supplement.npy')
    inference_nbr_existence = np.load(folder_path + 'inference_nbr_existence.npy')
    inference_nbr_existence_time = np.load(folder_path + 'inference_nbr_existence_time.npy')

    feature_destd = np.load(folder_path + 'features_min_max.npy')

    print("training_input.shape", training_input.shape)
    print("training_supplement.shape", training_supplement.shape)
    print("training_nbr_existence.shape", training_nbr_existence.shape)
    print("training_nbr_existence_time.shape", training_nbr_existence_time.shape)

    print("test_input.shape", test_input.shape)
    print("testing_supplement.shape", testing_supplement.shape)
    print("testing_nbr_existence.shape", testing_nbr_existence.shape)
    print("testing_nbr_existence_time.shape", testing_nbr_existence_time.shape)

    print("inference_input.shape", inference_input.shape)
    print("inference_supplement.shape", inference_supplement.shape)
    print("inference_nbr_existence.shape", inference_nbr_existence.shape)
    print("inference_nbr_existence_time.shape", inference_nbr_existence_time.shape)

    return training_input, training_supplement, training_nbr_existence, training_nbr_existence_time,\
        test_input, testing_supplement, testing_nbr_existence,testing_nbr_existence_time, \
        inference_input, inference_supplement, inference_nbr_existence,inference_nbr_existence_time, feature_destd

if __name__ == '__main__':
    load_data()