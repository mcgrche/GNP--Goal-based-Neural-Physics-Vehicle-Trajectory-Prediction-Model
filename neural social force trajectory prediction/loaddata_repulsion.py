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

# 废弃，原始数据的处理放在TUTR里
def traj_sup(TrajWithNbr):
    # TrajWithNbr, [num_seq, seq_len, feat_sum]
    num_seq, seq_len, feat_sum = TrajWithNbr.shape
    max_nei = 8  # Maximum number of neighbors

    # Initialize arrays
    traj = TrajWithNbr[:, :, :5]  # Extract current vehicle data
    supplement = np.zeros((num_seq, seq_len, max_nei, 4))  # To store neighbor features
    nei_existence = np.zeros((num_seq, seq_len, max_nei + 1))  # To store neighbor existence and their count

    # Iterate through each sequence to extract neighbor features and their existence
    for seq_idx in range(num_seq):
        for time_step in range(seq_len):
            nei_count = 0
            for nei_idx in range(max_nei):
                start_idx = 5 + nei_idx * 4  # Start index for current neighbor features
                nei_features = TrajWithNbr[seq_idx, time_step, start_idx:start_idx + 4]

                # Check if the neighbor exists (features are not all zeros)
                if np.any(nei_features != 0):
                    supplement[seq_idx, time_step, nei_count] = nei_features
                    nei_existence[seq_idx, time_step, nei_idx] = 1
                    nei_count += 1

            # Update the count of existing neighbors for the current sequence at the current time step
            nei_existence[seq_idx, time_step, -1] = nei_count

    return traj, supplement, nei_existence


# 废弃，原始数据的处理放在TUTR里
def load_data(sample_rate = 1):
    # 加载TrajWithNbr数据，[num_seq, seq_len, feat_sum](71032, 40, 37)
    TrajWithNbrs = np.load('TrajWithNbrs.npy')
    print(TrajWithNbrs.shape)

    # 将nei整理出来，生成一个没有nei的数据 [num_seq, seq_len, feat_sum==4]
    start = time.time()
    input_TOTAL, supplement, nbr_existence = traj_sup(TrajWithNbrs)
    end = time.time()
    traj_sup_duration = end - start
    print("# traj和supplement切分完毕")
    print('# traj和supplement切分时间：', traj_sup_duration)
    # 拆分为train和test两个部分
    seed = 1120  # 设置随机种子
    random.seed(seed)
    random.shuffle(input_TOTAL)

    if sample_rate:
        input_TOTAL = input_TOTAL[:int(input_TOTAL.shape[0] * sample_rate), ]
        supplement = supplement[:int(supplement.shape[0] * sample_rate), ]
        nbr_existence = nbr_existence[:int(nbr_existence.shape[0] * sample_rate), ]

    split_line1 = int(input_TOTAL.shape[0] * 0.8)

    training_input = input_TOTAL[:split_line1, ]
    training_supplement = supplement[:split_line1, ]
    training_nbr_existence = nbr_existence[:split_line1, ]

    test_input = input_TOTAL[split_line1:]
    testing_supplement = supplement[split_line1:]
    testing_nbr_existence = nbr_existence[split_line1:]

    print('\n', "#######数据划分结束，打印划分结果#########")
    print("training_input.shape", training_input.shape)
    print("training_supplement.shape", training_supplement.shape)
    print("training_nbr_existence.shape", training_nbr_existence.shape)

    print("test_input.shape", test_input.shape)
    print("testing_supplement.shape", testing_supplement.shape)
    print("testing_nbr_existence.shape", testing_nbr_existence.shape)

    folder_path = 'data/HighD/rep_sample/'
    np.save(folder_path + 'training_input.npy', training_input)
    np.save(folder_path + 'training_supplement.npy', training_supplement)
    np.save(folder_path + 'training_nbr_existence.npy', training_nbr_existence)
    np.save(folder_path + 'test_input.npy', test_input)
    np.save(folder_path + 'testing_supplement.npy', testing_supplement)
    np.save(folder_path + 'testing_nbr_existence.npy', testing_nbr_existence)

def dir_load_data():

    folder_path = 'E:/Pycharm Project/Goal-generation-TUTR-V4/data/HighD/rep_sample/' #Ngsim or HighD
    # 加载 .npz 文件，并将变量名改为 processed_data
    processed_data = np.load(folder_path + 'processed_trajectories_data/' + 'processed_trajectories_data_compressed.npz')
    # 通过键值对提取保存的数组
    training_input = processed_data['training_input']
    training_supplement = processed_data['training_supplement']
    training_nbr_existence = processed_data['training_nbr_existence']
    # training_nbr_existence_time = processed_data['training_nbr_existence_time']
    training_rele_current_trajectories = processed_data['training_rele_current_trajectories']
    training_rele_neighbor_trajectories = processed_data['training_rele_neighbor_trajectories']
    training_first_point = processed_data['training_first_point']

    test_input = processed_data['test_input']
    testing_supplement = processed_data['testing_supplement']
    testing_nbr_existence = processed_data['testing_nbr_existence']
    # testing_nbr_existence_time = processed_data['testing_nbr_existence_time']
    testing_rele_current_trajectories = processed_data['testing_rele_current_trajectories']
    testing_rele_neighbor_trajectories = processed_data['testing_rele_neighbor_trajectories']
    testing_first_point = processed_data['testing_first_point']

    inference_input = processed_data['inference_input']
    inference_supplement = processed_data['inference_supplement']
    inference_nbr_existence = processed_data['inference_nbr_existence']
    # inference_nbr_existence_time = processed_data['inference_nbr_existence_time']
    inference_rele_current_trajectories = processed_data['inference_rele_current_trajectories']
    inference_rele_neighbor_trajectories = processed_data['inference_rele_neighbor_trajectories']
    inference_first_point = processed_data['inference_first_point']

    feature_destd = processed_data['features_min_max']

    print("training_input.shape", training_input.shape)
    print("training_supplement.shape", training_supplement.shape)
    print("training_nbr_existence.shape", training_nbr_existence.shape)
    # print("training_nbr_existence_time.shape", training_nbr_existence_time.shape)
    print("training_rele_current_trajectories.shape", training_rele_current_trajectories.shape)
    print("training_rele_neighbor_trajectories.shape", training_rele_neighbor_trajectories.shape)
    print("training_first_point.shape", training_first_point.shape)

    print("test_input.shape", test_input.shape)
    print("testing_supplement.shape", testing_supplement.shape)
    print("testing_nbr_existence.shape", testing_nbr_existence.shape)
    # print("testing_nbr_existence_time.shape", testing_nbr_existence_time.shape)
    print("testing_rele_current_trajectories.shape", testing_rele_current_trajectories.shape)
    print("testing_rele_neighbor_trajectories.shape", testing_rele_neighbor_trajectories.shape)
    print("testing_first_point.shape", testing_first_point.shape)

    print("inference_input.shape", inference_input.shape)
    print("inference_supplement.shape", inference_supplement.shape)
    print("inference_nbr_existence.shape", inference_nbr_existence.shape)
    # print("inference_nbr_existence_time.shape", inference_nbr_existence_time.shape)
    print("inference_rele_current_trajectories.shape", inference_rele_current_trajectories.shape)
    print("inference_rele_neighbor_trajectories.shape", inference_rele_neighbor_trajectories.shape)
    print("inference_first_point.shape", inference_first_point.shape)

    return training_input, training_supplement, training_nbr_existence, training_rele_current_trajectories, training_rele_neighbor_trajectories, training_first_point, \
        test_input, testing_supplement, testing_nbr_existence, testing_rele_current_trajectories, testing_rele_neighbor_trajectories, testing_first_point, \
        inference_input, inference_supplement, inference_nbr_existence, inference_rele_current_trajectories, inference_rele_neighbor_trajectories, inference_first_point,\
        feature_destd


if __name__ == '__main__':
    load_data(0.05)