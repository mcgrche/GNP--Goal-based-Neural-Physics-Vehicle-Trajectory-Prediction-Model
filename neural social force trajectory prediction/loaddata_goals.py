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


def organize_data(TrajWithNbr):
    new_traj_data = []
    nbr_existence = []  # 记录每条序列的邻居存在情况
    num_seq, seq_len, feat_sum = TrajWithNbr.shape

    # 遍历每个序列
    for i in range(num_seq):
        sequence = TrajWithNbr[i]  # (40, 37)
        current_vehicle_feats = sequence[:, :4]
        new_traj_data.append(current_vehicle_feats)

        # 为当前序列记录邻居存在情况
        current_nbr_existence = np.zeros(8, dtype=int)

        # 检查并添加存在的邻居车辆的序列
        for nbr_idx in range(8):  # 对于每个邻居
            nbr_start_idx = 5 + nbr_idx * 4
            nbr_feats = sequence[:, nbr_start_idx:nbr_start_idx + 4]

            # 如果邻居的特征不全为0，则认为邻居存在
            if not np.all(nbr_feats == 0):
                new_traj_data.append(nbr_feats)
                current_nbr_existence[nbr_idx] = 1  # 标记邻居存在

        nbr_existence.append(current_nbr_existence)

    # 将list转换为numpy数组
    new_traj_data_np = np.array(new_traj_data)
    nbr_existence_np = np.array(nbr_existence)

    return new_traj_data_np, nbr_existence_np

def load_data(sample_rate = 0):
    # 加载TrajWithNbr数据，[num_seq, seq_len, feat_sum](71032, 40, 37)
    TrajWithNbrs = np.load('TrajWithNbrs.npy')
    print(TrajWithNbrs.shape)

    # 将nei整理出来，生成一个没有nei的数据 [num_seq, seq_len, feat_sum==4]
    input_TOTAL, nbr_existence = organize_data(TrajWithNbrs)

    # 拆分为train和test两个部分
    seed = 1120  # 设置随机种子
    random.seed(seed)
    random.shuffle(input_TOTAL)

    if sample_rate:
        input_TOTAL = input_TOTAL[:int(input_TOTAL.shape[0] * sample_rate), ]

    split_line1 = int(input_TOTAL.shape[0] * 0.8)

    training_input = input_TOTAL[:split_line1, ]
    test_input = input_TOTAL[split_line1:]
    print('\n', "#######数据划分结束，打印划分结果#########")
    print("training_input.shape", training_input.shape)
    print("training_target.shape", test_input.shape)

    return training_input, test_input

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

    return training_input, training_supplement, training_nbr_existence, training_rele_current_trajectories, \
        test_input, testing_supplement, testing_nbr_existence, testing_rele_current_trajectories, \
        inference_input, inference_supplement, inference_nbr_existence, inference_rele_current_trajectories, \
        feature_destd
