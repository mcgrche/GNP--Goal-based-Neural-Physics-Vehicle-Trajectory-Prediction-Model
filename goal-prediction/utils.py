
import numpy as np
from sklearn.cluster import KMeans
import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import random

def translation(seq, refer_point_index):  # rigid transformation
    # seq [N T 2]
    return seq - seq[:, refer_point_index:refer_point_index+1]

def get_rot_mats(thea_list):
    num = thea_list.shape[0]
    rot_mat_list = []
    for i in range(num):
        thea = thea_list[i]
        rot_mat_list.append(np.array([[np.cos(thea), -np.sin(thea)],
                                          [np.sin(thea), np.cos(thea)]]))
    return np.stack(rot_mat_list, axis=0)

def rotation(seq, refer_point_index=0):  # rigid transformation

    # seq [N T 2]
    angle = np.arctan2(seq[:, refer_point_index, 1], seq[:, refer_point_index, 0])
    rot_mat = get_rot_mats(angle)
    rot_seq = np.matmul(seq, rot_mat)

    return rot_seq, rot_mat

def simple_moving_average(input, windows_size):
    
    # input [N T 2]
    
    x_len = input.shape[1]
    x_list = []
    keep_num = int(np.floor(windows_size / 2))
    for i in range(windows_size):
        x_list.append(input[:, i:x_len-windows_size+1+i])
    x = sum(x_list) / windows_size
    x = np.concatenate((input[:, :keep_num], x, input[:, -keep_num:]), axis=1)
    return x

def dy_random_rotation(seqs):

    # seqs [N T 2]
    random_angle = -1 + 2 * np.random.rand(seqs.shape[0])
    random_angle = np.arcsin(random_angle)
    rot_mat = get_rot_mats(random_angle)
    rot_seq = np.matmul(seqs, rot_mat)
    return rot_seq

def kmeans_(seq, n_clusters=100):
    
    # seq [N T 2] T=pred_len

    input_data = seq.reshape(seq.shape[0], -1)
    # input_data = seq[:, -1] # destination
    clf = KMeans(n_clusters=n_clusters,
                 random_state=1, n_init=10
                 ).fit(input_data)

    centers = clf.cluster_centers_
    centers = centers.reshape(centers.shape[0], -1, 2)

    return centers

def trajectory_motion_modes(all_trajs, obs_len, n_units=120, smooth_size=3, random_rotation=False):

    # full_ego_trajs [B T 2]

    clustering_input = all_trajs[:, obs_len:]
    # clustering_input = all_trajs
    if smooth_size is not None:
        clustering_input = simple_moving_average(clustering_input, windows_size=smooth_size)
    if random_rotation:
        clustering_input = dy_random_rotation(clustering_input)
    motion_modes = kmeans_(clustering_input, n_units)
    return motion_modes


def get_motion_modes(dataset, obs_len, pred_len, n_clusters, dataset_path, dataset_name, smooth_size, random_rotation, traj_seg=False):
    trajs_list = []
    index1 = [0, 1, 2, 3, 4, 5]  # make full use of training data
    traj_scenarios = dataset.scenario_list
    for i in range(len(traj_scenarios)):
        curr_ped_obs = traj_scenarios[i][0][:, :2]
        curr_ped_pred = traj_scenarios[i][1]
        curr_traj = np.concatenate((curr_ped_obs, curr_ped_pred), axis=0)  # T 2
        if traj_seg:
            for i in index1:
                seq = curr_traj[i:i + pred_len + 2]
                pre_seq = np.repeat(seq[0:1], obs_len + pred_len - seq.shape[0], axis=0)
                seq = np.concatenate((pre_seq, seq), axis=0)
                trajs_list.append(seq)
        trajs_list.append(curr_traj)
    
    all_trajs = np.stack(trajs_list, axis=0) # [B T 2]
    all_trajs = translation(all_trajs, obs_len-1)
    all_trajs, _ = rotation(all_trajs, 0)
    motion_modes = trajectory_motion_modes(all_trajs, obs_len, n_units=n_clusters, 
                                      smooth_size=smooth_size, random_rotation=random_rotation) # motion_modes其实就是聚类的centers（100个）
    
    if not os.path.exists(dataset_path): 
        os.makedirs(dataset_path)
    save_path_file = dataset_path + dataset_name + '_motion_modes.pkl'
    f = open(save_path_file, 'wb')
    pickle.dump(motion_modes, f)
    f.close()
    print('Finished')

    return motion_modes

# 获取HighD车辆轨迹的motion modes
def get_motion_modes_veh(dataset, obs_len, pred_len, n_clusters, dataset_path, dataset_name, file_suffix, smooth_size, random_rotation,
                     traj_seg=False):
    data_agent, data_neis, _, _ = dataset.tensors
    # data_neis = data_neis.reshape(data_neis.size(0) * data_neis.size(1), data_neis.size(2), data_neis.size(3))
    # all_trajs =torch.cat((data_agent[:, :, 0:2], data_neis[:, :, 0:2]), dim=0)
    all_trajs = data_agent[:, :, 0:2]

    # 并不需要进行translation和rotation,preprocessing时已经对称+相对位置了
    # 每条轨迹已经相对seq的第一个点进行相对位置转换，不需要再对第obs_len个点做相对位置转换了
    # all_trajs = translation(all_trajs, obs_len - 1)
    # 车辆的方向只有向左或者向右两种，在处理数据时进行对称处理
    # all_trajs, _ = rotation(all_trajs, obs_len - 1)

    # 对数据的obs_len部分进行聚类
    motion_modes = trajectory_motion_modes(all_trajs, obs_len, n_units=n_clusters,
                                           smooth_size=smooth_size, random_rotation=random_rotation)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    save_path_file = dataset_path + dataset_name + file_suffix
    f = open(save_path_file, 'wb')
    pickle.dump(motion_modes, f)
    f.close()
    print('Finished')

    return motion_modes

# saving motion modes and closest motion modes
def saving_motion_modes(dataloader, motion_modes, obs_len, dataset_path, dataset_name):

    # motion_modes [K pred_len 2]
    closest_mode_indices_list = []
    cls_soft_label_list = []
    traj_scenes = dataloader.seq_array

    for i in range(traj_scenes.shape[0]):
        curr_scene = traj_scenes[i] # N T 2
        curr_traj = curr_scene[0:1]  # [1 T 2]
        norm_curr_traj = translation(curr_traj, obs_len-1)
        norm_curr_traj, _ = rotation(norm_curr_traj, 0)
        norm_curr_traj = norm_curr_traj[:, obs_len:]
        norm_curr_traj = norm_curr_traj.reshape(1, -1) #[1 pred_len*2]
        norm_curr_traj = np.repeat(norm_curr_traj, motion_modes.shape[0], axis=0) # [K pred_len*2]
        traj_units_ = motion_modes.reshape(motion_modes.shape[0], -1) # [K pred_len*2]
        distance = np.linalg.norm(norm_curr_traj - traj_units_, axis=-1) # [K]
        closest_unit_indices = np.argmin(distance)
        closest_unit_indices = np.expand_dims(closest_unit_indices, axis=0)
        closest_mode_indices_list.append(closest_unit_indices)
        cls_soft_label_list.append(-distance)
       
    closest_mode_indices_array = np.concatenate(closest_mode_indices_list, axis=0)
    cls_soft_label_array = np.stack(cls_soft_label_list, axis=0)

    np.save(dataset_path+dataset_name+'_motion_modes.npy', motion_modes)
    np.save(dataset_path+dataset_name+'_closest_mode_indices.npy', closest_mode_indices_array)
    np.save(dataset_path+dataset_name+'_cls_soft_label.npy', cls_soft_label_array)


def seed(seed: int):
    rand = seed is None
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = not rand
    torch.backends.cudnn.benchmark = rand


def get_rng_state(device):
    return (
        torch.get_rng_state(), 
        torch.cuda.get_rng_state(device) if torch.cuda.is_available and "cuda" in str(device) else None,
        np.random.get_state(),
        random.getstate(),
        )


def create_look_ahead_mask(size):
    # 创建一个上三角矩阵，对角线上方都是1，下方包括对角线都是0
    mask = torch.ones(size, size).triu(diagonal=1)
    # 使用一个非常大的负数来替代1，因为softmax后这些位置将接近于0
    mask = mask.masked_fill(mask == 1, float('-inf'))

    return mask

def calculate_best_match_error(pred_trajs, gt_trajs):
    """
    计算最佳匹配预测目的地与真实目的地之间的误差。
    :param pred_trajs: 预测的目的地，形状为 [B, num_k, 1, 2]
    :param gt_trajs: 真实的目的地，形状为 [B, 1, 2]
    :return: 最佳匹配误差的平均值
    """
    B, num_k, _ = pred_trajs.shape
    # 计算每个预测与真实目的地之间的欧几里得距离
    distances = torch.sqrt(((pred_trajs - gt_trajs.unsqueeze(1)) ** 2).sum(dim=-1))
    # 选择每个轨迹的最小距离
    min_distances = distances.min(dim=1)[0]
    # 计算平均最佳匹配误差
    avg_error = min_distances.mean().item()
    return avg_error

def calculate_coverage(pred_trajs, gt_trajs, threshold=2.0):
    """
    计算预测目的地覆盖真实目的地的比例。
    :param pred_trajs: 预测的目的地，形状为 [B, num_k, 1, 2]
    :param gt_trajs: 真实的目的地，形状为 [B, 1, 2]
    :param threshold: 覆盖的距离阈值
    :return: 覆盖率
    """
    distances = torch.sqrt(((pred_trajs - gt_trajs.unsqueeze(1)) ** 2).sum(dim=-1))
    coverage = (distances < threshold).any(dim=1).float().mean().item()
    return coverage

def show_loss(train_loss_record, test_loss_record, inference_loss):

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'gallery/training_test_loss_{current_time}.png'

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_record, label='Training Loss')
    plt.plot(test_loss_record, label='Test Loss')
    # 将inference的结果画成一条直线
    plt.axhline(y=inference_loss, color='r', linestyle='-', label='Inference Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def show_motion_modes(motion_modes, train_dataset):
    data_agent, data_neis, _ = train_dataset.tensors
    obs_data = data_agent[:, :]

def calculate_rmse(pred_trajs_denorm, gt_denorm):
    # 求每个预测点与真实点之间的平方差
    squared_errors = (pred_trajs_denorm - gt_denorm) ** 2
    # 按时间步求平均，然后求所有样本的平均
    mse = torch.mean(squared_errors)  # MSE
    rmse = torch.sqrt(mse)  # RMSE
    return rmse.item()