import time
import os
import numpy as np
import copy
import torch
import torch.nn as nn
import zipfile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def calculate_v(x_seq):
    length = x_seq.shape[1]
    peds = x_seq.shape[0]
    x_seq_velocity = np.zeros_like(x_seq)
    episa = 1e-6
    for i in range(1, length):
        for j in range(peds):
            position = x_seq[j][i]
            before_position = x_seq[j][i-1]
            position_norm = np.linalg.norm(position) #计算L2范数，即通过x y求距离
            before_position_norm = np.linalg.norm(before_position)
            if position_norm < episa:
                velocity = np.array([0,0])
            else:
                if before_position_norm < episa:
                    velocity = np.array([0, 0])
                else:
                    velocity = (position - before_position)/0.4 #计算速度，为什么是0.4？是时间步长吗
            x_seq_velocity[j][i] = velocity
    return x_seq_velocity

def translation(x_seq):
    first_frame = x_seq[:, 0, :]
    first_frame_new = first_frame[:, np.newaxis, :] #peds*1*2
    x_seq_translated = x_seq - first_frame_new
    return x_seq_translated

def augment_data(data):
    ks = [1, 2, 3]
    data_ = copy.deepcopy(data)  # data without rotation, used so rotated data can be appended to original df
    #k2rot = {1: '_rot90', 2: '_rot180', 3: '_rot270'}
    for k in ks:
        for t in range(len(data)):
            data_rot = rot(data[t], k)
            data_.append(data_rot)
    for t in range(27*4):
        data_flip = fliplr(data_[t])

        data_.append(data_flip)

    return data_

def rot(data_traj, k=1):
    xy = data_traj.copy()

    c, s = np.cos(-k * np.pi / 2), np.sin(-k * np.pi / 2)
    R = np.array([[c, s], [-s, c]])
    for i in range(20):
        xy[:, i, :] = np.dot(xy[:, i, :], R)

    return xy

def fliplr(data_traj):
    xy = data_traj.copy()

    R = np.array([[-1, 0], [0, 1]])
    for i in range(20):
        xy[:, i, :] = np.dot(xy[:, i, :], R)

    return xy

def calculate_loss(criterion, future, predictions):

    ADL_traj = criterion(future, predictions)  # better with l2 loss

    return ADL_traj

def calculate_loss_cvae(mean, log_var, criterion, future, predictions):
    # reconstruction loss
    ADL_traj = criterion(future, predictions) # better with l2 loss

    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())



    return KLD, ADL_traj
def translation_goals(goals, x_seq):
    first_frame = x_seq[:, 0, :]
    goals_translated = goals - first_frame #peds*2
    return goals_translated

# Custom initialization function with Kaiming initialization
def initialize_parameters_like(original_params):
    """
    Initializes new parameters with the same shape as the original model's parameters.
    Uses Kaiming initialization (He initialization) for weights and zeros for bias.
    """
    new_params = {}
    for key, param in original_params.items():
        if param.requires_grad:
            # Create new parameters with the same shape as the original
            new_param = torch.empty_like(param)
            if 'weight' in key:
                # Kaiming uniform initialization
                nn.init.kaiming_uniform_(new_param, nonlinearity='relu')
            elif 'bias' in key:
                # Initialize bias to zeros
                nn.init.zeros_(new_param)
            new_params[key] = new_param
        else:
            # Non-trainable parameters retain their original values
            new_params[key] = param
    return new_params

def select_para(model_complete):
    # selected parameters应该是指第二个网络（collision network）中的22(后一半)相关参数，只对这些参数进行优化
    params_totrain = []
    params_totrain.extend(model_complete.cell2.parameters())
    params_totrain.extend(model_complete.input_embedding_layer2.parameters())
    params_totrain.extend(model_complete.output_layer2.parameters())
    params_totrain.extend(model_complete.encoder_people_state.parameters())
    params_totrain.extend(model_complete.dec_para_people.parameters())
    return params_totrain

def new_point(checkpoint_t_dic, checkpoint_i_dic):
    point = checkpoint_i_dic
    dk_t = list(checkpoint_t_dic.keys())
    dk_i = list(point.keys())
    for k in range(22):
        point[dk_i[k]] = checkpoint_t_dic[dk_t[k]]
    return point    

def print_model_state_dict(model_state_dict):
    """
    Prints out the model's state_dict parameters including names and values.
    """
    total_keys = 0
    for name, param in model_state_dict.items():
        total_keys += 1
        print(f"Parameter Name: {name}")
        # print(f"Parameter Value: {param}")
        print(f"Shape: {param.shape}")
        print("-" * 80)
    print(f"Total Number of Parameters (Keys): {total_keys}")

def translation_supp(supplemnt, x_seq):
    first_frame = x_seq[:, 0, :]
    for ped in range(supplemnt.shape[0]):
        for frame in range(20):
            all_other_peds = int(supplemnt[ped, frame, -1, 1])
            supplemnt[ped, frame, :all_other_peds, :2] = supplemnt[ped, frame, :all_other_peds, :2] - first_frame[ped,:]
    return supplemnt

def nan_test(tensor):
    return torch.isnan(tensor).any().item()

def line_dis(traj ,upper_lane_marking, lower_lane_marking):
    """

    :param traj: 当前车辆的轨迹特征 [num_seq, num_len==40, num_feat==4]
    :param upper_lane_marking: 上方车道线y坐标 [8.51, 12.59, 16.43]
    :param lower_lane_marking: 下方车道线y坐标 [21.00, 24.96, 28.80]

    :return: traj_wdis [num_seq, num_len, num_feat+3]
    """
    initial_lane_ids = traj[:, 0, 4]
    belongs_to_upper = np.isin(initial_lane_ids, [2, 3])
    belongs_to_lower = np.isin(initial_lane_ids, [5, 6])

    y_coordinates = traj[:, :, 1]
    upper_distances = np.abs(y_coordinates[..., None] - np.array(upper_lane_marking))
    lower_distances = np.abs(y_coordinates[..., None] - np.array(lower_lane_marking))

    # Use broadcasting to create masks for entire batches
    upper_mask = belongs_to_upper[:, None, None]
    lower_mask = belongs_to_lower[:, None, None]

    # Use np.where to select distances for the whole batch without looping
    distances = np.where(upper_mask, upper_distances, np.zeros_like(upper_distances))
    distances += np.where(lower_mask, lower_distances, np.zeros_like(lower_distances))

    # Concatenate the original traj array with the distances array along the last dimension
    traj_wdis = np.concatenate([traj, distances], axis=-1)

    return traj_wdis

def manual_mse_loss(prediction, target):
    # 计算两个张量之间的差值
    diff = prediction - target
    # 计算差值的平方
    squared_diff = diff ** 2
    # 计算所有元素的均值
    mean_squared_error = torch.mean(squared_diff)
    return mean_squared_error


# 解析lane description并生成lane markings的字典
def parse_lane_data(lane_data):
    lane_markings = {}
    for i, row in lane_data.iterrows():
        location_id = row['Location ID']
        upper_lane_markings = [float(x) for x in row['upperLaneMarkings'].split(';')]
        lower_lane_markings = [float(x) for x in row['lowerLaneMarkings'].split(';')]
        lane_description = row['lane description']
        upper_lanes, lower_lanes = lane_description.split('/')
        upper_lanes = [int(x) for x in upper_lanes.split(',')]
        lower_lanes = [int(x) for x in lower_lanes.split(',')]
        lane_markings[location_id] = {
            'upper_lanes': upper_lanes,
            'lower_lanes': lower_lanes,
            'upper_lane_markings': upper_lane_markings,
            'lower_lane_markings': lower_lane_markings
        }
    return lane_markings


# 计算车辆与其他车道线的距离，并填充到相同的长度
def calculate_lane_distances_batch(traj_matrix, lane_markings, delta, block_size=1000):
    """
    计算车辆与当前方向所有车道线之间的距离，并计算车辆偏移 delta 后的车道线距离。
    支持分块处理大数组以节省内存。
    :param traj_matrix: 轨迹矩阵 (num_sequence, seq_length, num_feature)，其中包含laneid和locationid
    :param lane_markings: 各location的lane markings字典
    :param delta: y方向上的车辆偏移量
    :param block_size: 每次处理的块大小
    :return: 计算出的车辆与车道线的距离，以及偏移 delta 后的距离。返回结果包含当前和偏移后的距离。
    """
    num_sequence, seq_length, num_feature = traj_matrix.shape

    # 假设特征的顺序为：x, y, vx, vy, laneid, locationid, real_y
    laneid_idx = 4
    locationid_idx = 5
    real_y_idx = 6

    # 计算单方向车道线的最大数量（即不叠加上下方向）
    max_lanes = max([max(len(lane_markings[loc]['upper_lane_markings']),
                         len(lane_markings[loc]['lower_lane_markings']))
                     for loc in lane_markings.keys()])

    # 初始化一个数组用于存储车辆与车道线的距离和偏移后的距离
    all_distances = np.full((num_sequence, seq_length, max_lanes, 2), 0.0, dtype=np.float32)  # 当前距离和偏移距离

    # 分块处理
    for block_start in range(0, num_sequence, block_size):
        block_end = min(block_start + block_size, num_sequence)
        block_range = range(block_start, block_end)

        for seq_idx in block_range:
            locationid = int(traj_matrix[seq_idx, 0, locationid_idx])  # 轨迹中所有帧的locationid相同
            laneid = int(traj_matrix[seq_idx, 0, laneid_idx])  # 轨迹中所有帧的laneid相同

            # 根据 laneid 判断使用 upper lane markings 还是 lower lane markings
            if laneid in lane_markings[locationid]['upper_lanes']:
                current_lane_markings = lane_markings[locationid]['upper_lane_markings']
            else:
                current_lane_markings = lane_markings[locationid]['lower_lane_markings']

            # 遍历每一帧，计算与车道线的距离和偏移后的距离
            for t in range(seq_length):
                real_y = traj_matrix[seq_idx, t, real_y_idx]

                # 计算与所有车道线的距离
                distances = [abs(real_y - lane_marking) for lane_marking in current_lane_markings]

                # 计算车辆偏移 delta 后的 y 坐标
                shifted_real_y = real_y + delta
                shifted_distances = [abs(shifted_real_y - lane_marking) for lane_marking in current_lane_markings]

                # 将结果填入 all_distances
                all_distances[seq_idx, t, :len(distances), 0] = distances  # 当前距离
                all_distances[seq_idx, t, :len(shifted_distances), 1] = shifted_distances  # 偏移后的距离

    return all_distances # (num_sequence, seq_length, max_lanes, 2),第四个维度的 0 表示当前车辆与车道线的距离。第四个维度的 1 表示车辆偏移 delta 后的距离


def calculate_lane_distances_and_factors(traj_matrix, lane_markings, delta, mode='train', sigma_lines=1.22, block_size=1000,
                                         device='cpu', output_dir='data/HighD'):
    """
    计算车辆与当前方向所有车道线之间的距离、车辆偏移 delta 后的车道线距离，
    并计算 clines 和 boundary 的势场固定部分。
    支持分块处理大数组以节省内存。

    :param traj_matrix: 轨迹矩阵 (num_sequence, seq_length, num_feature)，其中包含 laneid 和 locationid
    :param lane_markings: 各 location 的 lane markings 字典
    :param delta: y方向上的车辆偏移量
    :param sigma_lines: 用于 clines 势场的固定参数 sigma
    :param block_size: 每次处理的块大小
    :param device: 设备（CPU 或 GPU）
    :return: boundary_factors 和 clines_factors，两个独立的张量
    """
    full_start = time.time()
    epsilon = 1e-7  # 防止除零错误
    large_value = 1e6  # 用于填充缺失的 clines 值
    num_sequence, seq_length, num_feature = traj_matrix.shape

    traj_matrix = torch.tensor(traj_matrix, dtype=torch.float32, device=device)

    # 假设特征的顺序为：x, y, vx, vy, laneid, locationid, real_y
    laneid_idx = 4
    locationid_idx = 5
    real_y_idx = 6

    # 计算单方向车道线的最大数量
    max_lanes = max([max(len(lane_markings[loc]['upper_lane_markings']),
                         len(lane_markings[loc]['lower_lane_markings']))
                     for loc in lane_markings.keys()])

    # 初始化 Tensor，用于存储车辆与车道线的距离和偏移后的距离
    all_distances = torch.zeros((num_sequence, seq_length, max_lanes, 2), dtype=torch.float32, device=device)

    # 初始化存储 boundary 和 clines 的结果
    boundary_dist_current_list = []
    boundary_dist_shifted_list = []
    clines_list_current = []
    clines_list_shifted = []

    # 分块处理数据
    for block_start in range(0, num_sequence, block_size):
        block_end = min(block_start + block_size, num_sequence)
        block_range = range(block_start, block_end)
        block_start_time = time.time()
        for seq_idx in block_range:
            locationid = int(traj_matrix[seq_idx, 0, locationid_idx].item())  # 轨迹中所有帧的locationid相同
            laneid = int(traj_matrix[seq_idx, 0, laneid_idx].item())  # 轨迹中所有帧的laneid相同

            # 根据 laneid 判断使用 upper lane markings 还是 lower lane markings
            if laneid in lane_markings[locationid]['upper_lanes']:
                current_lane_markings = lane_markings[locationid]['upper_lane_markings']
            else:
                current_lane_markings = lane_markings[locationid]['lower_lane_markings']

            current_lane_markings_tensor = torch.tensor(current_lane_markings, dtype=torch.float32, device=device)

            # 计算 real_y 和 shifted_real_y
            real_y = traj_matrix[seq_idx, :, real_y_idx].unsqueeze(-1)  # (seq_length, 1)
            shifted_real_y = real_y + delta  # (seq_length, 1)

            # 扩展 current_lane_markings 维度以便计算距离
            current_lane_markings_expanded = current_lane_markings_tensor.unsqueeze(0).expand(seq_length,
                                                                                              -1)  # (seq_length, max_lanes)

            # 计算当前距离和偏移后的距离
            distances = torch.abs(real_y - current_lane_markings_expanded)  # (seq_length, max_lanes)
            shifted_distances = torch.abs(shifted_real_y - current_lane_markings_expanded)  # (seq_length, max_lanes)

            # 将结果填入 all_distances
            all_distances[seq_idx, :, :current_lane_markings_tensor.size(0), 0] = distances  # 当前距离
            all_distances[seq_idx, :, :current_lane_markings_tensor.size(0), 1] = shifted_distances  # 偏移后的距离

            # 计算 boundary: 第一个非零值和最后一个非零值
            boundary_dist_current_seq = []
            boundary_dist_shifted_seq = []
            clines_dist_current_seq = []
            clines_dist_shifted_seq = []

            for t in range(seq_length):
                line_curr = distances[t, :] + epsilon
                line_shift = shifted_distances[t, :] + epsilon

                # 找出 boundary
                non_zero_curr = line_curr.nonzero(as_tuple=True)[0]
                non_zero_shift = line_shift.nonzero(as_tuple=True)[0]

                if len(non_zero_curr) > 2:
                    boundary_dist_current_seq.append(torch.tensor([line_curr[non_zero_curr[0]].item(),
                                                                   line_curr[non_zero_curr[-1]].item()]))
                else:
                    print("ERROR: number of lane markings < 3")  # 没有非零值时用极大值填充

                if len(non_zero_shift) > 2:
                    boundary_dist_shifted_seq.append(torch.tensor([line_shift[non_zero_shift[0]].item(),
                                                                   line_shift[non_zero_shift[-1]].item()]))
                else:
                    print("ERROR: number of lane markings < 3")  # 没有非零值时用极大值填充

                # 找出 clines 并补全至 3 个值
                clines_current = line_curr[non_zero_curr[1:-1]] if len(non_zero_curr) > 2 else torch.tensor([])
                clines_shifted = line_shift[non_zero_shift[1:-1]] if len(non_zero_shift) > 2 else torch.tensor([])

                clines_current_padded = torch.cat(
                    [clines_current, torch.full((3 - clines_current.size(0),), large_value, device=device)])
                clines_shifted_padded = torch.cat(
                    [clines_shifted, torch.full((3 - clines_shifted.size(0),), large_value, device=device)])

                clines_dist_current_seq.append(clines_current_padded)
                clines_dist_shifted_seq.append(clines_shifted_padded)

            # 堆叠每一帧的结果
            boundary_dist_current_list.append(torch.stack(boundary_dist_current_seq))
            boundary_dist_shifted_list.append(torch.stack(boundary_dist_shifted_seq))
            clines_list_current.append(torch.stack(clines_dist_current_seq))
            clines_list_shifted.append(torch.stack(clines_dist_shifted_seq))

        block_end_time = time.time()
        print("一个block的计算时长：", block_end_time - block_start_time, "s")
    # 将所有序列的结果堆叠为 Tensor
    boundary_dist_current = torch.stack(boundary_dist_current_list) + epsilon
    boundary_dist_shifted = torch.stack(boundary_dist_shifted_list) + epsilon
    clines_dist_current = torch.stack(clines_list_current) + epsilon
    clines_dist_shifted = torch.stack(clines_list_shifted) + epsilon

    # 计算 clines 和 boundary 的势场固定部分
    clines_factor_current = torch.exp(-torch.pow(clines_dist_current, 2) / (2 * sigma_lines ** 2))
    boundary_factor_current = 1 / torch.pow(boundary_dist_current, 2)

    clines_factor_shifted = torch.exp(-torch.pow(clines_dist_shifted, 2) / (2 * sigma_lines ** 2))
    boundary_factor_shifted = 1 / torch.pow(boundary_dist_shifted, 2)

    # 根据 mode 保存不同的文件名
    torch.save(boundary_factor_current, os.path.join(output_dir, f'{mode}_boundary_factor_current.pt'))
    torch.save(clines_factor_current, os.path.join(output_dir, f'{mode}_clines_factor_current.pt'))
    torch.save(boundary_factor_shifted, os.path.join(output_dir, f'{mode}_boundary_factor_shifted.pt'))
    torch.save(clines_factor_shifted, os.path.join(output_dir, f'{mode}_clines_factor_shifted.pt'))

    # 压缩文件
    zip_filename = os.path.join(output_dir, f'{mode}_lane_factors.zip')
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(os.path.join(output_dir, f'{mode}_boundary_factor_current.pt'), f'{mode}_boundary_factor_current.pt')
        zipf.write(os.path.join(output_dir, f'{mode}_clines_factor_current.pt'), f'{mode}_clines_factor_current.pt')
        zipf.write(os.path.join(output_dir, f'{mode}_boundary_factor_shifted.pt'), f'{mode}_boundary_factor_shifted.pt')
        zipf.write(os.path.join(output_dir, f'{mode}_clines_factor_shifted.pt'), f'{mode}_clines_factor_shifted.pt')

    # 打印总耗时
    full_end = time.time()
    print(f"calculate_lane_distances_and_factors ({mode}) 总计算时长：", full_end - full_start, " s")

    return zip_filename


# 加载预计算的势场数据
def load_precomputed_factors(mode, output_dir='data/HighD', device='cpu'):
    """
    加载之前保存的 train/test/inference 预计算势场文件
    :param mode: 'train', 'test', 'inference'
    :param output_dir: 保存文件的目录
    :return: boundary_factor_current, clines_factor_current, boundary_factor_shifted, clines_factor_shifted
    """
    boundary_factor_current = torch.load(os.path.join(output_dir, f'{mode}_boundary_factor_current.pt'))
    clines_factor_current = torch.load(os.path.join(output_dir, f'{mode}_clines_factor_current.pt'))
    boundary_factor_shifted = torch.load(os.path.join(output_dir, f'{mode}_boundary_factor_shifted.pt'))
    clines_factor_shifted = torch.load(os.path.join(output_dir, f'{mode}_clines_factor_shifted.pt'))

    return boundary_factor_current, clines_factor_current, boundary_factor_shifted, clines_factor_shifted


def visualize_trajectories_with_multiple_time_steps(params, traj, predictions, ade_min_idx, title, idx, neighbor_traj,
                                                    neighbor_existence, all_neighbor_factors, all_line_factors,
                                                    lane_markings, location_id, laneid):
    plot_start = time.time()

    # Define time intervals for visualization (0s, 1s, 2s, 3s, 4s, 5s)
    time_steps = [0, int(params['future_length'] * 0.2), int(params['future_length'] * 0.4), int(params['future_length'] * 0.6), int(params['future_length'] * 0.8), params['future_length'] - 1]  # 0s, 2s, 4s, and 5s
    time_labels = ["0s", "1s", "2s", "3s", "4s", "5s"]

    # Define a consistent marker size for all time steps
    consistent_marker_size = 60  # Keep a large, consistent size for better visibility
    # Define colors for different time steps
    marker_colors = ['#FF4500', '#FF6347', '#FFA07A', '#FFD700']  # Gradient colors for markers
    # Define colors for different neighbors (up to 8 neighbors)
    neighbor_colors = plt.cm.get_cmap('tab10', neighbor_traj.shape[1])
    # Adjust the figure size to ensure consistent width across all subplots
    fig = plt.figure(figsize=(12, 18))  # Adjusted to ensure the width remains the same for all subplots

    # Use gridspec with equal height ratios for all plots (5 total)
    gs = gridspec.GridSpec(7, 1, height_ratios=[1, 1, 1, 1, 1, 1, 1], hspace=0.3)

    # ----- Trajectory subplots (upper four plots) -----
    for subplot_idx, time_step in enumerate(time_steps):
        ax = fig.add_subplot(gs[subplot_idx])

        # Plot the past trajectory
        ax.plot(traj[:params['past_length'], 0], traj[:params['past_length'], 1], 'gray', label='Past Trajectory',
                linewidth=2)

        # Plot the future ground truth trajectory
        ax.plot(traj[params['past_length']:, 0], traj[params['past_length']:, 1], 'green', label='Future Ground Truth',
                linewidth=2)

        # Plot all predictions with low transparency, highlight best prediction
        for j in range(20):
            if j != ade_min_idx:
                ax.plot(predictions[j, :, 0], predictions[j, :, 1], '-', color='blue', alpha=0.2, label= 'Other Prediction' if j == 0 else "",linewidth=2)

        # Highlight the best prediction and add markers for the selected time step
        ax.plot(predictions[ade_min_idx, :, 0], predictions[ade_min_idx, :, 1], '-', color='red',
                label='Best Prediction',
                linewidth=3)
        ax.scatter(predictions[ade_min_idx, time_step, 0], predictions[ade_min_idx, time_step, 1],
                   s=consistent_marker_size, edgecolor='black', linewidth=2,  # Uniform size and black edge
                   color='y', alpha=1,
                   zorder=5)

        # Plot neighbor trajectories
        for neighbor_idx in range(neighbor_traj.shape[1]):
            if neighbor_existence[neighbor_idx] == 1:
                if time_step == 0:
                    # Get the past 10 frames ending at the current time (end of past trajectory)
                    start_frame = max(0, params['past_length'] - 10)
                    end_frame = params['past_length']
                else:
                    # For future steps, plot the 10 frames leading up to the current time step in the future
                    start_frame = max(params['past_length'], params['past_length'] + time_step - 10)
                    end_frame = params['past_length'] + time_step

                # Extract the segment of the neighbor trajectory
                neighbor_segment_x = neighbor_traj[start_frame:end_frame + 1, neighbor_idx, 0]
                neighbor_segment_y = neighbor_traj[start_frame:end_frame + 1, neighbor_idx, 1]

                # Define a huge value threshold to filter non-existing neighbors
                huge_value = 1e4

                # Replace non-existing neighbor positions with NaN to prevent lines from being drawn
                neighbor_segment_x[neighbor_segment_x >= huge_value] = np.nan
                neighbor_segment_y[neighbor_segment_y >= huge_value] = np.nan

                # Plot neighbor trajectory for the selected segment
                valid_mask_segment = ~np.isnan(neighbor_segment_x)
                ax.plot(neighbor_segment_x[valid_mask_segment], neighbor_segment_y[valid_mask_segment], '--',
                        color='lightseagreen', alpha=0.8,
                        label=f'Neighbor Past' if neighbor_idx == 0 else "")

                # Plot neighbor future position at the selected time step
                # if not np.isnan(neighbor_traj[end_frame, neighbor_idx, 0]) and not np.isnan(
                #         neighbor_traj[end_frame, neighbor_idx, 1]):
                #     ax.scatter(neighbor_traj[end_frame, neighbor_idx, 0], neighbor_traj[end_frame, neighbor_idx, 1],
                #                s=consistent_marker_size, color='lightseagreen', alpha=0.8,
                #                label=f'Neighbor {neighbor_idx + 1} (Future {time_labels[subplot_idx]})', zorder=5)

        # Plot lane markings
        if location_id in lane_markings:
            lane_data = lane_markings[location_id]
            lanes = lane_data['upper_lane_markings'] if laneid in lane_data['upper_lanes'] else lane_data[ 'lower_lane_markings']
            for i, lane_pos in enumerate(lanes):
                linestyle = '-' if i == 0 or i == len(lanes) - 1 else '--'
                ax.axhline(y=lane_pos, color='black' if linestyle == '-' else 'gray', linestyle=linestyle, linewidth=2)

        ax.set_xlim([0, 400])  # Ensure all plots share the same x-axis limits
        # ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect(5)  # Keep the x:y aspect ratio to 10 for clearer trajectory visualization
        if subplot_idx == 0:
            ax.legend()

        # Add the time label on the right side of each subplot
        ax.text(410, np.mean(ax.get_ylim()), f'{time_labels[subplot_idx]}', fontsize=12, verticalalignment='center')

    # ----- Forces plot (Bottom plot) -----
    ax_forces = fig.add_subplot(gs[6])  # Use the 5th grid spec for the forces plot (now same height as others)
    ax_forces.plot(traj[params['past_length']:, 0], all_neighbor_factors, color='purple', label='Neighbor Potential')
    ax_forces.plot(traj[params['past_length']:, 0], all_line_factors, color='cyan', label='Lane Potential')
    ax_forces.set_xlim([0, 400])  # Match x-axis limits with the upper plots
    ax_forces.set_xlabel('X (m)')
    ax_forces.set_ylabel('Potential Value')
    ax_forces.set_aspect('auto')  # Set the aspect ratio for the forces plot to auto
    # ax_forces.set_title('Forces Acting on the Vehicle')
    ax_forces.legend()

    # Save the combined plot
    os.makedirs('gallery/HighD/with_neighbors/selected_1002', exist_ok=True)  # Ensure directory exists
    plt.savefig(f'gallery/HighD/with_neighbors/selected_1002/{title}_{idx}.png', dpi=300, bbox_inches='tight')
    plt.close()

    plot_end = time.time()
    print(f"Scenarios plot for inference dataset idx {idx}: {plot_end - plot_start:.4f} seconds")





