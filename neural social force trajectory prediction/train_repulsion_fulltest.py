import time

import torch
import yaml
from model_repulsion import *
from utils import *
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import argparse
from loaddata_repulsion import *
from datetime import timedelta, datetime
import pandas as pd

lane_data = pd.DataFrame({
    'Location ID': [1, 2, 3, 4, 5, 6],
    'upperLaneMarkings': ["3.65;7.62;11.42;15.22", "8.51;12.59;16.43", "13.15;16.97;20.62;24.35",
                          "3.36;7.25;10.87;14.49", "7.68;11.77;15.61", "5.16;8.71;12.59;15.71;19.16"],
    'lowerLaneMarkings': ["20.88;24.86;28.48;32.54", "21.00;24.96;28.80", "26.35;30.09;33.65;37.56",
                          "19.35;22.97;26.60;30.66", "20.53;24.62;28.58", "23.92;27.20;30.57;34.55"],
    'lane description': ["2,3,4/6,7,8", "2,3/5,6", "2,3,4/6,7,8", "2,3,4/6,7,8", "2,3/5,6", "2,3,4,5/7,8,9"]
})

def train():

    model.train()
    total_loss = 0
    criterion = nn.MSELoss()
    total_batches = len(train_dataloader)

    total_start_time = time.time()

    for batch_idx, (batch_data, batch_supplement, batch_nbr_exist,
                    batch_boundary_factor_current, batch_clines_factor_current,
                    batch_boundary_factor_shifted, batch_clines_factor_shifted) in enumerate(train_dataloader):
        if batch_idx < total_batches-1:
            print("train batch idx: ", batch_idx)
            # start_time = time.time()  # 记录每个batch的开始时间

            # 针对lstm模型，把batch_size * num_nodes来和原human保持一致
            # data_proc_start = time.time()
            traj = batch_data # traj.shape [batch_size, 200, 7]
            # traj = traj.to(device)
            boundary_factor_current = batch_boundary_factor_current
            clines_factor_current = batch_clines_factor_current
            boundary_factor_shifted = batch_boundary_factor_shifted
            clines_factor_shifted = batch_clines_factor_shifted
            # line_dist = line_dist.to(device)
            supplement = batch_supplement
            # supplement = supplement.to(device)
            nbr_existence = batch_nbr_exist #.to(device)
            # data_proc_end = time.time()
            # print(f"Data processing time for batch {batch_idx}: {data_proc_end - data_proc_start:.4f} seconds")

            # physics_start = time.time()
            y = traj[:, params['past_length']:, :2] # batch_size * future_length * 2
            dest = y[:, -1, :] #.to(device) # num_frame * 2
            future = y.contiguous() #.to(device) # future是y的连续版本

            future_vel = (dest - traj[:, params['past_length'] - 1, :2]) / (torch.tensor(params['future_length']).to(device) * params['time_step']) #num_frame * 2
            future_vel_norm = torch.norm(future_vel, dim=-1)
            initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1) # num_frame * 1
            # physics_end = time.time()
            # print(f"Physics calculation time for batch {batch_idx}: {physics_end - physics_start:.4f} seconds")

            num_peds = traj.shape[0]
            numNodes = num_peds

            # forward_start = time.time()
            hidden_states1 = Variable(torch.zeros(numNodes, params['rnn_size'], device=device))
            cell_states1 = Variable(torch.zeros(numNodes, params['rnn_size'], device=device))
            # hidden_states1 = hidden_states1.to(device)
            # cell_states1 = cell_states1.to(device)
            hidden_states2 = Variable(torch.zeros(numNodes, params['rnn_size'], device=device))
            cell_states2 = Variable(torch.zeros(numNodes, params['rnn_size'], device=device))
            # hidden_states2 = hidden_states2.to(device)
            # cell_states2 = cell_states2.to(device)

            for m in range(1, params['past_length']):  #
                current_step = traj[:, m, :2]  # peds * 2
                current_vel = traj[:, m, 2:4]  # peds * 2
                input_lstm = torch.cat((current_step, current_vel), dim=1)  # peds * 4
                outputs_features1, hidden_states1, cell_states1, outputs_features2, hidden_states2, cell_states2 \
                    = model.forward_lstm(input_lstm, hidden_states1, cell_states1, hidden_states2, cell_states2)
            # forward_end = time.time()
            # print(f"Forward pass time for batch {batch_idx}: {forward_end - forward_start:.4f} seconds")

            # his_update_start = time.time()
            predictions = torch.zeros(num_peds, params['future_length'], 2, device=device)
            coefficients = model.forward_coefficient_veh(outputs_features2, supplement[:, params['past_length']-1, :, :],
                                                         nbr_existence[:, :-1], current_step, current_vel, device)  # peds*maxpeds*2, peds*(max_peds + 1)*4
            prediction, w_v, _, _ = model.forward_next_step_veh(current_step, current_vel, initial_speeds, dest,
                                                      outputs_features1, coefficients, supplement[:, params['past_length']-1, :, :],
                                                          boundary_factor_current[:, params['past_length'] - 1],
                                                          clines_factor_current[:, params['past_length'] - 1],
                                                          boundary_factor_shifted[:, params['past_length'] - 1],
                                                          clines_factor_shifted[:, params['past_length'] - 1],
                                                          sigma, params['time_step'],
                                                          k_boundary, k_clines, device=device)
            predictions[:, 0, :] = prediction #past_length预测出predictions的第一个

            current_step = prediction #peds * 2
            current_vel = w_v #peds * 2
            # his_update_end = time.time()
            # print(f"History part update time for batch {batch_idx}: {his_update_end - his_update_start:.4f} seconds")

            # future_start = time.time()
            for t in range(params['future_length'] - 1):
                input_lstm = torch.cat((current_step, current_vel), dim=1)
                outputs_features1, hidden_states1, cell_states1, outputs_features2, hidden_states2, cell_states2 \
                    = model.forward_lstm(input_lstm, hidden_states1, cell_states1, hidden_states2, cell_states2)

                future_vel = (dest - prediction) / ((params['future_length']-t-1) * params['time_step'])  # peds*2
                future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
                initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1

                coefficients = model.forward_coefficient_veh(outputs_features2, supplement[:, params['past_length'] + t, :, :],
                                                             nbr_existence[:, :-1], current_step, current_vel, device)
                prediction, w_v, _, _ = model.forward_next_step_veh(current_step, current_vel, initial_speeds, dest, outputs_features1,
                                                              coefficients, supplement[:, params['past_length'] + t, :, :],
                                                              boundary_factor_current[:, params['past_length'] + t],
                                                              clines_factor_current[:, params['past_length'] + t],
                                                              boundary_factor_shifted[:, params['past_length'] + t],
                                                              clines_factor_shifted[:, params['past_length'] + t],
                                                              sigma, params['time_step'],
                                                              k_boundary, k_clines, device=device)

                predictions[:, t+1, :] = prediction # 按照future_length依次预测出后面的每一步

                current_step = prediction  # peds*2
                current_vel = w_v # peds*2
            # future_end = time.time()
            # print(f"Future calculate time for batch {batch_idx}: {future_end - future_start:.4f} seconds")

            # backward_start = time.time()
            optimizer.zero_grad()
            loss = calculate_loss(criterion, future, predictions)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            # backward_end = time.time()
            # print(f"Backward pass time for batch {batch_idx}: {backward_end - backward_start:.4f} seconds")

            # batch_end_time = time.time()
            # print(f"Total time for batch {batch_idx}: {batch_end_time - start_time:.4f} seconds")

    total_end_time = time.time()
    print(f"Total training time: {total_end_time - total_start_time:.4f} seconds")
    return total_loss

def test(test_loader, goal_index, inference=False):
    model.eval()

    # 存储预测值以及势场值
    all_predictions = []
    all_neighbor_factors = []
    all_line_factors = []

    total_squared_errors = 0  # 累积平方误差
    total_num_sequences = 0  # 累积的轨迹数量
    ade_list = []
    fde_list = []
    squared_error_list = []
    metrics_ade = [[] for _ in range(5)]
    metrics_fde = [[] for _ in range(5)]
    squared_errors_metrics = [[] for _ in range(5)]

    # Time tracking variables
    batch_times = []
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (traj_batch, supplement_batch, nbr_existence_batch,
                        boundary_factor_current_batch, clines_factor_current_batch,
                        boundary_factor_shifted_batch, clines_factor_shifted_batch,
                        goal_batch) in enumerate(test_loader):

            batch_size = traj_batch.shape[0]  # Get actual batch size
            generated_goals = goal_batch[:, goal_index, :]  # 使用传入的 goal_index 选择目标

            y = traj_batch[:, params['past_length']:, :2]  # peds*future_length*2
            y = y.cpu().numpy()
            dest = generated_goals


            future_vel = (dest - traj_batch[:, params['past_length'] - 1, :2])# peds*2
            denominator = (torch.tensor(params['future_length']).to(device) * params['time_step'])
            future_vel = future_vel / denominator
            future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
            initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1

            batch_start_time = time.time()  # Start timing for this batch
            num_peds = traj_batch.shape[0]
            numNodes = num_peds
            hidden_states1 = Variable(torch.zeros(numNodes, params['rnn_size'], device=device))
            cell_states1 = Variable(torch.zeros(numNodes, params['rnn_size'], device=device))
            hidden_states2 = Variable(torch.zeros(numNodes, params['rnn_size'], device=device))
            cell_states2 = Variable(torch.zeros(numNodes, params['rnn_size'], device=device))

            if inference:
                # 初始化势场张量
                neighbor_factors = torch.zeros(num_peds, params['future_length'], device=device)
                line_factors = torch.zeros(num_peds, params['future_length'], device=device)

            batch_start = time.time()
            for m in range(1, params['past_length']):  #
                current_step = traj_batch[:, m, :2]  # peds*2
                current_vel = traj_batch[:, m, 2:4]  # peds*2
                input_lstm = torch.cat((current_step, current_vel), dim=1)  # peds*4
                outputs_features1, hidden_states1, cell_states1, outputs_features2, hidden_states2, cell_states2 \
                    = model.forward_lstm(input_lstm, hidden_states1, cell_states1, hidden_states2, cell_states2)

            predictions = torch.zeros(num_peds, params['future_length'], 2, device=device)

            coefficients = model.forward_coefficient_veh(outputs_features2, supplement_batch[:, params['past_length']-1, :, :], nbr_existence_batch[:, :-1],
                                                                       current_step, current_vel,
                                                                       device)  # peds*maxpeds*2, peds*(max_peds + 1)*4

            prediction, w_v, v_neighbors, v_lines = model.forward_next_step_veh(current_step, current_vel, initial_speeds, dest,
                                                      outputs_features1, coefficients, supplement_batch[:, params['past_length']-1, :, :],
                                                          boundary_factor_current_batch[:, params['past_length'] - 1],
                                                          clines_factor_current_batch[:, params['past_length'] - 1],
                                                          boundary_factor_shifted_batch[:, params['past_length'] - 1],
                                                          clines_factor_shifted_batch[:, params['past_length'] - 1], sigma, params['time_step'],
                                                          k_boundary, k_clines, device=device)

            predictions[:, 0, :] = prediction
            if inference:
                neighbor_factors[:, 0] = v_neighbors
                line_factors[:, 0] = v_lines

            current_step = prediction #peds*2
            current_vel = w_v #peds*2

            for t in range(params['future_length'] - 1):
                input_lstm = torch.cat((current_step, current_vel), dim=1)
                outputs_features1, hidden_states1, cell_states1, outputs_features2, hidden_states2, cell_states2 \
                    = model.forward_lstm(input_lstm, hidden_states1, cell_states1, hidden_states2, cell_states2)

                future_vel = (dest - prediction) / ((params['future_length'] - t - 1) * params['time_step'])  # peds*2
                future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
                initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1

                coefficients= model.forward_coefficient_veh(outputs_features2, supplement_batch[:, params['past_length'] + t, :, :],  nbr_existence_batch[:, :-1],
                                                                                  current_step, current_vel,
                                                                                  device=device) # 这里的输出应该是curr_supp而不是current_supplement吧？

                prediction, w_v, v_neighbors, v_lines = model.forward_next_step_veh(current_step, current_vel, initial_speeds, dest,
                                                          outputs_features1, coefficients, supplement_batch[:, params['past_length'] + t, :, :],
                                                              boundary_factor_current_batch[:, params['past_length'] + t],
                                                              clines_factor_current_batch[:, params['past_length'] + t],
                                                              boundary_factor_shifted_batch[:, params['past_length'] + t],
                                                              clines_factor_shifted_batch[:, params['past_length'] + t], sigma, params['time_step'],
                                                          k_boundary, k_clines,  device=device)

                predictions[:, t + 1, :] = prediction
                if inference:
                    neighbor_factors[:, t + 1] = v_neighbors
                    line_factors[:, t + 1] = v_lines

                current_step = prediction  # peds*2
                current_vel = w_v  # peds*2

            predictions = predictions.cpu().numpy()
            batch_end_time = time.time()  # End timing for this batch
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)
            total_samples += batch_size

            if inference:
                # 存储预测值和势场值
                all_predictions.append(predictions)
                all_neighbor_factors.append(neighbor_factors.cpu().numpy())
                all_line_factors.append(line_factors.cpu().numpy())

            # 计算整批次数据的ade fde rmse
            test_ade = np.mean(np.linalg.norm(y - predictions, axis=2), axis=1)  # B
            test_fde = np.linalg.norm((y[:, -1, :] - predictions[:, -1, :]), axis=1)  # B
            test_squared_errors = np.mean(np.linalg.norm(y - predictions, axis=2) ** 2, axis=1)  # 保存平方误差

            # 一次性输出预测1、2、3、4、5s的ade fde rmse指标
            metrics = {'ade': [], 'fde': [], 'squared_errors': []}
            for i in range(1, 6):
                interval = int(params['future_length'] * (i / 5))
                metrics['ade'].append(
                    np.mean(np.linalg.norm(y[:, :interval, :] - predictions[:, :interval, :], axis=2), axis=1))
                metrics['fde'].append(np.linalg.norm((y[:, interval - 1, :] - predictions[:, interval - 1, :]), axis=1))
                metrics['squared_errors'].append(np.mean(np.linalg.norm(y[:, :interval, :] - predictions[:, :interval, :], axis=2) ** 2, axis=1))

            ade_list.extend(test_ade)
            fde_list.extend(test_fde)
            squared_error_list.extend(test_squared_errors)

            for i in range(5):  # 保存1-5秒的预测结果
                metrics_ade[i].extend(metrics['ade'][i])
                metrics_fde[i].extend(metrics['fde'][i])
                squared_errors_metrics[i].extend(metrics['squared_errors'][i])

            total_squared_errors += np.sum(test_squared_errors)
            total_num_sequences += num_peds

        # Calculate and print timing statistics if inference
        if inference and batch_times:
            avg_batch_time = sum(batch_times) / len(batch_times)
            avg_sample_time = sum(batch_times) / total_samples if total_samples > 0 else 0

            print(f"\n=== Inference Time Statistics (Goal {goal_index}) ===")
            print(f"Total batches processed: {len(batch_times)}")
            print(f"Total samples processed: {total_samples}")
            print(f"Average time per batch: {avg_batch_time:.4f} seconds")
            print(f"Average time per sample: {avg_sample_time * 1000:.2f} milliseconds")
            print(f"Total inference time: {sum(batch_times):.2f} seconds")
            print("================================================\n")

    ade_array = np.array(ade_list)
    fde_array = np.array(fde_list)
    squared_errors_array = np.array(squared_error_list)

    # 处理 1-5 秒的预测结果
    metrics_ade_array = [np.array(metrics_ade[i]) for i in range(5)]
    metrics_fde_array = [np.array(metrics_fde[i]) for i in range(5)]
    squared_errors_metrics_array = [np.array(squared_errors_metrics[i]) for i in range(5)]

    if inference:
        return ade_array, fde_array, squared_errors_array, metrics_ade_array, metrics_fde_array, squared_errors_metrics_array,\
            all_predictions, all_neighbor_factors, all_line_factors

    else:
        return ade_array, fde_array, squared_errors_array, metrics_ade_array, metrics_fde_array, squared_errors_metrics_array



parser = argparse.ArgumentParser(description='GNP')
parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--save_file', '-sf', type=str, default='F_full_noline2.pt')
parser.add_argument('--mode', type=str, choices=['inference', 'plot'],
                    default='inference', help='Mode to run: inference or plot')
args = parser.parse_args()

# 读取yaml文件中的训练超参数
CONFIG_FILE_PATH = 'config/highd_rep.yaml'  # yaml config file containing all the hyperparameters
with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
print(params)

dtype = torch.float32
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
print(device)
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

# 加载处理好的train test inference data，包括agent supplement以及nbr_existence
training_input, training_supplement, training_nbr_existence, training_rele_current_trajectories, training_rele_neighbor_trajectories, training_first_point, \
        test_input, testing_supplement, testing_nbr_existence, testing_rele_current_trajectories, testing_rele_neighbor_trajectories, testing_first_point, \
        inference_input, inference_supplement, inference_nbr_existence, inference_rele_current_trajectories, inference_rele_neighbor_trajectories, inference_first_point,\
        feature_destd = dir_load_data()

# 解析lane description，得到车道标记的数值形式
line_dist_start = time.time()

lane_markings = parse_lane_data(lane_data)
delta = 1e-3 # 提前计算好偏移delta后的车道线势场

# 检查并加载预势场值文件，如果文件存在则加载，不存在则计算并保存
train_zip_filepath = "data/HighD/train_lane_factors.zip"
test_zip_filepath = "data/HighD/test_lane_factors.zip"
inference_zip_filepath = "data/HighD/inference_lane_factors.zip"

if os.path.exists(train_zip_filepath):
    print("找到训练集预势场值文件，开始加载...")
    train_boundary_factor_current, train_clines_factor_current, train_boundary_factor_shifted, train_clines_factor_shifted = load_precomputed_factors('train')
    train_boundary_factor_current = train_boundary_factor_current.to(device)
    train_clines_factor_current = train_clines_factor_current.to(device)
    train_boundary_factor_shifted = train_boundary_factor_shifted.to(device)
    train_clines_factor_shifted = train_clines_factor_shifted.to(device)
else:
    print("训练集预势场值文件不存在，开始计算并保存...")
    train_zip_filepath = calculate_lane_distances_and_factors(training_input, lane_markings, delta, mode='train', device=device)
    print("训练集文件保存路径:", train_zip_filepath)
    train_boundary_factor_current, train_clines_factor_current, train_boundary_factor_shifted, train_clines_factor_shifted = load_precomputed_factors('train')
    train_boundary_factor_current = train_boundary_factor_current.to(device)
    train_clines_factor_current = train_clines_factor_current.to(device)
    train_boundary_factor_shifted = train_boundary_factor_shifted.to(device)
    train_clines_factor_shifted = train_clines_factor_shifted.to(device)
if os.path.exists(test_zip_filepath):
    print("找到测试集预势场值文件，开始加载...")
    test_boundary_factor_current, test_clines_factor_current, test_boundary_factor_shifted, test_clines_factor_shifted = load_precomputed_factors('test')
    test_boundary_factor_current = test_boundary_factor_current.to(device)
    test_clines_factor_current = test_clines_factor_current.to(device)
    test_boundary_factor_shifted = test_boundary_factor_shifted.to(device)
    test_clines_factor_shifted = test_clines_factor_shifted.to(device)
else:
    print("测试集预势场值文件不存在，开始计算并保存...")
    test_zip_filepath = calculate_lane_distances_and_factors(test_input, lane_markings, delta, mode='test', device=device)
    print("测试集文件保存路径:", test_zip_filepath)
    test_boundary_factor_current, test_clines_factor_current, test_boundary_factor_shifted, test_clines_factor_shifted = load_precomputed_factors('test')
    test_boundary_factor_current = test_boundary_factor_current.to(device)
    test_clines_factor_current = test_clines_factor_current.to(device)
    test_boundary_factor_shifted = test_boundary_factor_shifted.to(device)
    test_clines_factor_shifted = test_clines_factor_shifted.to(device)
if os.path.exists(inference_zip_filepath):
    print("找到推理集预势场值文件，开始加载...")
    inference_boundary_factor_current, inference_clines_factor_current, inference_boundary_factor_shifted, inference_clines_factor_shifted = load_precomputed_factors('inference')
    inference_boundary_factor_current = inference_boundary_factor_current.to(device)
    inference_clines_factor_current = inference_clines_factor_current.to(device)
    inference_boundary_factor_shifted = inference_boundary_factor_shifted.to(device)
    inference_clines_factor_shifted = inference_clines_factor_shifted.to(device)
else:
    print("推理集预势场值文件不存在，开始计算并保存...")
    inference_zip_filepath = calculate_lane_distances_and_factors(inference_input, lane_markings, delta, mode='inference', device=device)
    print("推理集文件保存路径:", inference_zip_filepath)
    inference_boundary_factor_current, inference_clines_factor_current, inference_boundary_factor_shifted, inference_clines_factor_shifted = load_precomputed_factors('inference')
    inference_boundary_factor_current = inference_boundary_factor_current.to(device)
    inference_clines_factor_current = inference_clines_factor_current.to(device)
    inference_boundary_factor_shifted = inference_boundary_factor_shifted.to(device)
    inference_clines_factor_shifted = inference_clines_factor_shifted.to(device)

line_dist_end = time.time()
line_dist_duration = line_dist_end - line_dist_start
print("###TIME: Calculate line distance, ", line_dist_duration, "s")

# 反标准化操作，train_goals使用真实的数据进行训练？
mean = feature_destd[:, 0]
std = feature_destd[:, 1]

# training_input_denorm = training_input[:, :, 0:4] * std[np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, :]
# test_input_denorm = test_input[:, :, 0:4] * std[np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, :]
# inference_input_denorm = inference_input[:, :, 0:4] * std[np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, :]
# training_supplement_denorm = training_supplement * std[np.newaxis, np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, np.newaxis, :]
# test_supplement_denorm = testing_supplement * std[np.newaxis, np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, np.newaxis, :]
# inference_supplement_denorm = inference_supplement * std[np.newaxis, np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, np.newaxis, :]

training_input_denorm = training_rele_current_trajectories[:, :, 0:4]
test_input_denorm = testing_rele_current_trajectories[:, :, 0:4]
inference_input_denorm = inference_rele_current_trajectories[:, :, 0:4]
training_supplement_denorm = training_rele_neighbor_trajectories
test_supplement_denorm = testing_rele_neighbor_trajectories
inference_supplement_denorm = inference_rele_neighbor_trajectories

# 取XY VX VY，并转换为tensor
training_traj = torch.tensor(training_input_denorm, dtype=torch.float32).to(device)
testing_traj = torch.tensor(test_input_denorm, dtype=torch.float32).to(device)
inference_traj = torch.tensor(inference_input_denorm, dtype=torch.float32).to(device)
training_supplement_denorm = torch.tensor(training_supplement_denorm, dtype=torch.float32).to(device)
test_supplement_denorm = torch.tensor(test_supplement_denorm, dtype=torch.float32).to(device)
inference_supplement_denorm = torch.tensor(inference_supplement_denorm, dtype=torch.float32).to(device)
training_nbr_existence = torch.tensor(training_nbr_existence, dtype=torch.float32).to(device)
testing_nbr_existence = torch.tensor(testing_nbr_existence, dtype=torch.float32).to(device)
inference_nbr_existence = torch.tensor(inference_nbr_existence, dtype=torch.float32).to(device)

# 制作训练的train, test, inference dataloader
train_dataset = TensorDataset(
    training_traj,
    training_supplement_denorm,
    training_nbr_existence,
    train_boundary_factor_current,
    train_clines_factor_current,
    train_boundary_factor_shifted,
    train_clines_factor_shifted
)
train_dataloader = DataLoader(train_dataset, num_workers=0 ,batch_size=params['train_b_size'], shuffle=True)

# 加载TUTR训练好的goal
pred_test = np.load('E:/Pycharm Project/Goal-generation-TUTR-V4/checkpoint/HighD/test_pred_trajs.npy') # B, num_pred, pred_len, 2.    HighD or Ngsim
goal_test = torch.tensor(pred_test[:8633, :, -1, :], dtype=torch.float32).to(device) # 取最后一帧作为goal, [B, num_pred, 2]
inference_goals = np.load('E:/Pycharm Project/Goal-generation-TUTR-V4/checkpoint/HighD/inference_pred_trajs.npy') # B, num_pred, pred_len, 2.       HighD or Ngsim
goal_inference = torch.tensor(inference_goals[:8634, :, -1, :], dtype=torch.float32).to(device) # 取最后一帧作为goal, [B, num_pred, 2]
print("###相关数据已经加载完成，准备加载NSP模型")

test_dataset = TensorDataset(
    testing_traj,
    test_supplement_denorm,
    testing_nbr_existence,
    test_boundary_factor_current,
    test_clines_factor_current,
    test_boundary_factor_shifted,
    test_clines_factor_shifted,
    goal_test
)
test_dataloader = DataLoader(test_dataset, num_workers=0 ,batch_size=params['train_b_size'], shuffle=False)
inference_dataset = TensorDataset(
    inference_traj,
    inference_supplement_denorm,
    inference_nbr_existence,
    inference_boundary_factor_current,
    inference_clines_factor_current,
    inference_boundary_factor_shifted,
    inference_clines_factor_shifted,
    goal_inference
)
inference_dataloader = DataLoader(inference_dataset, num_workers=0 ,batch_size=params['train_b_size'], shuffle=False)

# 模型和训练参数加载
model = NSP(params["input_size"], params["embedding_size"], params["rnn_size"], params["output_size"],  params["enc_size"], params["dec_size"])
model = model.to(device)

# 加载train_goal训练好的参数，并组合形成train_repulsion的初始参数
load_path = 'saved_models/Fgoal.pt'
checkpoint_trained = torch.load(load_path)
# 保存模型的初始化参数
save_path = 'saved_models/SDD_nsp_wo_ini.pt'
torch.save({'hyper_params': params,
            'model_state_dict': model.state_dict()
                }, save_path)
load_path_ini = 'saved_models/SDD_nsp_wo_ini.pt' #？为什么这里有一个initial参数？ 可能是初始参数或者预训练参数，能够加速并提升训练精度。
checkpoint_ini = torch.load(load_path_ini)
# Initialize new parameters with the same structure as the initial model using Kaiming initialization
new_params = initialize_parameters_like(checkpoint_ini['model_state_dict'])
checkpoint_dic = new_point(checkpoint_trained['model_state_dict'], new_params) #合并上一步goal的模型参数与预训练的initial参数
model.load_state_dict(checkpoint_dic)
parameter_train = select_para(model)

# print("Printing checkpoint trained Parameters in model_state_dict:") # 22 keys
# print_model_state_dict(checkpoint_trained['model_state_dict'])
# print("Printing new initialized Parameters in model_state_dict:") # 44 keys
# print_model_state_dict(new_params)
# # Print each parameter name and its value in the combined model state dict
# print("Printing Combined Parameters in model_state_dict:") # 44 keys
# print_model_state_dict(checkpoint_dic)

sigma = torch.tensor(26)
# 定义两个可学习参数
k_boundary = torch.tensor(3.0, requires_grad=True, device=device)
k_clines = torch.tensor(2.0, requires_grad=True, device=device)

optimizer = optim.Adam([{'params': parameter_train}, {'params': [k_boundary, k_clines]}], lr = params["learning_rate"])

# best_ade = 7.85
# best_fde = 11.85
best_ade = 99
best_fde = 99
best_rmse = 99
best_epoch = 0
epsilon = 1e-8
mul = 1e3
train_time = [] # 记录每个epoch训练时长

print("######################  Training Starts ############################")
for e in range(params['num_epochs']):
    print("Start training epoch ", e)
    start_time = time.time()
    print("开始train()")
    total_loss = train()
    print("train()结束")

    ade_20 = np.zeros((20, len(testing_traj)))
    fde_20 = np.zeros((20, len(testing_traj)))
    squared_errors_20 = np.zeros((20, len(testing_traj)))
    metrics_ade_20 = [np.zeros((20, len(testing_traj))) for _ in range(5)]
    metrics_fde_20 = [np.zeros((20, len(testing_traj))) for _ in range(5)]
    squared_errors_metrics_20 = [np.zeros((20, len(testing_traj))) for _ in range(5)]
    total_squared_errors = 0
    total_num_sequences = 0

    print("开始test")
    test_start = time.time()
    for j in range(20): # 应该是测试20次取平均值
        # test_j_start = time.time()
        test_ade_, test_fde_, test_squared_errors_, metrics_ade_array, metrics_fde_array, squared_errors_metrics_array = test(test_dataloader, j) # 只在test中使用了goal
        ade_20[j, :] = test_ade_
        fde_20[j, :] = test_fde_
        squared_errors_20[j, :] = test_squared_errors_

        # 处理1-5秒的预测结果
        for i in range(5):
            metrics_ade_20[i][j, :] = metrics_ade_array[i]
            metrics_fde_20[i][j, :] = metrics_fde_array[i]
            squared_errors_metrics_20[i][j, :] = squared_errors_metrics_array[i]
        total_squared_errors += np.sum(test_squared_errors_)
        total_num_sequences += len(test_squared_errors_)

        # test_j_end = time.time()
        # print(f"Test time for predicted goal j {j}: {test_j_end - test_j_start:.4f} seconds")

    # 打印 1-5s 预测结果
    print('################## Print 1-5s results ########')
    for i in range(5):
        interval = (i + 1)
        ade_i = np.mean(np.min(metrics_ade_20[i], axis=0))
        fde_i = np.mean(np.min(metrics_fde_20[i], axis=0))
        rmse_i = np.sqrt(np.sum(np.min(squared_errors_metrics_20[i], axis=0)) / total_num_sequences)

        print(f"Test ADE for {interval}s: {ade_i:.4f}")
        print(f"Test FDE for {interval}s: {fde_i:.4f}")
        print(f"Test RMSE for {interval}s: {rmse_i:.4f}")

    print("20次test结束")
    test_end = time.time()
    print(f"Test time for epoch {e}: {test_end - test_start:.4f} seconds")

    # 计算总体的测试 ADE、FDE 和 RMSE
    test_ade = np.mean(np.min(ade_20, axis=0)) # 先从20个预测结果中求最小值，再对test整批次的数据求平均值
    test_fde = np.mean(np.min(fde_20, axis=0))
    test_rmse = np.sqrt(np.sum(np.min(squared_errors_20, axis=0)) / total_num_sequences)  # 先计算总的平方误差再取平方根

    if best_ade > test_ade:
        print(" Best Epoch so far: ", e)
        print('################## BEST PERFORMANCE {:0.2f} ########'.format(test_ade))
        best_epoch = e
        best_ade = test_ade
        best_fde = test_fde
        best_rmse = test_rmse
        save_path = 'saved_models/' + args.save_file
        torch.save({'hyper_params': params,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                        }, save_path)
        print("Saved model to:\n{}".format(save_path))

    end_time = time.time()  # Record the end time of the epoch
    epoch_duration = end_time - start_time
    train_time.append(epoch_duration)
    # Calculate total and average time up to the current epoch
    total_train_time = sum(train_time)
    avg_epoch_time = total_train_time / len(train_time)
    # Calculate estimated remaining time
    remaining_epochs = params['num_epochs'] - (e + 1)
    estimated_remaining_time = remaining_epochs * avg_epoch_time
    # Get the current time and add the estimated remaining time
    projected_end_time = datetime.now() + timedelta(seconds=estimated_remaining_time)

    print('Epoch:', e)
    print('Train time', epoch_duration)
    print(f"预计训练结束时间: {projected_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Train Loss (MSE)", total_loss)
    print("Test ADE", test_ade)
    print("Test FDE", test_fde)
    print("Test RMSE", test_rmse)
    print("Best Epoch So Far", best_epoch)
    print("Best ADE Loss So Far", best_ade)
    print("Best FDE Loss So Far", best_fde)
    print("Best RMSE Loss So Far", best_rmse)

if params['num_epochs'] > 0:
    print("训练总时长为：", sum(train_time))
    print("平均训练时长：", sum(train_time) / len(train_time))

if args.mode == 'inference':
    ####################################### 推理部分 ###############################################
    print("##########################开始推理（inference）########################################")

    # 加载训练好的模型参数
    checkpoint = torch.load('saved_models/' + args.save_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置模型为推理模式

    inference_ade_20 = np.zeros((20, len(inference_traj)))
    inference_fde_20 = np.zeros((20, len(inference_traj)))
    inference_squared_errors_20 = np.zeros((20, len(inference_traj)))
    inference_metrics_ade_20 = [np.zeros((20, len(inference_traj))) for _ in range(5)]
    inference_metrics_fde_20 = [np.zeros((20, len(inference_traj))) for _ in range(5)]
    inference_squared_errors_metrics_20 = [np.zeros((20, len(inference_traj))) for _ in range(5)]
    inference_total_squared_errors = 0
    inference_total_num_sequences = 0

    all_predictions_20 = np.zeros((20, len(inference_traj), params['future_length'], 2))  # 存储20个goal的预测
    all_neighbor_factors_20 = np.zeros((20, len(inference_traj), params['future_length']))  # 存储邻居势场结果
    all_line_factors_20 = np.zeros((20, len(inference_traj), params['future_length']))  # 存储车道线势场结果

    print("开始 inference")
    inference_start = time.time()

    for j in range(20):  # 和测试时类似，推理 20 次并取平均值
        inference_j_start = time.time()
        inference_ade_, inference_fde_, inference_squared_errors_, metrics_ade_array, metrics_fde_array, squared_errors_metrics_array,\
            all_predictions, all_neighbor_factors, all_line_factors = test(inference_dataloader, j, inference=True)
        inference_j_end = time.time()
        inference_ade_20[j, :] = inference_ade_
        inference_fde_20[j, :] = inference_fde_
        inference_squared_errors_20[j, :] = inference_squared_errors_

        all_predictions_20[j] = np.concatenate(all_predictions, axis=0)
        all_neighbor_factors_20[j] = np.concatenate(all_neighbor_factors, axis=0)
        all_line_factors_20[j] = np.concatenate(all_line_factors, axis=0)

        # 处理 1-5 秒的预测结果
        for i in range(5):
            inference_metrics_ade_20[i][j, :] = metrics_ade_array[i]
            inference_metrics_fde_20[i][j, :] = metrics_fde_array[i]
            inference_squared_errors_metrics_20[i][j, :] = squared_errors_metrics_array[i]

        inference_total_squared_errors += np.sum(inference_squared_errors_)
        inference_total_num_sequences += len(inference_squared_errors_)


        print(f"Inference time for predicted goal j {j}: {inference_j_end - inference_j_start:.4f} seconds")

    # 打印 1-5s 的预测结果
    print('################## Inference 1-5s results ########')
    for i in range(5):
        interval = (i + 1)
        inference_ade_i = np.mean(np.min(inference_metrics_ade_20[i], axis=0))
        inference_fde_i = np.mean(np.min(inference_metrics_fde_20[i], axis=0))
        inference_rmse_i = np.sqrt(np.sum(np.min(inference_squared_errors_metrics_20[i], axis=0)) / inference_total_num_sequences)

        print(f"Inference ADE for {interval}s: {inference_ade_i:.4f}")
        print(f"Inference FDE for {interval}s: {inference_fde_i:.4f}")
        print(f"Inference RMSE for {interval}s: {inference_rmse_i:.4f}")

    print("20次 inference 结束")
    inference_end = time.time()
    print(f"Inference time: {inference_end - inference_start:.4f} seconds")

    # 计算总体的 Inference ADE、FDE 和 RMSE
    inference_ade = np.mean(np.min(inference_ade_20, axis=0))  # 从 20 个预测结果中取最小值再取均值
    inference_fde = np.mean(np.min(inference_fde_20, axis=0))
    inference_rmse = np.sqrt(np.sum(np.min(inference_squared_errors_20, axis=0)) / inference_total_num_sequences)

    print(f"Inference ADE: {inference_ade:.4f}")
    print(f"Inference FDE: {inference_fde:.4f}")
    print(f"Inference RMSE: {inference_rmse:.4f}")

    np.savez_compressed('results.npz', all_predictions_20=all_predictions_20,
                        all_neighbor_factors_20=all_neighbor_factors_20,
                        all_line_factors_20=all_line_factors_20)

elif args.mode == 'plot':
    ############inference绘图阶段####################
    print("###############inference 绘图 start########################")
    data = np.load('results.npz')
    all_predictions_20 = data['all_predictions_20'] # TODO:修改为train goal的predictions结果，可能更加准确
    all_neighbor_factors_20 = data['all_neighbor_factors_20']
    all_line_factors_20 = data['all_line_factors_20']

    # idx_list = [0, 5, 10]
    # 从 inference_input 中筛选出 location_id == 6 的所有轨迹
    location_ids = inference_input[:, 0, 5]  # 获取 location_id 列
    # idx_list = np.where(location_ids == 6)[0]
    # idx_list = [274, 7431, 158, 1081, 2531, 3593, 4334, 4558, 4938, 5336, 5459, 6326, 7431, 7665, 8547,
    #             376, 447, 478, 490, 581, 6132, 2416, 8041,
    #             7264, 7163, 6816, 6627, 6326, 6220, 4999, 191, 316, 377, 447, 454, 490, 566, 615, 672, 897,
    #             6644, 6899, 7152, 7347, 7478, 8617,
    #             158, 7843, 7848]
    idx_list = [2531,6220]

    # idx_list = idx_list[0:15]# 获取符合条件的索引
    print(len(idx_list))

    for idx in idx_list:
        # Extract the specific sample for the given idx
        traj = inference_input_denorm[idx] # Trajectory of the sample
        predictions = all_predictions_20[:, idx, :, :]  # All 20 predictions of the sample

        # Calculate the ADE min index
        ade_min_idx = np.argmin(
            np.mean(np.linalg.norm(predictions - inference_traj[idx, params['past_length']:, :2].cpu().numpy(), axis=2), axis=1)
        )

        neighbor_traj = inference_supplement_denorm[idx].cpu().numpy()  # Neighbor trajectory of the sample
        neighbor_existence = inference_nbr_existence[idx].cpu().numpy()  # Neighbor existence status of the sample
        all_neighbor_factors = all_neighbor_factors_20[ade_min_idx, idx]  # Neighbor factors for the sample
        all_line_factors = all_line_factors_20[ade_min_idx, idx]  # Line factors for the sample

        # 利用first points转换为真实场景的X Y值
        first_point = inference_first_point[idx]
        real_traj = traj[:, :2] + first_point
        real_predictions = predictions + first_point
        real_neighbor_traj = neighbor_traj[:, :, :2] + first_point

        location_id = inference_input[idx, 0, 5] # Location ID of the sample
        laneid = inference_input[idx, 0, 4]  # Lane ID of the sample

        # Call the visualization function for this specific sample
        # visualize_trajectories_with_factors_and_lanes(
        #     params,
        #     traj=real_traj,
        #     predictions=real_predictions,
        #     ade_min_idx=ade_min_idx,
        #     title="Trajectory_Plot",
        #     idx=idx,
        #     neighbor_traj=real_neighbor_traj,
        #     neighbor_existence=neighbor_existence,
        #     all_neighbor_factors=all_neighbor_factors,
        #     all_line_factors=all_line_factors,
        #     lane_markings=lane_markings,
        #     location_id=location_id,
        #     laneid=laneid
        # )
        visualize_trajectories_with_multiple_time_steps(
            params,
            traj=real_traj,
            predictions=real_predictions,
            ade_min_idx=ade_min_idx,
            title="selected_potential_plot",
            idx=idx,
            neighbor_traj=real_neighbor_traj,
            neighbor_existence=neighbor_existence,
            all_neighbor_factors=all_neighbor_factors,
            all_line_factors=all_line_factors,
            lane_markings=lane_markings,
            location_id=location_id,
            laneid=laneid
        )
    print("绘图完成")