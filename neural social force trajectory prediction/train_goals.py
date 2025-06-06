import yaml
import os
import matplotlib.pyplot as plt
from model_goals import *
from utils import *
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import argparse
from loaddata_goals import *
import time
from PIL import Image
from datetime import timedelta, datetime

def train(train_dataloader):

    model.train()
    total_loss = 0
    criterion = nn.MSELoss()
    total_batches = len(train_dataloader)

    for batch_idx, batch_data in enumerate(train_dataloader):
        if batch_idx < total_batches-1:
            # 针对lstm模型，把batch_size * num_nodes来和原human保持一致
            traj = batch_data[0] # traj.shape [B, 40, 4]，当前车data
            # traj = traj.permute(0,2,1,3).contiguous().reshape(traj.shape[0] * 5, 40, 4)
            # traj = torch.squeeze(traj.double().to(device)) #torch.squeeze是挤压，去除tensor中维度为1的维度
            traj = traj.to(torch.float32).to(device)

            max_traj_x = torch.max(traj[:, :, 0])
            max_traj_y = torch.max(traj[:, :, 1])

            y = traj[:, params['past_length']:, :2] # B * future_length * 2

            dest = y[:, -1, :].to(device) # B * 2，真实的destination
            #dest_state = traj[:, -1, :]
            future = y.contiguous().to(device) # B * future_length * 2，future是y的连续版本，计算loss时使用
            future_vel = (dest - traj[:, params['past_length'] - 1, :2]) / (torch.tensor(params['future_length']).to(device) * params['time_step']) \
                #B * 2，过去最后一帧到dest的速度向量
            # future_vel = torch.tensor(dest.cpu().numpy()-traj[:, params['past_length'] - 1, :2].cpu().numpy()).to(device)
            # dest = dest * mul
            # trajk = traj[:, params['past_length'] - 1, :2] * mul
            # future_vel = dest # peds*2


            # denominator = (torch.tensor(params['future_length']).to(device) * 0.2 * mul)
            # future_vel = future_vel/ denominator
            # kk = (dest-trajk) / denominator
            future_vel_norm = torch.norm(future_vel, dim=-1) # B
            # kk_norm = torch.norm(kk, dim=-1)# num_frame
            initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1) # B * 1
            # kk_initial_speeds = torch.unsqueeze(kk_norm, dim=-1)

            num_peds = traj.shape[0] # B
            numNodes = num_peds # B
            hidden_states = Variable(torch.zeros(numNodes, params['rnn_size']))
            cell_states = Variable(torch.zeros(numNodes, params['rnn_size']))
            hidden_states = hidden_states.to(torch.float32).to(device)
            cell_states = cell_states.to(torch.float32).to(device)

            for m in range(1, params['past_length']):  #
                current_step = traj[:, m, :2]  # B * 2，LSTM中当前步的位置
                current_vel = traj[:, m, 2:]  # B * 2，LSTM中当前步的速度
                input_lstm = torch.cat((current_step, current_vel), dim=1)  # B * 4
                outputs_features, hidden_states, cell_states = model.forward_lstm(input_lstm, hidden_states, cell_states)\
                    # outputs_features [B, output_size]

            predictions = torch.zeros(num_peds, params['future_length'], 2).to(device) # B * future_length * 2
            prediction, w_v = model.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                      outputs_features, params['time_step'], device=device) # B * 2
            max_prediction_x = torch.max(prediction[:, 0])
            max_prediction_y = torch.max(prediction[:, 1])

            predictions[:, 0, :] = prediction # past_length预测出predictions的第一个

            current_step = prediction # B * 2, past_length预测出的第一个位置作为当前位置
            current_vel = w_v # B * 2, past_length预测出的第一个速度作为当前速度

            for t in range(params['future_length'] - 1):
                input_lstm = torch.cat((current_step, current_vel), dim=1) # B * 4
                outputs_features, hidden_states, cell_states = model.forward_lstm(input_lstm, hidden_states, cell_states)\
                    # outputs_features [B, output_size]
                future_vel = (dest - prediction) / ((params['future_length']-t-1) * params['time_step'])  # B * 2, 不断计算未来轨迹中当前帧到最后一帧的速度向量
                future_vel_norm = torch.norm(future_vel, dim=-1)  # B
                initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # B * 1

                prediction, w_v = model.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                          outputs_features, params['time_step'], device=device) # B * 2
                predictions[:, t+1, :] = prediction # B * future_length * 2，按照future_length依次预测出后面的每一步

                current_step = prediction  # B * 2
                current_vel = w_v # B * 2
            optimizer.zero_grad()

            # loss = calculate_loss(criterion, future, predictions)
            loss = manual_mse_loss(predictions, future)

            max_future = torch.max(future)
            max_predictions_x = torch.max(predictions[:, :, 0])
            max_predictions_y = torch.max(predictions[:, :, 1])
            # print(f"Maximum value in future tensor: {max_future.item()}")
            # print(f"Maximum value in prediction tensor: {max_predictions_x.item()}")
            # print(f"Maximum value in prediction tensor: {max_predictions_y.item()}")

            mse_x = manual_mse_loss(predictions[:, :, 0], future[:, :, 0])
            mse_y = manual_mse_loss(predictions[:, :, 1], future[:, :, 1])

            # print(f"MSE in X direction: {mse_x.item()}")
            # print(f"MSE in Y direction: {mse_y.item()}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 对梯度进行裁剪，1.0是一个常用的阈值

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.norm().item()
            #         print(f"{name}: {grad_norm}")
            #     if param.grad is not None and torch.isnan(param.grad).any():
            #         print(f"NaN gradient found in {name}")

            total_loss += loss.item()
            optimizer.step()

    return total_loss

def test(traj, generated_goals):
    model.eval()

    with torch.no_grad():
        # traj = traj.permute(0,2,1,3).contiguous().reshape(traj.shape[0] * 5, 40, 4)
        # generated_goals = generated_goals.reshape(generated_goals.shape[0] * 5, 2)
        traj = traj.to(torch.float32).to(device) # [B, seq_len==40, num_feat==4]
        generated_goals = generated_goals.to(torch.float32).to(device) # [B, 2]

        y = traj[:, params['past_length']:, :2]  # B * future_length * 2
        y = y.cpu().numpy()
        dest = generated_goals # [B, 2]

        future_vel = (dest - traj[:, params['past_length'] - 1, :2]) / (torch.tensor(params['future_length']).to(device) * params['time_step']) # B * 2
        future_vel_norm = torch.norm(future_vel, dim=-1)  # B
        initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # B * 1

        num_peds = traj.shape[0] # B
        numNodes = num_peds # B
        hidden_states = Variable(torch.zeros(numNodes, params['rnn_size'])) # B * rnn_size
        cell_states = Variable(torch.zeros(numNodes, params['rnn_size'])) # B * rnn_size
        hidden_states = hidden_states.to(torch.float32).to(device)
        cell_states = cell_states.to(torch.float32).to(device)

        for m in range(1, params['past_length']):  #
            current_step = traj[:, m, :2]  # B * 2
            current_vel = traj[:, m, 2:]  # B * 2
            input_lstm = torch.cat((current_step, current_vel), dim=1)  # B * 4
            outputs_features, hidden_states, cell_states = model.forward_lstm(input_lstm, hidden_states, cell_states)

        predictions = torch.zeros(num_peds, params['future_length'], 2).to(device) # B * future_length * 2

        prediction, w_v = model.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                              outputs_features, params['time_step'], device=device) # B * 2

        predictions[:, 0, :] = prediction # B * future_length * 2，利用his_len部分预测出predictions的一帧

        current_step = prediction # B * 2
        current_vel = w_v # B * 2

        for t in range(params['future_length'] - 1):
            input_lstm = torch.cat((current_step, current_vel), dim=1) # B * 4
            outputs_features, hidden_states, cell_states = model.forward_lstm(input_lstm, hidden_states, cell_states)

            future_vel = (dest - prediction) / ((params['future_length'] - t - 1) * params['time_step'])  # B * 2
            future_vel_norm = torch.norm(future_vel, dim=-1)  # B
            initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # B * 1

            prediction, w_v = model.forward_next_step(current_step, current_vel, initial_speeds, dest,
                                                  outputs_features, params['time_step'], device=device)
            predictions[:, t + 1, :] = prediction # B * future_length * 2

            current_step = prediction  # B * 2
            current_vel = w_v  # B * 2

        predictions = predictions.cpu().numpy()

        test_ade = np.mean(np.linalg.norm(y - predictions, axis = 2), axis=1) # B
        test_fde = np.linalg.norm((y[:,-1,:] - predictions[:, -1, :]), axis=1) # B
        test_rmse = np.sqrt(np.mean(np.linalg.norm(y - predictions, axis=2) ** 2, axis=1))

        # 一次性输出预测1、2、3、4、5s的ade fde rmse指标
        metrics = {'ade': [], 'fde': [], 'rmse': []}
        for i in range(1, 6):
            interval = int(params['future_length'] * (i / 5))
            metrics['ade'].append(np.mean(np.linalg.norm(y[:, :interval, :] - predictions[:, :interval, :], axis=2), axis=1))
            metrics['fde'].append(np.linalg.norm((y[:, interval - 1, :] - predictions[:, interval - 1, :]), axis=1))
            metrics['rmse'].append(np.sqrt(np.mean(np.linalg.norm(y[:, :interval, :] - predictions[:, :interval, :], axis=2) ** 2, axis=1)))

    return test_ade, test_fde, test_rmse, metrics, y, predictions

# 添加解析参数
parser = argparse.ArgumentParser(description='GNP')
parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--save_file', '-sf', type=str, default='Fgoal.pt') # Fgoal.pt for highd dataset
args = parser.parse_args()

# 读取yaml文件中的训练超参数
CONFIG_FILE_PATH = 'config/highd_goals.yaml'  # yaml config file containing all the hyperparameters, highd_goals.yaml, ngsim_goals.yaml
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
# 已完成：外围点清理，相对位置转换，std标准化
training_input, training_supplement, training_nbr_existence, training_rele_current_trajectories, \
    test_input, testing_supplement, testing_nbr_existence, testing_rele_current_trajectories, \
        inference_input, inference_supplement, inference_nbr_existence, inference_rele_current_trajectories, feature_destd = dir_load_data()

# 反标准化操作，train_goals使用真实的数据进行训练？
mean = feature_destd[:, 0]
std = feature_destd[:, 1]

# training_input_denorm = training_input[:, :, 0:4] * std[np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, :]
# test_input_denorm = test_input[:, :, 0:4] * std[np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, :]
# inference_input_denorm = inference_input[:, :, 0:4] * std[np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, :]
training_input_denorm = training_rele_current_trajectories[:, :, 0:4]
test_input_denorm = testing_rele_current_trajectories[:, :, 0:4]
inference_input_denorm = inference_rele_current_trajectories[:, :, 0:4]

# inference_supplement_denorm = inference_supplement * std[np.newaxis, np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, np.newaxis, :]

# 取XY VX VY，并转换为tensor
training_traj = torch.tensor(training_input_denorm)
testing_traj = torch.tensor(test_input_denorm)
inference_traj = torch.tensor(inference_input_denorm)

# 制作训练的train_batches
train_dataset = TensorDataset(training_traj)
train_dataloader = DataLoader(train_dataset, batch_size=params['train_b_size'], shuffle=True)

# 加载TUTR训练好的goal
pred_test = np.load('E:/Pycharm Project/Goal-generation-TUTR-V4/checkpoint/HighD/test_pred_trajs.npy') # B, num_pred, pred_len, 2.    HighD or Ngsim
goal_test = torch.tensor(pred_test[:, :, -1, :]) # 取最后一帧作为goal, [B, num_pred, 2]
inference_goals = np.load('E:/Pycharm Project/Goal-generation-TUTR-V4/checkpoint/HighD/inference_pred_trajs.npy') # B, num_pred, pred_len, 2.       HighD or Ngsim
goal_inference = torch.tensor(inference_goals[:, :, -1, :]) # 取最后一帧作为goal, [B, num_pred, 2]
print("相关数据已经加载完成，准备加载NSP模型")

# 模型和训练参数加载
model = NSP(params["input_size"], params["embedding_size"], params["rnn_size"], params["output_size"],\
            params["enc_dest_state_size"], params["dec_tau_size"])
model = model.to(torch.float32).to(device)
optimizer = optim.Adam(model.parameters(), lr = params["learning_rate"])
print("模型加载完成，准备进入epoch循环开始训练")

# best_ade = 7.85
# best_fde = 11.85
best_ade = 99
best_fde = 99
best_rmse = 99
best_epoch = 0
epsilon = 1e-8
mul = 1e3
# time_step = 0.08 # HighD数据为25hz(0.04)，Ngsim数据为10hz(0.1)
train_time = [] # 记录每个epoch训练时长

for e in range(params['num_epochs']):
    print("Start training epoch ", e)
    start_time = time.time()
    total_loss = train(train_dataloader)

    ade_20 = np.zeros((20, len(testing_traj)))
    fde_20 = np.zeros((20, len(testing_traj)))
    rmse_20 = np.zeros((20, len(testing_traj)))
    metrics_ade_20 = [np.zeros((20, len(testing_traj))) for _ in range(5)]
    metrics_fde_20 = [np.zeros((20, len(testing_traj))) for _ in range(5)]
    metrics_rmse_20 = [np.zeros((20, len(testing_traj))) for _ in range(5)]

    for j in range(20): # 取20个可能的goal的结果
        test_ade_, test_fde_, test_rmse_, metrics, _, _ = test(testing_traj, goal_test[:, j, :]) # test没有用batch，直接全部丢进去，只在test中使用了goal
        ade_20[j, :] = test_ade_
        fde_20[j, :] = test_fde_
        rmse_20[j, :] = test_rmse_

        for i in range(5):
            metrics_ade_20[i][j, :] = metrics['ade'][i]
            metrics_fde_20[i][j, :] = metrics['fde'][i]
            metrics_rmse_20[i][j, :] = metrics['rmse'][i]

    test_ade = np.mean(np.min(ade_20, axis=0))
    test_fde = np.mean(np.min(fde_20, axis=0))
    test_rmse = np.mean(np.min(rmse_20, axis=0))

    # 以ade为测试的目标
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

    # 打印模型预测1、2、3、4、5s分别的结果
    print('################## Print 1-5s results ########')
    for i in range(5):
        interval = (i + 1)
        ade_i = np.mean(np.min(metrics_ade_20[i], axis=0))
        fde_i = np.mean(np.min(metrics_fde_20[i], axis=0))
        rmse_i = np.mean(np.min(metrics_rmse_20[i], axis=0))
        print(f"Test ADE for {interval}s: {ade_i:.4f}")
        print(f"Test FDE for {interval}s: {fde_i:.4f}")
        print(f"Test RMSE for {interval}s: {rmse_i:.4f}")
    print("Epoch ", e, " finished")

if params['num_epochs'] > 0:
    print("训练总时长为：", sum(train_time))
    print("平均训练时长：", sum(train_time)/len(train_time))

####################################### 推理部分 ###############################################
print("开始推理（inference）")
# 加载最佳模型
checkpoint = torch.load('saved_models/' + args.save_file)
model.load_state_dict(checkpoint['model_state_dict'])

# 计算所有轨迹的最大最小值，确定统一的 xy 轴范围
x_min, y_min = np.min(inference_traj[:, :, :2].cpu().numpy(), axis=(0, 1))
x_max, y_max = np.max(inference_traj[:, :, :2].cpu().numpy(), axis=(0, 1))

inference_ade_20 = np.zeros((20, len(inference_traj)))
inference_fde_20 = np.zeros((20, len(inference_traj)))
inference_rmse_20 = np.zeros((20, len(inference_traj)))
inference_metrics_ade_20 = [np.zeros((20, len(inference_traj))) for _ in range(5)]
inference_metrics_fde_20 = [np.zeros((20, len(inference_traj))) for _ in range(5)]
inference_metrics_rmse_20 = [np.zeros((20, len(inference_traj))) for _ in range(5)]
inference_prediction_20 = np.zeros((20, len(inference_traj), params['future_length'], 2))

for j in range(20): # 取20个可能的goal的结果
    inference_ade_, inference_fde_, inference_rmse_, inference_metrics, inference_gt, inference_prediction = test(inference_traj, goal_inference[:, j, :])
    inference_prediction_20[j, :] = inference_prediction
    inference_ade_20[j, :] = inference_ade_
    inference_fde_20[j, :] = inference_fde_
    inference_rmse_20[j, :] = inference_rmse_

    for i in range(5):
        inference_metrics_ade_20[i][j, :] = inference_metrics['ade'][i]
        inference_metrics_fde_20[i][j, :] = inference_metrics['fde'][i]
        inference_metrics_rmse_20[i][j, :] = inference_metrics['rmse'][i]

inference_ade = np.mean(np.min(inference_ade_20, axis=0))
inference_fde = np.mean(np.min(inference_fde_20, axis=0))
inference_rmse = np.mean(np.min(inference_rmse_20, axis=0))

print("Inference ADE:", inference_ade)
print("Inference FDE:", inference_fde)
print("Inference RMSE:", inference_rmse)

# 打印1-5秒预测的三个指标结果
print('################## Print 1-5s results ########')
for i in range(5):
    interval = (i + 1)
    ade_i = np.mean(np.min(inference_metrics_ade_20[i], axis=0))
    fde_i = np.mean(np.min(inference_metrics_fde_20[i], axis=0))
    rmse_i = np.mean(np.min(inference_metrics_rmse_20[i], axis=0))
    print(f"Inference ADE for {interval}s: {ade_i:.4f}")
    print(f"Inference FDE for {interval}s: {fde_i:.4f}")
    print(f"Inference RMSE for {interval}s: {rmse_i:.4f}")

# 保存train goal的predictions用于最后的绘图
folder_path = 'saved_models'
save_path = os.path.join(folder_path, 'inference_prediction_20.npy')
np.save(save_path, inference_prediction_20)
print(f"Inference predictions saved successfully at {save_path}!")


# 用于可视化时，计算每个样本中20个预测结果的ade fde
def calculate_ade_fde(traj, predictions):
    """
    计算所有预测结果的 ADE 和 FDE。
    """
    future_len = predictions.shape[1]
    ade = np.mean(np.linalg.norm(predictions - traj[params['past_length']:, :2], axis=2), axis=1)
    fde = np.linalg.norm(predictions[:, -1, :] - traj[-1, :2], axis=1)
    return ade, fde

# 可视化部分
def visualize_trajectories(traj, predictions, ade_min_idx, title, idx):
    plt.figure(figsize=(10, 4))

    # 颜色列表
    color_list = ['#6EB576', '#A4B4B1', '#A78CC0', '#88BFC0', '#7A68AC', '#9077B6', '#F8BE7F', '#A5C7B0']
    pred_color = color_list[0]  # 预测轨迹颜色
    best_pred_color = color_list[2]  # 最佳预测轨迹颜色

    # 画历史部分
    plt.plot(traj[:params['past_length'], 0], traj[:params['past_length'], 1], 'gray', label='History', linewidth=2)

    # 画真实值
    plt.plot(traj[params['past_length']:, 0], traj[params['past_length']:, 1], 'k-', label='Ground Truth', linewidth=2)

    # 先画所有其他预测结果
    for j in range(20):
        if j != ade_min_idx:
            plt.plot(predictions[j, :, 0], predictions[j, :, 1], '-', color=pred_color, alpha=0.3,
                     label='Predictions' if j == 0 else "", linewidth=2)
            plt.plot(predictions[j, -1, 0], predictions[j, -1, 1], marker='*', markersize=5, color=pred_color,
                     alpha=0.3, label='Prediction End' if j == 0 else "", linewidth=2)

    # 再画最佳预测结果
    plt.plot(predictions[ade_min_idx, :, 0], predictions[ade_min_idx, :, 1], '-', color='#8B0000', alpha=0.9,
             label='Best Prediction', linewidth=3)
    plt.plot(predictions[ade_min_idx, -1, 0], predictions[ade_min_idx, -1, 1], marker='*', markersize=8, color='yellow',
             label='Best Prediction End', linewidth=2)

    # plt.legend()
    # plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-100, 200)
    plt.ylim(-4, 4)
    # plt.xticks(np.linspace(x_min, x_max, num=5))
    # plt.yticks(np.linspace(y_min, y_max, num=5))
    # 设置坐标轴比例相同
    plt.gca().set_aspect(15)
    os.makedirs('gallery/HighD', exist_ok=True)        # ngsim or HighD
    plt.savefig(f'gallery/HighD/{title}_{idx}.png')      # ngsim or HighD
    plt.close()

def visualize_on_highway(traj, predictions, ade_min_idx, title, idx, neighbor_traj, neighbor_existence, image_path='data/HighD/01_highway.png'):
    # 加载底图
    img = Image.open(image_path)
    img = img.resize((img.width, img.height * 2))
    img_width, img_height = img.size

    # 底图的物理尺寸（假设宽度为400米，y方向长度为45米）
    physical_width = 400
    physical_height = 45
    scale = img_width / physical_width
    y_scaling = img_height / physical_height

    # 设置 y 方向的 offset 为 y 方向总长度的 1/3
    y_offset = physical_height / 3 + 2.8

    plt.figure(figsize=(16, 9), dpi=300)
    plt.imshow(img, extent=[0, physical_width, 0, physical_height])

    # 将轨迹范围从 -150 到 250 调整到 0 到 400
    traj[:, 0] += 150
    traj[:, 1] += y_offset

    # 绘制历史轨迹
    history_x = traj[:params['past_length'], 0]
    history_y = traj[:params['past_length'], 1]
    plt.plot(history_x, history_y, 'k-', label='History', linewidth=4)

    # 绘制真实轨迹
    gt_x = traj[params['past_length']:, 0]
    gt_y = traj[params['past_length']:, 1]
    plt.plot(gt_x, gt_y, 'g-', label='Ground Truth', linewidth=4)

    # 颜色列表
    best_pred_color = '#8B0000'  # 深红色表示最佳预测轨迹颜色

    # 将预测结果范围从 -150 到 250 调整到 0 到 400
    predictions[:, :, 0] += 150
    predictions[:, :, 1] += y_offset

    # 仅绘制最佳预测结果
    pred_x = predictions[ade_min_idx, :, 0]
    pred_y = predictions[ade_min_idx, :, 1]
    plt.plot(pred_x, pred_y, '-', color=best_pred_color, alpha=0.7, label='Best Predicted Trajectory', linewidth=4)
    plt.plot(pred_x[-1], pred_y[-1], marker='*', markersize=10, color='y', label='Best Predicted Goal')

    # 绘制较大的点用于邻居轨迹标签展示
    plt.scatter([], [], c='b', alpha=1, s=30, label='Neighbor')
    # 绘制邻居车的历史轨迹（最后20个点，渐变颜色和大小）
    for neighbor_idx in range(neighbor_traj.shape[2]):
        if neighbor_existence[idx, neighbor_idx] == 1:
            neighbor_x = neighbor_traj[idx, -20:, neighbor_idx, 0] * scale  # 取最后20个点
            neighbor_y = neighbor_traj[idx, -20:, neighbor_idx, 1] + y_offset  # 取最后20个点
            for i in range(20):
                alpha = (i + 1) / 20
                size = (i + 1) * 1.2
                plt.scatter(neighbor_x[i], neighbor_y[i], c='b', alpha=alpha, s=size)

    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5)
    plt.title('(b)')
    plt.xlabel('Longitudinal Position (m)')
    plt.ylabel('Lateral Position (m)')
    plt.xlim(0, physical_width)
    plt.ylim(0, physical_height)

    os.makedirs('gallery/HighD/interpretability/chosen', exist_ok=True)  # ngsim or HighD
    plt.savefig(f'gallery/HighD/interpretability/chosen/{title}_{idx}.png', dpi=300, bbox_inches='tight')  # ngsim or HighD
    plt.close()

# fix_indices = np.arange(500, 1500)
# 用于可解释性的
# fix_indices = np.array([513, 574, 618, 619, 628, 636, 639, 695, 704, 723, 730, 737, 822, 891, 918, 1003, 1026, 1040, 1063, 1125, 1133, 1214, 1218, 1268, 1335])
# fix_indices = np.array([574])
# 用于多结果预测的
fix_indices = np.array([14, 205, 270, 277, 280, 343, 349, 368, 405, 507, 560, 599, 675, 730, 833, 834, 962,
                        22, 46, 55, 67, 77, 101, 176, 233, 301, 348, 386, 462, 464, 492, 755, 783, 874, 919, 1000,
                        25, 60, 122, 165, 173, 201, 324, 333, 371, 456, 479, 574, 618, 697, 842, 918])

# 随机选择一些数据进行可视化
indices = np.random.choice(len(inference_traj), 10, replace=False)

# 选择真实轨迹最后一个点y坐标绝对值最大和最小的轨迹进行可视化
sorted_indices = np.argsort(np.abs(inference_traj[:, -1, 1]))
indices = np.concatenate((indices, sorted_indices[:1], sorted_indices[-1:]))

for idx in fix_indices:
    traj = inference_traj[idx].cpu().numpy()
    predictions = np.array([inference_prediction_20[j, idx, :, :] for j in range(20)])
    ade, fde = calculate_ade_fde(traj, predictions)
    ade_min_idx = np.argmin(ade)
    fde_min_idx = np.argmin(fde)
    visualize_trajectories(traj, predictions, fde_min_idx, f'F_goal Predicted vs GT L23456', idx)
    # visualize_on_highway(traj, predictions, ade_min_idx, f'W highway Predicted vs GT', idx, inference_supplement_denorm, inference_nbr_existence)
print("绘图完成")

