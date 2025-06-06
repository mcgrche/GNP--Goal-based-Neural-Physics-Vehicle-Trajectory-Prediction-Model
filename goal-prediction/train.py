import argparse
import importlib
import json
import os

from torch.utils.data import DataLoader, TensorDataset

from model import TrajectoryModel
from torch import optim
import torch.nn.functional as F

from utils import calculate_coverage, calculate_best_match_error, show_loss, get_motion_modes_veh, calculate_rmse
# from load_data_2_240907 import *
from loaddata import *
# from loaddata_goal_gen import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

def load_motion_mode(args):
    file_suffix = '_motion_modes_125_highd_3.pkl'
    motion_modes_file = args.dataset_path + args.dataset_name + file_suffix

    if not os.path.exists(motion_modes_file):
        print('motion modes generating ... ')
        motion_modes = get_motion_modes_veh(train_dataset, args.obs_len, args.pred_len, hp_config.n_clusters,
                                        args.dataset_path, args.dataset_name,file_suffix,
                                        smooth_size=hp_config.smooth_size, random_rotation=hp_config.random_rotation,
                                        traj_seg=hp_config.traj_seg)
        motion_modes = torch.tensor(motion_modes, dtype=torch.float32).cuda()

    if os.path.exists(motion_modes_file):
        print('motion modes loading ... ')
        import pickle
        f = open(motion_modes_file, 'rb+')
        motion_modes = pickle.load(f)
        f.close()
        motion_modes = torch.tensor(motion_modes, dtype=torch.float32).cuda()
    file_name = file_suffix.rsplit('.', 1)[0]
    return motion_modes, file_name

def get_cls_label(gt, motion_modes, soft_label=True):
    # 根据给定的真实轨迹（gt）和一系列预测模式（motion_modes）来计算每个真实轨迹相对于这些预测模式的分类标签，例如概率分布等软标签

    # motion_modes [K pred_len 2]
    # gt [B,pred_len, 2]

    gt = gt.reshape(gt.shape[0], -1).unsqueeze(1)  # [B 1 pred_len*2]
    motion_modes = motion_modes.reshape(motion_modes.shape[0], -1).unsqueeze(0)  # [1 K pred_len*2]
    distance = torch.norm(gt - motion_modes, dim=-1)  # [B K], 计算每个真实goal与每个motion mode最后一帧之间的距离
    soft_label = F.softmax(-distance, dim=-1) # [B K],每个真实轨迹对应的预测模式的概率分布
    closest_mode_indices = torch.argmin(distance, dim=-1) # [B]，每个真实轨迹最接近的预测模式的索引
 
    return soft_label, closest_mode_indices

def plot_inference(pred_trajs_denorm, gt, gt_his, sample_indices,min_ade_indices):

    for index in sample_indices:
        plt.figure(figsize=(10, 8))

        # # 绘制所有预测轨迹
        # for pred_index in range(pred_trajs_denorm.shape[1]):
        #     pred_traj = pred_trajs_denorm[index, pred_index]
        #     plt.plot(pred_traj[:, 0], pred_traj[:, 1], label=f'Prediction {pred_index+1}', alpha=0.6)
        #     # 绘制预测轨迹的最后一个点
        #     plt.scatter(pred_traj[-1, 0], pred_traj[-1, 1], color='blue', s=20, zorder=5, marker='*')

        # 绘制min ade的预测轨迹
        pred_traj = pred_trajs_denorm[index, min_ade_indices[index]]
        plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', label='Min ADE Prediction', alpha=0.6)
        plt.scatter(pred_traj[-1, 0], pred_traj[-1, 1], color='blue', s=20, zorder=5, marker='*')

        # 绘制真实轨迹
        gt_traj = gt[index]
        gt_his_traj = gt_his[index]
        plt.plot(gt_traj[:, 0], gt_traj[:, 1], label='Ground Truth', color='black', linewidth=4)
        # 绘制真实轨迹的最后一个点
        plt.scatter(gt_traj[-1, 0], gt_traj[-1, 1], color='red', s=60, zorder=5, marker='*', label='Final Ground Truth')
        # 绘制历史轨迹
        plt.plot(gt_his_traj[:, 0], gt_his_traj[:, 1], label='His Ground Truth', color='gray', linewidth=4)

        # 添加图例和标题
        plt.ylim(-4, 4)
        # plt.legend()
        plt.title(f'Sample {index} Trajectory Comparison')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.legend()
        plot_path = "./gallery/pred_gt/" + args.dataset_name
        os.makedirs(plot_path, exist_ok=True)
        plot_filename = f"/trajectory_comparison_sample_{index}.png"
        plt.savefig(plot_path + plot_filename)
        plt.close()  # Close the figure to free up memory
        # plt.show()
    print("predictions与ground truth对比已保存在gallery/pred_gt/")

def train(epoch, model, reg_criterion, cls_criterion, mse_criterion, optimizer, train_dataloader, motion_modes):
    model.train()
    total_loss = []
    alfa = 0.2 # 原本的cls和reg loss的比重
    fde_weight = 0  # 这个权重因子可以根据你想要的 FDE 损失在总损失中的比重来调整

    for i, (ped, neis, mask, mask_time) in enumerate(train_dataloader):
        ped = ped.cuda() # [B, seq_len, 4]
        neis = neis.cuda() # [B, num_nei, seq_len, 4]
        mask = mask.cuda() # [B, num_nei], mask用来遮挡不存在的邻居信息
        mask_time = mask_time.cuda() # [B, num_nei, seq_len], mask用来遮挡不存在的邻居信息

        ped_obs = ped[:, :args.obs_len] # [B, OB_HORIZON, 4]
        gt = ped[:, args.obs_len:, 0:2] # [B, PRED_Horizon, 2]
        neis_obs = neis[:, :, :args.obs_len] # [B,num_nei, OB_HORIZON, 4]

        with torch.no_grad():
            soft_label, closest_mode_indices = get_cls_label(gt, motion_modes) #
        
        optimizer.zero_grad()
        pred_traj, scores = model(ped_obs, neis_obs, motion_modes, mask, mask_time, closest_mode_indices)
        reg_label = gt.reshape(pred_traj.shape) #[B, PRED_HORIZON * 2]
        # reg_label = gt
        reg_loss = reg_criterion(pred_traj, reg_label)
        # mse_loss = mse_criterion(pred_traj, reg_label)
        clf_loss = cls_criterion(scores.squeeze(),  soft_label) # 计算模型预测的轨迹概率（与motion mode数量相同）与gt的轨迹p_hat的差值

        pred_traj_reshaped = pred_traj.view(-1, args.pred_len, 2)
        reg_label_reshaped = reg_label.view(-1, args.pred_len, 2)
        fde_loss = reg_criterion(pred_traj_reshaped[:, -1], reg_label_reshaped[:, -1])
        loss = (1-alfa) * reg_loss + alfa * clf_loss + fde_weight * fde_loss
        # 切换为RMSE loss
        # loss = mse_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss.append(loss.item())

    return total_loss

def vis_predicted_trajectories(obs_traj, gt, pred_trajs, pred_probabilities, min_index):


    # obs_traj [B T_obs 2]
    # gt [B T_pred 2]
    # pred_trajs [B 20 T_pred 2]
    # pred_probabilities [B 20]

    for i in range(obs_traj.shape[0]):
        plt.clf()
        curr_obs = obs_traj[i].cpu().numpy() # [T_obs 2]
        curr_gt = gt[i].cpu().numpy()
        curr_preds = pred_trajs[i].cpu().numpy()
     
        curr_pros = pred_probabilities[i].cpu().numpy()
        curr_min_index = min_index[i].cpu().numpy()
        obs_x = curr_obs[:, 0]
        obs_y = curr_obs[:, 1]
        gt_x = np.concatenate((obs_x[-1:], curr_gt[:, 0]))
        gt_y = np.concatenate((obs_y[-1:], curr_gt[:, 1]))
        plt.plot(obs_x, obs_y, marker='o', color='green')
        plt.plot(gt_x, gt_y, marker='o', color='blue')
        plt.scatter(gt_x[-1], gt_y[-1], marker='*', color='blue', s=300)
       
        for j in range(curr_preds.shape[0]):
        
            pred_x = np.concatenate((obs_x[-1:], curr_preds[j][:, 0]))
            pred_y = np.concatenate((obs_y[-1:], curr_preds[j][:, 1]))
            if j == curr_min_index:
                plt.plot(pred_x, pred_y, ls='-.', lw=2.0, color='red')
                plt.scatter(pred_x[-1], pred_y[-1], marker='*', color='orange', s=300)
            else:
                plt.plot(pred_x, pred_y, ls='-.', lw=0.5, color='red')
                plt.scatter(pred_x[-1], pred_y[-1], marker='*', color='red', s=300)
            plt.text(pred_x[-1], pred_y[-1], str("%.2f" % curr_pros[j]),  ha='center')
            
        
        plt.tight_layout()
        save_path = './fig/' + args.dataset_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + '/' + str(time.time()) + '.png')

    return

def hex_to_rgb(hex_color):
    """Convert HEX color to RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

def show_motion_modes(motion_modes, train_dataset, args, feature_destd, file_suffix):
    # 颜色列表，按图中提供的顺序
    # color_list = ['#6EB576', '#A4B4B1', '#A78CC0', '#88BFC0', '#7A68AC', '#9077B6', '#F8BE7F', '#A5C7B0']
    # colors = [hex_to_rgb(color) for color in color_list]

    color_list = ['#8fd7d7', '#00b0be', '#ff8ca1', '#f45f74', '#bdd373', '#98c127', '#ffcd8e', '#ffb255']
    colors = [hex_to_rgb(color) for color in color_list]

    muted_color_list = ['#c8c8c8', '#f0c571', '#59a89c', '#0b81a2', '#e25759', '#9d2c00', '#7e4794', '#36b700']
    muted_colors = [hex_to_rgb(color) for color in muted_color_list]

    bright_color_list = ['#003a7d', '#008dff', '#ff73b6', '#c701ff', '#4ecb8d', '#ff9d3a', '#f9e858', '#d83034']
    bright_colors = [hex_to_rgb(color) for color in bright_color_list]

    # 解包数据集以获得预测数据
    data_agent, _, _, _ = train_dataset.tensors
    # 选取观察结束后的轨迹作为预测数据
    pred_data = data_agent[:, args.obs_len:, :2]
    full_traj_data = data_agent[:, :, :2]

    # 确保数据在CPU上，并转换为numpy数组
    motion_modes = motion_modes.cpu().numpy()
    pred_data = pred_data.cpu().numpy()
    full_traj_data = full_traj_data.cpu().numpy()
    feature_destd = feature_destd.cpu().numpy()

    # 反标准化操作，plot现实数据的情况
    mean = feature_destd[:2,0]
    std = feature_destd[:2,1]
    motion_modes = motion_modes * std[np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, :]
    pred_data = pred_data * std[np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, :]
    full_traj_data = full_traj_data * std[np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, :]

    # 随机选择几个预测轨迹来显示
    num_seqs_to_display = min(5, pred_data.shape[0])  # 最多展示5个序列
    chosen_indices = np.random.choice(pred_data.shape[0], num_seqs_to_display, replace=False)
    chosen_pred_data = pred_data[chosen_indices]
    # chosen_pred_data = full_traj_data[chosen_indices]

    # 创建一个图表以展示数据
    plt.figure(figsize=(6, 4))

    # 获取颜色映射
    # colors = cm.rainbow(np.linspace(0, 1, motion_modes.shape[0]))

    # 画出每个聚类中心的轨迹并显示最后一个点
    for i in range(motion_modes.shape[0]):
        color = muted_colors[i % len(muted_colors)]
        plt.plot(motion_modes[i, :, 0], motion_modes[i, :, 1], marker='o', markersize=1, color=color, linewidth=1)
        plt.plot(motion_modes[i, -1, 0], motion_modes[i, -1, 1], marker='*', markersize=10, color=color)

    # 画出每个聚类中心的轨迹
    # for i in range(motion_modes.shape[0]):
    #     plt.plot(motion_modes[i, :, 0], motion_modes[i, :, 1], marker='o', markersize=2, color=colors[i], label=f'Motion Mode {i+1}')

    # 画出选定的真实轨迹预测
    # for i in range(chosen_pred_data.shape[0]):
    #     plt.plot(chosen_pred_data[i, :, 0], chosen_pred_data[i, :, 1], linestyle='--', linewidth= 2, color='red', label=f'Pred Seq {i+1}')

    plt.xlim(0, 150)
    plt.ylim(-8,8)
    plt.gca().set_aspect(10)
    # plt.legend()
    plt.title('(b) Clustered Intention Modes for Ngsim Dataset')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True)
    plt.savefig('gallery/' + args.dataset_name + file_suffix + '_cluster.png')
    plt.show()

def test(model, test_dataloader, motion_modes, feature_destd, mse_criterion):

    model.eval()

    total_best_match_error = 0
    total_coverage = 0
    ade = 0
    fde = 0
    total_squared_error = 0.0
    num_traj = 0

    # 反标准化操作，plot现实数据的情况
    mean = feature_destd[:2,0]
    std = feature_destd[:2,1]

    for (ped, neis, mask, mask_time) in test_dataloader:

        ped = ped.cuda() # [B,  seq_len, 4]
        neis = neis.cuda() # [B, num_nei, seq_len, 4]
        mask = mask.cuda() # [B, num_nei], mask用来遮挡不存在的邻居信息
        mask_time = mask_time.cuda()

        ped_obs = ped[:, :args.obs_len]
        gt = ped[:, args.obs_len:, :2] # [B, PRED_HORIZON, 2]
        neis_obs = neis[:, :, :args.obs_len] # [B,num_nei, OB_HORIZON, 4]

        with torch.no_grad():
            
            num_traj += ped_obs.shape[0]
            pred_trajs, scores = model(ped_obs, neis_obs, motion_modes, mask, mask_time, closest_mode_indices=None, test=True)
            # top_k_scores = torch.topk(scores, k=20, dim=-1).values
            # top_k_scores = F.softmax(top_k_scores, dim=-1)
            pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[1], 2)

            #将pred_trajs和gt反标准化
            mean_expanded = mean.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(pred_trajs)
            std_expanded = std.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(pred_trajs)
            pred_trajs_denorm = pred_trajs * std_expanded + mean_expanded
            mean_expanded_gt = mean.unsqueeze(0).unsqueeze(0).expand(gt.size(0), gt.size(1), 2)
            std_expanded_gt = std.unsqueeze(0).unsqueeze(0).expand(gt.size(0), gt.size(1), 2)
            gt_denorm = gt * std_expanded_gt + mean_expanded_gt

            # print("检查test pred_trajs_denorm：", pred_trajs_denorm[0][0])
            # print("检查test gt_denorm：", gt_denorm[0])

            gt_ = gt_denorm.unsqueeze(1)
            norm_ = torch.norm(pred_trajs_denorm - gt_, p=2, dim=-1)
            ade_ = torch.mean(norm_, dim=-1) # [B, num_k]
            fde_ = norm_[:, :, -1]
            min_ade, min_ade_index = torch.min(ade_, dim=-1) #[B,] 从多个预测结果中，选择ade最小的
            min_fde, min_fde_index = torch.min(fde_, dim=-1)

            min_ade = torch.sum(min_ade) #[1]
            min_fde = torch.sum(min_fde)
            ade += min_ade.item()
            fde += min_fde.item()

            # TODO: 计算rmse指标时，或许应该选择最小值？
            gt_expanded = gt_.expand_as(pred_trajs_denorm)
            # 计算每个预测和真实数据之间的 MSE
            batch_mse = mse_criterion(pred_trajs_denorm, gt_expanded)  # [batch_size, n_predictions, seq_len, 2]
            batch_mse = torch.mean(batch_mse, dim=[2, 3])  # 对每个轨迹点求平均，[batch_size, n_predictions]
            min_mse = torch.min(batch_mse, dim=1)[0]  # [batch_size]
            # print(torch.sum(min_mse).item())
            total_squared_error += torch.sum(min_mse).item() # 累加bathch的所有最小 MSE

            # # TODO: 手写的mse计算函数，验证rmse计算过程是否有误
            # # 计算预测和真实轨迹之间的差异
            # diff = pred_trajs - gt_expanded
            # # 计算差异的平方
            # squared_diff = diff ** 2
            # # 对所有时间点的差异平方进行平均
            # mse_per_traj_per_pred = torch.mean(squared_diff, dim=(2, 3))  # [batch_size, n_predictions]
            # # 选择每个批次中最小的 MSE
            # min_mse_per_batch_ = torch.min(mse_per_traj_per_pred, dim=1)  # [batch_size]
            # min_mse_per_batch = min_mse_per_batch_[0]
            # # 累加最小 MSE
            # print(torch.sum(min_mse_per_batch).item())
            # total_squared_error += torch.sum(min_mse_per_batch).item()  # 累加 batch 的所有最小 MSE

            # 计算两种指标：1.准确率，即所有goal中的最小距离；2.覆盖率，即所有预测goal中在真实值附近2m的比例
            # best_match_error = calculate_best_match_error(pred_trajs, gt)
            # coverage = calculate_coverage(pred_trajs, gt, threshold=2.0)
            #
            # total_best_match_error += best_match_error * ped.shape[0] # 计算一个batch的总误差，因为每个best_match_error是batch的平均值
            # total_coverage += coverage * ped.shape[0]

    # avg_best_match_error = total_best_match_error / num_traj
    # avg_coverage = total_coverage / num_traj
    # return avg_best_match_error, avg_coverage, num_traj
    ade = ade / num_traj
    fde = fde / num_traj
    total_mse = total_squared_error / num_traj
    total_rmse = torch.sqrt(torch.tensor(total_mse))
    total_rmse = total_rmse.item()
    return ade, fde, total_rmse, num_traj

# inference函数，使用单独的一批数据，用最优模型参数进行结果推断
def inference(model, inference_dataloader, motion_modes, feature_destd, mse_criterion, device='cuda'):
    # 加载模型
    model = model # 初始化你的模型结构
    model_path = args.checkpoint + args.dataset_name
    model.load_state_dict(torch.load(model_path + args.save_param_name))
    # model = model.to(device)
    model.eval()  # 设置为评估模式

    # total_best_match_error = 0
    # total_coverage = 0
    ade = 0
    fde = 0
    num_traj = 0
    total_squared_error = 0.0

    # 反标准化操作，plot现实数据的情况
    mean = feature_destd[:2, 0]
    std = feature_destd[:2, 1]

    all_pred_trajs = []
    all_scores = []

    for (ped, neis, mask, mask_time) in inference_dataloader:
        # 将数据转移到设备上
        ped = ped.cuda()   # [B, seq_len, 4]
        neis = neis.cuda()   # [B, num_nei, seq_len, 4]
        mask = mask.cuda()   # [B, num_nei], mask用来遮挡不存在的邻居信息
        mask_time = mask_time.cuda()

        ped_obs = ped[:, :args.obs_len]  # 观测到的行人数据
        gt = ped[:, args.obs_len:, :2]  # 真实的最后位置 [B, PRED_HORIZON, 2]
        neis_obs = neis[:, :, :args.obs_len]  # 观测到的邻居数据 [B, num_nei, OB_HORIZON, 4]

        with torch.no_grad():
            num_traj += ped_obs.shape[0]
            pred_trajs, scores = model(ped_obs, neis_obs, motion_modes, mask, mask_time, closest_mode_indices=None, test=True)

            all_pred_trajs.append(pred_trajs)
            all_scores.append(scores)

            pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[1], 2)

            # 进行反归一化
            mean_expanded = mean.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(pred_trajs)
            std_expanded = std.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(pred_trajs)
            pred_trajs_denorm = pred_trajs * std_expanded + mean_expanded
            mean_expanded_gt = mean.unsqueeze(0).unsqueeze(0).expand(gt.size(0), gt.size(1), 2)
            std_expanded_gt = std.unsqueeze(0).unsqueeze(0).expand(gt.size(0), gt.size(1), 2)
            gt_denorm = gt * std_expanded_gt + mean_expanded_gt

            gt_ = gt_denorm.unsqueeze(1)
            norm_ = torch.norm(pred_trajs_denorm - gt_, p=2, dim=-1)
            ade_ = torch.mean(norm_, dim=-1) # [B, num_k]
            fde_ = norm_[:, :, -1]
            min_ade, min_ade_index = torch.min(ade_, dim=-1) #[B,] 从多个预测结果中，选择ade最小的
            min_fde, min_fde_index = torch.min(fde_, dim=-1)

            min_ade = torch.sum(min_ade) #[1]
            min_fde = torch.sum(min_fde)
            ade += min_ade.item()
            fde += min_fde.item()

            # TODO: 计算rmse指标
            gt_expanded = gt_.expand_as(pred_trajs_denorm)
            # 计算每个预测和真实数据之间的 MSE
            batch_mse = mse_criterion(pred_trajs_denorm, gt_expanded)  # [batch_size, n_predictions, seq_len, 2]
            batch_mse = torch.mean(batch_mse, dim=[2, 3])  # 对每个轨迹点求平均，[batch_size, n_predictions]
            min_mse = torch.min(batch_mse, dim=1)[0]  # [batch_size]
            total_squared_error += torch.sum(min_mse).item() # 累加bathch的所有最小 MSE

            # 计算误差指标
            # best_match_error = calculate_best_match_error(pred_trajs, gt)
            # coverage = calculate_coverage(pred_trajs, gt, threshold=2.0)
            #
            # total_best_match_error += best_match_error * ped.shape[0]
            # total_coverage += coverage * ped.shape[0]

    ade = ade / num_traj
    fde = fde / num_traj
    total_mse = total_squared_error / num_traj
    total_rmse = torch.sqrt(torch.tensor(total_mse))
    total_rmse = total_rmse.item()

    # 保存最后的推断结果
    all_pred_trajs = torch.cat(all_pred_trajs, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    np.save(model_path + '/inference_pred_trajs.npy', all_pred_trajs.cpu())
    np.save(model_path + '/inference_scores.npy', all_scores.cpu())

    # 计算平均误差和覆盖率
    # avg_best_match_error = total_best_match_error / num_traj
    # avg_coverage = total_coverage / num_traj

    # 返回结果
    return ade, fde, total_rmse, num_traj

# 240909 由于inference和test部分数据集变大，还是需要采用分批次的方式
def inference2(model, inference_dataloader, motion_modes, feature_destd, mse_criterion, output=False, device='cuda'):
    # 加载模型
    model = model  # 初始化你的模型结构
    model_path = args.checkpoint + args.dataset_name
    model.load_state_dict(torch.load(model_path + args.save_param_name))
    model.eval()  # 设置为评估模式

    ade = 0
    fde = 0
    num_traj = 0
    total_squared_error = 0.0

    # 反标准化操作，plot现实数据的情况
    mean = feature_destd[:2, 0]
    std = feature_destd[:2, 1]

    all_pred_trajs = []
    all_scores = []
    # Time tracking variables
    batch_times = []
    total_samples = 0

    for (ped, neis, mask, mask_time) in inference_dataloader:
        # 将数据转移到设备上
        ped = ped.cuda()  # [B, seq_len, 4]
        neis = neis.cuda()  # [B, num_nei, seq_len, 4]
        mask = mask.cuda()  # [B, num_nei], mask用来遮挡不存在的邻居信息
        mask_time = mask_time.cuda()

        ped_obs = ped[:, :args.obs_len]  # 观测到的行人数据
        gt = ped[:, args.obs_len:, :2]  # 真实的最后位置 [B, PRED_HORIZON, 2]
        gt_his = ped[:, :args.obs_len, :2]  # 历史轨迹
        neis_obs = neis[:, :, :args.obs_len]  # 观测到的邻居数据 [B, num_nei, OB_HORIZON, 4]

        with torch.no_grad():
            num_traj += ped_obs.shape[0]
            model_start_time = time.time()

            pred_trajs, scores = model(ped_obs, neis_obs, motion_modes, mask, mask_time, closest_mode_indices=None, test=True)

            model_end_time = time.time()

            pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[1], 2)

            # 进行反归一化
            mean_expanded = mean.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(pred_trajs)
            std_expanded = std.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(pred_trajs)
            pred_trajs_denorm = pred_trajs * std_expanded + mean_expanded
            mean_expanded_gt = mean.unsqueeze(0).unsqueeze(0).expand(gt.size(0), gt.size(1), 2)
            std_expanded_gt = std.unsqueeze(0).unsqueeze(0).expand(gt.size(0), gt.size(1), 2)
            gt_denorm = gt * std_expanded_gt + mean_expanded_gt
            mean_expanded_gt_his = mean.unsqueeze(0).unsqueeze(0).expand(gt_his.size(0), gt_his.size(1), 2)
            std_expanded_gt_his = std.unsqueeze(0).unsqueeze(0).expand(gt_his.size(0), gt_his.size(1), 2)
            gt_his_denorm = gt_his * std_expanded_gt_his + mean_expanded_gt_his

            # 保存 denormalized 后的预测结果
            all_pred_trajs.append(pred_trajs_denorm)
            all_scores.append(scores)

            gt_ = gt_denorm.unsqueeze(1)
            norm_ = torch.norm(pred_trajs_denorm - gt_, p=2, dim=-1)
            ade_ = torch.mean(norm_, dim=-1)  # [B, num_k]
            fde_ = norm_[:, :, -1]
            min_ade, min_ade_index = torch.min(ade_, dim=-1)  # [B,] 从多个预测结果中，选择ade最小的
            min_fde, min_fde_index = torch.min(fde_, dim=-1)

            min_ade = torch.sum(min_ade)  # [1]
            min_fde = torch.sum(min_fde)
            ade += min_ade.item()
            fde += min_fde.item()

            # 计算 rmse 指标
            gt_expanded = gt_.expand_as(pred_trajs_denorm)
            batch_mse = mse_criterion(pred_trajs_denorm, gt_expanded)  # [batch_size, n_predictions, seq_len, 2]
            batch_mse = torch.mean(batch_mse, dim=[2, 3])  # 对每个轨迹点求平均，[batch_size, n_predictions]
            min_mse = torch.min(batch_mse, dim=1)[0]  # [batch_size]
            total_squared_error += torch.sum(min_mse).item()  # 累加 batch 的所有最小 MSE

        # End timing for this batch
        batch_time = model_end_time - model_start_time
        batch_times.append(batch_time)
        total_samples += ped.shape[0]

    ade = ade / num_traj
    fde = fde / num_traj
    total_mse = total_squared_error / num_traj
    total_rmse = torch.sqrt(torch.tensor(total_mse)).item()

    # Calculate average times
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    avg_sample_time = sum(batch_times) / total_samples if total_samples > 0 else 0

    # Print timing information
    print(f"\n=== Inference Time Statistics ===")
    print(f"Total batches processed: {len(batch_times)}")
    print(f"Total samples processed: {total_samples}")
    print(f"Average time per batch: {avg_batch_time:.4f} seconds")
    print(f"Average time per sample: {avg_sample_time * 1000:.2f} milliseconds")
    print(f"Total inference time: {sum(batch_times):.2f} seconds")
    print("================================\n")

    # 保存最后的推断结果
    all_pred_trajs = torch.cat(all_pred_trajs, dim=0)
    all_scores = torch.cat(all_scores, dim=0)

    if output:  # 表示需要输出train_goals需要的test goal部分
        np.save(model_path + '/test_pred_trajs.npy', all_pred_trajs.cpu())
        np.save(model_path + '/test_scores.npy', all_scores.cpu())
    else:  # 正常的inference
        np.save(model_path + '/inference_pred_trajs.npy', all_pred_trajs.cpu())
        np.save(model_path + '/inference_scores.npy', all_scores.cpu())

        # 可视化预测轨迹与真实轨迹
        last_y_coordinates = gt_denorm[:, -1, 1]  # Select the last frame's y-coordinate
        _, sorted_indices = torch.sort(last_y_coordinates, descending=True)  # Sort by y-coordinate
        top_indices = sorted_indices[:50]  # Select top trajectories based on y-coordinate
        plot_inference(pred_trajs_denorm.cpu(), gt_denorm.cpu(), gt_his_denorm.cpu(), top_indices, min_ade_index.cpu())

    # 返回结果
    return ade, fde, total_rmse, num_traj

# 重写一个inference函数，直接加载全部数据，不需要进行dataloader
def inference_all(model, inference_traj, inference_supplement, inference_mask, inference_mask_time,  motion_modes, feature_destd, mse_criterion, output=False, device='cuda'):
    # 加载模型
    model = model # 初始化你的模型结构
    model_path = args.checkpoint + args.dataset_name
    model.load_state_dict(torch.load(model_path + args.save_param_name))
    # model = model.to(device)
    model.eval()  # 设置为评估模式

    ade = 0
    fde = 0
    num_traj = 0
    total_squared_error = 0.0

    # 反标准化操作，plot现实数据的情况
    mean = feature_destd[:2, 0]
    std = feature_destd[:2, 1]

    all_pred_trajs = []
    all_scores = []

    ped = inference_traj.cuda()  # [B, seq_len, 4]
    neis = inference_supplement.cuda()  # [B, num_nei, seq_len, 4]
    mask = inference_mask.cuda()  # [B, num_nei], mask用来遮挡不存在的邻居信息
    mask_time = inference_mask_time.cuda()

    ped_obs = ped[:, :args.obs_len]  # 观测到的行人数据
    gt = ped[:, args.obs_len:, :2]  # 真实的最后位置 [B, PRED_HORIZON, 2]
    gt_his = ped[:, :args.obs_len, :2]
    neis_obs = neis[:, :, :args.obs_len]  # 观测到的邻居数据 [B, num_nei, OB_HORIZON, 4]

    with torch.no_grad():
        num_traj += ped_obs.shape[0]
        pred_trajs, scores = model(ped_obs, neis_obs, motion_modes, mask, mask_time, closest_mode_indices=None, test=True)

        # all_pred_trajs.append(pred_trajs)
        all_scores.append(scores)

        pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[1], 2)

        # 进行反归一化
        mean_expanded = mean.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(pred_trajs)
        std_expanded = std.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(pred_trajs)
        pred_trajs_denorm = pred_trajs * std_expanded + mean_expanded
        mean_expanded_gt = mean.unsqueeze(0).unsqueeze(0).expand(gt.size(0), gt.size(1), 2)
        std_expanded_gt = std.unsqueeze(0).unsqueeze(0).expand(gt.size(0), gt.size(1), 2)
        gt_denorm = gt * std_expanded_gt + mean_expanded_gt
        mean_expanded_gt_his = mean.unsqueeze(0).unsqueeze(0).expand(gt_his.size(0), gt_his.size(1), 2)
        std_expanded_gt_his = std.unsqueeze(0).unsqueeze(0).expand(gt_his.size(0), gt_his.size(1), 2)
        gt_his_denorm = gt_his * std_expanded_gt_his + mean_expanded_gt_his

        all_pred_trajs.append(pred_trajs_denorm)

        gt_ = gt_denorm.unsqueeze(1)
        norm_ = torch.norm(pred_trajs_denorm - gt_, p=2, dim=-1)
        ade_ = torch.mean(norm_, dim=-1)  # [B, num_k]
        fde_ = norm_[:, :, -1]
        min_ade, min_ade_index = torch.min(ade_, dim=-1)  # [B,] 从多个预测结果中，选择ade最小的
        min_fde, min_fde_index = torch.min(fde_, dim=-1)

        min_ade = torch.sum(min_ade)  # [1]
        min_fde = torch.sum(min_fde)
        ade += min_ade.item()
        fde += min_fde.item()

        # TODO: 计算rmse指标
        gt_expanded = gt_.expand_as(pred_trajs_denorm)
        # 计算每个预测和真实数据之间的 MSE
        batch_mse = mse_criterion(pred_trajs_denorm, gt_expanded)  # [batch_size, n_predictions, seq_len, 2]
        batch_mse = torch.mean(batch_mse, dim=[2, 3])  # 对每个轨迹点求平均，[batch_size, n_predictions]
        min_mse = torch.min(batch_mse, dim=1)[0]  # [batch_size]
        total_squared_error += torch.sum(min_mse).item()  # 累加bathch的所有最小 MSE

    ade = ade / num_traj
    fde = fde / num_traj
    total_mse = total_squared_error / num_traj
    total_rmse = torch.sqrt(torch.tensor(total_mse))
    total_rmse = total_rmse.item()

    # 保存最后的推断结果
    all_pred_trajs = torch.cat(all_pred_trajs, dim=0)
    all_scores = torch.cat(all_scores, dim=0)

    # 保存test或inference的 goal
    if output: # 表示需要输出train_goals需要的test goal部分
        np.save(model_path + '/test_pred_trajs.npy', all_pred_trajs.cpu())
        np.save(model_path + '/test_scores.npy', all_scores.cpu())
    else: # 正常的inference
        np.save(model_path + '/inference_pred_trajs.npy', all_pred_trajs.cpu())
        np.save(model_path + '/inference_scores.npy', all_scores.cpu())

        # 画图对比预测轨迹与真实轨迹
        # sample_indices = [1000,1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        # Sort trajectories by the last y-coordinate and pick the ones with the largest values
        last_y_coordinates = gt_denorm[:, -1, 1]  # Select the last frame's y-coordinate
        _, sorted_indices = torch.sort(last_y_coordinates, descending=True)  # Sort by y-coordinate
        top_indices = sorted_indices[:50]  # Select top trajectories based on y-coordinate
        plot_inference(pred_trajs_denorm.cpu(), gt_denorm.cpu(), gt_his_denorm.cpu(), top_indices, min_ade_index.cpu())

    # 计算平均误差和覆盖率
    # avg_best_match_error = total_best_match_error / num_traj
    # avg_coverage = total_coverage / num_traj

    # 返回结果
    return ade, fde, total_rmse, num_traj

# min_avg_coverage = 9999
# min_avg_best_match_error = 9999
min_ade = 99
min_fde = 99
min_rmse = 999

if __name__ == '__main__':

    # 可能需要添加的 freeze_support() 调用
    # from multiprocessing import freeze_support
    # freeze_support()

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='./dataset/')
    parser.add_argument('--dataset_name', type=str, default='HighD')   # HighD or Ngsim
    parser.add_argument('--save_param_name', type=str, default='/bestL23456_250524.pth')
    parser.add_argument("--hp_config", type=str, default='./config/HighD.py', help='hyper-parameter') # HighD or Ngsim
    parser.add_argument('--lr_scaling', action='store_true', default=True)
    parser.add_argument('--num_works', type=int, default=8)
    parser.add_argument('--obs_len', type=int, default=75)
    parser.add_argument('--pred_len', type=int, default=125)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--data_scaling', type=list, default=[1.9, 0.4])
    parser.add_argument('--dist_threshold', type=float, default=2)
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/')

    args = parser.parse_args()

    # json_file_path = os.path.join('./config/', 'args_ngsim.json')
    # with open(json_file_path, 'w') as f:
    #     json.dump(vars(args), f)
    # print(f"Parameters saved to {json_file_path}")

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(args)

    # python train.py --dataset_name sdd --gpu 0 --hp_config config/sdd.py
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    spec = importlib.util.spec_from_file_location("hp_config", args.hp_config)
    hp_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hp_config)

    # 加载处理好的train test inference data，包括agent supplement以及nbr_existence
    # 已完成：外围点清理，相对位置转换，minmax归一化
    training_input, training_supplement, training_nbr_existence,training_nbr_existence_time,\
        test_input, testing_supplement, testing_nbr_existence, testing_nbr_existence_time,\
        inference_input, inference_supplement, inference_nbr_existence,inference_nbr_existence_time, feature_destd = dir_load_data()

    # 取XY VX VY，并转换为tensor
    training_traj = torch.tensor(training_input[:, :, 0:4], dtype=torch.float32)
    training_supplement = torch.tensor(training_supplement, dtype=torch.float32)
    training_supplement = training_supplement.permute(0, 2, 1, 3).contiguous()

    testing_traj = torch.tensor(test_input[:, :, 0:4], dtype=torch.float32)
    testing_supplement = torch.tensor(testing_supplement, dtype=torch.float32)
    testing_supplement = testing_supplement.permute(0, 2, 1, 3).contiguous()

    inference_traj = torch.tensor(inference_input[:, :, 0:4], dtype=torch.float32)
    inference_supplement = torch.tensor(inference_supplement, dtype=torch.float32)
    inference_supplement = inference_supplement.permute(0, 2, 1, 3).contiguous()

    feature_destd = torch.tensor(feature_destd, dtype=torch.float32).cuda()
    print(feature_destd)
    num_feat = training_supplement.shape[-1]

    # 通过mask遮挡不存在的邻居，第一种mask，seq中每个位置用01表示
    train_mask = training_nbr_existence
    train_mask = torch.tensor(train_mask[:,:-1], dtype=torch.float32)

    test_mask = testing_nbr_existence
    test_mask = torch.tensor(test_mask[:,:-1], dtype=torch.float32)

    inference_mask = inference_nbr_existence
    inference_mask = torch.tensor(inference_mask[:,:-1], dtype=torch.float32)

    # 第二种mask，每个step的邻居情况都有
    train_mask_time = training_nbr_existence_time
    train_mask_time = torch.tensor(train_mask_time, dtype=torch.float32)

    test_mask_time = testing_nbr_existence_time
    test_mask_time = torch.tensor(test_mask_time, dtype=torch.float32)

    inference_mask_time = inference_nbr_existence_time
    inference_mask_time = torch.tensor(inference_mask_time, dtype=torch.float32)

    # 制作dataset和dataloader
    train_dataset = TensorDataset(training_traj, training_supplement, train_mask, train_mask_time)
    test_dataset = TensorDataset(testing_traj, testing_supplement, test_mask, test_mask_time)
    inference_dataset = TensorDataset(inference_traj, inference_supplement, inference_mask, inference_mask_time)

    train_dataloader = DataLoader(train_dataset, batch_size=hp_config.batch_size, shuffle=True,
                                  num_workers=args.num_works)
    test_dataloader = DataLoader(test_dataset, batch_size=hp_config.batch_size, shuffle=False,
                                 num_workers=args.num_works)
    inference_dataloader = DataLoader(inference_dataset, batch_size=hp_config.batch_size, shuffle=False,
                                 num_workers=args.num_works)

    # 加载motion modes
    motion_modes, mode_file_suffix = load_motion_mode(args)

    # Show motion modes
    show_motion_modes(motion_modes, train_dataset, args, feature_destd, mode_file_suffix)

    model = TrajectoryModel(in_size=num_feat, out_size=2, obs_len=args.obs_len, pred_len=args.pred_len,
                            embed_size=hp_config.model_hidden_dim,
                            enc_num_layers=2, int_num_layers_list=[1, 1], heads=4, forward_expansion=2)
    model = model.cuda()
    # model = model.float()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hp_config.lr)
    reg_criterion = torch.nn.SmoothL1Loss().cuda()
    mse_criterion = torch.nn.MSELoss(reduction='none').cuda()
    cls_criterion = torch.nn.CrossEntropyLoss().cuda()

    if args.lr_scaling:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 95, 125], gamma=0.5)

    # 构建loss可视化，观察模型的收敛情况
    train_loss_record = []
    test_loss_record = []
    train_time = []
    fde_weight = 1.0

    for ep in range(hp_config.epoch):

        start_time = time.time()
        total_loss = train(ep, model, reg_criterion, cls_criterion, mse_criterion, optimizer, train_dataloader, motion_modes)
        ade, fde, rmse, num_traj = test(model, test_dataloader, motion_modes, feature_destd, mse_criterion)
        if args.lr_scaling:
            scheduler.step()

        if not os.path.exists(args.checkpoint + args.dataset_name):
            os.makedirs(args.checkpoint + args.dataset_name)

        if min_fde * fde_weight + min_ade > fde_weight * fde + ade:
            min_fde = fde
            min_ade = ade
            min_epoch = ep
            torch.save(model.state_dict(), args.checkpoint + args.dataset_name + args.save_param_name)  # OK

        # 当采用rmse作为指标时
        # if min_rmse > rmse:
        #     min_rmse = rmse
        #     min_epoch = ep
        #     torch.save(model.state_dict(), args.checkpoint + args.dataset_name + '/best_rmse.pth')  # OK

        # test_loss = avg_best_match_error
        # if min_avg_best_match_error > test_loss:
        #     min_avg_best_match_error = avg_best_match_error
        #     # min_avg_coverage = avg_coverage
        #     min_epoch = ep
        #     torch.save(model.state_dict(), args.checkpoint + args.dataset_name + '/best.pth')  # OK

        test_loss = ade + fde
        # test_loss = rmse
        train_loss = sum(total_loss) / len(total_loss)
        train_loss_record.append(train_loss)
        test_loss_record.append(test_loss)

        # 将loss随epoch的变化保存下来用于打印
        np.save(args.checkpoint + args.dataset_name + '/train_loss_record.npy', np.array(train_loss_record))
        np.save(args.checkpoint + args.dataset_name + '/test_loss_record.npy', np.array(test_loss_record))

        end_time = time.time()
        epoch_duration = end_time-start_time
        train_time.append(epoch_duration)

        print('轮次Epoch:', ep, '训练时长:', epoch_duration, 'train_loss:', train_loss, 'test_loss:', test_loss)
        print('ADE:', ade,  'FDE:', fde, "RMSE: ", rmse,'\n',"目前最小的test loss(后一个)：",min_fde + min_ade, min_rmse,
              "最佳epoch为:", min_epoch, '\n')

    # 进行结果推断
    i_start_time = time.time()
    # i_ade, i_fde, i_rmse, i_num_traj = inference(model, inference_dataloader, motion_modes, feature_destd, mse_criterion)
    i_ade, i_fde, i_rmse, i_num_traj = inference2(model, inference_dataloader, motion_modes, feature_destd, mse_criterion)
    # 保存test部分预测的goal数据，用于下一步train_goals
    i_end_time = time.time()
    _, _, _, _ = inference2(model,  test_dataloader, motion_modes, feature_destd, mse_criterion, output=True)


    inference_loss = i_ade + i_fde
    # inference_loss = i_rmse
    # 将train test inference的loss变化情况进行绘图
    # train_loss_record = np.load(args.checkpoint + args.dataset_name + 'train_loss_record.npy')
    # test_loss_record = np.load(args.checkpoint + args.dataset_name + 'test_loss_record.npy')

    show_loss(train_loss_record, test_loss_record, inference_loss)
    print("*推断inference loss:", inference_loss, "ade:", i_ade, "fde:", i_fde)
    print("*推断的rmse:", i_rmse)
    if hp_config.epoch > 0:
        print("训练总时长为：", sum(train_time))
        print("平均训练时长：", sum(train_time)/len(train_time))
    print("推断总时长inference time duration：", i_end_time-i_start_time)
