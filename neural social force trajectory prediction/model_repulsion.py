import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import pdb
from torch.distributions.normal import Normal
import math
import numpy as np
import copy

def stateutils_desired_directions(current_step, generated_dest):
    # 计算行人的期望运动方向
    destination_vectors = generated_dest - current_step #peds*2
    norm_factors = torch.norm(destination_vectors, dim=-1) #peds
    norm_factors = torch.unsqueeze(norm_factors, dim=-1)
    directions = destination_vectors / (norm_factors + 1e-8) #peds*2
    return directions

def value_p_p_neighbors(c1, pedestrians, coefficients, sigma):
    """
    计算来自邻近车辆的势场
    :param c1: 当前邻近车辆的位置，形状为 (peds, maxpeds, 2)
    :param pedestrians: 当前步行者的位置，形状为 (peds, 1, 2)
    :param coefficients: 势场系数，通常是可学习的参数
    :param sigma: 控制势场的衰减
    :return: 势场值，形状为 (peds,)
    """
    d_p_c1 = pedestrians - c1  # 计算邻近车辆与当前车辆的距离向量
    d_p_c1_norm = torch.norm(d_p_c1, dim=-1)  # 计算邻近车辆的距离标量

    potential_nei = 7 * sigma * coefficients * torch.exp(-d_p_c1_norm / sigma)  # 势场计算公式
    out = torch.sum(potential_nei, 1)  # 汇总所有邻近车辆的势场
    return out

def f_ab_fun_neighbors(current_step, coefficients, current_supplement, sigma, device):
    """
    计算邻近车辆产生的斥力
    :param current_step: 当前车辆位置，形状为 (peds, 2)
    :param coefficients: 势场系数
    :param current_supplement: 邻近车辆位置，形状为 (peds, maxpeds, 2)
    :param sigma: 势场的衰减系数
    :param device: 设备，CPU 或 GPU
    :return: 邻近车辆的斥力，形状为 (peds, 2)
    """
    c1 = current_supplement[:,:,:2]  # 获取邻近车辆的位置
    pedestrians = torch.unsqueeze(current_step, dim=1)  # 添加维度

    # 计算邻近车辆的势场
    v_neighbors = value_p_p_neighbors(c1, pedestrians, coefficients, sigma)

    delta = torch.tensor(1e-3).to(device)
    dx = torch.tensor([[[delta, 0.0]]]).to(device)  # 在x方向上的微小偏移
    dy = torch.tensor([[[0.0, delta]]]).to(device)  # 在y方向上的微小偏移

    # 计算 x 和 y 方向的偏导数
    dvdx_neighbors = (value_p_p_neighbors(c1, pedestrians + dx, coefficients, sigma) - v_neighbors) / delta
    dvdy_neighbors = (value_p_p_neighbors(c1, pedestrians + dy, coefficients, sigma) - v_neighbors) / delta

    # 计算斥力
    grad_r_ab_neighbors = torch.stack((dvdx_neighbors, dvdy_neighbors), dim=-1)
    F_neighbors = -1.0 * grad_r_ab_neighbors
    return F_neighbors, v_neighbors

def f_ab_fun_lines(boundary_factor_current, clines_factor_current,
                   boundary_factor_shifted, clines_factor_shifted,
                   k_boundary, k_clines, device):
    """
    计算车道线产生的斥力，车道线只影响y方向，x方向的力为0
    :param boundary_factor_current: 预先计算好的当前时刻 boundary 势场因素
    :param clines_factor_current: 预先计算好的当前时刻 clines 势场因素
    :param boundary_factor_shifted: 预先计算好的偏移 delta 后的 boundary 势场因素
    :param clines_factor_shifted: 预先计算好的偏移 delta 后的 clines 势场因素
    :param k_boundary: 对应 boundary 的可学习参数
    :param k_clines: 对应 clines 的可学习参数
    :param device: 设备，CPU 或 GPU
    :return: 车道线产生的斥力
    """

    # 将当前的 boundary 和 clines 势场乘上对应的可学习参数
    # 计算 boundary 的势场
    boundary_potential_current = k_boundary * boundary_factor_current.sum(dim=1)  # 对 2 个 boundary 做求和
    boundary_potential_shifted = k_boundary * boundary_factor_shifted.sum(dim=1)
    # 计算 clines 的势场
    clines_potential_current = k_clines * clines_factor_current.sum(dim=1)  # 对 3 个 clines 做求和
    clines_potential_shifted = k_clines * clines_factor_shifted.sum(dim=1)
    # 合并 boundary 和 clines 的势场
    potential_current = boundary_potential_current + clines_potential_current
    potential_shifted = boundary_potential_shifted + clines_potential_shifted

    # 计算偏微分得到力，只对 y 方向有影响（因为车道线只在 y 方向产生影响）
    delta = torch.tensor(1e-3).to(device)
    dvdx_lines = torch.zeros_like(potential_current)  # 对x方向的偏导数为0
    dvdy_lines = (potential_shifted - potential_current) / delta

    # 只对 y 方向有非零值的力
    grad_r_ab_lines = torch.stack((dvdx_lines, dvdy_lines), dim=-1)  # peds*2, 其中 x 分量为 0
    F_lines = -1.0 * grad_r_ab_lines

    return F_lines, potential_current

'''MLP model'''
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x

class NSP(nn.Module):

    def __init__(self, input_size, embedding_size, rnn_size, output_size, enc_size, dec_size):
        '''
        Args:
            size parameters: Dimension sizes
            sigma: Standard deviation used for sampling N(0, sigma)
            past_length: Length of past history (number of timesteps)
            future_length: Length of future trajectory to be predicted
        '''
        super(NSP, self).__init__()

        self.max_peds = 8
        self.r_pixel = 100
        self.costheta = np.cos(np.pi / 3)

        # The Goal-Network
        self.cell1 = nn.LSTMCell(embedding_size, rnn_size)
        self.input_embedding_layer1 = nn.Linear(input_size, embedding_size)
        self.output_layer1 = nn.Linear(rnn_size, output_size)

        self.encoder_dest_state = MLP(input_dim = 2, output_dim = output_size, hidden_size=enc_size)
        self.dec_tau = MLP(input_dim = 2*output_size, output_dim = 1, hidden_size=dec_size)

        # The Collision-Network
        self.cell2 = nn.LSTMCell(embedding_size, rnn_size)
        self.input_embedding_layer2 = nn.Linear(input_size, embedding_size)
        self.output_layer2 = nn.Linear(rnn_size, output_size)

        self.encoder_people_state = MLP(input_dim=4, output_dim=output_size, hidden_size=enc_size)
        self.dec_para_people = MLP(input_dim=2 * output_size, output_dim=1, hidden_size=dec_size)


        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()


    def forward_lstm(self, input_lstm, hidden_states1, cell_states1, hidden_states2, cell_states2):
        #input_lstm: peds*4

        # LSTM1
        input_embedded1 = self.relu(self.input_embedding_layer1(input_lstm)) #peds*embedding_size
        h_nodes1, c_nodes1 = self.cell1(input_embedded1, (hidden_states1, cell_states1)) #h_nodes/c_nodes: peds*rnn_size
        outputs1 = self.output_layer1(h_nodes1) #peds*output_size

        # LSTM2
        input_embedded2 = self.relu(self.input_embedding_layer2(input_lstm)) #peds*embedding_size
        h_nodes2, c_nodes2 = self.cell2(input_embedded2, (hidden_states2, cell_states2)) #h_nodes/c_nodes: peds*rnn_size
        outputs2 = self.output_layer2(h_nodes2) #peds*output_size


        return outputs1, h_nodes1, c_nodes1, outputs2, h_nodes2, c_nodes2

    def forward_coefficient_veh(self, outputs_features2, supplement, nbr_existence, current_step, current_vel, device):

        batch_size = outputs_features2.size()[0]

        encoding_part1 = torch.unsqueeze(outputs_features2, dim=1).repeat(1, self.max_peds, 1) #peds*25*16
        features_others = self.encoder_people_state(supplement) #peds*25*16
        input_coefficients = torch.cat((encoding_part1, features_others), dim=-1) #peds*25*32
        coefficients = torch.squeeze(100 * self.sigmoid(self.dec_para_people(input_coefficients)))  # peds*25

        # 将没有邻居位置上对应的coefficients的位置设置为0，表示没有这个邻居
        # for i in range(batch_size):
        #     num_nei = int(nbr_existence[i]) # nbr_existence的9的最后一位表示neighbors的个数
        #     coefficients[i, num_nei:] = torch.zeros(self.max_peds - num_nei)
        coefficients = coefficients * nbr_existence  # Masking out non-existing neighbors

        return coefficients


    def forward_next_step_veh(self, current_step, current_vel, initial_speeds, dest, features_lstm1, coefficients,
                              current_supplement, boundary_factor_current, clines_factor_current,
                          boundary_factor_shifted, clines_factor_shifted, sigma, time_step, k_boundary, k_clines, device=torch.device('cpu')):

        delta_t = torch.tensor(time_step).to(device)
        e = stateutils_desired_directions(current_step, dest).to(device)  # peds*2, 应该是速度向量的方向e

        features_dest = self.encoder_dest_state(dest)
        features_tau = torch.cat((features_lstm1, features_dest), dim = -1)
        tau = self.sigmoid(self.dec_tau(features_tau)) + time_step

        F0 = 1.0 / tau * (initial_speeds * e - current_vel).to(device)  #peds*2
        F1_neighbors, v_neighbors = f_ab_fun_neighbors(current_step, coefficients, current_supplement, sigma, device)  # 邻近车辆的斥力
        F1_lines, v_lines = f_ab_fun_lines(boundary_factor_current, clines_factor_current,
            boundary_factor_shifted, clines_factor_shifted,
            k_boundary, k_clines, device) # 车道线的斥力

        # F = F0 + F1_neighbors +  F1_lines #peds*2
        F = F0 + F1_neighbors   #peds*2
        # F = F0

        w_v = current_vel + delta_t * F  #peds*2

        # update state
        prediction = current_step + w_v * delta_t  # peds*2

        return prediction, w_v, v_neighbors, v_lines