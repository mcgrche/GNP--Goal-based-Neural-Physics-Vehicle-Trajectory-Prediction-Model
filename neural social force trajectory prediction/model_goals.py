import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

epsilon = 1e-8

def stateutils_desired_directions(current_step, generated_dest):

    destination_vectors = generated_dest - current_step #peds*2
    norm_factors = torch.norm(destination_vectors, dim=-1) #peds
    norm_factors = torch.unsqueeze(norm_factors, dim=-1)
    directions = destination_vectors / (norm_factors + 1e-8) #peds*2
    return directions

def f_ab_fun(current_step, coefficients, current_supplement, device):
    disp_p_x = torch.zeros(1,2)
    disp_p_y = torch.zeros(1,2)
    disp_p_x[0, 0] = 0.1
    disp_p_y[0, 1] = 0.1

    c1 = current_supplement[:,:-1,:2] #peds*maxpeds*2
    pedestrians = torch.unsqueeze(current_step, dim=1)  # peds*1*2

    v = value_p_p(c1, pedestrians, coefficients) # peds

    delta = torch.tensor(1e-3).to(device)
    dx = torch.tensor([[[delta, 0.0]]]).to(device) #1*1*2
    dy = torch.tensor([[[0.0, delta]]]).to(device) #1*1*2

    dvdx = (value_p_p(c1 + dx, pedestrians, coefficients) - v) / delta # peds
    dvdy = (value_p_p(c1 + dy, pedestrians, coefficients) - v) / delta # peds

    grad_r_ab = torch.stack((dvdx, dvdy), dim=-1) # peds*2
    out = -1.0 * grad_r_ab

    return out

def value_p_p(c1, pedestrians, coefficients):
    #potential field function : pf = K*exp(-norm(p-p1))

    d_p_c1 = pedestrians - c1  # peds*maxpeds*2
    d_p_c1_norm = torch.norm(d_p_c1, dim=-1) # peds*maxpeds

    potential = coefficients * torch.exp(-d_p_c1_norm.detach()) #peds*maxpeds

    out = torch.sum(potential, 1) #peds

    return out

class MLP2(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, activation='relu', discrim=False, dropout=0.5):
        super(MLP2, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        layer_sizes = [input_dim] + list(hidden_size) + [output_dim]

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                if activation == 'relu':
                    self.activations.append(nn.ReLU())
                elif activation == 'sigmoid':
                    self.activations.append(nn.Sigmoid())

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.final_activation = nn.Sigmoid() if discrim else None

        self.initialize_weights(activation)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation and dropout to all but last layer
                x = self.activations[i](x)
                if self.dropout:
                    x = self.dropout(x)
            elif self.final_activation:
                x = self.final_activation(x)
        return x

    def initialize_weights(self, activation):
        for layer in self.layers:
            if activation == 'relu':
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif activation == 'sigmoid':
                init.xavier_uniform_(layer.weight)
            init.constant_(layer.bias, 0)

'''MLP model'''
# class MLP(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
#         super(MLP, self).__init__()
#         dims = []
#         dims.append(input_dim)
#         dims.extend(hidden_size)
#         dims.append(output_dim)
#         self.layers = nn.ModuleList()
#         for i in range(len(dims)-1):
#             self.layers.append(nn.Linear(dims[i], dims[i+1]))
#
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'sigmoid':
#             self.activation = nn.Sigmoid()
#
#         self.sigmoid = nn.Sigmoid() if discrim else None
#         self.dropout = dropout
#
#     def forward(self, x):
#         for i in range(len(self.layers)):
#             x = self.layers[i](x)
#             if i != len(self.layers)-1:
#                 x = self.activation(x)
#                 if self.dropout != -1:
#                     x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
#             elif self.sigmoid:
#                 x = self.sigmoid(x)
#         return x

class NSP(nn.Module):

    def __init__(self, input_size, embedding_size, rnn_size, output_size, encoder_dest_size, dec_tau_size):
        '''
        Args:
            input_size: Dimension of the state
            embedding_size: Dimension of the input of LSTM
            rnn_size: Dimension of LSTM
            output_size: Dimension of linear transformation of output of LSTM
            encoder_dest_size: Hitten size of MLP encoding destinations
            dec_tau_size: Hitten size of MLP decoding extracted features to tau

        '''
        super(NSP, self).__init__()

        # The Goal-Network
        self.cell = nn.LSTMCell(embedding_size, rnn_size)
        self.input_embedding_layer = nn.Linear(input_size, embedding_size)
        self.output_layer = nn.Linear(rnn_size, output_size)

        self.encoder_dest = MLP2(input_dim = 2, output_dim = output_size, hidden_size=encoder_dest_size, activation='relu')
        self.dec_tau = MLP2(input_dim = 2*output_size, output_dim = 1, hidden_size=dec_tau_size, activation='sigmoid')


        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

        self.initialize_lstm_weights()


    def forward_lstm(self, input_lstm, hidden_states_current, cell_states_current):
        #input_lstm: B * 4

        # Embed inputs
        input_embedded = self.dropout(self.relu(self.input_embedding_layer(input_lstm))) # B * embedding_size
        h_nodes, c_nodes = self.cell(input_embedded, (hidden_states_current, cell_states_current)) # h_nodes/c_nodes: B * rnn_size
        outputs = self.output_layer(h_nodes) # B * output_size

        return outputs, h_nodes, c_nodes

    def forward_next_step(self, current_step, current_vel, initial_speeds, dest, features_lstm, time_step,  device=torch.device('cpu')):

        delta_t = torch.tensor(time_step)
        e = stateutils_desired_directions(current_step, dest)  # 计算方向，B * 2

        features_dest = self.encoder_dest(dest)
        features_tau = torch.cat((features_lstm, features_dest), dim = -1)
        tau = self.sigmoid(self.dec_tau(features_tau)) + time_step
        F0 = 1.0 / (tau ) * (initial_speeds * e - current_vel)   # B * 2

        F = F0 # B * 2

        w_v = current_vel + delta_t * F  # B * 2

        # update state
        prediction = current_step + w_v * delta_t  # B * 2

        return prediction, w_v

    def initialize_lstm_weights(self):
        for name, param in self.cell.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                init.constant_(param.data, 0)




