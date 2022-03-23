import torch
import torch.nn as nn

torch.manual_seed(10)

class SingleLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, pred_len, drop_out):
        super(SingleLSTM, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len

        # embedding [x, y, z] to embedding_dim
        # self.embedding = nn.Linear(3, embedding_dim)
        self.hidden2pos = nn.Linear(hidden_dim, 3)

        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_dim, num_layers=num_layers)

    def forward(self, obs_traj_rel):

        last_pos_rel = obs_traj_rel[-1]
        batch_size = last_pos_rel.shape[0]

        # construct state tuple with shape [num_layers, batch_size, hidden_size]
        h_0 = torch.zeros((self.num_layers, obs_traj_rel.shape[1], self.hidden_dim)).cuda()
        c_0 = torch.zeros((self.num_layers, obs_traj_rel.shape[1], self.hidden_dim)).cuda()

        # input_embedding = self.embedding(obs_traj_rel)
        lstm_output, state_tuple = self.lstm(obs_traj_rel, (h_0, c_0))

        pred_traj_fake_rel = []
        for i in range(self.pred_len):
            last_pos_rel = torch.unsqueeze(last_pos_rel, 0)                     # increase one dimension (seq_length) for batched input [batch, features] -> [seq_len, batch, features]

            # last_pos_rel = self.embedding(last_pos_rel)                         # embeds the input [x, y, z] -> [embedding_dim]
            
            output, state_tuple = self.lstm(last_pos_rel, state_tuple)
            output = output.view(-1, self.hidden_dim)                           # removes the "seq_len" dimension: [seq_len, batch, hidden_dim] -> [batch, hidden_dim]
            pred_traj_rel = self.hidden2pos(output)                             # [embedding_dim] -> [x, y, z]

            last_pos_rel = pred_traj_rel

            pred_traj_fake_rel.append(pred_traj_rel.view(batch_size, -1))       # reshape back to each pred_traj with each batch elements


        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel