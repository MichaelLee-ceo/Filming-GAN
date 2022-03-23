import torch
import torch.nn as nn

torch.manual_seed(10)

class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, drop_out):
        super(Encoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # embedding [x, y, z] to embedding_dim
        self.embedding = nn.Linear(3, embedding_dim)
        self.encoder = nn.LSTM(input_size=3, hidden_size=hidden_dim, num_layers=num_layers)

        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        # construct state tuple with shape [num_layers, batch_size, hidden_size]
        h_0 = torch.zeros((self.num_layers, x.shape[1], self.hidden_dim)).cuda()
        c_0 = torch.zeros((self.num_layers, x.shape[1], self.hidden_dim)).cuda()

        # encoder_embedding = self.embedding(x)
        # encoder_embedding = self.dropout(encoder_embedding)

        encoder_output, state_tuple = self.encoder(x, (h_0, c_0))

        return encoder_output, state_tuple

class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, pred_len, drop_out):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len

        # embedding [x, y, z] to [embedding_dim]
        self.embedding = nn.Linear(3, embedding_dim)
        self.decoder = nn.LSTM(input_size=3, hidden_size=hidden_dim, num_layers=num_layers)
        self.hidden2pos = nn.Linear(hidden_dim, 3)

        self.dropout = nn.Dropout(drop_out)

    def forward(self, obs_traj_rel, state_tuple):
        decoder_input = obs_traj_rel[-1]
        batch_size = decoder_input.shape[0]

        pred_traj_fake_rel = []

        for i in range(self.pred_len):
            # decoder_embedding = self.embedding(decoder_input)
            # decoder_embedding = self.dropout(decoder_embedding)

            decoder_embedding = torch.unsqueeze(decoder_input, 0)                           # increase one dimension (seq_length) for batched input [batch, features] -> [seq_len, batch, features]
            decoder_output, state_tuple = self.decoder(decoder_embedding, state_tuple)

            decoder_output = decoder_output.view(-1, self.hidden_dim)                       # removes the "seq_len" dimension: [seq_len, batch, hidden_dim] -> [batch, hidden_dim]
            decoder_input = self.hidden2pos(decoder_output)

            pred_traj_fake_rel.append(decoder_input.view(batch_size, -1))
            # print('decoder output', pred_traj_fake_rel[-1].shape)

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)

        return pred_traj_fake_rel

class Seq2Seq(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, pred_len, drop_out):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, drop_out=drop_out)
        self.decoder = Decoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, pred_len=pred_len, drop_out=drop_out)

    def forward(self, obs_traj_rel):
        encoder_output, state_tuple = self.encoder(obs_traj_rel)
        decoder_output = self.decoder(obs_traj_rel, state_tuple)

        return decoder_output
