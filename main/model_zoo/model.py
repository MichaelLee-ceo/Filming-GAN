import torch
import torch.nn as nn

torch.manual_seed(10)

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0.0):
    layers = []
    for i in range(len(dim_list) - 1):
        d_in, d_out = dim_list[i], dim_list[i+1]
        layers.append(nn.Linear(d_in, d_out))

        if batch_norm:
            layers.append(nn.BatchNorm1d(d_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout != 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

def get_noise(noise_shape, noise_type):
    # print('[INFO] Using noise type:', noise_type)
    if noise_type == 'gaussian':
        return torch.randn(*noise_shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*noise_shape).sub_(0.5).mul(2.0).cuda()


class Encoder(nn.Module):
    '''
    Basically composed of a LSTM model
    '''
    def __init__(self, embedding_dim=16, hidden_dim=64, num_layers=1, dropout=0.0):
        super(Encoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # self.embedding = nn.Linear(3, embedding_dim)
        self.encoder = nn.LSTM(3, hidden_dim, num_layers, dropout=dropout)

    def forward(self, obs_traj):
        '''
        Inputs:
        - obs_traj(obs_traj_rel): Tensor of shape (obs_len, batch, 3)
        Outpus:
        - final_h: Tensor of shape (self.num_layers, batch, self.hidden_dim)
        '''

        batch_size = obs_traj.shape[1]
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda()
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda()

        output, (h_n, c_n) = self.encoder(obs_traj, (h_0, c_0))

        return h_n


class Decoder(nn.Module):
    '''
    Basically composed of a LSTM model
    '''
    def __init__(self, embedding_dim=16, hidden_dim=64, num_layers=1, dropout=0.0, pred_len=8):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len

        # self.embedding = nn.Linear(3, embedding_dim)
        self.decoder = nn.LSTM(3, hidden_dim, num_layers, dropout=dropout)
        self.hidden2pos = nn.Linear(hidden_dim, 3)

    def forward(self, obs_traj_rel, state_tuple):
        '''
        Inputs:
        - last_pos: Tensor of shape (batch, 3)
        - state_tuple: (hn, cn) each Tensor of shape (num_layers, batch, hidden_dim)
        Outpus:
        - pred_traj: Tensor of shape (self.pred_len, batch, 3)
        '''
        decoder_input = obs_traj_rel[-1]
        batch_size = decoder_input.shape[0]

        pred_traj_fake_rel = []

        for i in range(self.pred_len):
            # decoder_embedding = self.embedding(decoder_input)
            decoder_embedding = torch.unsqueeze(decoder_input, 0)
            decoder_output, state_tuple = self.decoder(decoder_embedding, state_tuple)

            decoder_output = decoder_output.view(-1, self.hidden_dim)
            decoder_input = self.hidden2pos(decoder_output)

            pred_traj_fake_rel.append(decoder_input.view(batch_size, -1))

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)

        return pred_traj_fake_rel


class Generator(nn.Module):
    def __init__(self, embedding_dim=16, encoder_h_dim=32, decoder_h_dim=32, num_layers=1, mlp_dim=64, dropout=0.0, 
        obs_len=8, pred_len=8, noise_dim=8, noise_type='gaussian', activation='relu', batch_norm=True):
        super(Generator, self).__init__()

        self.embedding_dim = embedding_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.noise_dim = noise_dim
        self.noise_type = noise_type

        self.encoder = Encoder(embedding_dim=embedding_dim, hidden_dim=encoder_h_dim, num_layers=num_layers, dropout=dropout)
        self.decoder = Decoder(embedding_dim=embedding_dim, hidden_dim=decoder_h_dim, num_layers=num_layers, dropout=dropout, pred_len=pred_len)

        mlp_context_dims = [encoder_h_dim, mlp_dim, decoder_h_dim-self.noise_dim]
        self.encoder_compression = make_mlp(mlp_context_dims, activation=activation, batch_norm=batch_norm, dropout=dropout)

    def add_noise(self, _input):
        '''
        Inputs:
        - _input: Tensor of shape ( _, decoder_h_dim - noise_dim )
        - custom_noise: for testing different types of noise
        Outputs:
        - decoder_hidden_state: Tensor of shape ( _, decoder_h_dim )
        '''

        noise_shape = (_input.size(0), self.noise_dim)
        z = get_noise(noise_shape, self.noise_type)

        decoder_h = torch.cat([_input, z], dim=1)
        return decoder_h

    def forward(self, obs_traj_rel):
        '''
        Inputs:
        - obs_traj_rel: Tensor of shape (obs_len, batch, 3)
        Outputs:
        - pred_traj_rel: Tensor of shape (pred_len, batch, 3)
        '''
        batch_size = obs_traj_rel.shape[1]

        encoder_hn = self.encoder(obs_traj_rel)
        encoder_hn = encoder_hn.view(-1, self.encoder_h_dim)

        # Add noise
        compressed_encoder_hn = self.encoder_compression(encoder_hn)
        
        decoder_h = self.add_noise(compressed_encoder_hn)
        decoder_h = torch.unsqueeze(decoder_h, 0)
        decoder_c = torch.zeros(self.num_layers, batch_size, self.decoder_h_dim).cuda()

        pred_traj_fake_rel = self.decoder(obs_traj_rel, (decoder_h, decoder_c))

        return pred_traj_fake_rel


class Discriminator(nn.Module):
    def __init__(self, embedding_dim=16, encoder_h_dim=64, num_layers=1, mlp_dim=64, dropout=0.0, activation='relu', batch_norm=True):
        super(Discriminator, self).__init__()

        self.embedding_dim = embedding_dim
        self.encoder_h_dim = encoder_h_dim
        self.mlp_dim = mlp_dim

        self.encoder = Encoder(embedding_dim=embedding_dim, hidden_dim=encoder_h_dim, num_layers=num_layers, dropout=dropout)

        classifier_dims = [encoder_h_dim, mlp_dim, 1]
        self.classifier = make_mlp(classifier_dims, activation=activation, batch_norm=batch_norm, dropout=dropout)

    def forward(self, traj_rel):
        '''
        Inputs:
        - traj_rel: Tensor of shpae ( obs_len + pred_len, batch, 3 )
        Outputs:
        - scores: Tensor of shape ( batch, ) with real/fake scores
        '''
        encoder_hn = self.encoder(traj_rel)
        scores = self.classifier(encoder_hn.squeeze())
        return scores