from utils.encoders import IEncoder

import numpy as np

import torch
import torch.nn as nn

# for now i ported only FasterGRUEncoder, LSTM + classic GRU in in TODO list

class GruLayer(nn.Module):
    def __init__(self, in_count, out_count, bias):
        super().__init__()
        self.net = nn.GRU(
                in_count, out_count,
                batch_first=True,
                bidirectional=False,
                num_layers=1,
                bias=bias)
    def forward(self, states, hidden):
        return self.net(states, hidden)

class FasterGRUEncoder(IEncoder):
    """
    most likely we can not do this with LSTM
    as GRU provide us full hidden state ( in out ), while LSTM only last one for t~seq_len
    - this is problematic to extract features in fast fashion ( 1 forward pass )

    - also we banned now full RNN state forwarding; as in extract features will need
      more complicated design ( stacking outputs together.. now not worth implement ~ overengineering )
    """
    def __init__(self, size_in, n_history, n_features, rnn_n_layers):
        assert 0 == n_features % rnn_n_layers, "history_features must be fraction of rnn_n_layers!"

        self.n_layers = rnn_n_layers
        self.history_count = n_history
        self.state_size = size_in
        self.n_features_ex = n_features // self.n_layers

        super().__init__(size_in=n_history*size_in, size_out=(n_history-1)*size_in+self.n_features_ex, n_features=n_features)
#        super().__init__(size_in=n_history*size_in, size_out=n_history*size_in+n_features, n_features=n_features)

        self.rnn = [ GruLayer(
            self.state_size if not i else self.n_features_ex, self.n_features_ex, bool(i + 1 < self.n_layers)
            ) for i in range(self.n_layers) ]

        for i, rnn in enumerate(self.rnn):
            self.add_module("fast_gru_%i"%i, rnn)

    def has_features(self):
        return True

    def _forward_impl(self, out, features):
        features = features.view(-1, self.n_layers, self.n_features_ex).transpose(0, 1)

        memory = []
        for rnn, feats in zip(self.rnn, features):
            out, _ = rnn(out, feats.unsqueeze(0))
            memory.append(out)

        return torch.stack(memory)

    def forward(self, states, features):
#        states = torch.stack([states, states, states])#.transpose(0, 1)
#        features = torch.stack([features, features, features])
#        memory = self._forward_impl(states, features)
        sequence_batch = states.view(-1, self.history_count, self.state_size)
        memory = self._forward_impl(sequence_batch, features)

        out = memory[-1, :, -1, :] # last layer, all states, final output, all features
        # if we want to go all out, then following line is way to go
        #  out = memory[-1, :, :, :].view(memory.size(1), -1)
        #  out = memory[:, :, -1, :]

        #  assert states.size(0) == memory.size(1), "size assumption mismatch {} :: {} vs {}".format(
        #          states.size(), memory.size(), features.size())

        # all layers, all states, first hidden state, all features
#        print("===>", memory[:, :, 0, :].shape, memory[:, :, 0, :].transpose(0, 1).shape, memory[:, :, 0, :].transpose(0, 1).reshape(memory.size(1), -1).shape)
        hidden = memory[:, :, 0, :].transpose(0, 1).reshape(memory.size(1), -1)
#        return torch.cat([states.view(memory.size(1), -1), features.reshape(memory.size(1), -1)], 1), hidden
        return out, hidden
        states = states.view(memory.size(1), self.history_count, -1)
        states = states[:, 1:, :].view(memory.size(1), -1)
        return torch.cat([states, out], 1), hidden

    def extract_features(self, states):
        assert states.size(-1) == self.size_in, "...{}->{}".format(states.size(), self.size_in)
        hidden = torch.zeros(1, self.n_layers, self.n_features_ex)
        sequence = torch.cat([
            states[0].view(self.history_count, self.state_size), # fist state compose of initial states as well for sequence
            states[1:, -self.state_size:]]).unsqueeze(0) # later on extract actual state from sequence

        memory = self._forward_impl(sequence, hidden)

        #  last layer, from first state take whole history sequence of features
        out = memory[-1, 0, self.history_count-1:, :]#.transpose(0, 1)#.reshape(states.size(0), -1) # last layer, from first state take whole history sequence of features
        assert states.size(0)-1==memory[:, 0, :-self.history_count, :].transpose(0, 1).shape[0]
        #  out = memory[:, 0, self.history_count-1:, :].transpose(0, 1).reshape(states.size(0), -1) # last layer, from first state take whole history sequence of features
        hidden = torch.cat([
            hidden.view(1, -1), # our first hidden state is obviouselly initial one ...
            memory[:, 0, :-self.history_count, :].transpose(0, 1).reshape(states.size(0)-1, -1)
            ], 0)

#        return torch.cat([states, hidden], 1), hidden.cpu().numpy()
        return out, hidden.cpu().numpy()
        assert self.history_count > 1, "RNN extract feats, forwarding 2nd+ state failed!"
        states = states.view(states.size(0), self.history_count, -1)
        states = states[:, 1:, :].view(states.size(0), -1)
        return torch.cat([states, out], 1), hidden.cpu().numpy()
