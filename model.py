import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain


class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_encoder = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x: torch.Tensor,lens , h_0: torch.Tensor)->torch.Tensor:
        # x.shape: Batch x sequence length x input_size
        # h_0.shape:


        X = torch.nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True)
        output, h_n = self.rnn_encoder(X, h_0)

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output


class DetectModel(nn.Module):
    def __init__(self, input_size,
                 hidden_size, rnn_layers,
                 out_channels, height, cnn_layers,
                 linear_hidden_size, linear_layers, output_size):
        super(DetectModel, self).__init__()
        self.output_size = output_size
        self.rnn_encoder = RNNEncoder(input_size=input_size, hidden_size=hidden_size, num_layers=rnn_layers)
        self.linear  = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)
    
    
    def forward(self, x, lens, h0):

        with open('outout.txt', 'a', encoding='utf8') as file:
            file.write(str(tuple(x.shape)))
            batch_size = x.shape[0]
            seq_len = x.shape[1]

            rnn_output = self.rnn_encoder(x,lens, h0)
            file.write('\n\n&&&\n\n')
            file.write(str(torch.Tensor(rnn_output).detach().numpy()[:,-1,:]))
            file.write('***\n***\n')
            file.write(str(tuple(rnn_output.shape)))
            output = rnn_output.contiguous()
            file.write('***\n***\n')
            file.write(str(tuple(rnn_output.shape)))
            output = output.view(-1, output.shape[2])
            output = self.linear(output)


            file.write('\n#####\n')
            file.write(str(tuple(output.shape)))
            file.write(str(batch_size))
            file.write(str(seq_len))
            file.write(str(self.output_size))
            file.write('\n####\n')

        hidden = output.view(batch_size, seq_len, self.output_size)
        output = F.softmax(output, dim=1)
        output = output.view(batch_size, seq_len, self.output_size)
        


        return output,hidden


if __name__ == '__main__':
    batch_size = 5
    sequence_length =10
    input_size = 15
    hidden_size = 5
    out_channels = 10
    height = 3
    linear_hidden_size = 20
    linear_layers = 5
    output_size = 2

    x = torch.randn(batch_size, sequence_length, input_size)
    h0 = torch.zeros(1, batch_size, hidden_size)
    model = DetectModel(input_size, hidden_size, 1, out_channels, height, 1, linear_hidden_size, linear_layers, output_size)
    output = model(x, h0)
    print(output)
