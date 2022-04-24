from torch import *

#test
class BiDeepSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiDeepSeq2Seq, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim)
        self.decoder = nn.LSTM(input_size=output_dim, hidden_size=hidden_dim)
        self.h2pt = nn.Linear(in_features=hidden_dim, out_features=output_dim)
