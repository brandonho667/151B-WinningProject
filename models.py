import torch

class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, dropout=0.1, batch_first=True)
        self.linear = torch.nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.device = device
    
    def forward(self, x):
        if self.training:
            out, _ = self.lstm(x)
            y = self.linear(out)
            return y
        else:        
            y = torch.zeros((x.size(0), x.size(1)+59, x.size(2)), device=self.device)
            out, (h, c) = self.lstm(x)
            y[:, :x.size(1), :] = self.linear(out)
            for i in range(59):
                out, (h, c) = self.lstm(y[:, x.size(1)+i-1:x.size(1)+i, :], (h, c))
                y[:, x.size(1)+i:x.size(1)+i+1, :] = self.linear(out)
            return y[:, x.size(1)-1:, :]
