import torch

class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, device):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, dropout=dropout, batch_first=True)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.device = device
    
    def forward(self, x):
        if self.training:
            out, _ = self.lstm(x)
            out = self.dropout(out)
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
        
class LSTMAttention(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, device):
        super(LSTMAttention, self).__init__()
        
        self.enc = torch.nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=2, dropout=dropout, batch_first=True, bidirectional=True)
        self.enc_dropout = torch.nn.Dropout(p=dropout)
        
        self.attn = torch.nn.Linear(hidden_dim*2, hidden_dim)
        self.softmax = torch.nn.Softmax(dim=1)
            
        self.dec1_fwd = torch.nn.LSTM(
            input_size=hidden_dim*2, hidden_size=hidden_dim, batch_first=True)
        self.dec1_bkwd = torch.nn.LSTM(
            input_size=hidden_dim*2, hidden_size=hidden_dim, batch_first=True)
        self.dec1_dropout = torch.nn.Dropout(p=dropout)
        
        self.dec2 = torch.nn.LSTM(
            input_size=2*hidden_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.dec2_dropout = torch.nn.Dropout(p=dropout)
        
        self.linear = torch.nn.Linear(in_features=2*hidden_dim, out_features=output_dim)
        self.device = device
       
    def forward(self, x):
        enc_out, _ = self.enc(x)
        if self.training:
            enc_out = self.enc_dropout(enc_out)
        
        keys = self.attn(enc_out).transpose(1, 2).contiguous()
        
        dec1_fwd_out = torch.zeros((x.size(0), 60, enc_out.size(2)//2), device=self.device)
        dec1_bkwd_out = torch.zeros((x.size(0), 60, enc_out.size(2)//2), device=self.device)
        h_fwd = torch.zeros((1, x.size(0), enc_out.size(2)//2), device=self.device)
        c_fwd = torch.zeros((1, x.size(0), enc_out.size(2)//2), device=self.device)
        h_bkwd = torch.zeros((1, x.size(0), enc_out.size(2)//2), device=self.device)
        c_bkwd = torch.zeros((1, x.size(0), enc_out.size(2)//2), device=self.device)
        for i in range(60):
            # compute alignment weights, manipulating dimensions as necessary
            align_fwd = torch.bmm(h_fwd.transpose(0, 1).contiguous(), keys).transpose(1, 2).contiguous()
            align_bkwd = torch.bmm(h_bkwd.transpose(0, 1).contiguous(), keys).transpose(1, 2).contiguous()
            
            attn_fwd = self.softmax(align_fwd)
            attn_bkwd = self.softmax(align_bkwd)
            
            context_fwd = torch.sum(attn_fwd * enc_out, dim=1, keepdim=True)
            context_bkwd = torch.sum(attn_bkwd * enc_out, dim=1, keepdim=True)
            
            dec1_fwd_out[:, i:i+1, :], (h_fwd, c_fwd) = self.dec1_fwd(context_fwd, (h_fwd, c_fwd))
            dec1_bkwd_out[:, 60-i-1:60-i, :], (h_bkwd, c_bkwd) = self.dec1_bkwd(context_bkwd, (h_bkwd, c_bkwd))
        
        dec2_in = torch.cat((dec1_fwd_out, dec1_bkwd_out), dim=2)
        if self.training:
            dec2_in = self.dec1_dropout(dec2_in)
        dec_out, _ = self.dec2(dec2_in)
        if self.training:
            dec_out = self.dec2_dropout(dec_out)
        y = self.linear(dec_out)
        return y
        
            
            
if __name__ == "__main__":
    model = LSTMAttention(4, 128, 4, 0, torch.device("cpu"))
    model(torch.rand((4, 50, 4)))