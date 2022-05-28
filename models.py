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
            input_size=input_dim, hidden_size=hidden_dim, num_layers=3, dropout=dropout, batch_first=True, bidirectional=True)
        self.enc_dropout = torch.nn.Dropout(p=dropout)
        
        self.attn = torch.nn.Linear(hidden_dim*2, hidden_dim)
        self.softmax = torch.nn.Softmax(dim=1)
            
        self.dec1_fwd = torch.nn.LSTM(
            input_size=hidden_dim*2, hidden_size=hidden_dim, batch_first=True)
        self.dec1_bkwd = torch.nn.LSTM(
            input_size=hidden_dim*2, hidden_size=hidden_dim, batch_first=True)
        self.dec1_dropout = torch.nn.Dropout(p=dropout)
        
        self.dec2 = torch.nn.LSTM(
            input_size=2*hidden_dim, hidden_size=hidden_dim, num_layers=2, dropout=dropout, batch_first=True, bidirectional=True)
        self.dec2_dropout = torch.nn.Dropout(p=dropout)
        
        self.linear = torch.nn.Linear(in_features=2*hidden_dim, out_features=output_dim)
        self.device = device
       
    def forward(self, x):
        enc_out, _ = self.enc(x)
#         if self.training:
#             enc_out = self.enc_dropout(enc_out)
        
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
#         if self.training:
#             dec_out = self.dec2_dropout(dec_out)
        y = self.linear(dec_out)
        return y
        

class Conv2Seq(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, device):
        super(Conv2Seq, self).__init__()
        
        channel_sizes = [input_dim, 8, 16, 32]
        kernel_sizes = [4, 8, 16]
        self.conv_enc = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=channel_sizes[0], out_channels=channel_sizes[1], kernel_size=kernel_sizes[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv1d(in_channels=channel_sizes[1], out_channels=channel_sizes[2], kernel_size=kernel_sizes[1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv1d(in_channels=channel_sizes[2], out_channels=channel_sizes[3], kernel_size=kernel_sizes[2]),
            torch.nn.ReLU()
        )
        
        self.hidden_dim = hidden_dim
        
        self.attn = torch.nn.Linear(channel_sizes[3], hidden_dim)
        self.softmax = torch.nn.Softmax(dim=1)
            
        self.dec1_fwd = torch.nn.LSTM(
            input_size=channel_sizes[3], hidden_size=hidden_dim, batch_first=True)
        self.dec1_bkwd = torch.nn.LSTM(
            input_size=channel_sizes[3], hidden_size=hidden_dim, batch_first=True)
        self.dec1_dropout = torch.nn.Dropout(p=dropout)
        
        self.dec2 = torch.nn.LSTM(
            input_size=2*hidden_dim, hidden_size=hidden_dim, num_layers=2, dropout=dropout, batch_first=True, bidirectional=True)
        self.dec2_dropout = torch.nn.Dropout(p=dropout)
        
        self.linear = torch.nn.Linear(in_features=2*hidden_dim, out_features=output_dim)
        self.device = device
        
    def forward(self, x, label=None):
        # Deactivate dropout as needed
        if not self.training:
            self.conv_enc.eval()
        else:
            self.conv_enc.train()
        
        # Encode with convolutional layers
        x_reshaped = x.transpose(1, 2).contiguous()
        conv_enc_out = self.conv_enc(x_reshaped)
        conv_enc_out = conv_enc_out.transpose(1, 2).contiguous()
        
        keys = self.attn(conv_enc_out).transpose(1, 2).contiguous()
        
        dec1_fwd_out = torch.zeros((x.size(0), 60, self.hidden_dim), device=self.device)
        dec1_bkwd_out = torch.zeros((x.size(0), 60, self.hidden_dim), device=self.device)
        h_fwd = torch.zeros((1, x.size(0), self.hidden_dim), device=self.device)
        c_fwd = torch.zeros((1, x.size(0), self.hidden_dim), device=self.device)
        h_bkwd = torch.zeros((1, x.size(0), self.hidden_dim), device=self.device)
        c_bkwd = torch.zeros((1, x.size(0), self.hidden_dim), device=self.device)
        for i in range(60):
            # compute alignment weights, manipulating dimensions as necessary
            align_fwd = torch.bmm(h_fwd.transpose(0, 1).contiguous(), keys).transpose(1, 2).contiguous()
            align_bkwd = torch.bmm(h_bkwd.transpose(0, 1).contiguous(), keys).transpose(1, 2).contiguous()
            
            attn_fwd = self.softmax(align_fwd)
            attn_bkwd = self.softmax(align_bkwd)
            
            context_fwd = torch.sum(attn_fwd * conv_enc_out, dim=1, keepdim=True)
            context_bkwd = torch.sum(attn_bkwd * conv_enc_out, dim=1, keepdim=True)
            
            dec1_fwd_out[:, i:i+1, :], (h_fwd, c_fwd) = self.dec1_fwd(context_fwd, (h_fwd, c_fwd))
            dec1_bkwd_out[:, 60-i-1:60-i, :], (h_bkwd, c_bkwd) = self.dec1_bkwd(context_bkwd, (h_bkwd, c_bkwd))
        
        dec2_in = torch.cat((dec1_fwd_out, dec1_bkwd_out), dim=2)
        if self.training:
            dec2_in = self.dec1_dropout(dec2_in)
        dec_out, _ = self.dec2(dec2_in)
#         if self.training:
#             dec_out = self.dec2_dropout(dec_out)
        y = self.linear(dec_out)
        return y

class LSTMAttention2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, device):
        super(LSTMAttention2, self).__init__()
        
        self.enc = torch.nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=2, dropout=dropout, batch_first=True, bidirectional=True)
        self.enc_dropout = torch.nn.Dropout(p=dropout)
        
        self.hidden_dim = hidden_dim
        self.attn1 = torch.nn.Linear(hidden_dim*2, hidden_dim*2)
        self.attn2 = torch.nn.Linear(hidden_dim*2, hidden_dim*2)
        self.attn3 = torch.nn.Linear(hidden_dim*2, hidden_dim*2)
        self.attn_drp = torch.nn.Dropout(p=dropout)
        self.softmax = torch.nn.Softmax(dim=1)
            
        self.dec1 = torch.nn.LSTM(
            input_size=hidden_dim*2+input_dim, hidden_size=hidden_dim*2, batch_first=True)
        self.drp1 = torch.nn.Dropout(p=dropout)
        self.dec2 = torch.nn.LSTM(
            input_size=hidden_dim*4, hidden_size=hidden_dim*2, batch_first=True)
        self.drp2 = torch.nn.Dropout(p=dropout)
        self.dec3 = torch.nn.LSTM(
            input_size=hidden_dim*4, hidden_size=hidden_dim*2, batch_first=True)
        
        self.linear = torch.nn.Linear(in_features=hidden_dim*2, out_features=output_dim)
        self.device = device
       
    def forward(self, x, label=None):
        enc_out, _ = self.enc(x)
        if self.training:
            enc_out = self.enc_dropout(enc_out)
        
        # cache some values to use throughout loop
        keys1 = self.attn1(enc_out).transpose(1, 2).contiguous()
        keys2 = self.attn2(enc_out).transpose(1, 2).contiguous()
        keys3 = self.attn3(enc_out).transpose(1, 2).contiguous()
        h1 = torch.zeros((1, x.size(0), self.hidden_dim*2), device=self.device)
        h2 = torch.zeros((1, x.size(0), self.hidden_dim*2), device=self.device)
        h3 = torch.zeros((1, x.size(0), self.hidden_dim*2), device=self.device)
        c1 = torch.zeros((1, x.size(0), self.hidden_dim*2), device=self.device)
        c2 = torch.zeros((1, x.size(0), self.hidden_dim*2), device=self.device)
        c3 = torch.zeros((1, x.size(0), self.hidden_dim*2), device=self.device)
        y = torch.zeros((x.size(0), 60, x.size(2)), device=self.device)
        
        for i in range(60):
            if self.training:
                keys1_d = self.attn_drp(keys1)
                keys2_d = self.attn_drp(keys2)
                keys3_d = self.attn_drp(keys3)
            else:
                keys1_d = keys1
                keys2_d = keys2
                keys3_d = keys3
            # perform attention for each LSTM layer
            align1 = torch.bmm(h1.transpose(0, 1).contiguous(), keys1_d).transpose(1, 2).contiguous()
            align2 = torch.bmm(h2.transpose(0, 1).contiguous(), keys2_d).transpose(1, 2).contiguous()
            align3 = torch.bmm(h3.transpose(0, 1).contiguous(), keys3_d).transpose(1, 2).contiguous()
            
            attn1 = self.softmax(align1)
            attn2 = self.softmax(align2)
            attn3 = self.softmax(align3)
            
            context1 = torch.sum(attn1 * enc_out, dim=1, keepdim=True)
            context2 = torch.sum(attn2 * enc_out, dim=1, keepdim=True)
            context3 = torch.sum(attn3 * enc_out, dim=1, keepdim=True)
            
            # evaluate cells for each layer once
            if i == 0:
                inp1 = x[:, -1, :].unsqueeze(1)
            elif self.training:
                inp1 = label[:, i-1, :].unsqueeze(1) #teacher forcing
            else:
                inp1 = y[:, i-1, :].unsqueeze(1) #auto regression
            dec1_out, (h1, c1) = self.dec1(torch.cat((context1, inp1), dim=2), (h1, c1))
            if self.training:
                dec1_out = self.drp1(dec1_out)
            dec2_out, (h2, c2) = self.dec2(torch.cat((context2, dec1_out), dim=2), (h2, c2))
            if self.training:
                dec2_out = self.drp2(dec2_out)
            dec3_out, (h3, c3) = self.dec3(torch.cat((context3, dec2_out), dim=2), (h3, c3))
            
            # store output in respective location
            y[:, i:i+1, :] = self.linear(dec3_out)
        
        return y
            
if __name__ == "__main__":
    model = Conv2Seq(6, 128, 6, 0, torch.device("cpu"))
    print(model(torch.rand((6, 50, 6)))[:, -5:, :])