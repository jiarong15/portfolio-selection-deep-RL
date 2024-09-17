import torch.nn as nn

## Building the temporal correlational
## convolution block
class TCCB(nn.Module):
    def __init__(self, num_assets, c_in, c_out, k_size, s_size,
                 p_size, dilation_rate, dropout_rate):
        super(TCCB, self).__init__()
        self.num_assets = num_assets

        self.tccb_net = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k_size, stride=s_size, padding=p_size, dilation=dilation_rate),
            nn.Dropout2d(p=dropout_rate),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=k_size, stride=s_size, padding=p_size, dilation=dilation_rate),
            nn.Dropout2d(p=dropout_rate),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=(self.num_assets, 1), stride=s_size, padding='same'),
            nn.Dropout2d(p=dropout_rate),
            nn.ReLU()
        )

    def forward(self, X):
        return self.tccb_net(X)


## Building the correlation
## information net
class CIN(nn.Module):
    def __init__(self, no_assets, time_horizon):
        super(CIN, self).__init__()
        self.tccb1 = TCCB(num_assets=no_assets, c_in=4, c_out=8, k_size=(1,5), s_size=1, p_size=(0,2), dilation_rate=1, dropout_rate=0.2)
        self.tccb2  = TCCB(num_assets=no_assets, c_in=8, c_out=16, k_size=(1,5), s_size=1, p_size=(0,4), dilation_rate=2, dropout_rate=0.2)
        self.tccb3  = TCCB(num_assets=no_assets, c_in=16, c_out=16, k_size=(1,5), s_size=1, p_size=(0,8), dilation_rate=4, dropout_rate=0.2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,time_horizon), padding='valid', stride=1),
            nn.ReLU()
        )

    def forward(self, X):
        X = X.permute(0,2,1).unsqueeze(2)
        X = self.tccb1(X)
        X = self.tccb2(X)
        X = self.tccb3(X)
        X = self.conv4(X)
        out = X.squeeze(2).permute(0,2,1)
        return out

## Building the sequential
## information net
class SIN(nn.Module):
    def __init__(self):
        super(TCCB, self).__init__()
        self.network = nn.Sequential(

        )






class GradientPolicyPPN(nn.Module):
    pass