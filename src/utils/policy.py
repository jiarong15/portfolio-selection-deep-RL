import torch.nn as nn

## Building the temporal correlational
## convolution block
class TCCB(nn.Module):
    def __init__(self, c_in, c_out, k_size, s_size,
                 p_size, dilation_rate, dropout_rate):
        super(TCCB, self).__init__()
        self.tccb_net = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k_size, stride=s_size, padding=p_size, dilation=dilation_rate),
            nn.Dropout2d(p=dropout_rate),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=k_size, stride=s_size, padding=p_size, dilation=dilation_rate),
            nn.Dropout2d(p=dropout_rate),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=(c_in, 1), stride=s_size, padding=p_size),
            nn.Dropout2d(p=dropout_rate),
            nn.ReLU()
        )

    def forward(self, X):
        return self.tccb_net(X)


## Building the correlation
## information net
class CIN(nn.Module):
    def __init__(self, num_assets):
        super(CIN, self).__init__()
        self.tccb1 = TCCB(c_in=num_assets, c_out=8, k_size=(1,3),
                          s_size=1, p_size=2, dilation_rate=1, dropout_rate=0.2)
        self.tccb2 = TCCB(c_in=8, c_out=16, k_size=(1,3),
                          s_size=1, p_size=4, dilation_rate=2, dropout_rate=0.2)
        self.tccb3 = TCCB(c_in=16, c_out=16, k_size=(1,3),
                          s_size=1, p_size=8, dilation_rate=4, dropout_rate=0.2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,3), stride=1),
            nn.ReLU()
        )

    def forward(self, X):
        X = self.tccb1(X)
        X = self.tccb2(X)
        X = self.tccb3(X)
        return self.conv4(X)

        




class GradientPolicyPPN(nn.Module):
    pass