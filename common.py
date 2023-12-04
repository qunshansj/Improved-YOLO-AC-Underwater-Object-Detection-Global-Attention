python

class CBS(nn.Module):
    """ Convolution -> BatchNorm -> SiLU """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(CBS, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.layer(x)


class MP1(nn.Module):
    """ MP1 block """

    def __init__(self, in_channels):
        super(MP1, self).__init__()
        # Upper branch
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cbs_upper = CBS(in_channels, in_channels // 2, 1, 1)

        # Lower branch
        self.cbs_lower1 = CBS(in_channels, in_channels // 2, 1, 1)
        self.cbs_lower2 = CBS(in_channels // 2, in_channels // 2, 3, 2)

    def forward(self, x):
        # Upper branch
        x_upper = self.maxpool(x)
        x_upper = self.cbs_upper(x_upper)

        # Lower branch
        x_lower = self.cbs_lower1(x)
        x_lower = self.cbs_lower2(x_lower)

        # Concatenate
        return torch.cat([x_upper, x_lower], dim=1)


class MP2(nn.Module):
    """ MP2 block, similar to MP1 but with different output channels """

    def __init__(self, in_channels):
        super(MP2, self).__init__()
        # Upper branch
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cbs_upper = CBS(in_channels, in_channels, 1, 1)

        # Lower branch
        self.cbs_lower1 = CBS(in_channels, in_channels, 1, 1)
        self.cbs_lower2 = CBS(in_channels, in_channels, 3, 2)

    def forward(self, x):
        # Upper branch
        x_upper = self.maxpool(x)
        x_upper = self.cbs_upper(x_upper)

        # Lower branch
        x_lower = self.cbs_lower1(x)
        x_lower = self.cbs_lower2(x_lower)

        # Concatenate
        return torch.cat([x_upper, x_lower], dim=1)

class ChannelAttention(nn.Module):
    """ Channel Attention Submodule for GAM """

    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.SiLU(),
            nn.Linear(in_channels // 2, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pool = torch.mean(x, dim=[2, 3])
        return self.mlp(avg_pool).unsqueeze(2).unsqueeze(3) * x


class SpatialAttention(nn.Module):
    """ Spatial Attention Submodule for GAM """

    def __init__(self, in_channels, out_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x) * x


class GAM(nn.Module):
    """ Global Attention Module (GAM) """

    def __init__(self, in_channels):
        super(GAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention(in_channels, in_channels // 2)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
    ......
