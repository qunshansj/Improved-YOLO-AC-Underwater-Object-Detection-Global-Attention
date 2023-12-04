python

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
