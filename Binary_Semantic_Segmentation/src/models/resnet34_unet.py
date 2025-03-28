'''
參考網址：
1. https://ithelp.ithome.com.tw/m/articles/10333931
2. https://blog.csdn.net/weixin_44350337/article/details/115474009
3. https://www.researchgate.net/figure/UNet-architecture-with-a-ResNet-34-encoder-The-output-of-the-additional-1x1-convolution_fig3_350858002
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

############################
#     ResNet34 Encoder     #
############################

'''定義 ResNet 的 BasicBlock(兩層 3x3 卷積，即架構圖中每次跳接箭頭的中間都會夾的2塊模組)，和DoubleConv很類似'''
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) # 3x3 conv, 64
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False) # 3x3 conv, 64
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels: # 當stride不等於1 or 通道數不相等時，處理跳躍連接(虛線箭頭的部分)
            self.downsample = nn.Sequential( # 需要做1x1卷積下採樣來調整通道，讓後續可以作加法操作
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        skip_connection = x

        # 第1層卷積
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第2層卷積
        out = self.conv2(out)
        out = self.bn2(out)

        # 當stride不等於1 or 通道數不相等時，處理跳躍連接(虛線箭頭的部分)
        if self.downsample is not None:
            skip_connection = self.downsample(x) # 用1x1卷積對skip_connection做調整

        # 殘差連接(Residual Connection)
        out += skip_connection
        return self.relu(out)

'''ResNet34 編碼器，取代原本UNet的下採樣'''
class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(ResNet34Encoder, self).__init__()
        # 根據架構圖，先過一次7x7卷積，stride=2代表架構圖中的/2部分，把特徵圖縮小一半(7x7 conv, 64, /2)
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), # 在卷積後進行批次正規化，目的是用來提升訓練穩定性
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 對輸入進行 3×3 的最大池化(pool, /2)

        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1) # layer1：有3個block，沒有虛線箭頭跳接
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2) # layer2：有4個block，有虛線箭頭跳接
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2) # layer3：有6個block，有虛線箭頭跳接
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2) # layer4：有3個block，有虛線箭頭跳接

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride), # 第1個BasicBlock可能會有跳接
            *[BasicBlock(out_channels, out_channels, stride=1) for _ in range(1, blocks)]
        )

    def forward(self, x):
        # 初始模組: 卷積 -> BN -> ReLU，輸出 x0 (尺寸: H/2, 通道: 64)
        x0 = self.initial(x)
        # 最大池化: 進一步下採樣，尺寸從 H/2 變為 H/4，並保留 x0 作為跳連接特徵
        x = self.maxpool(x0)  # 尺寸: H/4, 64

        # 分別通過四個由 BasicBlock 組成的層
        x1 = self.layer1(x)   # layer1 輸出: 尺寸保持 H/4, 通道: 64
        x2 = self.layer2(x1)  # layer2 輸出: 尺寸下採樣至 H/8, 通道: 128
        x3 = self.layer3(x2)  # layer3 輸出: 尺寸下採樣至 H/16, 通道: 256
        x4 = self.layer4(x3)  # layer4 輸出: 尺寸下採樣至 H/32, 通道: 512
        return x0, x1, x2, x3, x4 # 回傳所有層的特徵圖

###########################
#      UNet Decoder       #
###########################

'''定義兩層卷積塊(Conv -> ReLU -> Conv -> ReLU)，即論文Unet架構圖中常見的連續2個深藍色箭頭(conv 3x3, ReLU)'''
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # 定義一個Sequential容器，內含兩層卷積，每層卷積後接 BatchNorm 和 ReLU 激活函數
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

'''定義上採樣模塊：上採樣後拼接對應層的特徵，再進行雙卷積。即綠箭頭(up-conv 2x2) + 藍箭頭 * 2'''
class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(Up, self).__init__()
        # 上採樣過程中使用雙線性插值(雙線性插值)，可以讓圖較為平滑
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # 拼接後的通道數為 in_channels + skip_channels，接著用雙卷積塊融合特徵，輸出通道為 out_channels
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip): # x從下面上採樣得到、skip從先前左側得到
        x = self.up(x)
        # 若尺寸不匹配則做padding
        diffY = skip.size()[2] - x.size()[2] # 計算兩邊長度差
        diffX = skip.size()[3] - x.size()[3] # 計算兩邊寬度差
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2]) # 根據長度和寬度差，在x的周圍補0
        
        x = torch.cat([skip, x], dim=1) # 左側跳接和從下面上採樣的張量拼接
        return self.conv(x) # 拼接後再照慣例過2次藍箭頭(DoubleConv)

'''定義最後輸出層，即藍綠色箭頭(conv 1x1)'''
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

###########################
#   ResNet34_UNet 模型    #
###########################

'''建立 ResNet34_UNet 模型'''
class ResNet34_UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ResNet34_UNet, self).__init__()
        self.encoder = ResNet34Encoder(in_channels=n_channels)
        # decoder 的上採樣模組
        self.up1 = Up(in_channels=512, skip_channels=256, out_channels=256)  # 融合 x4 與 x3
        self.up2 = Up(in_channels=256, skip_channels=128, out_channels=128)  # 融合下層輸出與 x2
        self.up3 = Up(in_channels=128, skip_channels=64,  out_channels=64)   # 融合下層輸出與 x1
        self.up4 = Up(in_channels=64,  skip_channels=64,  out_channels=32)   # 融合下層輸出與 x0
        self.final_up = nn.UpsamplingBilinear2d(scale_factor=2) # 最後一層上採樣至原始尺寸
        self.outc = OutConv(32, n_classes) # 最後再做一次卷積，通道數：32 -> n_channels

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x) # 做ResNet34下採樣
        d1 = self.up1(x4, x3)        # 融合 x4 和 x3 得到 d1，尺寸為H/16，通道數 256
        d2 = self.up2(d1, x2)        # 融合 d1 和 x2 得到 d2，尺寸為H/8，通道數 128
        d3 = self.up3(d2, x1)        # 融合 d2 和 x1 得到 d3，尺寸為H/4，通道數 64
        d4 = self.up4(d3, x0)        # 融合 d3 和 x0 得到 d4，尺寸為H/2，通道數 64
        up_final = self.final_up(d4) # 上採樣回原始尺寸
        return self.outc(up_final)   # 最頂層：最後再做一次卷積
