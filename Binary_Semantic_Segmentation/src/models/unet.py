'''
參考網址：
1. https://www.youtube.com/watch?v=I9MPzQCd4o4
2. https://ithelp.ithome.com.tw/articles/10240314
3. https://blog.csdn.net/knighthood2001/article/details/138075554?spm=1001.2014.3001.5506
4. https://tomohiroliu22.medium.com/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92paper%E7%B3%BB%E5%88%97-05-u-net-41be7533c934
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

'''定義兩層卷積塊(Conv -> ReLU -> Conv -> ReLU)，即論文Unet架構圖中常見的連續2個深藍色箭頭(conv 3x3, ReLU)'''
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # 定義一個Sequential容器，內含兩層卷積，每層卷積後接 BatchNorm 和 ReLU 激活函數
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # 第一個卷積層：使用 3x3 kernel，padding=1
            nn.BatchNorm2d(out_channels), # 在卷積後進行批次正規化，目的是用來提升訓練穩定性
            nn.ReLU(inplace=True), # 使用 ReLU 激活函數，inplace=True 代表直接在輸入的數據上進行修改，比較可以省空間

            # 第二個卷積層：再使用 3x3 kernel，輸入與輸出通道數相同
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x) # 根據__init__所訂的結構去處理x

'''定義下採樣模塊：使用最大池化(論文中的紅色箭頭)後接雙卷積塊。即紅箭頭(max pool 2x2) + 藍箭頭 * 2'''
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), # 對輸入進行 2×2 的最大池化(每個2x2區塊只保留最大值作為該區域代表，因此長寬會減半)
            DoubleConv(in_channels, out_channels) # 接著再進雙卷積模組，對池化後的特徵進一步提取，將通道數轉成out_channels
        )

    def forward(self, x):
        return self.maxpool_conv(x) # 根據__init__所訂的結構去處理x

'''定義上採樣模塊：上採樣後拼接對應層的特徵，再進行雙卷積。即綠箭頭(up-conv 2x2) + 藍箭頭 * 2'''
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        # 上採樣過程中使用雙線性插值，可以讓圖較為平滑
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1) # 調整通道數

        # 拼接後的通道數為 2*out_channels，接著用雙卷積塊融合特徵，輸出通道為 out_channels
        self.conv = DoubleConv(2 * out_channels, out_channels)

    def forward(self, x1, x2): # x1從下面上採樣得到、x2從先前左側得到
        x1 = self.conv1x1(self.up(x1)) # 用bilinear後要過1x1卷積改通道數

        # 進行尺寸匹配，若 x1 與 x2 尺寸不一致，則做 padding
        diffY = x2.size()[2] - x1.size()[2] # 計算兩邊長度差
        diffX = x2.size()[3] - x1.size()[3] # 計算兩邊寬度差
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2]) # 根據長度和寬度差，在x1的周圍補0
        
        # 將上採樣的特徵與編碼層對應特徵做拼接
        x = torch.cat([x2, x1], dim=1) # 根據論文，左側跳接和從下面上採樣的張量拼接
        return self.conv(x) # 拼接後再照慣例過2次藍箭頭(DoubleConv)

'''定義最後輸出層，即藍綠色箭頭(conv 1x1)'''
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) # 最後的卷積層：使用 3x3 kernel

    def forward(self, x):
        return self.conv(x) # 根據__init__所訂的結構去處理x

'''建立 UNet 模型'''
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64) # 按照論文架構，先做一次雙卷積，通道數：n_channels -> 64
        self.down1 = Down(64, 128)   # 下採樣 + 雙卷積，通道數：64 -> 128
        self.down2 = Down(128, 256)  # 下採樣 + 雙卷積，通道數：128 -> 256
        self.down3 = Down(256, 512)  # 下採樣 + 雙卷積，通道數：256 -> 512
        self.down4 = Down(512, 1024) # 最底層下採樣 + 雙卷積，通道數：512 -> 1024
        self.up1 = Up(1024, 512)     # 上採樣 + 雙卷積，通道數：1024 -> 512
        self.up2 = Up(512, 256)      # 上採樣 + 雙卷積，通道數：512 -> 256
        self.up3 = Up(256, 128)      # 上採樣 + 雙卷積，通道數：256 -> 128
        self.up4 = Up(128, 64)       # 上採樣 + 雙卷積，通道數：128 -> 64
        self.outc = OutConv(64, n_classes) # 按照論文架構，最後再做一次卷積，通道數：64 -> n_channels

    def forward(self, x):
        x1 = self.inc(x)      # 編碼層 1：先做一次雙卷積
        x2 = self.down1(x1)   # 編碼層 2：下採樣 64 -> 128
        x3 = self.down2(x2)   # 編碼層 3：下採樣 128 -> 256
        x4 = self.down3(x3)   # 編碼層 4：下採樣 256 -> 512
        x5 = self.down4(x4)   # 最底層：下採樣 512 -> 1024

        x = self.up1(x5, x4)  # 解碼層 1：上採樣 1024 -> 512
        x = self.up2(x, x3)   # 解碼層 2：上採樣 512 -> 256
        x = self.up3(x, x2)   # 解碼層 3：上採樣 256 -> 128
        x = self.up4(x, x1)   # 解碼層 4：上採樣 128 -> 64
        return self.outc(x)   # 最頂層：最後再做一次卷積