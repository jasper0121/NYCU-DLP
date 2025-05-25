import torch
from utils import dice_score

def evaluate(net, data, device, criterion):
    net.eval()  # 將模型設置為驗證模式
    total_loss, total_dice = 0.0, 0.0  # 初始化累積loss與累積dice score

    '''開始evaluate'''
    with torch.no_grad():  # 關閉梯度計算以節省記憶體
        for batch in data:  # 逐批讀取驗證資料
            # 將圖片與對應的 mask 轉換為 float 並移至指定設備
            images = batch["image"].float().to(device)
            masks  = batch["mask"].float().to(device)
            outputs = net(images)  # 使用模型進行前向傳播，獲得預測結果
            loss = criterion(outputs, masks)  # 計算loss
            total_loss += loss.item() * images.size(0) # 計算train loss
            for i in range(images.size(0)): # 逐張計算圖片的dise score並累加
                total_dice += dice_score(outputs[i], masks[i])

    # 計算驗證集的平均loss與平均dice score
    return total_loss / len(data.dataset), total_dice / len(data.dataset)
