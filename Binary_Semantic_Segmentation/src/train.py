import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from oxford_pet import load_dataset
from evaluate import evaluate
from utils import dice_score, plot_training_curves, save_model
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet

def train(args):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''載入資料集'''
    train_dataset = load_dataset(args.data_path, mode="train", augmentation=args.use_augmentation)
    valid_dataset = load_dataset(args.data_path, mode="valid", augmentation=args.use_augmentation)

    '''建立 DataLoader，訓練集資料洗牌，驗證集不洗牌'''
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False) 
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(valid_dataset)}")

    '''選擇模型架構'''
    model_dict = {"unet": UNet,"resnet34_unet": ResNet34_UNet}
    try:
        model = model_dict[args.model_type](n_channels=3, n_classes=1)
    except KeyError:
        raise ValueError("Unknown model type")
    
    '''模型訓練配置參數初始化'''
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) # 設置Adam優化器
    scheduler = torch.optim.lr_scheduler.OneCycleLR( # 使用OneCycleLR動態調整學習率
                    optimizer, 
                    max_lr=args.learning_rate,  # 最高學習率
                    total_steps=args.epochs * len(train_loader),  # 總步數 = epoch 數 × 每個 epoch 的 batch 數
                    pct_start=0.3,  # 前 30% 時間內增加學習率
                    anneal_strategy='cos',  # 餘弦退火
                    final_div_factor=1e4  # 最後學習率 = max_lr / 10000
                )
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.BCEWithLogitsLoss() # 使用二元交叉熵作為損失函式

    '''開始訓練'''
    train_losses, valid_losses, train_dice_scores, valid_dice_scores = [], [], [], [] # 存每個epoch的loss和dice score
    for epoch in range(args.epochs):
        model.train() # 設定訓練模式
        train_loss, train_dice = 0.0, 0.0 # 累計當前epoch的loss、累計當前epoch的train dice score
        print(f"\nEpoch {epoch+1}/{args.epochs}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"): # 逐批訓練
            # 將 batch 中的image與mask轉換成浮點數並移到指定設備
            images = batch["image"].float().to(device)
            masks  = batch["mask"].float().to(device)
            optimizer.zero_grad() # 清空前一輪梯度
            outputs = model(images) # 前向傳播得到預測結果
            loss = criterion(outputs, masks) # 計算loss
            loss.backward() # 反向傳播計算梯度
            optimizer.step() # 更新模型參數
            scheduler.step() # OneCycleLR 用來更新學習率
            train_loss += loss.item() * images.size(0) # 計算train loss
            train_dice += dice_score(outputs, masks) * images.size(0) # 計算訓練Dice Score
        
        epoch_loss = train_loss / len(train_loader.dataset) # 計算平均loss
        avg_train_dice = train_dice / len(train_loader.dataset) # 計算平均dice score
        train_losses.append(epoch_loss)
        train_dice_scores.append(avg_train_dice)

        # 驗證：回傳驗證loss及平均Dice分數
        val_epoch_loss, avg_dice = evaluate(model, valid_loader, device, criterion)
        valid_losses.append(val_epoch_loss)
        valid_dice_scores.append(avg_dice)
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, "
              f"Train Dice: {avg_train_dice:.4f}, Val Dice: {avg_dice:.4f}"
        )

        #scheduler.step(val_epoch_loss) # ReduceLROnPlateau用
        if epoch >= args.epochs - 5: # 只存最後5個模型
            save_model(model, args.model_type, epoch + 1)
    
    # 繪製訓練曲線
    plot_training_curves(train_losses, valid_losses, train_dice_scores, valid_dice_scores, args.model_type, args.use_augmentation)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'resnet34_unet'], help='choose model architecture')
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4, help='weight decay for optimizer')
    parser.add_argument("--use_augmentation", action="store_true", help="enable data augmentation")
    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)
