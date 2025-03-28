import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

def dice_score(pred_mask, gt_mask, smooth=1e-6):
    # 如果是 tensor，就先做 sigmoid、二值化與轉 numpy
    if torch.is_tensor(pred_mask):
        pred_mask = (torch.sigmoid(pred_mask) > 0.5).float().cpu().numpy()
    if torch.is_tensor(gt_mask):
        gt_mask = gt_mask.cpu().numpy()
    pred_mask = pred_mask.astype(np.float32)
    gt_mask = gt_mask.astype(np.float32)

    # 計算 Dice Score，加上 smooth 以避免分母為 0 的錯誤
    return 2.0 * np.sum(pred_mask * gt_mask) / (np.sum(pred_mask) + np.sum(gt_mask) + smooth)

def plot_training_curves(train_losses, valid_losses, train_dice_scores, valid_dice_scores, model_name, use_augmentation):
    epochs = range(1, len(train_losses) + 1) # 取得epoch數
    
    # 設定後綴名稱，若使用 augmentation 則加上 "_augmentation"
    suffix = "_augmentation" if use_augmentation else ""
    augmentation_text = " with Data Augmentation" if use_augmentation else ""

    '''畫train和val的loss和dice score'''
    plots = [ # 第1組圖：畫loss、第2組：畫dice score
        (["Train Loss", "Val Loss"], [train_losses, valid_losses], "Training Loss", f"training_loss_{model_name}{suffix}.png"),
        (["Train Dice Score", "Val Dice Score"], [train_dice_scores, valid_dice_scores], "Dice Score", f"dice_score_{model_name}{suffix}.png")
    ]
    
    for labels, data, ylabel, filename in plots:
        plt.figure() # 建立新圖表
        for d, label in zip(data, labels): # 畫train和val
            plt.plot(epochs, d, label=label)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} ({model_name}){augmentation_text}")
        plt.legend()
        plt.savefig(filename)
        plt.close()

def save_model(model, model_type, epoch):
    """儲存模型並額外存一份最新的模型"""
    os.makedirs("saved_models", exist_ok=True)
    checkpoint_name = f"{model_type}_epoch{epoch}.pth"
    latest_checkpoint = f"{model_type}_latest.pth"
    torch.save(model.state_dict(), os.path.join("saved_models", checkpoint_name))
    torch.save(model.state_dict(), os.path.join("saved_models", latest_checkpoint))

def save_overlay_image(image, output, save_dir, index):
    # 將模型輸出轉換成二值遮罩並映射到 0-255 範圍
    pred_mask = ((torch.sigmoid(output) > 0.5).cpu().numpy().squeeze() * 255).astype(np.uint8)

    # 將圖像轉為 numpy (HWC 格式) 並轉換為 uint8
    img = image.cpu().permute(1, 2, 0).numpy()
    img = (img * 255 if img.max() <= 1.0 else img).astype("uint8")
    green_mask = np.zeros_like(img) # 建立一個與img相同形狀和型態的全零陣列
    green_mask[:, :, 1] = pred_mask # 將pred_mask的內容賦值給綠色通道(index = 1)

    '''疊加原圖與遮罩(alpha = 0.5)，並轉換為BGR格式儲存'''
    overlay = cv2.addWeighted(img, 1.0, green_mask, 0.5, 0)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, f"overlay_{index}.png"), overlay_bgr)

def save_overlay_ground_truth(image, gt_mask, pred_mask, save_dir, index):
    gt = (gt_mask.squeeze().cpu().numpy() * 255 * 0.7).astype(np.uint8)  # 0.7是調整白色亮度

    # 將模型預測輸出經 sigmoid 二值化為遮罩，再映射到 0 ~ 255
    pred = (torch.sigmoid(pred_mask).squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

    '''疊加ground_truth與遮罩(alpha = 0.5)'''
    base = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB) # 將灰階的GT遮罩轉為RGB三通道圖像
    base[:, :, 1] = cv2.addWeighted(base[:, :, 1], 1.0, pred, 0.5, 0)  # 疊加綠色遮罩
    cv2.imwrite(os.path.join(save_dir, f"ground_truth_{index}.png"), base)
