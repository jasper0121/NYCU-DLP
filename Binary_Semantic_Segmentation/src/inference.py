import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from oxford_pet import load_dataset
from utils import dice_score, save_overlay_image, save_overlay_ground_truth

def get_args():
    parser = argparse.ArgumentParser(description='Inference script for segmentation model')
    parser.add_argument('--model', default='saved_models/unet_epoch5.pth', help='Path to the saved model weights')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'resnet34_unet'], help='Choose model architecture')
    return parser.parse_args()

def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    '''載入資料集，建立 DataLoader，測試集不洗牌'''
    test_dataset = load_dataset(args.data_path, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    '''選擇模型架構'''
    model_dict = {'unet': UNet, 'resnet34_unet': ResNet34_UNet}
    if args.model_type not in model_dict:
        raise ValueError("Unknown model type: {}".format(args.model_type))
    model = model_dict[args.model_type](n_channels=3, n_classes=1)
    
    # 檢查指定的模型權重檔案是否存在
    if not os.path.exists(args.model):
        raise FileNotFoundError("Model checkpoint not found at: {}".format(args.model))
    
    # 載入模型權重，並將模型移至運算裝置
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval() # 設定模型為驗證模式
    
    '''建立輸出資料夾'''
    result_root = os.path.join("result", args.model_type)
    overlay_dir = os.path.join(result_root, "overlay")
    gt_dir = os.path.join(result_root, "ground_truth")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    
    '''開始inference'''
    total_dice, num_samples = 0.0, 0
    with torch.no_grad(): # 關閉梯度計算以節省記憶體
        for batch in tqdm(test_loader, desc="Inference Progress"):
            # 將 batch 中的圖片與 mask 轉換成 float 並移至裝置
            images = batch["image"].float().to(device)
            masks  = batch["mask"].float().to(device)
            outputs = model(images)  # 前向傳播取得模型輸出
            
            # 對每個樣本計算 Dice 分數並保存遮罩影像
            for i in range(images.size(0)):
                dice = dice_score(outputs[i], masks[i]) # 計算該樣本的 Dice 分數
                total_dice += dice
                num_samples += 1
                
                # 把推論出來的區域疊加在test原圖上、並且也疊加在ground_truth上
                save_overlay_image(images[i], outputs[i], overlay_dir, num_samples)
                save_overlay_ground_truth(images[i], masks[i], outputs[i], gt_dir, num_samples)
    
    # 計算並輸出測試集上的平均dice score
    avg_dice = total_dice / num_samples if num_samples > 0 else 0.0
    print("Average Dice Score on test set: {:.4f}".format(avg_dice))

if __name__ == '__main__':
    args = get_args()
    inference(args)
