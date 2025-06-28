import os
import wandb
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from diffusers import DDPMScheduler

from data_loader import ICLEVRDataset
from model import ConditionalDDPMModel
from evaluator import evaluation_model
from utils import plot_metrics, save_best_model

def get_args():
    parser = argparse.ArgumentParser(description="Train Conditional DDPM on i-CLEVR dataset")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--image_dir', type=str, default='./iclevr')
    parser.add_argument('--train_json', type=str, default='./train.json')
    parser.add_argument('--objects_path', type=str, default='./objects.json')
    parser.add_argument('--test_json', type=str, default='./test.json')
    parser.add_argument('--new_test_json', type=str, default='./new_test.json')
    parser.add_argument("--beta_schedule", choices=["linear", "squaredcos_cap_v2"], default="linear")
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

'''在每個 epoch 結束後進行生成並評估分類器準確度'''
def evaluate_dataset(model, scheduler, evaluator, args, json_path):
    loader = DataLoader(
        ICLEVRDataset(json_path=json_path, objects_path=args.objects_path, image_dir=args.image_dir, is_train=False),
        batch_size=args.test_batch_size, shuffle=False, num_workers=4
    )
    model.eval()
    total_acc = 0.0
    # 遍歷每個 batch
    for labels in loader: # 對每批標籤生成影像並評估
        labels = labels.to(args.device)
        generated_images = torch.randn(labels.size(0), 3, 64, 64).to(args.device) # 從純噪聲開始生成影像
        # 執行去噪過程
        for timestep in tqdm(scheduler.timesteps, desc="Denoising", leave=False): # 逆向去噪步驟
            t_tensor = torch.full((labels.size(0),), timestep, device=labels.device, dtype=torch.long)
            with torch.no_grad():
                pred_noise = model(generated_images, t_tensor, labels)
            
            # 更新影像
            generated_images = scheduler.step(pred_noise, timestep, generated_images).prev_sample
        total_acc += evaluator.eval(generated_images, labels) # 計算本批次的分類準確度
    avg_acc = total_acc / len(loader)
    return avg_acc

def evaluate(model, scheduler, evaluator, args, epoch, avg_loss, current_lr):
    acc_test = evaluate_dataset(model, scheduler, evaluator, args, args.test_json)
    acc_new_test = evaluate_dataset(model, scheduler, evaluator, args, args.new_test_json)
    avg_acc = (acc_test + acc_new_test) / 2.0
    print(f"Epoch {epoch:03} | Loss: {avg_loss:.4f} | Test Acc: {acc_test:.4f} | New Test Acc: {acc_new_test:.4f} | Avg Acc: {avg_acc:.4f} | LR: {current_lr:.6f}")
    wandb.log({"epoch": epoch, "train_loss": avg_loss, "test_accuracy": acc_test,
               "new_test_accuracy": acc_new_test, "avg_accuracy": avg_acc, "learning_rate": current_lr})
    return avg_acc

'''主要訓練'''
def train(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    dataloader = DataLoader( # 建立訓練集 DataLoader，包含影像與標籤
        ICLEVRDataset(args.train_json, args.objects_path, args.image_dir, is_train=True),
        batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # 初始化模型及訓練相關工具
    model = ConditionalDDPMModel().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule=args.beta_schedule)
    evaluator = evaluation_model()

    lr_scheduler = OneCycleLR( # 動態學習率：OneCycleLR
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(dataloader),
        epochs=args.epochs,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    model.optimizer = optimizer

    loss_list, acc_list, lr_list = [], [], [] # 記錄各項指標
    best_acc = 0.0

    print("Start training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            images, labels = images.to(args.device), labels.to(args.device)

            # 隨機噪聲與時間步索引
            noise = torch.randn_like(images).to(args.device)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (images.size(0),), device=args.device)
            noisy_images = scheduler.add_noise(images, noise, timesteps) # 添加噪聲到影像

            optimizer.zero_grad()
            loss = F.mse_loss(model(noisy_images, timesteps, labels), noise) # 訓練目標：預測噪聲
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        current_lr = lr_scheduler.get_last_lr()[0]
        loss_list.append(avg_loss)
        lr_list.append(current_lr)

        # 同時使用 test.json 與 new_test.json 評估模型
        acc = evaluate(model, scheduler, evaluator, args, epoch, avg_loss, current_lr)
        acc_list.append(acc)
        if acc >= best_acc:
            best_acc = acc
            save_best_model(model, args.save_dir, epoch)

    plot_metrics(range(1, args.epochs + 1), loss_list, acc_list, lr_list) # 繪製訓練曲線
    print("Training complete. Metrics plot saved to training_metrics.png")

if __name__ == '__main__':
    wandb.init(project="Lab6", name="DDPM", save_code=True)
    train(get_args())
