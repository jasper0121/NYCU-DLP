import os
import matplotlib.pyplot as plt
import torch

# 儲存最佳模型
def save_best_model(model, save_dir, epoch):
    save_path = os.path.join(save_dir, 'model_latest.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Saved best model at epoch {epoch:04} -> {save_path}")

# 繪製並儲存訓練過程的圖表
def plot_metrics(epochs_range, loss_list, acc_list, lr_list):
    metrics = [("Training Loss", loss_list, "Loss", 'tab:blue'),
               ("Evaluation Accuracy", acc_list, "Accuracy", 'tab:orange'),
               ("Learning Rate Schedule", lr_list, "Learning Rate", 'tab:green')]

    plt.figure(figsize=(15, 5))
    for i, (title, values, ylabel, color) in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        plt.plot(epochs_range, values, label=ylabel, color=color)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
    plt.tight_layout()
    plt.savefig('training_metrics.png')