import os
import argparse
import torch
from tqdm import tqdm
from diffusers import DDPMScheduler, DDIMScheduler
from torchvision.utils import save_image, make_grid
from model import ConditionalDDPMModel
from evaluator import evaluation_model
from data_loader import ICLEVRDataset
from torch.utils.data import DataLoader

class Inference:
    def __init__(self, args):
        # 初始化參數與隨機種子，並設定運算裝置
        self.args = args
        self.device = args.device
        torch.manual_seed(args.seed)
        
        # 建立儲存圖片的目錄
        os.makedirs(self.args.sample_dir, exist_ok=True)
        self.images_root = os.path.join(self.args.sample_dir, "images")
        os.makedirs(self.images_root, exist_ok=True)

        # 選擇 DDIM 或 DDPM 調度器並初始化
        Scheduler = DDIMScheduler if args.sampler == 'ddim' else DDPMScheduler
        scheduler_kwargs = {
            'num_train_timesteps': args.timesteps,
            'prediction_type': 'epsilon',
            'beta_schedule': args.beta_schedule
        }
        if args.sampler == 'ddim':
            scheduler_kwargs.update({'clip_sample': False})
        self.scheduler = Scheduler(**scheduler_kwargs)
        self.scheduler.set_timesteps(args.timesteps)

        # 載入model與evaluator
        self.model = ConditionalDDPMModel().to(self.device).eval()
        self.model.load_state_dict(torch.load(args.checkpoint, map_location=self.device))
        self.evaluator = evaluation_model()
        self.objects_map = None

    def test_model_on_conditions(self, dataloader, prefix):
        # 以指定條件 (prefix) 測試模型並儲存結果
        print(f"\n生成 {prefix} 圖片中...")
        out = os.path.join(self.images_root, prefix)
        os.makedirs(out, exist_ok=True)

        # 同時跑完所有 batch 並收集生成的影像與對應標籤
        batches = [(self.sample_images(b.to(self.device)), b.to(self.device))
                for b in tqdm(dataloader, desc=f"Generating {prefix}")]
        
        imgs, labels = torch.cat([i for i,_ in batches]), torch.cat([l for _,l in batches])

        # 存整組合併 grid
        gp = os.path.join(self.args.sample_dir, f"{prefix}_grid.png")
        save_image(make_grid(imgs * 0.5 + 0.5, nrow=8), gp)
        print(f"已儲存 {prefix} 合成圖至 {gp}")

        # 存每張小圖
        for i, img in enumerate(imgs):
            save_image(img * 0.5 + 0.5, f"{out}/{i}.png")

        # 使用 evaluator 計算準確度並回傳
        acc = self.evaluator.eval(imgs, labels)
        print(f"Evaluator Accuracy on {prefix}: {acc:.4f}")
        return acc
    
    def sample_images(self, cond_batch):
        # 根據條件向量批次生成影像張量
        bsz = cond_batch.size(0)
        gen_img = torch.randn(bsz, 3, self.args.image_size, self.args.image_size, device=self.device)
        
        # 逐步去噪
        for t in self.scheduler.timesteps:
            t_tensor = torch.full((bsz,), t, device=self.device, dtype=torch.long)
            with torch.no_grad():
                noise = self.model(gen_img, t_tensor, cond_batch)
                out = self.scheduler.step(noise, t, gen_img, eta=self.args.eta) \
                    if self.args.sampler == 'ddim' else self.scheduler.step(noise, t, gen_img)
                gen_img = out.prev_sample
        return gen_img.clamp(-1, 1)

    def sample_denoising_process(self):
        # 構造指定物件條件向量，選擇 red sphere、cyan cylinder、cyan cube
        cond = torch.zeros(len(self.objects_map), device=self.device)
        for name in ["red sphere", "cyan cylinder", "cyan cube"]:
            cond[self.objects_map[name]] = 1.0
        cond = cond.unsqueeze(0)  # shape: [1, cond_dim]

        # 初始化純隨機噪聲樣本 x，shape: [1, 3, H, W]
        x = torch.randn(1, 3, self.args.image_size, self.args.image_size, device=self.device)
        # 選擇去噪步驟函式
        if self.args.sampler == 'ddim':
            step_fn = lambda noise, t, x: self.scheduler.step(noise, t, x, eta=self.args.eta)
        else:
            step_fn = lambda noise, t, x: self.scheduler.step(noise, t, x)

        # 擷取等間隔的中間去噪結果
        timesteps = self.scheduler.timesteps
        step_size = max(1, len(timesteps) // 10)
        images = []
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)
            with torch.no_grad():
                noise = self.model(x, t_tensor, cond)
                x = step_fn(noise, t, x).prev_sample
            if i % step_size == 0 or i + 1 == len(timesteps):
                images.append(x.clamp(-1, 1))

        grid = make_grid(torch.cat([img * 0.5 + 0.5 for img in images]), nrow=len(images))
        save_image(grid, os.path.join(self.args.sample_dir, "denoising_process.png"))
        print(f"去噪過程儲存至 {self.args.sample_dir}/denoising_process.png")

    def run(self):
        # 對兩組 JSON 執行測試
        for json_path, prefix in [(self.args.test_json, "test"), (self.args.new_test_json, "new_test")]:
            dataset = ICLEVRDataset(json_path=json_path, objects_path=self.args.objects_json, image_dir="", is_train=False)
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)
            self.objects_map = dataset.objects_map
            self.test_model_on_conditions(dataloader, prefix) # 執行條件生成測試：生成影像、儲存與評估
        self.sample_denoising_process() # 所有測試完成後，執行去噪過程範例並儲存中間結果

def get_args():
    parser = argparse.ArgumentParser(description="Test Conditional DDPM/DDIM Model (Batch Mode)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_latest.pth")
    parser.add_argument("--test_json", type=str, default="test.json")
    parser.add_argument("--new_test_json", type=str, default="new_test.json")
    parser.add_argument("--objects_json", type=str, default="objects.json")
    parser.add_argument("--sample_dir", type=str, default="test_samples")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", choices=["linear", "squaredcos_cap_v2"], default="linear")
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grid_nrow", type=int, default=8)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

if __name__ == "__main__":
    Inference(get_args()).run()
