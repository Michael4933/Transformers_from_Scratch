# # filename: train.py
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer
# from datasets import load_dataset
# import torch.nn.functional as F
# import torch.optim as optim
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from torch.optim.lr_scheduler import CosineAnnealingLR
# import os
# from tqdm import tqdm

# from modeling_nini import NiniForCausalLM, NiniConfig, get_nini_tokenizer


# # ========== 1. 自定义Dataset ==========
# class TextDataset(Dataset):
#     def __init__(self, texts, tokenizer, block_size=128):
#         self.examples = []
#         for t in tqdm(texts):
#             tokens = tokenizer(t['text'], truncation=True, max_length=block_size,
#                                padding="max_length", return_tensors="pt")
#             self.examples.append(tokens["input_ids"].squeeze(0))

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, i):
#         ids = self.examples[i]
#         return {"input_ids": ids, "labels": ids.clone()}


# # ========== 3. 训练函数 ==========
# def train(model, dataloader, optimizer, scheduler, device, epochs=3, save_dir="results"):
#     model.train()
#     os.makedirs(save_dir, exist_ok=True)
#     losses = []
#     lrs = []

#     for epoch in range(epochs):
#         epoch_loss = 0.0
#         pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
#         for batch in pbar:
#             input_ids = batch["input_ids"].to(device)
#             labels = batch["labels"].to(device)

#             outputs = model(input_ids=input_ids, labels=labels)
#             loss = outputs["loss"]

#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
            
#             # 记录当前学习率
#             current_lr = optimizer.param_groups[0]['lr']
#             lrs.append(current_lr)

#             epoch_loss += loss.item()
#             pbar.set_postfix({"loss": loss.item(), "lr": f"{current_lr:.6f}"})

#         # 每个epoch结束后更新学习率
#         scheduler.step()
        
#         avg_loss = epoch_loss / len(dataloader)
#         losses.append(avg_loss)
#         print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}, Learning rate: {current_lr:.6f}")

#         torch.save(model.state_dict(), os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt"))

#     # 画 loss 曲线
#     plt.figure(figsize=(12, 5))
    
#     # 损失曲线
#     plt.subplot(1, 2, 1)
#     plt.plot(range(1, epochs + 1), losses, marker='o')
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Training Loss Curve")
    
#     # 学习率曲线
#     plt.subplot(1, 2, 2)
#     plt.plot(range(1, len(lrs) + 1), lrs)
#     plt.xlabel("Step")
#     plt.ylabel("Learning Rate")
#     plt.title("Learning Rate Schedule (Cosine Annealing)")
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, "loss_lr_curve.png"))
#     print(f"Loss and LR curves saved to {save_dir}/loss_lr_curve.png")


# # ========== 4. 主函数 ==========
# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # 加载 tokenizer
#     tokenizer = get_nini_tokenizer()
#     vocab_size = len(tokenizer)

#     # 加载中文语料
#     texts = load_dataset("austenjs/ClueCorpusSmallDataset", split="train[:1%]")

#     # 构造 Dataset
#     dataset = TextDataset(texts, tokenizer, block_size=128)
#     dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

#     # 构造模型
#     config = NiniConfig(
#         vocab_size=vocab_size,
#         hidden_size=768,
#         num_hidden_layers=16,
#         num_attention_heads=8,
#         intermediate_size=4096,
#         dropout=0.1,
#     )
#     model = NiniForCausalLM(config).to(device)

#     # 优化器
#     optimizer = optim.AdamW(model.parameters(), lr=3e-4)
#     total_steps = 3 * len(dataloader)
    
#     # 余弦退火调度器
#     scheduler = CosineAnnealingLR(
#         optimizer,
#         T_max=total_steps,
#         eta_min=3e-5  # 3e-4 * 0.1 = 3e-5
#     )

#     # 训练（传入scheduler）
#     train(model, dataloader, optimizer, scheduler, device, epochs=3, save_dir="results")


# if __name__ == "__main__":
#     main()


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import wandb  # 导入WandB
from datetime import datetime

from modeling_nini import NiniForCausalLM, NiniConfig, get_nini_tokenizer


# ========== 1. 自定义Dataset ==========
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=128):
        self.examples = []
        for t in tqdm(texts, desc="Processing dataset"):
            tokens = tokenizer(t['text'], truncation=True, max_length=block_size,
                              padding="max_length", return_tensors="pt")
            self.examples.append(tokens["input_ids"].squeeze(0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids = self.examples[i]
        return {"input_ids": ids, "labels": ids.clone()}


# ========== 3. 训练函数 ==========
def train(model, dataloader, optimizer, scheduler, device, epochs=3, save_dir="results"):
    model.train()
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化记录列表
    step_losses = []  # 记录每个step的loss
    epoch_losses = []  # 记录每个epoch的平均loss

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for step, batch in enumerate(pbar, 1):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()  # 移到这里，每个step更新一次学习率
            
            # 记录当前学习率和loss
            current_lr = optimizer.param_groups[0]['lr']
            step_loss = loss.item()
            
            # 保存step级别的指标
            step_losses.append(step_loss)
            epoch_loss += step_loss
            
            # 实时记录到WandB (每步都记录)
            wandb.log({
                "step_loss": step_loss,
                "learning_rate": current_lr,
                "epoch": epoch + 1,
                "global_step": epoch * len(dataloader) + step
            })

            # 更新进度条
            pbar.set_postfix({
                "step_loss": f"{step_loss:.4f}",
                "avg_epoch_loss": f"{(epoch_loss/step):.4f}",
                "lr": f"{current_lr:.6f}"
            })

        # 计算epoch平均loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)
        
        # 记录epoch级别的指标到WandB
        wandb.log({
            "avg_epoch_loss": avg_epoch_loss,
            "epoch_complete": epoch + 1
        })
        
        print(f"Epoch {epoch+1} complete - Average loss: {avg_epoch_loss:.4f}")

        # 保存模型 checkpoint
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        wandb.save(checkpoint_path)  # 将checkpoint同步到WandB

    # 绘制本地loss曲线
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Curve (per epoch)")
    
    # 步骤级损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(step_losses) + 1), step_losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve (per step)")
    
    plt.tight_layout()
    loss_curve_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(loss_curve_path)
    wandb.log({"loss_curves": wandb.Image(loss_curve_path)})  # 上传图片到WandB
    print(f"Loss curves saved to {loss_curve_path}")

    return epoch_losses, step_losses


# ========== 4. 主函数 ==========
def main():
    # 初始化WandB
    run_name = f"nini_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="nini-language-model",  # 项目名称
        name=run_name,                  # 运行名称
        config={                        # 记录超参数
            "batch_size": 128,
            "block_size": 128,
            "hidden_size": 768,
            "num_hidden_layers": 16,
            "num_attention_heads": 8,
            "intermediate_size": 4096,
            "dropout": 0.1,
            "learning_rate": 3e-4,
            "min_learning_rate": 3e-5,
            "epochs": 3,
            "dataset": "austenjs/ClueCorpusSmallDataset (1%)"
        }
    )
    config = wandb.config  # 可以通过wandb.config获取配置

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb.log({"device": device})  # 记录使用的设备

    # 加载 tokenizer
    tokenizer = get_nini_tokenizer()
    vocab_size = len(tokenizer)
    wandb.log({"vocab_size": vocab_size})

    # 加载中文语料
    texts = load_dataset("austenjs/ClueCorpusSmallDataset", split="train[:1%]")
    wandb.log({"dataset_size": len(texts)})

    # 构造 Dataset
    dataset = TextDataset(texts, tokenizer, block_size=config.block_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    wandb.log({"num_batches_per_epoch": len(dataloader)})

    # 构造模型
    model_config = NiniConfig(
        vocab_size=vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        dropout=config.dropout,
    )
    model = NiniForCausalLM(model_config).to(device)
    
    # 记录模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    })

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = config.epochs * len(dataloader)
    
    # 余弦退火调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=config.min_learning_rate
    )

    # 训练
    save_dir = os.path.join("results", run_name)
    train(model, dataloader, optimizer, scheduler, device, 
          epochs=config.epochs, save_dir=save_dir)
    
    # 完成训练
    wandb.finish()
    print("Training completed. Results logged to WandB.")


if __name__ == "__main__":
    main()