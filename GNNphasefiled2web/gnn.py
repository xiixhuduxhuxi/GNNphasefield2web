# -*- coding: utf-8 -*-
"""
完整 GNN 版本脚本：基于 KNN 构图的轻量 GCN（纯 PyTorch 实现）
增加模型保存与加载功能
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import griddata

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -------------------- 设置 --------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('predictions', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)
os.makedirs('models', exist_ok=True)  # 新增模型保存目录

# -------------------- 1. 数据加载 --------------------
disp_df = pd.read_csv("disp/加载位移.csv", header=0, encoding='utf-8')
displacements = disp_df["加载位移"].values[0:293]   
steps = list(range(2, 295, 2))                     
n_steps = len(steps) 

sample_step = steps[0]
sample_df = pd.read_csv(f"data/node_data_step_{sample_step}.csv", encoding='utf-8')
nodes_per_step = len(sample_df)
n_total_nodes = nodes_per_step * n_steps

print(f"steps 数量: {n_steps}, nodes_per_step: {nodes_per_step}, 总节点数: {n_total_nodes}")

node_features = np.empty((n_total_nodes, 3), dtype=np.float32)   
targets = np.empty((n_total_nodes, 3), dtype=np.float32)         

for i, (step, disp) in enumerate(zip(steps, displacements)):
    df = pd.read_csv(f"data/node_data_step_{step}.csv", encoding='utf-8')
    s = i * nodes_per_step
    e = s + nodes_per_step
    node_features[s:e, :2] = df[['X', 'Y']].values
    node_features[s:e, 2] = disp
    targets[s:e, :] = df[['U', 'V', 'Phi']].values
    del df

scaler_X = StandardScaler()
scaler_y = StandardScaler()
node_features_scaled = scaler_X.fit_transform(node_features)
targets_scaled = scaler_y.fit_transform(targets)

# -------------------- 2. 构图（KNN） --------------------
coords_base = node_features[:nodes_per_step, :2]
K = 4
knn = NearestNeighbors(n_neighbors=K).fit(coords_base)
dist, knn_idx = knn.kneighbors(coords_base)

edge_list = []
for i in range(nodes_per_step):
    for j in knn_idx[i]:
        edge_list.append([i, j])
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

row, col = edge_index[0], edge_index[1]
deg = torch.zeros(nodes_per_step, dtype=torch.float)
deg.index_add_(0, col, torch.ones_like(col, dtype=torch.float))
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

# -------------------- 3. Dataset / DataLoader --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

class GraphStepDataset(Dataset):
    def __init__(self, step_indices):
        self.step_indices = step_indices

    def __len__(self):
        return len(self.step_indices)

    def __getitem__(self, idx):
        step_idx = self.step_indices[idx]
        s = step_idx * nodes_per_step
        e = s + nodes_per_step
        x = torch.tensor(node_features_scaled[s:e], dtype=torch.float32)
        y = torch.tensor(targets_scaled[s:e], dtype=torch.float32)
        return x, y, step_idx

all_step_indices = list(range(n_steps))
train_idx, test_idx = train_test_split(all_step_indices, test_size=0.2, random_state=42)

train_loader = DataLoader(GraphStepDataset(train_idx), batch_size=1, shuffle=True)
test_loader = DataLoader(GraphStepDataset(test_idx), batch_size=1, shuffle=False)

# -------------------- 4. 轻量 GCN 实现 --------------------
class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, edge_index, deg_inv_sqrt):
        row, col = edge_index[0], edge_index[1]
        x_j = x[row]
        norm = (deg_inv_sqrt[row].unsqueeze(1) * deg_inv_sqrt[col].unsqueeze(1))
        msg = x_j * norm
        agg = torch.zeros_like(x)
        agg.index_add_(0, col, msg)
        out = self.linear(agg)
        return F.elu(out)

class GNNModel(nn.Module):
    def __init__(self, in_dim=3, hidden=64, out_dim=3, dropout=0.2):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.gcn3 = GCNLayer(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, deg_inv_sqrt):
        x = self.gcn1(x, edge_index, deg_inv_sqrt)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index, deg_inv_sqrt)
        x = self.dropout(x)
        x = self.gcn3(x, edge_index, deg_inv_sqrt)
        out = self.fc_out(x)
        return out

# -------------------- 5. 训练设置 --------------------
model = GNNModel(in_dim=3, hidden=64, out_dim=3, dropout=0.2).to(device)
edge_index = edge_index.to(device)
deg_inv_sqrt = deg_inv_sqrt.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99, eps=1e-08, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)

train_loss_history = []
test_loss_history = []
lr_history = []

if torch.cuda.is_available():
    print(f"初始 GPU 显存: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

n_epochs = 2000
best_test_loss = float('inf')

print("开始训练 GNN 模型...")
for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        x_batch, y_batch, _ = batch
        x_batch = x_batch[0].to(device)
        y_batch = y_batch[0].to(device)

        optimizer.zero_grad()
        pred = model(x_batch, edge_index, deg_inv_sqrt)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch, _ = batch
            x_batch = x_batch[0].to(device)
            y_batch = y_batch[0].to(device)
            pred = model(x_batch, edge_index, deg_inv_sqrt)
            test_loss += criterion(pred, y_batch).item()
    avg_test_loss = test_loss / len(test_loader)

    scheduler.step(avg_test_loss)
    
    # 保存最佳模型
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_loss': avg_test_loss,
        }, 'models/gnn_model_best.pth')

    train_loss_history.append(avg_train_loss)
    test_loss_history.append(avg_test_loss)
    lr_history.append(optimizer.param_groups[0]['lr'])

    if (epoch + 1) % 20 == 0 or epoch == 0:
        mem_info = f" | GPU mem: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB" if torch.cuda.is_available() else ""
        print(f"Epoch {epoch+1:04d} | Train: {avg_train_loss:.6e} | Test: {avg_test_loss:.6e} | LR: {optimizer.param_groups[0]['lr']:.3e}{mem_info}")

# 保存最终模型和完整信息
torch.save({
    'epoch': n_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss_history': train_loss_history,
    'test_loss_history': test_loss_history,
    'lr_history': lr_history,
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'edge_index': edge_index,
    'deg_inv_sqrt': deg_inv_sqrt,
}, 'models/gnn_model_final.pth')

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"训练完成，最终 GPU 显存: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
print("\n模型已保存至 models/ 文件夹")

# -------------------- 6. 绘图：训练损失与学习率 --------------------
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(train_loss_history)+1), train_loss_history, label='训练损失')
plt.plot(range(1, len(test_loss_history)+1), test_loss_history, label='测试损失')
plt.xlabel('训练轮次')
plt.ylabel('MSE 损失')
plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('visualizations/loss_curve_gnn.png', dpi=150)
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(lr_history)+1), lr_history, label='学习率')
plt.xlabel('训练轮次')
plt.ylabel('学习率')
plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('visualizations/learning_rate_gnn.png', dpi=150)
plt.close()

# -------------------- 7. 预测并保存结果 & 可视化 --------------------
print("\n开始生成预测结果并保存...")

error_list = []
all_errors = []

model.eval()
with torch.no_grad():
    for step_idx in range(n_steps):
        s = step_idx * nodes_per_step
        e = s + nodes_per_step
        x_step = torch.tensor(node_features_scaled[s:e], dtype=torch.float32).to(device)
        pred_scaled = model(x_step, edge_index, deg_inv_sqrt).cpu().numpy()
        pred_original = scaler_y.inverse_transform(pred_scaled)
        true_original = targets[s:e]
        coords = node_features[s:e, :2]

        result_df = pd.DataFrame({
            'X': coords[:, 0], 'Y': coords[:, 1],
            'U_pred': pred_original[:, 0], 'V_pred': pred_original[:, 1], 'Phi_pred': pred_original[:, 2],
            'U_true': true_original[:, 0], 'V_true': true_original[:, 1], 'Phi_true': true_original[:, 2],
            'step': steps[step_idx], 'displacement': displacements[step_idx]
        })
        result_df.to_csv(f"predictions/pred_step_{steps[step_idx]}_gnn.csv", index=False, encoding='utf-8')

        error = np.abs(pred_original - true_original)
        all_errors.append(error)
        mae_U = np.mean(error[:, 0]); mae_V = np.mean(error[:, 1]); mae_P = np.mean(error[:, 2])
        rmse_U = np.sqrt(np.mean(error[:, 0]**2)); rmse_V = np.sqrt(np.mean(error[:, 1]**2)); rmse_P = np.sqrt(np.mean(error[:, 2]**2))

        error_list.append([steps[step_idx], displacements[step_idx], mae_U, mae_V, mae_P, rmse_U, rmse_V, rmse_P])

        if steps[step_idx] in list(range(0, 295, 20)):
            variables = ['U', 'V', 'Phi']
            units = ['位移(mm)', '位移(mm)', '']
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            grid_x, grid_y = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

            for i_var, (var, unit) in enumerate(zip(variables, units)):
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                combined_min = min(true_original[:, i_var].min(), pred_original[:, i_var].min())
                combined_max = max(true_original[:, i_var].max(), pred_original[:, i_var].max())

                grid_true = griddata(coords, true_original[:, i_var], (grid_x, grid_y), method='linear')
                im1 = axes[0].imshow(grid_true.T, extent=[x_min, x_max, y_min, y_max],
                                     cmap='RdBu_r', origin='lower', aspect='equal',
                                     vmin=combined_min, vmax=combined_max)
                axes[0].set_title(f'{var} 真实值\n范围: [{true_original[:, i_var].min():.3e}, {true_original[:, i_var].max():.3e}] {unit}')
                axes[0].set_xlabel('X 坐标'); axes[0].set_ylabel('Y 坐标')
                plt.colorbar(im1, ax=axes[0], format='%.2e', fraction=0.046, pad=0.04)

                grid_pred = griddata(coords, pred_original[:, i_var], (grid_x, grid_y), method='linear')
                im2 = axes[1].imshow(grid_pred.T, extent=[x_min, x_max, y_min, y_max],
                                     cmap='RdBu_r', origin='lower', aspect='equal',
                                     vmin=combined_min, vmax=combined_max)
                axes[1].set_title(f'{var} 预测值\n范围: [{pred_original[:, i_var].min():.3e}, {pred_original[:, i_var].max():.3e}] {unit}')
                axes[1].set_xlabel('X 坐标'); axes[1].set_ylabel('Y 坐标')
                plt.colorbar(im2, ax=axes[1], format='%.2e', fraction=0.046, pad=0.04)

                grid_error = griddata(coords, error[:, i_var], (grid_x, grid_y), method='linear')
                im3 = axes[2].imshow(grid_error.T, extent=[x_min, x_max, y_min, y_max],
                                     cmap='Oranges', origin='lower', aspect='equal')
                mae = np.mean(error[:, i_var]); rmse = np.sqrt(np.mean(error[:, i_var]**2))
                axes[2].set_title(f'{var} 绝对误差\nMAE: {mae:.3e}, RMSE: {rmse:.3e} {unit}')
                axes[2].set_xlabel('X 坐标'); axes[2].set_ylabel('Y 坐标')
                plt.colorbar(im3, ax=axes[2], format='%.2e', fraction=0.046, pad=0.04)

                plt.tight_layout()
                plt.savefig(f'visualizations/step_{steps[step_idx]}_{var}_gnn_analysis.png', dpi=150, bbox_inches='tight')
                plt.close()

# -------------------- 8. 整体误差统计与可视化 --------------------
error_df = pd.DataFrame(error_list, columns=[
    'step', 'displacement', 'U_MAE', 'V_MAE', 'Phi_MAE',
    'U_RMSE', 'V_RMSE', 'Phi_RMSE'
])

all_errors_np = np.vstack(all_errors)

overall_stats = pd.DataFrame({
    'Variable': ['U', 'V', 'Phi'],
    'MAE': [error_df['U_MAE'].mean(), error_df['V_MAE'].mean(), error_df['Phi_MAE'].mean()],
    'RMSE': [error_df['U_RMSE'].mean(), error_df['V_RMSE'].mean(), error_df['Phi_RMSE'].mean()],
    'Max Error': [np.max(all_errors_np[:, 0]), np.max(all_errors_np[:, 1]), np.max(all_errors_np[:, 2])],
    'Relative MAE (%)': 100 * np.array([error_df['U_MAE'].mean(), error_df['V_MAE'].mean(), error_df['Phi_MAE'].mean()]) / np.ptp(targets, axis=0)
})

plt.figure(figsize=(14, 8))
variables = ['U', 'V', 'Phi']
for i, var in enumerate(variables):
    plt.subplot(2, 1, 1)
    plt.plot(error_df['displacement'], error_df[f'{var}_MAE'], label=var)
    plt.ylabel('平均绝对误差')
    plt.grid(True); plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(error_df['displacement'], error_df[f'{var}_RMSE'], label=var)
    plt.xlabel('位移'); plt.ylabel('均方根误差')
    plt.grid(True); plt.legend()

plt.tight_layout()
plt.savefig('visualizations/error_vs_displacement_gnn.png', dpi=150)
plt.close()

plt.figure(figsize=(15, 5))
for i, var in enumerate(variables):
    plt.subplot(1, 3, i+1)
    plt.hist(np.log10(all_errors_np[:, i] + 1e-12), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('log10(误差值)'); plt.ylabel('频数'); plt.grid(True)
plt.tight_layout()
plt.savefig('visualizations/error_distribution_gnn.png', dpi=150)
plt.close()

plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(3)
plt.bar(index, overall_stats['MAE'], bar_width, label='MAE', alpha=0.8)
plt.bar(index + bar_width, overall_stats['RMSE'], bar_width, label='RMSE', alpha=0.8)
plt.xticks(index + bar_width/2, variables)
plt.yscale('log')
plt.legend(); plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('visualizations/overall_error_comparison_gnn.png', dpi=150)
plt.close()

# -------------------- 9. 输出结果 --------------------
print("\n预测结果已保存到 predictions/ 文件夹中（每步 CSV，文件名含 _gnn）")
print("可视化图表已保存到 visualizations/ 文件夹中（文件名含 _gnn）")
print("\n整体预测误差统计 (GNN):")
print(overall_stats.to_string(index=False))
print("\n全部任务完成！")