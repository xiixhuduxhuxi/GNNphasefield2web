# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata
import torch.nn as nn
from matplotlib import ticker

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建保存结果的文件夹
os.makedirs('predictions', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# 1. 数据加载与预处理
displacement_df = pd.read_csv("disp\加载位移.csv", header=0, encoding='utf-8')
displacements = displacement_df["加载位移"].values[0:294]
steps = list(range(2, 585, 2))
n_steps = len(steps)

# 获取单个step的节点数
sample_df = pd.read_csv(f"data/node_data_step_{steps[0]}.csv", encoding='utf-8')
nodes_per_step = len(sample_df)
n_total_nodes = nodes_per_step * n_steps

# 预分配数组
node_features = np.empty((n_total_nodes, 3), dtype=np.float32)  # X,Y,displacement
targets = np.empty((n_total_nodes, 3), dtype=np.float32)  # U,V,Phi

for i, (step, disp) in enumerate(zip(steps, displacements)):
    df = pd.read_csv(f"data/node_data_step_{step}.csv", encoding='utf-8')
    start = i * nodes_per_step
    end = start + nodes_per_step
    node_features[start:end, :2] = df[['X', 'Y']].values
    node_features[start:end, 2] = disp
    targets[start:end] = df[['U', 'V', 'Phi']].values
    del df

# 标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
node_features_scaled = scaler_X.fit_transform(node_features)
targets_scaled = scaler_y.fit_transform(targets)

# 自动检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 2. 数据集构建
class CNNDataSet(Dataset):
    def __init__(self, indices, features, targets, nodes_per_step, device):
        self.indices = indices
        self.features = features
        self.targets = targets
        self.nodes_per_step = nodes_per_step
        self.device = device
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        step_idx = self.indices[idx]
        start = step_idx * self.nodes_per_step
        end = start + self.nodes_per_step
        x = torch.tensor(self.features[start:end], dtype=torch.float, device=self.device)
        y = torch.tensor(self.targets[start:end], dtype=torch.float, device=self.device)
        return x, y

# 数据集划分
all_step_indices = list(range(n_steps))
train_indices, test_indices = train_test_split(all_step_indices, test_size=0.2, random_state=42)

# 创建数据集
train_dataset = CNNDataSet(train_indices, node_features_scaled, targets_scaled, nodes_per_step, device)
test_dataset = CNNDataSet(test_indices, node_features_scaled, targets_scaled, nodes_per_step, device)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

class MLPModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, output_dim)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        return self.fc5(x)

# 4. 模型训练
model = MLPModel().to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=1e-5)
criterion = nn.MSELoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

train_loss_history = []
test_loss_history = []
lr_history = []

if torch.cuda.is_available():
    print(f"初始GPU显存: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

for epoch in range(1000):
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x_flat = batch_x.view(-1, batch_x.shape[-1])
        batch_y_flat = batch_y.view(-1, batch_y.shape[-1])
        
        optimizer.zero_grad()
        out = model(batch_x_flat)
        loss = criterion(out, batch_y_flat)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:  
            batch_x_flat = batch_x.view(-1, batch_x.shape[-1])
            batch_y_flat = batch_y.view(-1, batch_y.shape[-1])
            
            pred = model(batch_x_flat)
            test_loss += criterion(pred, batch_y_flat).item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_test_loss = test_loss / len(test_loader)
    scheduler.step(avg_test_loss)
    
    train_loss_history.append(avg_train_loss)
    test_loss_history.append(avg_test_loss)
    lr_history.append(optimizer.param_groups[0]['lr'])
    
    if (epoch + 1) % 20 == 0:
        lr = optimizer.param_groups[0]['lr']
        memory_info = f" | GPU 显存: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB" if torch.cuda.is_available() else ""
        print(f"Epoch {epoch+1:03d} | 训练损失: {avg_train_loss:.6f} | 测试损失: {avg_test_loss:.6f} | 学习率: {lr:.6f}{memory_info}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"训练完成，最终 GPU 显存: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

# 5. 绘制训练曲线
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(train_loss_history)+1), train_loss_history, label='训练损失', color='blue', alpha=0.7)
plt.plot(range(1, len(test_loss_history)+1), test_loss_history, label='测试损失', color='red', alpha=0.7)
plt.xlabel('训练轮次', fontsize=12)
plt.ylabel('MSE 损失', fontsize=12)
plt.yscale('log')

# 关键修复：设置对数坐标轴的刻度格式，确保负号使用正确字体
formatter = ticker.ScalarFormatter(useMathText=False)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))  # 当指数超出-1到1范围时使用科学计数法
plt.gca().yaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_minor_formatter(formatter)

# 确保刻度标签使用中文字体
plt.gca().tick_params(axis='both', labelsize=10)
for label in plt.gca().get_yticklabels():
    label.set_fontname(['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans'])

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/loss_curve_mlp.png', dpi=150)
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(lr_history)+1), lr_history, label='学习率', color='green')
plt.xlabel('训练轮次', fontsize=12)
plt.ylabel('学习率', fontsize=12)
plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/learning_rate_mlp.png', dpi=150)
plt.close()

# 6. 预测与可视化
print("\n开始生成预测结果...")
model.eval()
error_list = []
all_errors = []

with torch.no_grad():
    for step_idx in all_step_indices:
        step = steps[step_idx]
        disp = displacements[step_idx]
        start = step_idx * nodes_per_step
        end = start + nodes_per_step
        
        x_scaled = torch.tensor(node_features_scaled[start:end], dtype=torch.float, device=device)
        pred_scaled = model(x_scaled)
        
        pred_original = scaler_y.inverse_transform(pred_scaled.cpu().numpy())
        true_original = targets[start:end]
        coords = node_features[start:end, :2]
        
        result_df = pd.DataFrame({
            'X': coords[:, 0], 'Y': coords[:, 1],
            'U_pred': pred_original[:, 0], 'V_pred': pred_original[:, 1], 'Phi_pred': pred_original[:, 2],
            'U_true': true_original[:, 0], 'V_true': true_original[:, 1], 'Phi_true': true_original[:, 2],
            'step': step, 'displacement': disp
        })
        
        error = np.abs(pred_original - true_original)
        all_errors.append(error)
        error_list.append([
            step, disp,
            np.mean(error[:, 0]), np.mean(error[:, 1]), np.mean(error[:, 2]),
            np.sqrt(np.mean(error[:, 0]**2)), np.sqrt(np.mean(error[:, 1]** 2)), np.sqrt(np.mean(error[:, 2]**2))
        ])
        
        if step in list(range(0, 584, 50)):
            variables = ['U', 'V', 'Phi']
            units = ['位移(mm)', '位移(mm)', '']
            
            for i, (var, unit) in enumerate(zip(variables, units)):
                x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
                y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
                grid_x, grid_y = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                
                combined_min = min(true_original[:, i].min(), pred_original[:, i].min())
                combined_max = max(true_original[:, i].max(), pred_original[:, i].max())
                
                grid_true = griddata(coords, true_original[:, i], (grid_x, grid_y), method='linear')
                im1 = axes[0].imshow(grid_true.T, extent=[x_min, x_max, y_min, y_max], 
                                    cmap='RdBu_r', origin='lower', aspect='equal',
                                    vmin=combined_min, vmax=combined_max)
                axes[0].set_title(f'{var} 真实值\n范围: [{true_original[:, i].min():.3e}, {true_original[:, i].max():.3e}] {unit}', 
                                 fontsize=10)
                axes[0].set_xlabel('X 坐标', fontsize=9)
                axes[0].set_ylabel('Y 坐标', fontsize=9)
                plt.colorbar(im1, ax=axes[0], format='%.2e', fraction=0.046, pad=0.04)
                
                grid_pred = griddata(coords, pred_original[:, i], (grid_x, grid_y), method='linear')
                im2 = axes[1].imshow(grid_pred.T, extent=[x_min, x_max, y_min, y_max], 
                                    cmap='RdBu_r', origin='lower', aspect='equal',
                                    vmin=combined_min, vmax=combined_max)
                axes[1].set_title(f'{var} 预测值\n范围: [{pred_original[:, i].min():.3e}, {pred_original[:, i].max():.3e}] {unit}', 
                                 fontsize=10)
                axes[1].set_xlabel('X 坐标', fontsize=9)
                axes[1].set_ylabel('Y 坐标', fontsize=9)
                plt.colorbar(im2, ax=axes[1], format='%.2e', fraction=0.046, pad=0.04)
                
                grid_error = griddata(coords, error[:, i], (grid_x, grid_y), method='linear')
                im3 = axes[2].imshow(grid_error.T, extent=[x_min, x_max, y_min, y_max], 
                                    cmap='Oranges', origin='lower', aspect='equal')
                mae = np.mean(error[:, i])
                rmse = np.sqrt(np.mean(error[:, i]**2))
                axes[2].set_title(f'{var} 绝对误差\nMAE: {mae:.3e}, RMSE: {rmse:.3e} {unit}', 
                                 fontsize=10)
                axes[2].set_xlabel('X 坐标', fontsize=9)
                axes[2].set_ylabel('Y 坐标', fontsize=9)
                plt.colorbar(im3, ax=axes[2], format='%.2e', fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                plt.savefig(f'visualizations/step_{step}_{var}_mlp_analysis.png', dpi=150, bbox_inches='tight')
                plt.close()

error_df = pd.DataFrame(error_list, columns=[
    'step', 'displacement', 'U_MAE', 'V_MAE', 'Phi_MAE',
    'U_RMSE', 'V_RMSE', 'Phi_RMSE'
])
overall_stats = pd.DataFrame({
    'Variable': ['U', 'V', 'Phi'],
    'MAE': [error_df['U_MAE'].mean(), error_df['V_MAE'].mean(), error_df['Phi_MAE'].mean()],
    'RMSE': [error_df['U_RMSE'].mean(), error_df['V_RMSE'].mean(), error_df['Phi_RMSE'].mean()],
    'Max Error': [np.max(np.abs(pred_original[:, i] - true_original[:, i])) for i in range(3)],
    'Relative MAE (%)': 100 * np.array([error_df['U_MAE'].mean(), error_df['V_MAE'].mean(), error_df['Phi_MAE'].mean()]) / np.ptp(targets, axis=0)
})

plt.figure(figsize=(14, 8))
variables = ['U', 'V', 'Phi']
colors = ['blue', 'green', 'red']

for i, var in enumerate(variables):
    plt.subplot(2, 1, 1)
    plt.plot(error_df['displacement'], error_df[f'{var}_MAE'], label=f'{var}', color=colors[i], alpha=0.7, linewidth=1.5)
    plt.ylabel('平均绝对误差', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    plt.subplot(2, 1, 2)
    plt.plot(error_df['displacement'], error_df[f'{var}_RMSE'], label=f'{var}', color=colors[i], alpha=0.7, linewidth=1.5)
    plt.xlabel('位移', fontsize=12)
    plt.ylabel('均方根误差', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/error_vs_displacement_mlp.png', dpi=150)
plt.close()

all_errors_np = np.vstack(all_errors)
plt.figure(figsize=(15, 5))
for i, (var, color) in enumerate(zip(variables, colors)):
    plt.subplot(1, 3, i+1)
    plt.hist(np.log10(all_errors_np[:, i] + 1e-12), bins=50, color=color, alpha=0.7, edgecolor='black')
    plt.xlabel('log10(误差值)', fontsize=10)
    plt.ylabel('频数', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('visualizations/error_distribution_mlp.png', dpi=150)
plt.close()

plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(3)

plt.bar(index, overall_stats['MAE'], bar_width, label='MAE', color='skyblue', alpha=0.8, edgecolor='black')
plt.bar(index + bar_width, overall_stats['RMSE'], bar_width, label='RMSE', color='orange', alpha=0.8, edgecolor='black')

plt.xlabel('变量', fontsize=12)
plt.ylabel('误差值', fontsize=12)
plt.xticks(index + bar_width / 2, variables, fontsize=12)
plt.yscale('log')
plt.legend(fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

def add_labels(positions, heights, labels, offset=0):
    for pos, height, label in zip(positions, heights, labels):
        plt.text(pos + offset, height, f'{label:.2e}', ha='center', va='bottom', fontsize=10)

add_labels(index, overall_stats['MAE'], overall_stats['MAE'], -bar_width/2)
add_labels(index, overall_stats['RMSE'], overall_stats['RMSE'], bar_width/2)

plt.tight_layout()
plt.savefig('visualizations/overall_error_comparison_mlp.png', dpi=150)
plt.close()

print(f"\n预测结果已保存到 predictions/ 文件夹中")
print(f"可视化完成！图表已保存到 visualizations/ 文件夹中，包括：")
print(f"- 训练与测试损失曲线")
print(f"- 学习率调整曲线")
print(f"- 各关键步骤的U、V、Phi独立云图分析（真实值、预测值、误差）")
print(f"- 误差随位移变化趋势图")
print(f"- 误差分布直方图")
print(f"- 整体误差统计柱状图")
print("\n整体预测误差统计 (MLP):")
print(overall_stats.to_string(index=False))
print("\n所有任务完成！")