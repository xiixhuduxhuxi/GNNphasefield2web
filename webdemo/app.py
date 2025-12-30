# -*- coding: utf-8 -*-
"""
GNN相场预测Streamlit应用
实现网页端实时交互预测
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import griddata
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import os

# -------------------- 模型定义 --------------------
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

# -------------------- 缓存加载函数 --------------------
@st.cache_resource
def load_model(model_path='models/gnn_model_final.pth'):
    """缓存加载模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = GNNModel().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device, checkpoint

@st.cache_data
def load_data():
    """缓存加载数据"""
    # 加载位移数据
    disp_df = pd.read_csv("disp/加载位移.csv", header=0, encoding='utf-8')
    displacements = disp_df["加载位移"].values[:293]
    steps = list(range(2, 295, 2))
    
    # 获取节点数
    sample_df = pd.read_csv(f"data/node_data_step_{steps[0]}.csv", encoding='utf-8')
    nodes_per_step = len(sample_df)
    
    # 加载所有数据
    node_features = np.empty((len(steps) * nodes_per_step, 3), dtype=np.float32)
    targets = np.empty((len(steps) * nodes_per_step, 3), dtype=np.float32)
    
    for i, (step, disp) in enumerate(zip(steps, displacements)):
        df = pd.read_csv(f"data/node_data_step_{step}.csv", encoding='utf-8')
        s = i * nodes_per_step
        e = s + nodes_per_step
        node_features[s:e, :2] = df[['X', 'Y']].values
        node_features[s:e, 2] = disp
        targets[s:e, :] = df[['U', 'V', 'Phi']].values
    
    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    node_features_scaled = scaler_X.fit_transform(node_features)
    targets_scaled = scaler_y.fit_transform(targets)
    
    # 构建图结构
    coords_base = node_features[:nodes_per_step, :2]
    knn = NearestNeighbors(n_neighbors=4).fit(coords_base)
    _, knn_idx = knn.kneighbors(coords_base)
    
    edge_list = [[i, j] for i in range(nodes_per_step) for j in knn_idx[i]]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # 计算归一化度矩阵
    row, col = edge_index[0], edge_index[1]
    deg = torch.zeros(nodes_per_step, dtype=torch.float)
    deg.index_add_(0, col, torch.ones_like(col, dtype=torch.float))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    
    return (node_features_scaled, edge_index, deg_inv_sqrt, scaler_y, 
            steps, nodes_per_step, targets)

# -------------------- 预测函数 --------------------
def predict_step(step_value, model, device, node_features_scaled, edge_index, 
                deg_inv_sqrt, scaler_y, steps, nodes_per_step):
    """预测指定步骤"""
    step_idx = steps.index(step_value)
    s = step_idx * nodes_per_step
    e = s + nodes_per_step
    
    x_step = torch.tensor(node_features_scaled[s:e], dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_step, edge_index.to(device), deg_inv_sqrt.to(device))
        pred_scaled = pred_scaled.cpu().numpy()
    
    return scaler_y.inverse_transform(pred_scaled)

# -------------------- 可视化函数 --------------------
def plot_phi_contours(step_value, pred_phi, true_phi, coords):
    """绘制相场等高线图"""
    # 创建网格
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    grid_x, grid_y = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    
    # 插值
    grid_true = griddata(coords, true_phi, (grid_x, grid_y), method='linear')
    grid_pred = griddata(coords, pred_phi, (grid_x, grid_y), method='linear')
    
    # 设置统一的色标范围
    combined_min = min(true_phi.min(), pred_phi.min())
    combined_max = max(true_phi.max(), pred_phi.max())
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 真实值
    im1 = axes[0].imshow(grid_true.T, extent=[x_min, x_max, y_min, y_max],
                         cmap='RdBu_r', origin='lower', aspect='equal',
                         vmin=combined_min, vmax=combined_max)
    axes[0].set_title(f'步骤 {step_value} - 真实相场 Φ')
    axes[0].set_xlabel('X 坐标'); axes[0].set_ylabel('Y 坐标')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 预测值
    im2 = axes[1].imshow(grid_pred.T, extent=[x_min, x_max, y_min, y_max],
                         cmap='RdBu_r', origin='lower', aspect='equal',
                         vmin=combined_min, vmax=combined_max)
    axes[1].set_title(f'步骤 {step_value} - 预测相场 Φ')
    axes[1].set_xlabel('X 坐标'); axes[1].set_ylabel('Y 坐标')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig

# -------------------- Streamlit界面 --------------------
def main():
    """主界面"""
    st.set_page_config(page_title="GNN相场预测", layout="wide")
    st.title("🧠 GNN相场预测交互平台")
    
    # 侧边栏
    st.sidebar.header("模型与数据配置")
    
    # 模型选择
    model_files = [f for f in os.listdir('models') if f.endswith('.pth')] if os.path.exists('models') else []
    if not model_files:
        st.error("未找到模型文件！请将模型文件放入 'models' 目录")
        return
    
    selected_model = st.sidebar.selectbox(
        "选择模型文件",
        model_files,
        index=0
    )
    
    # 加载模型
    try:
        model, device, checkpoint = load_model(f"models/{selected_model}")
        st.sidebar.success(f"模型加载成功！设备: {device}")
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return
    
    # 加载数据
    try:
        (node_features_scaled, edge_index, deg_inv_sqrt, scaler_y, 
         steps, nodes_per_step, targets) = load_data()
        st.sidebar.success(f"数据加载成功！共 {len(steps)} 个步骤")
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return
    
    # 显示模型信息
    with st.sidebar.expander("模型信息"):
        st.write(f"- 输入维度: 3")
        st.write(f"- 隐藏层: 64")
        st.write(f"- 输出维度: 3")
        st.write(f"- 训练轮次: {checkpoint.get('epoch', '未知')}")
        st.write(f"- 测试损失: {checkpoint.get('test_loss', '未知'):.6f}" if checkpoint.get('test_loss') else "- 测试损失: 未知")
    
    # 主区域
    st.header("📊 相场预测")
    
    # 步骤选择
    col1, col2 = st.columns([2, 1])
    
    with col1:
        step_value = st.slider(
            "选择加载步骤",
            min_value=min(steps),
            max_value=max(steps),
            value=100,
            step=2,
            help="拖动滑块选择要预测的加载步"
        )
    
    with col2:
        st.metric("当前步骤", f"{step_value}")
        st.metric("对应位移", f"{pd.read_csv('disp/加载位移.csv')['加载位移'].iloc[steps.index(step_value)]:.4f}")
    
    # 预测按钮
    if st.button("🔍 开始预测", type="primary"):
        with st.spinner('正在预测中...'):
            # 获取预测结果
            pred_original = predict_step(
                step_value, model, device, node_features_scaled, 
                edge_index, deg_inv_sqrt, scaler_y, steps, nodes_per_step
            )
            
            # 获取真实值和坐标
            df = pd.read_csv(f"data/node_data_step_{step_value}.csv", encoding='utf-8')
            coords = df[['X', 'Y']].values
            true_phi = df['Phi'].values
            pred_phi = pred_original[:, 2]
            
            # 计算误差
            error_phi = np.abs(pred_phi - true_phi)
            mae = np.mean(error_phi)
            rmse = np.sqrt(np.mean(error_phi**2))
            max_error = np.max(error_phi)
            
            # 显示误差指标
            st.subheader("误差统计")
            col_err1, col_err2, col_err3 = st.columns(3)
            with col_err1:
                st.metric("MAE", f"{mae:.2e}")
            with col_err2:
                st.metric("RMSE", f"{rmse:.2e}")
            with col_err3:
                st.metric("最大误差", f"{max_error:.2e}")
            
            # 绘制图形
            st.subheader("可视化结果")
            fig = plot_phi_contours(step_value, pred_phi, true_phi, coords)
            st.pyplot(fig)
            
            # 显示数据表格（可展开）
            with st.expander("查看原始数据"):
                result_df = pd.DataFrame({
                    'X': coords[:, 0],
                    'Y': coords[:, 1],
                    '真实Φ': true_phi,
                    '预测Φ': pred_phi,
                    '绝对误差': error_phi
                })
                st.dataframe(result_df, use_container_width=True)

if __name__ == "__main__":
    main()