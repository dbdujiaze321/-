import argparse
import tkinter as tk
from tkinter import ttk
import requests
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import os
import logging
from sklearn.preprocessing import MinMaxScaler


# 配置日志记录
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Binance API URL for SOL price and volume
API_URL_PRICE = "https://api.binance.com/api/v3/ticker/price?symbol=SOLUSDT"
API_URL_VOLUME = "https://api.binance.com/api/v3/ticker/24hr?symbol=SOLUSDT"

# 初始化价格、交易量和时间列表
prices = []
volumes = []
timestamps = []
# 线程锁，用于保护共享资源
price_lock = threading.Lock()

# 模型文件路径
MODEL_FILE ='sol_price_rnn_model.pth'

# 检测设备并设置 device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# 定义更复杂的 LSTM 模型，进一步调整超参数
class SolPriceComplexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(SolPriceComplexLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 尝试加载已保存的模型或创建新模型
def load_or_create_model():
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    input_size = 2  # 价格和交易量两个特征
    hidden_size = 512  # 进一步增加隐藏单元数量
    num_layers = 6  # 进一步增加层数
    output_size = 1
    model = SolPriceComplexLSTM(input_size, hidden_size, num_layers, output_size)
    model.to(device)

    if os.path.exists(MODEL_FILE):
        try:
            model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
            logging.info("成功加载现有模型")
        except Exception as e:
            logging.error(f"加载模型时出错: {e}")
    else:
        logging.info("未找到模型文件，创建新模型")
        # 添加一些初始训练逻辑
        mock_prices = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
        mock_volumes = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
        mock_data = np.column_stack((mock_prices, mock_volumes))
        X, y = prepare_data(mock_data)
        if X is not None and y is not None:
            train_model(model, X, y, num_epochs=300)  # 增加训练轮数
            logging.info("新模型训练完成并保存")

    return model


# 获取 SOL 价格和交易量
def get_sol_price_and_volume():
    try:
        response_price = requests.get(API_URL_PRICE)
        response_price.raise_for_status()
        data_price = response_price.json()
        price = float(data_price['price'])

        response_volume = requests.get(API_URL_VOLUME)
        response_volume.raise_for_status()
        data_volume = response_volume.json()
        volume = float(data_volume['volume'])

        return price, volume
    except requests.RequestException as e:
        logging.error(f"请求 API 时出错: {e}")
    except KeyError as e:
        logging.error(f"数据格式错误，未找到键 {e}，原始数据: {data_price if 'data_price' in locals() else data_volume}")
    except Exception as e:
        logging.error(f"发生未知错误: {e}")
    return None, None


# 准备训练数据，增加批量处理
def prepare_data(data, batch_size=32):
    if len(data) < 2:
        return None, None
    scaler = MinMaxScaler()
    try:
        scaled_data = scaler.fit_transform(data)
    except ValueError as e:
        logging.error(f"数据缩放时出错: {e}")
        return None, None
    look_back = 60
    X = []
    y = []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:i + look_back])
        y.append(scaled_data[i + look_back, 0])  # 预测价格
    X = np.array(X)
    y = np.array(y)
    # 按批次划分数据
    num_batches = len(X) // batch_size
    X = X[:num_batches * batch_size].reshape(num_batches, batch_size, look_back, 2)
    y = y[:num_batches * batch_size].reshape(num_batches, batch_size, 1)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    return X, y, scaler


# 训练 RNN 模型，使用批次数据
def train_model(model, X, y, num_epochs=100):
    best_loss = float('inf')
    patience = 20
    counter = 0
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # 减小学习率
    for epoch in range(num_epochs):
        model.train()
        if X.dim()!= 4:
            logging.error(f"输入 X 的维度错误，期望 4D，实际 {X.dim()}D，形状为 {X.shape}")
            continue
        epoch_loss = 0
        for batch in range(X.size(0)):
            try:
                outputs = model(X[batch])
                loss = criterion(outputs, y[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            except Exception as e:
                logging.error(f"训练模型时出错: {e}")
                continue

            if loss.item() < best_loss:
                best_loss = loss.item()
                counter = 0
                try:
                    torch.save(model.state_dict(), MODEL_FILE)
                except Exception as e:
                    logging.error(f"保存模型时出错: {e}")
            else:
                counter += 1

            if counter >= patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break

        if X.size(0) > 0:
            avg_loss = epoch_loss / X.size(0)
            if (epoch + 1) % 10 == 0:
                logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')


# RNN 预测
def predict_trend_rnn(data, scaler):
    if len(data) < 2:
        return "数据不足，无法预测"
    look_back = 60
    last_sequence = data[-look_back:]
    try:
        last_sequence_scaled = scaler.transform(last_sequence)
    except ValueError as e:
        logging.error(f"数据缩放时出错: {e}")
        return "数据缩放错误，无法预测"
    last_sequence_tensor = torch.tensor(last_sequence_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    if last_sequence_tensor.dim()!= 4:
        logging.error(f"预测输入的维度错误，期望 4D，实际 {last_sequence_tensor.dim()}D，形状为 {last_sequence_tensor.shape}")
        return "输入维度错误，无法预测"
    model.eval()
    with torch.no_grad():
        try:
            predicted_price_scaled = model(last_sequence_tensor).item()
        except Exception as e:
            logging.error(f"预测时出错: {e}")
            return "预测错误，无法得出结果"
    prediction = scaler.inverse_transform([[predicted_price_scaled, 0]])[0, 0]
    last_price = data[-1, 0]
    return "预计上涨" if prediction > last_price else "预计下跌" if prediction < last_price else "趋势不明"


# 预测大概量
def predict_amount(data, scaler):
    if len(data) < 2:
        return "数据不足，无法预测"
    look_back = 60
    last_sequence = data[-look_back:]
    try:
        last_sequence_scaled = scaler.transform(last_sequence)
    except ValueError as e:
        logging.error(f"数据缩放时出错: {e}")
        return "数据缩放错误，无法预测"
    last_sequence_tensor = torch.tensor(last_sequence_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    if last_sequence_tensor.dim()!= 4:
        logging.error(f"预测输入的维度错误，期望 4D，实际 {last_sequence_tensor.dim()}D，形状为 {last_sequence_tensor.shape}")
        return "输入维度错误，无法预测"
    model.eval()
    with torch.no_grad():
        try:
            predicted_price_scaled = model(last_sequence_tensor).item()
        except Exception as e:
            logging.error(f"预测时出错: {e}")
            return "预测错误，无法得出结果"
    prediction = scaler.inverse_transform([[predicted_price_scaled, 0]])[0, 0]
    last_price = data[-1, 0]
    price_std = np.std(data[:, 0]) if len(data) > 1 else 0
    amount = price_std * 0.5 if prediction > last_price else -price_std * 0.5 if prediction < last_price else 0
    return amount


# 更新价格、走势图和预测结果
def update_price_and_chart():
    global prices, volumes, timestamps
    price, volume = get_sol_price_and_volume()
    if price is not None and volume is not None:
        with price_lock:
            prices.append(price)
            volumes.append(volume)
            timestamps.append(time.time())
        price_label.config(text=f"SOL 价格: ${price:.2f}")

        data = np.column_stack((prices, volumes))
        if len(data) >= 2:
            result = prepare_data(data)
            if result is not None:
                X, y, scaler = result
                train_model(model, X, y, num_epochs=10)  # 每次更新数据后进行一定轮数的训练
                trend = predict_trend_rnn(data, scaler)
                amount = predict_amount(data, scaler)
                show_prediction_in_window(trend, amount)
                update_chart()
    # 每 1 秒更新一次
    root.after(1000, update_price_and_chart)


# 更新走势图
def update_chart():
    with price_lock:
        ax.clear()
        ax.plot(timestamps, prices, marker='o')
        ax.set_xlabel('时间')
        ax.set_ylabel('SOL 价格 (美元)')
        ax.set_title('SOL 价格走势')
        canvas.draw()


# 开始刷新
def start_refresh():
    global refresh_thread
    if not refresh_thread or not refresh_thread.is_alive():
        refresh_thread = threading.Thread(target=update_price_and_chart)
        refresh_thread.start()


# 停止刷新
def stop_refresh():
    global refresh_thread
    if refresh_thread and refresh_thread.is_alive():
        try:
            # 取消定时任务
            root.after_cancel(refresh_id)
        except NameError:
            pass
        refresh_thread = None


# 在窗口显示预测结果和时间
def show_prediction_in_window(trend, amount):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction_text = f"时间: {current_time}\n预测趋势: {trend}"
    amount_text = f"预测大概量: {'上涨' if amount > 0 else '下跌' if amount < 0 else '无变化'} {abs(amount):.2f} 美元"
    prediction_label.config(text=f"{prediction_text}\n{amount_text}")


# 创建主窗口
root = tk.Tk()
root.title("SOL 实时价格查看器")

# 创建价格标签
price_label = tk.Label(root, text="正在获取价格...", font=("Arial", 24))
price_label.pack(pady=20)

# 创建开始和停止按钮
start_button = tk.Button(root, text="开始刷新", command=start_refresh)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="停止刷新", command=stop_refresh)
stop_button.pack(pady=10)

# 创建 Matplotlib 图形和轴
fig, ax = plt.subplots(figsize=(8, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=20)

# 创建预测结果窗口
prediction_window = tk.Toplevel(root)
prediction_window.title("SOL 价格预测结果")
prediction_label = tk.Label(prediction_window, text="等待数据", font=("Arial", 18))
prediction_label.pack(pady=20)

# 全局变量来保存刷新线程
refresh_thread = None
# 用于保存定时任务的 ID
refresh_id = None

# 加载或创建模型
model = load_or_create_model()

# 训练 RNN 模型
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # 减小学习率

# 定义程序关闭时的处理函数
def on_closing():
    stop_refresh()
    try:
        if hasattr(model,'state_dict'):
            torch.save(model.state_dict(), MODEL_FILE)
    except Exception as e:
        logging.error(f"保存模型时出错: {e}")
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# 启动 GUI 事件循环
refresh_id = root.after(0, update_price_and_chart)
root.mainloop()
