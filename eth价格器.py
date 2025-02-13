import tkinter as tk
from tkinter import ttk, messagebox
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Binance API URL for ETH price and volume
API_URL_PRICE = "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT"
API_URL_VOLUME = "https://api.binance.com/api/v3/ticker/24hr?symbol=ETHUSDT"

# 初始化价格、交易量和时间列表
prices = []
volumes = []
timestamps = []
# 线程锁，用于保护共享资源
price_lock = threading.Lock()

# 模型文件路径
MODEL_FILE = 'eth_price_rnn_model.pth'

# 数据文件路径
DATA_FILE = 'eth_price_volume_data.csv'

# 检测设备并设置device，目前先设为cpu，因为Windows核显支持有限
device = torch.device("cpu")

# 警报阈值
ALERT_UPPER_THRESHOLD = None
ALERT_LOWER_THRESHOLD = None

# 定义更复杂的LSTM模型
class EthPriceComplexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(EthPriceComplexLSTM, self).__init__()
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
    try:
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn')

        input_size = 2  # 价格和交易量两个特征
        hidden_size = 256  # 增加隐藏单元数量
        num_layers = 4  # 增加层数
        output_size = 1
        model = EthPriceComplexLSTM(input_size, hidden_size, num_layers, output_size)
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
            X, y, scaler = prepare_data(mock_data)
            if X is not None and y is not None:
                train_model(model, X, y, num_epochs=300)  # 增加训练轮数
                logging.info("新模型训练完成并保存")
    except Exception as e:
        logging.error(f"加载或创建模型时出现严重错误: {e}")
        return None
    return model


# 获取ETH价格和交易量
def get_eth_price_and_volume():
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
        logging.error(f"请求API时出错: {e}")
    except KeyError as e:
        logging.error(f"数据格式错误，未找到键{e}，原始数据: {data_price if 'data_price' in locals() else data_volume}")
    except Exception as e:
        logging.error(f"发生未知错误: {e}")
    return None, None


# 准备训练数据
def prepare_data(data):
    try:
        if len(data) < 2:
            return None, None
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        look_back = 60
        X = []
        y = []
        for i in range(len(scaled_data) - look_back):
            X.append(scaled_data[i:i + look_back])
            y.append(scaled_data[i + look_back, 0])  # 预测价格
        X = np.array(X)
        y = np.array(y)
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
        return X, y, scaler
    except Exception as e:
        logging.error(f"准备数据时出错: {e}")
        return None, None


# 训练RNN模型
def train_model(model, X, y, num_epochs=100):
    try:
        best_loss = float('inf')
        patience = 20
        counter = 0
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # 减小学习率
        for epoch in range(num_epochs):
            model.train()
            if X.dim() != 3:
                logging.error(f"输入X的维度错误，期望3D，实际{X.dim()}D，形状为{X.shape}")
                continue
            outputs = model(X)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                counter = 0
                torch.save(model.state_dict(), MODEL_FILE)
            else:
                counter += 1

            if counter >= patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    except Exception as e:
        logging.error(f"训练模型时出错: {e}")


# RNN预测
def predict_trend_rnn(data, scaler, model):
    try:
        if len(data) < 2:
            return "数据不足，无法预测"
        look_back = 60
        last_sequence = data[-look_back:]
        last_sequence_scaled = scaler.transform(last_sequence)
        last_sequence_tensor = torch.tensor(last_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        if last_sequence_tensor.dim() != 3:
            logging.error(f"预测输入的维度错误，期望3D，实际{last_sequence_tensor.dim()}D，形状为{last_sequence_tensor.shape}")
            return "输入维度错误，无法预测"
        model.eval()
        with torch.no_grad():
            predicted_price_scaled = model(last_sequence_tensor).item()
        prediction_input = np.zeros((1, 2))
        prediction_input[0, 0] = predicted_price_scaled
        prediction = scaler.inverse_transform(prediction_input)[0, 0]
        last_price = data[-1, 0]
        if prediction > last_price:
            return "预计上涨"
        elif prediction < last_price:
            return "预计下跌"
        else:
            return "趋势不明"
    except Exception as e:
        logging.error(f"预测趋势时出错: {e}")
        return "预测出错"


# 预测大概量
def predict_amount(data, scaler, model):
    try:
        if len(data) < 2:
            return "数据不足，无法预测"
        look_back = 60
        last_sequence = data[-look_back:]
        last_sequence_scaled = scaler.transform(last_sequence)
        last_sequence_tensor = torch.tensor(last_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        if last_sequence_tensor.dim() != 3:
            logging.error(f"预测输入的维度错误，期望3D，实际{last_sequence_tensor.dim()}D，形状为{last_sequence_tensor.shape}")
            return "输入维度错误，无法预测"
        model.eval()
        with torch.no_grad():
            predicted_price_scaled = model(last_sequence_tensor).item()
        prediction_input = np.zeros((1, 2))
        prediction_input[0, 0] = predicted_price_scaled
        prediction = scaler.inverse_transform(prediction_input)[0, 0]
        last_price = data[-1, 0]
        # 简单估算大概量，这里假设根据历史价格波动的标准差来估算
        price_std = np.std(data[:, 0]) if len(data) > 1 else 0
        if prediction > last_price:
            amount = price_std * 0.5  # 假设上涨时大概量为标准差的0.5倍
        elif prediction < last_price:
            amount = -price_std * 0.5  # 假设下跌时大概量为标准差的 -0.5倍
        else:
            amount = 0
        return amount
    except Exception as e:
        logging.error(f"预测大概量时出错: {e}")
        return "预测出错"


# 保存数据到文件
def save_data_to_file(price, volume):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(DATA_FILE, 'a') as f:
        f.write(f"{timestamp},{price},{volume}\n")


# 检查警报阈值
def check_alert_threshold(price):
    global ALERT_UPPER_THRESHOLD, ALERT_LOWER_THRESHOLD
    if ALERT_UPPER_THRESHOLD is not None and price >= ALERT_UPPER_THRESHOLD:
        messagebox.showwarning("价格警报", f"ETH价格已超过上限阈值 {ALERT_UPPER_THRESHOLD} 美元！")
    if ALERT_LOWER_THRESHOLD is not None and price <= ALERT_LOWER_THRESHOLD:
        messagebox.showwarning("价格警报", f"ETH价格已低于下限阈值 {ALERT_LOWER_THRESHOLD} 美元！")


# 耗时操作线程函数
def update_data_thread(model, price_label, ax, ax2, canvas, prediction_label):
    global prices, volumes, timestamps
    price, volume = get_eth_price_and_volume()
    if price is not None and volume is not None:
        with price_lock:
            prices.append(price)
            volumes.append(volume)
            timestamps.append(time.time())
        price_label.config(text=f"ETH价格: ${price:.2f}")

        # 保存数据到文件
        save_data_to_file(price, volume)

        # 检查警报阈值
        check_alert_threshold(price)

        data = np.column_stack((prices, volumes))
        if len(data) >= 2:
            result = prepare_data(data)
            if result is not None:
                X, y, scaler = result
                # 启动训练线程
                train_thread = threading.Thread(target=train_model, args=(model, X, y, 10))
                train_thread.start()

                def update_prediction_and_chart():
                    trend = predict_trend_rnn(data, scaler, model)
                    amount = predict_amount(data, scaler, model)
                    show_prediction_in_window(trend, amount, prediction_label)
                    update_chart(ax, ax2, canvas)

                price_label.after(0, update_prediction_and_chart)


# 更新价格、走势图和预测结果
def update_price_and_chart(model, price_label, ax, ax2, canvas, prediction_label):
    update_thread = threading.Thread(target=update_data_thread, args=(model, price_label, ax, ax2, canvas, prediction_label))
    update_thread.start()
    # 每1秒更新一次
    price_label.after(1000, update_price_and_chart, model, price_label, ax, ax2, canvas, prediction_label)


# 更新走势图
def update_chart(ax, ax2, canvas):
    with price_lock:
        ax.clear()
        ax.plot(timestamps, prices, marker='o', label='价格')
        ax.set_xlabel('时间')
        ax.set_ylabel('ETH价格 (美元)')
        ax.set_title('ETH价格走势')
        ax2.clear()
        ax2.bar(timestamps, volumes, alpha=0.3, color='orange', label='交易量')
        ax2.set_ylabel('交易量')

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        canvas.draw()


# 开始刷新
def start_refresh(model, price_label, ax, ax2, canvas, prediction_label, refresh_id):
    global refresh_thread
    if not refresh_thread or not refresh_thread.is_alive():
        refresh_thread = threading.Thread(target=update_price_and_chart, args=(model, price_label, ax, ax2, canvas, prediction_label))
        refresh_thread.start()


# 停止刷新
def stop_refresh(refresh_id):
    global refresh_thread
    if refresh_thread and refresh_thread.is_alive():
        try:
            # 取消定时任务
            root.after_cancel(refresh_id)
        except NameError:
            pass
        refresh_thread = None


# 在窗口显示预测结果和时间
def show_prediction_in_window(trend, amount, prediction_label):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(amount, str):
        amount_text = f"预测大概量: {amount}"
    else:
        amount_text = f"预测大概量: {'上涨' if amount > 0 else '下跌' if amount < 0 else '无变化'} {abs(amount):.2f}美元"
    prediction_text = f"时间: {current_time}\n预测趋势: {trend}"
    prediction_label.config(text=f"{prediction_text}\n{amount_text}")

    # 根据预测趋势改变文本颜色
    if trend == "预计上涨":
        prediction_label.config(fg='green')
    elif trend == "预计下跌":
        prediction_label.config(fg='red')
    else:
        prediction_label.config(fg='black')


# 设置警报阈值
def set_alert_thresholds(upper_threshold_entry, lower_threshold_entry):
    global ALERT_UPPER_THRESHOLD, ALERT_LOWER_THRESHOLD
    try:
        upper = float(upper_threshold_entry.get())
        lower = float(lower_threshold_entry.get())
        if upper > lower:
            ALERT_UPPER_THRESHOLD = upper
            ALERT_LOWER_THRESHOLD = lower
            messagebox.showinfo("设置成功", "警报阈值设置成功！")
        else:
            messagebox.showerror("设置失败", "上限阈值必须大于下限阈值！")
    except ValueError:
        messagebox.showerror("设置失败", "请输入有效的数字！")


# 加载历史数据
def load_historical_data():
    global prices, volumes, timestamps
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            lines = f.readlines()
            for line in lines:
                timestamp, price, volume = line.strip().split(',')
                timestamps.append(datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").timestamp())
                prices.append(float(price))
                volumes.append(float(volume))
        messagebox.showinfo("加载成功", "历史数据加载成功！")
    else:
        messagebox.showwarning("加载失败", "未找到历史数据文件！")


# 创建主窗口
root = tk.Tk()
root.title("ETH实时价格查看器")

# 创建价格标签
price_label = tk.Label(root, text="正在获取价格...", font=("Arial", 24))