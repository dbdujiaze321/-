import requests
import tkinter as tk
from tkinter import ttk
import threading
import time

# 币安 API 基础 URL
BINANCE_API_BASE_URL = "https://api.binance.com"

def get_recent_trades(symbol="SOLUSDT", limit=10):
    """
    获取指定交易对的最新成交信息
    :param symbol: 交易对，默认为 SOLUSDT
    :param limit: 获取的成交记录数量，默认为 10
    :return: 成交记录列表
    """
    url = f"{BINANCE_API_BASE_URL}/api/v3/trades"
    params = {
        "symbol": symbol,
        "limit": limit
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"请求发生错误: {e}")
    return []

def update_trade_info(text_widget):
    """
    定时更新交易信息并显示在文本框中
    :param text_widget: 用于显示交易信息的文本框
    """
    while True:
        trades = get_recent_trades()
        if trades:
            text_widget.delete(1.0, tk.END)  # 清空文本框
            for trade in trades:
                text = f"交易 ID: {trade['id']}, 价格: {trade['price']}, 数量: {trade['qty']}, 时间: {trade['time']}\n"
                text_widget.insert(tk.END, text)
        time.sleep(10)  # 每 10 秒更新一次

def main():
    root = tk.Tk()
    root.title("币安 SOLUSDT 交易情况实时查看")

    # 创建一个文本框用于显示交易信息
    text_widget = tk.Text(root, wrap=tk.WORD)
    text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # 创建一个线程来更新交易信息
    update_thread = threading.Thread(target=update_trade_info, args=(text_widget,))
    update_thread.daemon = True
    update_thread.start()

    root.mainloop()

if __name__ == "__main__":
    main()