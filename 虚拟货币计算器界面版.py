import requests
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox


def get_all_crypto_prices():
    url = 'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        prices = {item['symbol']: item['current_price'] for item in data}
        return prices
    except requests.RequestException as e:
        messagebox.showerror("错误", f"请求出错: {e}")
        return {}


def calculate_crypto_value():
    symbol = symbol_entry.get().lower()
    if symbol not in crypto_prices:
        messagebox.showwarning("警告", "不支持的虚拟货币符号，请重新输入。")
        return
    try:
        amount = float(amount_entry.get())
        value = amount * crypto_prices[symbol]
        total_value.set(f"你拥有的 {symbol} 价值为: {value} 美元")
    except ValueError:
        messagebox.showerror("错误", "请输入有效的数字。")


def update_prices():
    global crypto_prices
    crypto_prices = get_all_crypto_prices()
    if crypto_prices:
        messagebox.showinfo("提示", "价格更新成功")
    else:
        messagebox.showerror("错误", "价格更新失败")


# 创建主窗口
root = tk.Tk()
root.title("高级虚拟货币计算器")
root.geometry("600x400")

# 样式设置
style = ttk.Style()
style.theme_use('clam')

# 获取初始价格
crypto_prices = get_all_crypto_prices()

# 符号标签和输入框
symbol_label = ttk.Label(root, text="输入虚拟货币符号:")
symbol_label.pack(pady=10)
symbol_entry = ttk.Entry(root)
symbol_entry.pack(pady=10)

# 数量标签和输入框
amount_label = ttk.Label(root, text="输入拥有的数量:")
amount_label.pack(pady=10)
amount_entry = ttk.Entry(root)
amount_entry.pack(pady=10)

# 计算按钮
calculate_button = ttk.Button(root, text="计算价值", command=calculate_crypto_value)
calculate_button.pack(pady=15)

# 显示结果变量
total_value = tk.StringVar()
result_label = ttk.Label(root, textvariable=total_value)
result_label.pack(pady=10)

# 更新价格按钮
update_button = ttk.Button(root, text="更新价格", command=update_prices)
update_button.pack(pady=15)

root.mainloop()
