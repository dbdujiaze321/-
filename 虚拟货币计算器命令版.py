import requests


def get_all_crypto_prices():
    url = 'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        prices = {item['symbol']: item['current_price'] for item in data}
        return prices
    except requests.RequestException as e:
        print(f"请求出错: {e}")
        return {}


def calculate_crypto_value(prices):
    total_value = 0
    while True:
        symbol = input("请输入虚拟货币符号（如btc、eth，输入q结束）: ").lower()
        if symbol == 'q':
            break
        if symbol not in prices:
            print("不支持的虚拟货币符号，请重新输入。")
            continue
        try:
            amount = float(input(f"请输入你拥有的 {symbol} 数量: "))
            value = amount * prices[symbol]
            total_value += value
            print(f"你拥有的 {symbol} 价值为: {value} 美元")
        except ValueError:
            print("请输入有效的数字。")
    print(f"你拥有的所有虚拟货币总价值为: {total_value} 美元")


if __name__ == "__main__":
    all_crypto_prices = get_all_crypto_prices()
    if all_crypto_prices:
        calculate_crypto_value(all_crypto_prices)