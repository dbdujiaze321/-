import argparse
import psutil
import wmi
import time
import os
import atexit
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from GPUtil import getGPUs


console = Console()
log_file = None


def setup_logging():
    global log_file
    log_file = open('system_monitor.log', 'w', buffering=1)
    atexit.register(log_file.close)


def get_system_uptime():
    return time.time() - psutil.boot_time()


def get_login_users():
    return psutil.users()


def get_system_basic_info():
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_freq = psutil.cpu_freq().current
    mem = psutil.virtual_memory()
    mem_percent = mem.percent
    mem_used = mem.used / (1024.0 ** 3)
    disk = psutil.disk_usage('/')
    disk_percent = disk.percent
    disk_used = disk.used / (1024.0 ** 3)
    net = psutil.net_io_counters()
    net_sent = net.bytes_sent / (1024.0 ** 2)
    net_recv = net.bytes_recv / (1024.0 ** 2)
    uptime = get_system_uptime()
    users = get_login_users()
    return cpu_percent, cpu_freq, mem_percent, mem_used, disk_percent, disk_used, net_sent, net_recv, uptime, users


def print_system_info():
    cpu_percent, cpu_freq, mem_percent, mem_used, disk_percent, disk_used, net_sent, net_recv, uptime, users = get_system_basic_info()

    table = Table(title="系统信息")
    table.add_column("指标", justify="right", style="cyan", no_wrap=True)
    table.add_column("数值", style="magenta")
    table.add_row("CPU使用率", f"{cpu_percent}%")
    table.add_row("CPU频率", f"{cpu_freq:.2f}MHz")
    table.add_row("内存使用率", f"{mem_percent}%")
    table.add_row("内存使用量", f"{mem_used:.2f}GB")
    table.add_row("磁盘使用率", f"{disk_percent}%")
    table.add_row("磁盘使用量", f"{disk_used:.2f}GB")
    table.add_row("网络发送量", f"{net_sent:.2f}MB")
    table.add_row("网络接收量", f"{net_recv:.2f}MB")
    table.add_row("系统运行时间", time.strftime("%H:%M:%S", time.gmtime(uptime)))
    table.add_row("当前登录用户", ', '.join([user.name for user in users]))
    console.print(table)


def print_battery_info():
    if psutil.sensors_battery():
        battery = psutil.sensors_battery()
        percent = battery.percent
        plugged = battery.power_plugged
        table = Table(title="电池信息")
        table.add_column("指标", justify="right", style="yellow", no_wrap=True)
        table.add_column("数值", style="magenta")
        table.add_row("电量", f"{percent}%")
        table.add_row("充电状态", "已充电" if plugged else "未充电")
        console.print(table)


def print_gpu_info():
    try:
        gpus = getGPUs()
        for i, gpu in enumerate(gpus):
            table = Table(title=f"GPU信息 (设备 {i})")
            table.add_column("指标", justify="right", style="magenta", no_wrap=True)
            table.add_column("数值", style="cyan")
            table.add_row("GPU名称", gpu.name)
            table.add_row("GPU内存使用量", f"{gpu.memoryUsed:.2f}MB / {gpu.memoryTotal:.2f}MB")
            console.print(table)
    except ImportError:
        console.print("[red]未安装GPUtil库，无法获取GPU信息。")
    except Exception as e:
        console.print(f"[red]获取GPU信息时出错: {e}")


def get_sensor_info(namespace, sensor_type):
    try:
        c = wmi.WMI(namespace=namespace)
        sensors = c.Sensor()
        sensor_data = []
        for sensor in sensors:
            if sensor.SensorType == sensor_type:
                sensor_data.append((sensor.Name, sensor.Value))
        return sensor_data
    except wmi.x_wmi as wmi_err:
        console.print(f"[red]WMI查询出错: {wmi_err}")
    except Exception as e:
        console.print(f"[red]获取传感器信息时出错: {e}")
    return []


def print_fan_speed_info():
    sensor_data = get_sensor_info(r'root\OpenHardwareMonitor', 'Fan')
    if sensor_data:
        table = Table(title="风扇转速信息")
        table.add_column("传感器名称", style="green")
        table.add_column("转速", style="cyan")
        for name, value in sensor_data:
            table.add_row(name, f"{value} RPM")
        console.print(table)


def print_voltage_info():
    sensor_data = get_sensor_info(r'root\OpenHardwareMonitor', 'Voltage')
    if sensor_data:
        table = Table(title="主板电压信息")
        table.add_column("传感器名称", style="green")
        table.add_column("电压", style="cyan")
        for name, value in sensor_data:
            table.add_row(name, f"{value} V")
        console.print(table)


def print_temperature_info():
    sensor_data = get_sensor_info(r'root\OpenHardwareMonitor', 'Temperature')
    if sensor_data:
        table = Table(title="传感器温度")
        table.add_column("传感器名称", style="green")
        table.add_column("温度", style="cyan")
        for name, value in sensor_data:
            table.add_row(name, f"{value}°C")
        console.print(table)


def print_process_info(filter_name=None):
    table = Table(title="进程信息")
    table.add_column("名称", style="magenta")
    table.add_column("CPU使用率", style="cyan")
    table.add_column("内存使用率", style="cyan")
    table.add_column("线程数", style="cyan")
    for proc in psutil.process_iter(['name', 'cpu_percent','memory_percent', 'num_threads']):
        try:
            info = proc.info
            name = info['name']
            if filter_name and filter_name.lower() not in name.lower():
                continue
            cpu_percent = info['cpu_percent']
            memory_percent = info['memory_percent']
            num_threads = info['num_threads']
            table.add_row(name, f"{cpu_percent}%", f"{memory_percent}%", str(num_threads))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    console.print(table)


def log_system_info():
    cpu_percent, cpu_freq, mem_percent, mem_used, disk_percent, disk_used, net_sent, net_recv, uptime, users = get_system_basic_info()

    log_file.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"CPU使用率: {cpu_percent}%\n")
    log_file.write(f"CPU频率: {cpu_freq:.2f}MHz\n")
    log_file.write(f"内存使用率: {mem_percent}%\n")
    log_file.write(f"内存使用量: {mem_used:.2f}GB\n")
    log_file.write(f"磁盘使用率: {disk_percent}%\n")
    log_file.write(f"磁盘使用量: {disk_used:.2f}GB\n")
    log_file.write(f"网络发送量: {net_sent:.2f}MB\n")
    log_file.write(f"网络接收量: {net_recv:.2f}MB\n")
    log_file.write(f"系统运行时间: {uptime} 秒\n")
    log_file.write(f"当前登录用户: {', '.join([user.name for user in users])}\n\n")


def main():
    parser = argparse.ArgumentParser(description='系统监控工具')
    parser.add_argument('--filter', '-f', help='按进程名过滤进程信息')
    parser.add_argument('--interval', '-i', type=float, default=1.0, help='监控刷新间隔时间（秒）')
    args = parser.parse_args()

    setup_logging()
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        console.print("=" * 80)
        console.print("系统监控信息".center(80))
        console.print("=" * 80)
        with Progress() as progress:
            task = progress.add_task("[cyan]正在监控...", total=None)
            print_system_info()
            print_battery_info()
            print_gpu_info()
            print_fan_speed_info()
            print_voltage_info()
            print_temperature_info()
            print_process_info(args.filter)
            log_system_info()
            time.sleep(args.interval)
            progress.update(task, advance=1)
        console.print("=" * 80)


if __name__ == "__main__":
    main()