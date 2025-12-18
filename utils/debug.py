"""
Debug utility module for console output with color support and extensibility.
"""

import sys
from typing import Any
from datetime import datetime


class DebugConfig:
    """配置类，方便后续扩展功能"""
    enabled = True
    use_color = True
    log_to_file = False
    log_file_path = "debug.log"
    show_timestamp = True
    show_caller_info = False


class Colors:
    """ANSI颜色代码"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def debug(*args, **kwargs):
    """
    增强的调试输出函数，支持彩色输出和日志记录
    
    参数:
        *args: 要输出的内容, 与print类似
        **kwargs: 额外的关键字参数
            - color: 指定颜色 ('red', 'green', 'yellow', 'blue', 'magenta', 'cyan')
            - level: 日志级别 ('INFO', 'WARNING', 'ERROR', 'DEBUG')
            - sep: 分隔符，默认为空格
            - end: 结束符，默认为换行
    
    示例:
        debug("这是一条调试信息")
        debug("警告信息", level="WARNING", color="yellow")
        debug("变量x =", x, "变量y =", y)
    """
    if not DebugConfig.enabled:
        return
    
    color = kwargs.pop('color', 'red')
    level = kwargs.pop('level', 'DEBUG')
    sep = kwargs.pop('sep', ' ')
    end = kwargs.pop('end', '\n')
    
    message_parts = []
    
    if DebugConfig.show_timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message_parts.append(f"[{timestamp}]")
    
    message_parts.append(f"[{level}]")
    
    message = sep.join(str(arg) for arg in args)
    message_parts.append(message)
    
    full_message = " ".join(message_parts)
    
    if DebugConfig.use_color:
        color_code = getattr(Colors, color.upper(), Colors.RED)
        colored_message = f"{Colors.BOLD}{color_code}{full_message}{Colors.RESET}"
        print(colored_message, end=end, file=sys.stderr, **kwargs)
    else:
        print(full_message, end=end, file=sys.stderr, **kwargs)
    
    if DebugConfig.log_to_file:
        try:
            with open(DebugConfig.log_file_path, 'a', encoding='utf-8') as f:
                f.write(full_message + end)
        except Exception as e:
            print(f"日志写入失败: {e}", file=sys.stderr)


def enable_debug(enabled: bool = True):
    """启用或禁用debug输出"""
    DebugConfig.enabled = enabled


def enable_log_file(file_path: str = "debug.log"):
    """启用文件日志记录"""
    DebugConfig.log_to_file = True
    DebugConfig.log_file_path = file_path


def disable_log_file():
    """禁用文件日志记录"""
    DebugConfig.log_to_file = False


def set_color_output(enabled: bool = True):
    """设置是否使用彩色输出"""
    DebugConfig.use_color = enabled


def set_timestamp(enabled: bool = True):
    """设置是否显示时间戳"""
    DebugConfig.show_timestamp = enabled


# if __name__ == "__main__":
#     print("=== Debug函数使用示例 ===\n")
    
#     # 基本使用
#     debug("这是一条默认的调试信息（红色）")
    
#     # 不同颜色
#     debug("这是绿色信息", color="green")
#     debug("这是黄色警告", color="yellow", level="WARNING")
#     debug("这是蓝色信息", color="blue", level="INFO")
    
#     # 多个参数
#     x, y = 10, 20
#     debug("变量值:", "x =", x, "y =", y)
    
#     # 启用文件日志
#     enable_log_file("test_debug.log")
#     debug("这条消息会同时输出到控制台和文件")
    
#     # 禁用时间戳
#     set_timestamp(False)
#     debug("没有时间戳的消息")
    
#     # 禁用debug
#     enable_debug(False)
#     debug("这条消息不会显示")
    
#     enable_debug(True)
#     debug("重新启用debug功能")