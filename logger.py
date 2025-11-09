"""
日志记录模块 - 同时输出到控制台和文件
"""
import sys
import os
from datetime import datetime


class TeeLogger:
    """同时输出到控制台和文件的日志记录器"""
    
    def __init__(self, log_file_path):
        """
        Args:
            log_file_path: 日志文件路径
        """
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.log_file_path = log_file_path
        
        # 写入开始标记
        self.write(f"\n{'='*60}\n")
        self.write(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.write(f"{'='*60}\n\n")
    
    def write(self, message):
        """写入消息到控制台和文件"""
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # 立即刷新到文件
    
    def flush(self):
        """刷新缓冲区"""
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        """关闭日志文件"""
        if self.log_file:
            self.write(f"\n{'='*60}\n")
            self.write(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.write(f"{'='*60}\n")
            self.log_file.close()
            print(f"\n日志已保存到: {self.log_file_path}")
    
    def __del__(self):
        """析构函数，确保文件被关闭"""
        self.close()

