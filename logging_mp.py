"""
logging_mp shim — 轻量兼容层，用标准 logging 模块替代原始 logging_mp 包。
提供与原始 logging_mp 相同的 API：basicConfig, getLogger, 以及日志等级常量。
"""
import logging

# 日志等级常量，与标准 logging 一致
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

def basicConfig(**kwargs):
    logging.basicConfig(**kwargs)

def getLogger(name=None):
    return logging.getLogger(name)
