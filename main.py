# =========================== 1. 導入一般模組 ===========================
import sys
import os
import json
import cv2
import mss
import time
import numpy as np
import logging
import zipfile
import io
import ctypes
from ultralytics import YOLO
from pynput.mouse import Controller as MouseController
import torch
import warnings
import requests
from datetime import datetime, timedelta
import threading
import psutil
import subprocess
import socket
from pathlib import Path
import random
import atexit

# =========================
# UI 元件導入
# =========================
from PySide6.QtCore import Qt, Signal, QObject, QPoint, QRect, QRectF, QTimer, QDateTime, QUrl, QThread
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QFrame, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QGraphicsDropShadowEffect,
    QMessageBox, QRadioButton, QCheckBox, QSlider, QTextEdit, 
    QStackedWidget, QSizeGrip, QGraphicsBlurEffect,
    QListWidget, QListWidgetItem, QComboBox, QGroupBox, 
    QScrollArea, QSpinBox
)
from PySide6.QtGui import QPainter, QPen, QColor, QScreen, QPixmap, QPainterPath, QIcon, QBrush, QFont
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

# =========================
# 配置系統導入
# =========================
try:
    from config import get_config
    logging.info("✅ 配置系統已加載")
except ImportError:
    logging.warning("⚠️ 找不到 config.py，使用内置配置")
    def get_config():
        """備用配置函數"""
        class DefaultConfig:
            def get(self, key, default=None):
                return default
            def get_section(self, section):
                return {}
            def set(self, key, value):
                return True
            def save_config(self):
                return True
        return DefaultConfig()
log_path = os.path.join(os.path.dirname(sys.executable), "aimbot.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.info("=== 程序初始化中 ===")
time.sleep(1)
try:
    # 1. 首选：GPU 版本
    from aimbot_gpu import AimbotOverlay
    logging.info("🚀 运行环境：GPU 加速已启用")
except (ImportError, Exception) as e: 
    # 如果 GPU 导入失败或驱动报错，尝试 CPU
    try:
        from aimbot_cpu import AimbotOverlay
        logging.warning(f"⚠️ GPU 启动失败 ({type(e).__name__})，已切换至 CPU 模式")
    except ImportError:
        logging.error("❌ 严重错误：找不到任何 AimbotOverlay 模块！")
        # 保持你之前的 sys.exit(1) 逻辑，确保程序不会带着错误继续跑
        sys.exit(1)

# atexit 注册将在定义 stop_all_function_threads 后执行，见文件下方

# ==================== 全局模型轉換標誌 ====================
ONNX_CONVERSION_LOCK = threading.Lock()
ONNX_CONVERSION_DONE = False
CACHED_MODEL_PATH = None
# ========================================================


# 简单的 QThread 封装，用于在 Qt 事件循环中运行普通函数
# 全局追踪所有 FunctionThread 實例以便退出時清理
FUNCTION_THREADS = []


class FunctionThread(QThread):
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        # 註冊到全局列表
        try:
            FUNCTION_THREADS.append(self)
        except Exception:
            pass
        # 當線程結束時嘗試從全局列表移除
        try:
            self.finished.connect(lambda: FUNCTION_THREADS.remove(self) if self in FUNCTION_THREADS else None)
        except Exception:
            pass

    def run(self):
        try:
            self.func(*self.args, **self.kwargs)
        except Exception:
            logging.exception("FunctionThread run error")
        finally:
            # 確保在結束時將自己從全局列表移除
            try:
                if self in FUNCTION_THREADS:
                    FUNCTION_THREADS.remove(self)
            except Exception:
                pass


def stop_all_function_threads(timeout_ms=2000):
    """嘗試優雅停止並等待所有註冊的 FunctionThread。"""
    try:
        threads = list(FUNCTION_THREADS)
    except Exception:
        threads = []
    logging.info(f"準備停止 {len(threads)} 個后台 FunctionThread 實例")
    for t in threads:
        try:
            if not isinstance(t, QThread):
                continue
            try:
                t.requestInterruption()
            except Exception:
                pass
            try:
                t.quit()
            except Exception:
                pass
            try:
                t.wait(timeout_ms)
            except Exception:
                pass
            if t.isRunning():
                try:
                    t.terminate()
                except Exception:
                    pass
        except Exception:
            # 忽略单个线程处理中的任何异常，继续处理其它线程
            pass
    logging.info("已發送停止信號至所有 FunctionThread")

# 註冊 atexit 清理，確保在解釋器退出時也會嘗試停止后台線程
try:
    atexit.register(stop_all_function_threads)
except Exception:
    pass


#  DPI 警告修正
os.environ["QT_LOGGING_RULES"] = "qt.qpa.window=false"
# --- 1. 環境設定與 DPI 解決方案 ---
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1) 
except Exception:
    pass
        
# 強制設定使用系統縮放，避免進入 Per-Monitor V2 模式
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_FONT_DPI"] = "96"

# --- 1. 透明 ESP  ---
class ESPOverlay(QWidget):
    def __init__(self):
        super().__init__()
        
        # 初始化FOV值
        self.fov = 150 
        # 是否顯示 FOV（可由菜單開關控制）
        self.fov_enabled = False
        self.targets = [] # 格式: [(x, y, w, h, is_lock)]
        # ESP 配置引用（會由 ModernModMenu 注入）
        self.esp_config = {}
        
        # 120 = WA_TranslucentBackground
        # 150 = WA_TransparentForInput
        # 84  = WA_AlwaysStackOnTop
        try:
            # 嘗試使用標準寫法
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForInput, True)
        except:
            # 強制使用整數代碼轉換
            self.setAttribute(Qt.WidgetAttribute(120), True)
            self.setAttribute(Qt.WidgetAttribute(150), True)
        # 0x00000800 = FramelessWindowHint
        # 0x00040000 = WindowStaysOnTopHint
        # 0x0000000a = Tool
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.Tool
        )
        # 3. 獲取螢幕大小並全螢幕化
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen) # 直接覆蓋整個螢幕
        
        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # 抗鋸齒
        
        center_x = self.width() // 2
        center_y = self.height() // 2

        # --- 繪製 FOV 圓圈 ---
        if getattr(self, 'fov_enabled', True) and hasattr(self, 'fov') and self.fov > 0:
            # 绘制外圆圈（不透明的红色）
            pen_fov = QPen(QColor(255, 0, 0, 255))
            pen_fov.setWidth(1)
            painter.setPen(pen_fov)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center_x - self.fov, center_y - self.fov, self.fov * 2, self.fov * 2)
            
            # 绘制中心十字
            cross_size = 10
            painter.setPen(QPen(QColor(255, 0, 0, 200), 1))
            painter.drawLine(center_x - cross_size, center_y, center_x + cross_size, center_y)
            painter.drawLine(center_x, center_y - cross_size, center_x, center_y + cross_size)

        # --- 根據 ESP 配置繪製目標 ---
        for (x, y, w, h, is_lock) in self.targets:
            lock_color = QColor(255, 0, 0, 255) if is_lock else QColor(0, 255, 0, 150)
            lock_width = 2 if is_lock else 1
            
            # 1. 显示方框 (Box ESP)
            if self.esp_config.get('box', False):
                painter.setPen(QPen(lock_color, lock_width))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(int(x), int(y), int(w), int(h))
            
            # 2. 显示射線 (Line ESP) - 從屏幕中心到目標中心的射線
            if self.esp_config.get('line', False):
                target_cx = int(x + w / 2)
                target_cy = int(y + h / 2)
                painter.setPen(QPen(lock_color, 1))
                painter.drawLine(center_x, center_y, target_cx, target_cy)
            
            # 3. 显示骨架 (Skeleton ESP) - 人体关键点连线
            if self.esp_config.get('skeleton', False):
                painter.setPen(QPen(QColor(255, 255, 0, 200), 1))
                head_x, head_y = int(x + w/2), int(y + h*0.2)
                torso_mid_y = int(y + h*0.5)
                leg_y = int(y + h)
                # 畫頭
                painter.drawEllipse(head_x - 5, head_y - 5, 10, 10)
                # 畫脊椎
                painter.drawLine(head_x, head_y, head_x, torso_mid_y)
                painter.drawLine(head_x, torso_mid_y, int(x + w*0.3), leg_y)  # 左腿
                painter.drawLine(head_x, torso_mid_y, int(x + w*0.7), leg_y)  # 右腿
            
            # 4. 显示盒子 (Boxes ESP) - 填充半透明方框
            if self.esp_config.get('boxes', False):
                painter.setPen(QPen(lock_color, lock_width))
                painter.setBrush(QBrush(QColor(lock_color.red(), lock_color.green(), lock_color.blue(), 30)))
                painter.drawRect(int(x), int(y), int(w), int(h))
            
            # 5. 显示距離 (Distance ESP) - 文字顯示目標到中心的距離
            if self.esp_config.get('distance', False):
                target_cx = int(x + w / 2)
                target_cy = int(y + h / 2)
                dist = int(((target_cx - center_x)**2 + (target_cy - center_y)**2)**0.5)
                painter.setPen(QPen(QColor(100, 255, 255, 255), 1))
                font = painter.font()
                font.setPointSize(8)
                painter.setFont(font)
                painter.drawText(int(x) + 5, int(y) + 15, f"D:{dist}px")
            
            # 6. 显示受攻擊預警 (Threat ESP) - 周圍繪製警告三角形
            if self.esp_config.get('threat', False):
                painter.setPen(QPen(QColor(255, 100, 0, 255), 2))
                target_cx = int(x + w / 2)
                target_cy = int(y + h / 2)
                # 繪製三個警告三角形（頂部、左下、右下）
                triangle_size = 15
                top_tri = [QPoint(target_cx, target_cy - triangle_size), 
                          QPoint(target_cx - triangle_size//2, target_cy), 
                          QPoint(target_cx + triangle_size//2, target_cy)]
                painter.drawPolygon(top_tri)
            
            # 7. 显示投擲物 (Grenade ESP) - 繪製特殊標記
            if self.esp_config.get('grenade', False):
                painter.setPen(QPen(QColor(255, 165, 0, 255), 2))
                painter.setBrush(QBrush(QColor(255, 165, 0, 100)))
                # 繪製菱形標記
                cx = int(x + w / 2)
                cy = int(y + h / 2)
                diamond_size = 8
                diamond = [QPoint(cx, cy - diamond_size),
                          QPoint(cx + diamond_size, cy),
                          QPoint(cx, cy + diamond_size),
                          QPoint(cx - diamond_size, cy)]
                painter.drawPolygon(diamond)

        # --- 显示 CAESAR UI - 悬浮在视窗顶端的红色文字 ---
        if self.esp_config.get('XUANS', True):  # 默认启用
            painter.setPen(QPen(QColor(255, 0, 0, 255), 2))  # 红色画笔
            font = painter.font()
            font.setPointSize(20)  # 大字体
            font.setBold(True)  # 加粗
            font.setItalic(True)  # 微斜体
            painter.setFont(font)
            
            # 在视窗顶端中央绘制"XUANS"文字
            text = "XUANS"  # 替换为你想显示的文字
            text_rect = painter.fontMetrics().boundingRect(text)
            text_x = (self.width() - text_rect.width()) // 2  # 水平居中
            text_y = 80  # 距离顶端80px（增加间距）
            painter.drawText(text_x, text_y, text)
            
            # 显示人数 - 在CAESAR文字上方
            num_targets = len(self.targets)
            count_text = "1" if num_targets > 0 else "0"
            count_font = painter.font()
            count_font.setPointSize(20)  # 和CAESAR一样大
            count_font.setBold(True)
            count_font.setItalic(True)  # 同样微斜体
            painter.setFont(count_font)
            count_rect = painter.fontMetrics().boundingRect(count_text)
            count_x = (self.width() - count_rect.width()) // 2  # 水平居中
            count_y = text_y - 25  # 在CAESAR上方25px
            painter.drawText(count_x, count_y, count_text)

        

# --- 簡單十字架小部件 ---
class CrosshairWidget(QWidget):
    def __init__(self, size=80, color=QColor(255, 0, 0, 200), thickness=2, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.color = color
        self.thickness = thickness
        self._visible = True

    def set_visible(self, visible: bool):
        self._visible = bool(visible)
        self.setVisible(self._visible)
        self.update()

    def paintEvent(self, event):
        if not self._visible:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self.color, self.thickness)
        painter.setPen(pen)
        cx = self.width() // 2
        cy = self.height() // 2
        arm = min(self.width(), self.height()) // 3
        # 水平線
        painter.drawLine(cx - arm, cy, cx + arm, cy)
        # 垂直線
        painter.drawLine(cx, cy - arm, cx, cy + arm)


class CrosshairOverlay(QWidget):
    """頂層透明十字架覆蓋，固定在主螢幕中心且允許點擊穿透。"""
    def __init__(self, size=80, color=QColor(255, 0, 0, 200), thickness=2):
        super().__init__(None)
        self.size = size
        self.color = color
        self.thickness = thickness
        self._visible = False
        self.rotation = 0.0

        # 視窗屬性：無框、最上層、工具窗
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # 試著讓視窗不攔截滑鼠事件
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForInput, True)
        except Exception:
            try:
                self.setAttribute(Qt.WidgetAttribute(150), True)
            except Exception:
                pass

        self.setFixedSize(self.size, self.size)
        self.reposition_to_center()

    def reposition_to_center(self):
        screen = QApplication.primaryScreen().geometry()
        cx = screen.width() // 2
        cy = screen.height() // 2
        self.move(cx - self.width() // 2, cy - self.height() // 2)

    def set_visible(self, visible: bool):
        self._visible = bool(visible)
        if self._visible:
            self.reposition_to_center()
            self.show()
        else:
            self.hide()

    def paintEvent(self, event):
        if not self._visible:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self.color, self.thickness)
        painter.setPen(pen)
        cx = self.width() // 2
        cy = self.height() // 2
        arm = min(self.width(), self.height()) // 3
        # 使用座標系平移與旋轉，讓線段繞中心旋轉
        painter.translate(cx, cy)
        painter.rotate(getattr(self, 'rotation', 0))
        painter.drawLine(-arm, 0, arm, 0)
        painter.drawLine(0, -arm, 0, arm)
        painter.resetTransform()
    
    # 動態調整方法
    def set_size(self, size: int):
        try:
            size = int(size)
            size = max(10, min(size, 2000))
            self.size = size
            self.setFixedSize(self.size, self.size)
            self.reposition_to_center()
            self.update()
        except Exception:
            pass

    def set_thickness(self, thickness: int):
        try:
            thickness = int(thickness)
            thickness = max(1, min(thickness, 50))
            self.thickness = thickness
            self.update()
        except Exception:
            pass

    def set_color(self, qcolor: QColor):
        try:
            self.color = qcolor
            self.update()
        except Exception:
            pass

    def set_rotation(self, angle: float):
        try:
            # 允許小數角度
            self.rotation = float(angle) % 360.0
            self.update()
        except Exception:
            pass
# ============================================================
# 1. 登入視窗類別 (LoginWindow)
# ============================================================
class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        # 1. 基本設定：無邊框、置頂、透明背景
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(650, 400)
        
        self.is_authenticated = False
        self.drag_pos = None
        self.my_hwid = self.get_hwid()
        
        # 2. 載入背景
        self.bg_pixmap = None
        self.load_background("login_bg.png")
        
        # 3. 初始化 UI
        self.init_ui()
        
        # 4. 啟動計時器更新時間
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)
    
    def get_hwid(self):
        try:
            cmd = 'wmic csproduct get uuid'
            output = subprocess.check_output(cmd, shell=True).decode().split()
            return output[1] if len(output) > 1 else "UNKNOWN_DEVICE"
        except Exception:
            return "UNKNOWN_DEVICE"

    def load_background(self, image_path):
        if os.path.exists(image_path):
            self.bg_pixmap = QPixmap(image_path)
            if not self.bg_pixmap.isNull():
                logging.info(f"✅ 背景圖片已加載: {image_path}")
            else:
                self.bg_pixmap = None
        else:
            logging.info(f"ℹ️ 背景圖不存在，使用默認深色背景")

    def update_time(self):
        current_time = QDateTime.currentDateTime().toString("HH:mm:ss")
        if hasattr(self, 'time_label'):
            self.time_label.setText(f"<span style='color: #64748B;'>[目前時間]：</span><span style='color: #38BDF8;'>{current_time}</span>")

    def init_ui(self):
        # --- 主容器  ---
        self.main_container = QFrame(self)
        self.main_container.setGeometry(0, 0, 650, 400)
        self.main_container.setStyleSheet("""
            QFrame#MainContainer { 
                background-color: transparent; 
            }
            QLabel { color: #F8FAFC; font-family: 'Poppins', 'Microsoft JhengHei'; }
        """)
        self.main_container.setObjectName("MainContainer")
        
        main_layout = QHBoxLayout(self.main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- 左側 Panel (半透明深色) ---
        self.left_panel = QFrame()
        self.left_panel.setFixedWidth(260)
        self.left_panel.setStyleSheet("""
            background-color: rgba(30, 41, 59, 200);
            border-top-left-radius: 20px;
            border-bottom-left-radius: 20px;
            border-right: 1px solid rgba(51, 65, 85, 100);
        """)
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # Logo 區塊
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap("logo.png")
        if not pixmap.isNull():
            pixmap = pixmap.scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.logo_label.setPixmap(pixmap)
            glow = QGraphicsDropShadowEffect(self.logo_label)
            glow.setBlurRadius(35)
            glow.setColor(QColor(56, 189, 248, 180))
            glow.setOffset(0, 0)
            self.logo_label.setGraphicsEffect(glow)
            self.logo_label.setContentsMargins(30, 30, 30, 30)
        else:
            self.logo_label.setText("XUANS")
            self.logo_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #38BDF8;")

        self.logo_text = QLabel("困暄")
        self.logo_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo_text.setStyleSheet("font-size: 32px; font-weight: 900; color: #38BDF8; background: transparent;")

        self.info_label = QLabel("本工具由困暄製作\n版本 v1.0.0 (付費版)")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("color: #94A3B8; font-size: 13px; background: transparent;")

        self.time_label = QLabel()
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet("font-size: 12px; margin-bottom: 20px; background: transparent;")

        left_layout.addStretch()
        left_layout.addWidget(self.logo_label)
        left_layout.addWidget(self.logo_text)
        left_layout.addWidget(self.info_label)
        left_layout.addStretch()
        left_layout.addWidget(self.time_label)

        # --- 右側 Panel (透明的背景) ---
        self.right_panel = QFrame()
        self.right_panel.setStyleSheet("background-color: rgba(15, 23, 42, 150); border-top-right-radius: 20px; border-bottom-right-radius: 20px;")
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(45, 40, 60, 40)
        right_layout.setSpacing(15)

        # 右上角關閉按鈕
        self.close_btn = QPushButton("✕", self.right_panel)
        self.close_btn.setFixedSize(30, 30)
        self.close_btn.move(350, 10)
        self.close_btn.setStyleSheet("QPushButton { color: #94A3B8; font-size: 18px; border: none; background: transparent; } QPushButton:hover { color: #F8FAFC; }")
        self.close_btn.clicked.connect(self.close)

        self.title = QLabel("Welcome Back")
        self.title.setStyleSheet("font-size: 26px; font-weight: 700; background: transparent;")
        
        self.status_label = QLabel("請輸入您的授權資訊")
        self.status_label.setStyleSheet("color: #94A3B8; font-size: 12px; background: transparent;")

        edit_style = """
            QLineEdit {
                background-color: rgba(15, 23, 42, 200);
                border: 1.5px solid #334155;
                border-radius: 10px;
                padding: 10px;
                font-size: 13px;
                color: #F8FAFC;
            }
            QLineEdit:focus { border: 1.5px solid #38BDF8; }
        """

        self.acc_input = QLineEdit()
        self.acc_input.setPlaceholderText("請輸入帳號")
        self.acc_input.setStyleSheet(edit_style)

        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("請輸入卡密")
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_input.setStyleSheet(edit_style)

        self.login_btn = QPushButton("登入系統")
        self.login_btn.setFixedHeight(45)
        self.login_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.login_btn.setStyleSheet("QPushButton { background-color: #0284C7; border-radius: 10px; font-weight: 700; color: white; } QPushButton:hover { background-color: #0EA5E9; }")
        self.login_btn.clicked.connect(self.check_key)

        self.copy_btn = QPushButton("📋 複製目前的機器碼 (HWID)")
        self.copy_btn.setFixedHeight(30)
        self.copy_btn.setStyleSheet("QPushButton { background-color: #334155; border-radius: 5px; font-size: 11px; color: #94A3B8; } QPushButton:hover { background-color: #475569; color: #F8FAFC; }")
        self.copy_btn.clicked.connect(self.copy_to_clipboard)

        right_layout.addWidget(self.title)
        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.acc_input)
        right_layout.addWidget(self.key_input)
        right_layout.addWidget(self.login_btn)
        right_layout.addWidget(self.copy_btn)
        right_layout.addStretch()

        main_layout.addWidget(self.left_panel)
        main_layout.addWidget(self.right_panel)

    # --- 事件處理 ---
    def paintEvent(self, event):
     painter = QPainter(self)
     painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    
     # 建立圓角裁切路徑
     path = QPainterPath()
     path.addRoundedRect(QRectF(self.rect()), 20.0, 20.0)
     painter.setClipPath(path)

     if self.bg_pixmap and not self.bg_pixmap.isNull():
        # 繪製背景圖，並拉伸至視窗大小
        painter.drawPixmap(self.rect(), self.bg_pixmap)
     else:
        # 如果沒圖，畫一個顯眼的顏色標記（除錯用）
        painter.fillPath(path, QColor("#FF00FF")) 
        logging.error("❌ 背景圖對象為空或無效")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.drag_pos is not None:
            delta = event.globalPosition().toPoint() - self.drag_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.drag_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self.drag_pos = None

    # 複製剪貼簿功能
    def copy_to_clipboard(self):
        text = self.key_input.text().strip()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            self.status_label.setText("✅ 卡密已複製到剪貼簿")
        else:
            self.status_label.setText("⚠️ 沒有可複製的內容")

    # 自動貼上剪貼簿內容
    def auto_paste_from_clipboard(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text().strip()
        if text:
            self.key_input.setText(text)
            self.status_label.setText("✏️ 已自動貼上剪貼簿卡密")
            
    # --- 事件處理 ---
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 如果有背景圖片，先繪製背景圖片
        if self.bg_pixmap and not self.bg_pixmap.isNull():
            # 縮放圖片以充滿整個窗口
            scaled_pixmap = self.bg_pixmap.scaled(
                self.width(), self.height(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            p.drawPixmap(0, 0, scaled_pixmap)
        else:
            # 如果沒有背景圖片，使用默認顏色
            p.setBrush(QColor(15, 23, 42))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(self.rect(), 20, 20)

    # --- 核心驗證邏輯 ---
    def check_key(self):
        # 修正：使用 init_ui 中定義的變數名
        account = self.acc_input.text().strip()
        user_key = self.key_input.text().strip()

        if not account or not user_key:
            self.status_label.setText("⚠️ 帳號與卡密均不能為空")
            return

        GIST_URL = "https://gist.githubusercontent.com/cyrog120hz-hub/dd425f38a5fa9c8c16a8083a294a06be/raw/ab47836c6a1b951fdde7a5ba169b2af27a80bbd9/keys.json"
        local_auth_file = os.path.join(os.environ['APPDATA'], "cyrog_auth.dat")

        try:
            self.status_label.setText("⏳ 正在連線雲端...")
            QApplication.processEvents()
            
            res = requests.get(GIST_URL, timeout=8)
            key_db = res.json() # 直接用 .json() 解析

            if user_key in key_db:
                days = int(key_db[user_key])
                expiry_time = datetime.now() + timedelta(days=days)
                
                new_auth = {
                    "account": account,
                    "key": user_key,
                    "expiry": expiry_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "hwid": self.my_hwid
                }
                
                with open(local_auth_file, "w") as f:
                    json.dump(new_auth, f)

                self.status_label.setText(f"🚀 驗證成功！天數: {days} 天")
                self.status_label.setStyleSheet("color: #34C759;")
                self.is_authenticated = True
                
                # 延時關閉並進入主程式
                QTimer.singleShot(1500, self.close) 
            else:
                self.status_label.setText("❌ 無效的卡密，請檢查後再試")
                self.status_label.setStyleSheet("color: #FF453A;")
        except Exception as e:
            self.status_label.setText(f"🌐 網路異常或雲端格式錯誤")

    # --- 補充缺失的方法 ---
    def accept_login(self):
        """關閉登入視窗，由 main 函數啟動主選單"""
        self.close()
# --- 2. 信號傳輸系統 ---
class AimbotSignal(QObject):
    # 在 PySide6 中，必須使用 Signal
    update_log = Signal(str) 
    update_status = Signal(str)
    screen_recording_detected_signal = Signal()  # 防錄屏檢測信號
    restore_windows_signal = Signal()  # 恢復窗口信號 

# 實例化
aimbot_signal = AimbotSignal()

# --- 3. 主介面類別 ---
class ModernModMenu(QWidget):
    def __init__(self, model_path=None):
        super().__init__()
        self.my_hwid = self.get_hwid()
        # ⭐ 主题状态
        self.is_dark_theme = True
        
        # ⭐ 程序初始化状态标志
        self.is_program_initialized = False
        
        # ========== 加載配置系統 ==========
        self.cfg = get_config()
        logging.info("📋 配置系統初始化完成")
        
        # 核心變數
        self.overlay = ESPOverlay()
        
        # ========== 從配置加載FOV設置 ==========
        fov_config = self.cfg.get_section('fov')
        self.overlay.fov = fov_config.get('radius', 150)
        
        # 初始化 ESP 配置字典
        esp_section = self.cfg.get_section('esp')
        self.esp_config = {
            'box': esp_section.get('box', False),
            'line': esp_section.get('line', False),
            'skeleton': esp_section.get('skeleton', False),
            'boxes': esp_section.get('boxes', False),
            'distance': esp_section.get('distance', False),
            'threat': esp_section.get('threat', False),
            'grenade': esp_section.get('grenade', False),
            'caesar': esp_section.get('caesar', False)
        }
        self.overlay.esp_config = self.esp_config  # 注入到 overlay
        
        self.aimbot_active = False
        self.aimbot_thread = None
        self.yolo_model = None
        self.model_path = model_path  # 接收預先設置好的模型路徑
        self.mouse = MouseController()
        self.drag_pos = None
        self.target_device = "cpu" 
        self.status_label = None 
        self.log_output = None
        # 背景圖片
        self.bg_pixmap = None
        self.bg_pixmap_current = None  # 当前显示的背景
        self.bg_pixmap_target = None   # 目标背景
        self.bg_opacity = 1.0          # 背景透明度（用于过渡）
        
        # ========== 從配置加載AI自瞄參數 ==========
        aimbot_cfg = self.cfg.get_section('aimbot')
        fov_cfg = self.cfg.get_section('fov')
        
        # FOV 相關屬性大小 (半徑)
        self.fov_radius = fov_cfg.get('radius', 150)
        self.fov_enabled = self.cfg.get('features.fov_enabled', True)
        fov_color = fov_cfg.get('color', [255, 0, 0])
        self.fov_color = QColor(fov_color[0], fov_color[1], fov_color[2])
        
        # Aimbot 优化参数（從配置加載）
        use_gpu = self.cfg.get('model.use_gpu', True)
        # 自動判斷是GPU還是CPU模式，並根據配置選擇不同的參數值
        if use_gpu:
            self.aim_smoothing = aimbot_cfg.get('smoothing', 0.18)
            self.aim_infer_interval = aimbot_cfg.get('infer_interval', 0.04)
            self.aim_max_move = aimbot_cfg.get('max_move', 80)
        else:
            self.aim_smoothing = aimbot_cfg.get('smoothing_cpu', 0.25)
            self.aim_infer_interval = aimbot_cfg.get('infer_interval_cpu', 0.08)
            self.aim_max_move = aimbot_cfg.get('max_move_cpu', 60)
        # 保存基线与最大推理间隔，用于自适应调节
        self._base_aim_infer_interval = float(self.aim_infer_interval)
        self._max_aim_infer_interval = float(self.cfg.get('aimbot.max_infer_interval', 0.5))
            
        self._last_infer = 0.0
        self._target_offset_x = 0.0
        self._target_offset_y = 0.0
        self.aim_sensitivity = 0.5  # 缩放目标偏移到鼠标移动的倍数
        
        self.initUI()
        # 連接信號槽
        aimbot_signal.update_log.connect(self.update_log_display)
        aimbot_signal.update_status.connect(self.update_status_display)
        aimbot_signal.screen_recording_detected_signal.connect(self.on_screen_recording_detected)
        aimbot_signal.restore_windows_signal.connect(self.restore_windows)
        # 使用 QThread 封装替代 threading.Thread，便于与 Qt 事件循环集成
        # 保存對象以便在退出時等待
        self.system_monitor_thread = FunctionThread(self.system_monitor_loop)
        self.system_monitor_thread.start()
        # 在 initUI 末尾加入
        self.security_timer = QTimer(self)
        
        # ⭐ 添加强制刷新定时器，确保paintEvent被频繁调用
        self.repaint_timer = QTimer(self)
        self.repaint_timer.timeout.connect(self.update)  # 调用update()触发paintEvent
        self.repaint_timer.start(100)  # 優化：改為100ms，降低CPU占用
        
        # ⭐ 添加全屏ESPOverlay刷新定时器
        self.overlay_update_timer = QTimer(self)
        # 优化：当自瞄激活或任一 ESP 配置被启用时更新 overlay
        self.overlay_update_timer.timeout.connect(lambda: self.overlay.update() if (getattr(self, 'aimbot_active', False) or any(getattr(self, 'esp_config', {}).values())) else None)
        self.overlay_update_timer.start(100)  # 优化：改为100ms，从60fps降至10fps，降低CPU占用
        
        # ⭐ 添加背景过渡动画定时器
        self.bg_fade_timer = QTimer(self)
        self.bg_fade_timer.timeout.connect(self._update_bg_fade)
        self.bg_fade_timer.start(100)  # 优化：从66ms改为100ms，约10fps，背景过渡不需要高刷新率
        
        # 载入持久化设置
        try:
            self.load_settings()
        except Exception:
            pass
        # 启动后台的防录屏检测线程 (改为 QThread 封装)
        self.screen_recording_thread = FunctionThread(self.screen_recording_monitor_loop)
        self.screen_recording_thread.start()
        # 启动全局热键監控線程（用于快速启/禁防录屏）
        self.hotkey_thread = FunctionThread(self.hotkey_monitor_loop)
        self.hotkey_thread.start()

        # 停止控制旗標（用于优雅退出）
        self._stop_threads = False

        # 连接退出钩子以便优雅停止所有后台线程
        try:
            app = QApplication.instance()
            if app is not None:
                app.aboutToQuit.connect(self._shutdown_threads)
        except Exception:
            pass
    def _shutdown_threads(self):
        """优雅停止所有后台线程，等待它们退出。"""
        logging.info("開始優雅關閉後台線程...")
        try:
            # 標記停止
            self._stop_threads = True
            # 先停用自瞄
            try:
                self.aimbot_active = False
            except Exception:
                pass

            # 尝试请求线程退出并等待
            threads = [
                getattr(self, 'aimbot_thread', None),
                getattr(self, 'system_monitor_thread', None),
                getattr(self, 'screen_recording_thread', None),
                getattr(self, 'hotkey_thread', None)
            ]
            # 可能還有臨時線程
            threads += [getattr(self, 'model_loader_thread', None), getattr(self, 'stats_thread', None)]
            for t in threads:
                if t is None:
                    continue
                try:
                    # 如果是 QThread，等待结束（对关键线程延长等待时间）
                    if isinstance(t, QThread):
                        # 对模型加载与自瞄主循环给予更长的等待时间
                        timeout = 10000 if (getattr(self, 'aimbot_thread', None) is t or getattr(self, 'model_loader_thread', None) is t) else 2000
                        try:
                            t.wait(timeout)
                        except Exception:
                            logging.debug("等待 QThread 时发生异常")
                    # 如果是 threading.Thread，join
                    elif hasattr(t, 'join'):
                        try:
                            t.join(timeout=2)
                        except Exception:
                            pass
                except Exception as e:
                    logging.debug(f"等待線程結束時出錯: {e}")
        except Exception as e:
            logging.error(f"_shutdown_threads 發生錯誤: {e}")
        logging.info("後台線程已發送停止信號，主程序將繼續退出")
    def update_log_display(self, message):
        """
        处理来自信号的日志信息并更新到 UI 界面
        """
        if self.log_output:
            # 在文本框末尾添加新日志
            self.log_output.append(message)
            # 自动滚动到底部
            self.log_output.verticalScrollBar().setValue(
                self.log_output.verticalScrollBar().maximum()
            )
        else:
            # 如果 log_output 还没初始化好，先打印到控制台
            print(f"[UI Log] {message}")

    def get_server_latency(self):
        """測量到伺服器的連線延遲"""
        import time
        import requests
        try:
            start_time = time.time()
            # 測試一個輕量級的 URL 以獲取延遲，使用 GitHub 的 favicon 作為測試對象
            requests.get("https://github.com/favicon.ico", timeout=2)
            latency = int((time.time() - start_time) * 1000)
            return latency
        except:
            return -1 # 連線失敗

    def sync_fov_value(self, v):
        """同步FOV滑桿數值到UI和绘制"""
        # ✅ 檢查程序是否已初始化
        if not getattr(self, 'is_program_initialized', False):
            aimbot_signal.update_log.emit("❌ 請先點擊「系統公告」頁面的「初始化」按鈕來啟動程序！")
            logging.warning("❌ FOV 功能被禁用：程序未初始化")
            return
        
        self.fov_label.setText(f"瞄准范围 (FOV): {v}")
        self.fov_radius = v 
        
        # ⭐ 最关键：同步到全屏ESP层
        if hasattr(self, 'overlay') and self.overlay:
            self.overlay.fov = v
            self.overlay.update()
        
        logging.info(f"FOV 已更新为: {v} 像素")
        self.update()
    def toggle_fov_enabled(self, state):
        """開關 FOV 顯示：同步菜單與 overlay"""
        # ✅ 檢查程序是否已初始化
        if not getattr(self, 'is_program_initialized', False):
            aimbot_signal.update_log.emit("❌ 請先點擊「系統公告」頁面的「初始化」按鈕來啟動程序！")
            logging.warning("❌ FOV 功能被禁用：程序未初始化")
            # 取消複選框狀態
            if hasattr(self, 'fov_checkbox'):
                self.fov_checkbox.blockSignals(True)
                self.fov_checkbox.setChecked(False)
                self.fov_checkbox.blockSignals(False)
            return
        
        enabled = bool(state)
        self.fov_enabled = enabled
        if hasattr(self, 'overlay') and self.overlay:
            self.overlay.fov_enabled = enabled
            # 如果啟用，確保 overlay 可見並置頂；如果關閉則隱藏以節省資源
            try:
                if enabled:
                    self.overlay.show()
                    try:
                        self.overlay.raise_()
                    except Exception:
                        pass
                else:
                    self.overlay.hide()
            except Exception:
                pass
            self.overlay.update()
        logging.info(f"FOV 顯示開關：{'開' if enabled else '關'}")
        # 儲存設定
        try:
            self.save_settings()
        except Exception:
            pass

    def save_settings(self):
        """將部分 UI 設定儲存到 settings.json（simple）"""
        try:
            cfg = {
                "fov_enabled": bool(getattr(self, 'fov_enabled', True)),
                "screen_protection_enabled": bool(getattr(self, 'screen_protection_enabled', True))
            }
            with open("settings.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f)
        except Exception as e:
            logging.warning(f"儲存設定失敗: {e}")

    def load_settings(self):
        """從 settings.json 讀取設定並應用到 UI/overlay"""
        try:
            if os.path.exists("settings.json"):
                with open("settings.json", "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                self.fov_enabled = bool(cfg.get("fov_enabled", True))
                self.screen_protection_enabled = bool(cfg.get("screen_protection_enabled", True))
            else:
                self.fov_enabled = True
                self.screen_protection_enabled = True
        except Exception as e:
            logging.warning(f"讀取設定失敗: {e}")
            self.fov_enabled = True
            self.screen_protection_enabled = True

        # 應用到 overlay 與 checkbox
        if hasattr(self, 'overlay') and self.overlay:
            self.overlay.fov_enabled = self.fov_enabled
            self.overlay.update()
        if hasattr(self, 'fov_checkbox'):
            self.fov_checkbox.setChecked(self.fov_enabled)
        # 應用防錄屏設定
        if hasattr(self, 'cb_Screenrecordingprotection'):
            try:
                self.cb_Screenrecordingprotection.setChecked(self.screen_protection_enabled)
            except Exception:
                pass
    def get_hwid(self):
        """獲取硬體唯一標識符 (HWID)"""
        try:
            import subprocess
            # 使用 wmic 獲取主板序列號或 UUID
            cmd = 'wmic csproduct get uuid'
            output = subprocess.check_output(cmd, shell=True).decode().split('\n')
            for line in output:
                if line.strip() and "UUID" not in line:
                    return line.strip()
            return "UNKNOWN-DEVICE-ID"
        except Exception as e:
            print(f"獲取 HWID 失敗: {e}")
            return "DEFAULT-HWID-0000"
    
    # ================= 配置管理方法 =================
    
    def save_config_to_file(self):
        """保存當前UI設置到 settings.json"""
        try:
            # 保存FOV设置
            self.cfg.set('fov.radius', self.fov_radius)
            self.cfg.set('features.fov_enabled', self.fov_enabled)
            
            # 保存ESP设置
            for key, value in self.esp_config.items():
                self.cfg.set(f'esp.{key}', value)
            
            # 保存AI自瞄参数
            self.cfg.set('aimbot.smoothing', self.aim_smoothing)
            self.cfg.set('aimbot.infer_interval', self.aim_infer_interval)
            self.cfg.set('aimbot.max_move', self.aim_max_move)
            
            # 保存到文件
            if self.cfg.save_config():
                logging.info("✅ 配置已成功保存到 settings.json")
                self.update_log_display("✅ 設定已保存")
                return True
            else:
                logging.warning("⚠️ 配置保存失败")
                return False
        except Exception as e:
            logging.error(f"❌ 保存配置时出错: {e}")
            return False

    def load_config_from_file(self):
        """從 settings.json 重新加載配置"""
        try:
            from config import ConfigManager
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
            self.cfg = ConfigManager(config_path)
            
            # 加载FOV设置
            fov_cfg = self.cfg.get_section('fov')
            self.fov_radius = fov_cfg.get('radius', 150)
            self.fov_enabled = self.cfg.get('features.fov_enabled', True)
            fov_color = fov_cfg.get('color', [255, 0, 0])
            self.fov_color = QColor(fov_color[0], fov_color[1], fov_color[2])
            self.overlay.fov = self.fov_radius
            self.overlay.fov_enabled = self.fov_enabled
            
            # 加载ESP设置
            esp_section = self.cfg.get_section('esp')
            self.esp_config = {
                'box': esp_section.get('box', False),
                'line': esp_section.get('line', False),
                'skeleton': esp_section.get('skeleton', False),
                'boxes': esp_section.get('boxes', False),
                'distance': esp_section.get('distance', False),
                'threat': esp_section.get('threat', False),
                'grenade': esp_section.get('grenade', False)
            }
            self.overlay.esp_config = self.esp_config
            
            # 加载AI自瞄参数
            aimbot_cfg = self.cfg.get_section('aimbot')
            self.aim_smoothing = aimbot_cfg.get('smoothing', 0.18)
            self.aim_infer_interval = aimbot_cfg.get('infer_interval', 0.04)
            self.aim_max_move = aimbot_cfg.get('max_move', 80)
            
            logging.info("✅ 配置已從 settings.json 重新加載")
            self.update_log_display("✅ 設定已重新加載")
            return True
        except Exception as e:
            logging.error(f"❌ 加载配置时出错: {e}")
            return False

    def export_config_to_file(self, export_path=None):
        """導出配置到指定的JSON文件（供其他人使用）"""
        try:
            if export_path is None:
                export_path = os.path.join(os.path.expanduser("~"), "Desktop", "my_aimbot_config.json")
            
            # 先保存当前设置
            self.save_config_to_file()
            
            # 复制配置到导出路径
            export_config = self.cfg.config.copy()
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_config, f, indent=2, ensure_ascii=False)
            
            logging.info(f"✅ 配置已導出到: {export_path}")
            self.update_log_display(f"✅ 配置已導出到桌面")
            return export_path
        except Exception as e:
            logging.error(f"❌ 导出配置时出错: {e}")
            return None

    def import_config_from_file(self, import_path=None):
        """從指定的JSON文件導入配置"""
        try:
            if import_path is None:
                # 从桌面寻找默认配置文件
                import_path = os.path.join(os.path.expanduser("~"), "Desktop", "my_aimbot_config.json")
            
            if not os.path.exists(import_path):
                logging.warning(f"⚠️ 配置文件不存在: {import_path}")
                self.update_log_display(f"⚠️ 找不到配置文件")
                return False
            
            # 加载导入的配置
            with open(import_path, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)
            
            # 更新当前配置
            self.cfg.config = imported_config
            self.cfg.save_config()
            
            # 重新加载配置到UI
            self.load_config_from_file()
            
            logging.info(f"✅ 配置已從以下文件導入: {import_path}")
            self.update_log_display(f"✅ 配置已成功導入")
            return True
        except Exception as e:
            logging.error(f"❌ 导入配置时出错: {e}")
            return False

    def sync_fov_to_config(self, radius, enabled=None, color=None):
        """同步FOV設置到配置"""
        try:
            self.fov_radius = radius
            if enabled is not None:
                self.fov_enabled = enabled
            if color is not None:
                self.fov_color = color
            
            self.overlay.fov = radius
            if enabled is not None:
                self.overlay.fov_enabled = enabled
            
            self.save_config_to_file()
        except Exception as e:
            logging.warning(f"⚠️ FOV同步失败: {e}")

    def sync_esp_to_config(self, esp_key, value):
        """同步ESP設置到配置"""
        try:
            if esp_key in self.esp_config:
                self.esp_config[esp_key] = value
                self.cfg.set(f'esp.{esp_key}', value)
                self.cfg.save_config()
                logging.debug(f"✓ ESP '{esp_key}' 同步: {value}")
        except Exception as e:
            logging.warning(f"⚠️ ESP同步失败: {e}")

    def sync_aimbot_to_config(self, param_name, value):
        """同步AI自瞄參數到配置"""
        try:
            if param_name == 'smoothing':
                self.aim_smoothing = value
            elif param_name == 'infer_interval':
                self.aim_infer_interval = value
            elif param_name == 'max_move':
                self.aim_max_move = value
            
            self.cfg.set(f'aimbot.{param_name}', value)
            self.cfg.save_config()
            logging.debug(f"✓ Aimbot '{param_name}' 同步: {value}")
        except Exception as e:
            logging.warning(f"⚠️ Aimbot同步失败: {e}")

    def get_config_as_dict(self):
        """獲取完整配置字典（用於展示或分享）"""
        return self.cfg.config.copy()

    def reset_config_to_default(self):
        """重置配置到默認值"""
        try:
            self.cfg.reset_to_default()
            self.load_config_from_file()
            logging.info("✅ 配置已重置至默認值")
            self.update_log_display("✅ 配置已重置")
            return True
        except Exception as e:
            logging.error(f"❌ 重置配置时出错: {e}")
            return False
        
    # ================= 配置管理方法結束 =================
        
    def start_typing_effect(self, text):
        """啟動打字機效果"""
        self.full_account_text = text
        self.char_index = 0
        self.account_info_label.setText("")
        
        if hasattr(self, 'typing_timer'):
            self.typing_timer.stop()
        else:
            self.typing_timer = QTimer(self)
            self.typing_timer.timeout.connect(self._type_letter)
        
        self.typing_timer.start(100) # 每 15ms 打印一個字符

    def _type_letter(self):
        if self.char_index <= len(self.full_account_text):
            # 取得目前的子字串
            current_substring = self.full_account_text[:self.char_index]

            if current_substring.endswith("<"):
                end_tag_idx = self.full_account_text.find(">", self.char_index)
                if end_tag_idx != -1:
                    self.char_index = end_tag_idx + 1
                    current_substring = self.full_account_text[:self.char_index]

            self.account_info_label.setText(current_substring)
            self.char_index += 1
        else:
            self.typing_timer.stop()
    # --- 防錄屏功能 ---
    def set_screen_protection_enabled(self, state):
        """切換防錄屏保護 (UI 回調)"""
        # ✅ 檢查程序是否已初始化
        if not getattr(self, 'is_program_initialized', False):
            aimbot_signal.update_log.emit("❌ 請先點擊「系統公告」頁面的「初始化」按鈕來啟動程序！")
            logging.warning("❌ 防錄屏功能被禁用：程序未初始化")
            # 取消複選框狀態
            if hasattr(self, 'cb_Screenrecordingprotection'):
                self.cb_Screenrecordingprotection.blockSignals(True)
                self.cb_Screenrecordingprotection.setChecked(False)
                self.cb_Screenrecordingprotection.blockSignals(False)
            return
        
        enabled = bool(state)
        self.screen_protection_enabled = enabled
        try:
            self.save_settings()
        except Exception:
            pass
        logging.info(f"防錄屏保護：{'開' if enabled else '關'}")


    def restore_windows(self):
        """恢復窗口顯示"""
        logging.info("防復原過程：開始恢復窗口顯示 [START]")
        try:
            import ctypes
            user32 = ctypes.windll.user32
            SW_SHOW = 5
            
            # 恢復 ModernModMenu（主菜單）和 MainController（按鈕）窗口
            target_classes = ['ModernModMenu', 'MainController']
            widget_count = 0
            for w in QApplication.topLevelWidgets():
                # 檢查窗口類型是否在目標清單中
                if w.__class__.__name__ not in target_classes:
                    continue
                
                widget_count += 1
                window_name = w.__class__.__name__
                try:
                    hwnd = int(w.winId())
                    logging.info(f"防復原 [{window_name}]：開始恢復 (hwnd={hwnd})")
                    
                    # Qt 顯示
                    w.show()
                    w.setVisible(True)
                    logging.info(f"防復原 [{window_name}]：Qt show/setVisible 成功")
                    
                    # Win32 顯示
                    res = user32.ShowWindow(hwnd, SW_SHOW)
                    logging.info(f"防復原 [{window_name}]：ShowWindow(hwnd={hwnd}, SW_SHOW) 返回 {res}")
                    
                    # 帶到前台
                    w.raise_()
                    w.activateWindow()
                    logging.info(f"防復原 [{window_name}]：raise/activateWindow 完成")
                        
                except Exception as e:
                    logging.warning(f"防復原 [{window_name}]：處理異常: {e}")
            
            if widget_count == 0:
                logging.info("防復原：未找到 ModernModMenu 或 MainController 窗口")
            else:
                logging.info(f"防復原：共恢復 {widget_count} 個窗口")

            # 強制刷新 UI
            try:
                QApplication.processEvents()
                logging.info("防復原：已調用 QApplication.processEvents()")
            except Exception:
                pass

            self.hidden_by_protection = False
            logging.info("防復原：已設置 hidden_by_protection = False")
            logging.info("防復原過程：完成窗口顯示恢復 [END]")
            return
        except Exception as e:
            logging.error(f"防復原發生未預期的異常: {e}", exc_info=True)
            return

    def detect_screen_recording(self):
        """檢查已知的錄屏/串流進程是否存在，返回 True 表示疑似錄屏中"""
        try:
            # 常見錄屏/串流程式名（小寫）
            suspects = [
                'obs64.exe', 'obs32.exe', 'obs.exe', 'streamlabs.exe', 'streamlabsobs.exe',
                'xsplit', 'xsplit.core.exe', 'bandicam.exe', 'fraps.exe', 'camtasiaStudio.exe',
                'dxtory.exe', 'action.exe', 'nvcamera.exe', 'nvspcap64.dll', 'gamesbar.exe',
                'gamebar.exe', 'xboxgamebar.exe', 'rtmp.exe'
            ]
            for proc in psutil.process_iter(['name']):
                name = (proc.info.get('name') or '').lower()
                if not name:
                    continue
                for s in suspects:
                    s_base = s.replace('.exe', '').lower()

                    if s_base in name or name == s.lower():
                        logging.info(f"偵測到可疑錄屏進程: {proc.info.get('name')}")
                        return True
        except Exception:
            pass
        return False

    def on_screen_recording_detected(self):
        """在主線程呼叫：發現錄屏時的處理 - 隱藏窗口"""
        logging.info("防錄屏處理：隱藏窗口 [START]")
        try:
            aimbot_signal.update_log.emit("⚠️ 偵測到錄屏/串流軟件，已隱藏窗口（按 Ctrl+Alt+P 恢復）")

            # 隱藏透明 overlay
            try:
                if hasattr(self, 'overlay') and self.overlay:
                    self.overlay.hide()
                    logging.info("防錄屏：已隱藏 overlay")
            except Exception as e:
                logging.warning(f"防錄屏：隱藏 overlay 失敗: {e}")

            # 直接隱藏窗口
            try:
                import ctypes
                user32 = ctypes.windll.user32
                SW_HIDE = 0
                
                # 隱藏 ModernModMenu（主菜單）和 MainController（按鈕）窗口
                target_classes = ['ModernModMenu', 'MainController']
                widget_count = 0
                
                # 先列出所有當前的頂層 widgets 進行調試
                all_widgets = QApplication.topLevelWidgets()
                logging.info(f"防錄屏：當前頂層 widgets 數量: {len(all_widgets)}")
                for idx, w in enumerate(all_widgets):
                    logging.info(f"防錄屏：  [{idx}] {w.__class__.__name__} - {w.objectName() if hasattr(w, 'objectName') else 'N/A'}")
                
                for w in all_widgets:
                    # 檢查窗口類型是否在目標清單中
                    if w.__class__.__name__ not in target_classes:
                        continue
                    
                    widget_count += 1
                    window_name = w.__class__.__name__
                    logging.info(f"防錄屏：開始隱藏 [{window_name}]...")
                    try:
                        hwnd = int(w.winId())
                        # Qt 隱藏
                        w.hide()
                        w.setVisible(False)
                        logging.info(f"防錄屏 [{window_name}]：Qt 隱藏成功")
                        
                        # Win32 隱藏
                        res = user32.ShowWindow(hwnd, SW_HIDE)
                        logging.info(f"防錄屏 [{window_name}]：ShowWindow(hwnd={hwnd}, SW_HIDE) 返回 {res}")
                        self.hidden_by_protection = True
                    except Exception as e:
                        logging.warning(f"防錄屏 [{window_name}]：隱藏失敗: {e}")
                
                if widget_count == 0:
                    logging.info("防錄屏：未找到 ModernModMenu 或 MainController 窗口")
                else:
                    logging.info(f"防錄屏：共隱藏 {widget_count} 個窗口")
                
                # 強制刷新 UI
                try:
                    QApplication.processEvents()
                    logging.info("防錄屏：已調用 QApplication.processEvents()")
                except Exception as e:
                    logging.warning(f"防錄屏：processEvents 失敗: {e}")
                
            except Exception as e:
                logging.error(f"防錄屏：隱藏窗口時發生異常: {e}")

            logging.info("防錄屏處理完成：窗口已隱藏 [END]")
            return
        except Exception as e:
            logging.error(f"防錄屏處理發生未預期的異常: {e}", exc_info=True)
            return
            try:
                import ctypes
                user32 = ctypes.windll.user32
                SW_HIDE = 0
                SWP_NOSIZE = 0x0001
                SWP_NOZORDER = 0x0004
                
                # 隱藏 ModernModMenu（主菜單）和 MainController（按鈕）
                target_classes = ['ModernModMenu', 'MainController']
                widget_count = 0
                
                # 先列出所有當前的頂層 widgets 進行調試
                all_widgets = QApplication.topLevelWidgets()
                logging.info(f"防錄屏：當前頂層 widgets 數量: {len(all_widgets)}")
                for idx, w in enumerate(all_widgets):
                    logging.info(f"防錄屏：  [{idx}] {w.__class__.__name__} - {w.objectName() if hasattr(w, 'objectName') else 'N/A'}")
                
                for w in all_widgets:
                    # 檢查窗口類型是否在目標清單中
                    if w.__class__.__name__ not in target_classes:
                        continue
                    
                    widget_count += 1
                    window_name = w.__class__.__name__
                    logging.info(f"防錄屏：開始隱藏 [{window_name}]...")
                    try:
                        # 1) Qt hide/minimize
                        try:
                            w.hide()
                            w.setWindowState(w.windowState() | Qt.WindowState.WindowMinimized)
                            w.setVisible(False)
                            logging.info(f"防錄屏 [{window_name}]：Qt 隱藏成功 (hide/minimize/setVisible)")
                        except Exception as e:
                            logging.warning(f"防錄屏 [{window_name}]：Qt 隱藏失敗: {e}")

                        # 2) Win32 ShowWindow
                        try:
                            hwnd = int(w.winId())
                            res = user32.ShowWindow(hwnd, SW_HIDE)
                            logging.info(f"防錄屏 [{window_name}]：ShowWindow(hwnd={hwnd}, SW_HIDE) 返回 {res}")
                        except Exception as e:
                            logging.warning(f"防錄屏 [{window_name}]：ShowWindow 失敗: {e}")
                    except Exception as e:
                        logging.warning(f"防錄屏 [{window_name}]：處理異常: {e}")
                
                if widget_count == 0:
                    logging.info("防錄屏：未找到 ModernModMenu 或 MainController 窗口")
                else:
                    logging.info(f"防錄屏：共隱藏 {widget_count} 個窗口")
                
                # 強制刷新 UI
                try:
                    QApplication.processEvents()
                    logging.info("防錄屏：已調用 QApplication.processEvents()")
                except Exception as e:
                    logging.warning(f"防錄屏：processEvents 失敗: {e}")
                
                # 標記已被保護隱藏
                self.hidden_by_protection = True
                logging.info("防錄屏：已設置 hidden_by_protection = True")
            except Exception as e:
                logging.error(f"防錄屏：使用 Win32 隐藏窗口時發生頂層異常: {e}")

            logging.info("防錄屏處理完成：所有頂層視窗嘗試隱藏（多重手段）[END]")
            return
        except Exception as e:
            logging.error(f"防錄屏處理發生未預期的異常: {e}", exc_info=True)
            return

    def screen_recording_monitor_loop(self):
        """背景執行緒：定時檢查是否有錄屏行為，發現則隱藏；消失則恢復"""
        logging.info("防錄屏監測線程已啟動")
        last_detected = False  # 記錄上一次檢測狀態
        while True:
            if getattr(self, '_stop_threads', False):
                logging.info("screen_recording_monitor_loop 停止標誌已設置，退出線程")
                break
            try:
                enabled = getattr(self, 'screen_protection_enabled', True)
                if not enabled:
                    # 如果使用者關閉防錄屏，短暫休息後繼續檢查設定
                    time.sleep(2)
                    continue

                # 每次迭代先記錄一次檢查行為（低頻）
                logging.debug("檢查錄屏進程中...")
                detected = self.detect_screen_recording()
                
                # 檢測狀態發生變化
                if detected and not last_detected:
                    # 剛偵測到錄屏 → 隱藏窗口
                    logging.info("防錄屏：偵測到錄屏程式，觸發隱藏")
                    try:
                        aimbot_signal.screen_recording_detected_signal.emit()
                        logging.info("防錄屏：隱藏信號已發射")
                    except Exception as e:
                        logging.error(f"防錄屏：隱藏信號發射失敗: {e}")
                        try:
                            if hasattr(self, 'on_screen_recording_detected'):
                                self.on_screen_recording_detected()
                        except Exception as e2:
                            logging.error(f"防錄屏：直接隱藏調用失敗: {e2}")
                    last_detected = True
                    time.sleep(2)  # 優化：檢測間隔改為 2 秒
                elif not detected and last_detected:
                    # 錄屏已停止 → 恢復窗口
                    logging.info("防錄屏：錄屏進程已消失，觸發恢復")
                    try:
                        # 發射恢復信號給主線程
                        aimbot_signal.restore_windows_signal.emit()
                        logging.info("防錄屏：恢復信號已發射")
                    except Exception as e:
                        logging.error(f"防錄屏：恢復信號發射失敗: {e}")
                    last_detected = False
                    time.sleep(1)  # 優化：恢復後等待 1 秒
                else:
                    # 未檢測到錄屏，定期掃描
                    time.sleep(3)  # 常態檢測間隔為 3 秒，平衡即時性與資源消耗
            except Exception as e:
                logging.warning(f"防錄屏監測線程錯誤: {e}")
                time.sleep(2)

    def hotkey_monitor_loop(self):
        """後台執行緒：監控全局熱鍵（Ctrl+Alt+P）來快速啟/禁防錄屏保護"""
        logging.info("全局熱鍵監控線程已啟動 (Ctrl+Alt+P 切換防錄屏)")
        while True:
            if getattr(self, '_stop_threads', False):
                logging.info("hotkey_monitor_loop 停止標誌已設置，退出線程")
                break
            try:
                # 使用 ctypes 監控按鍵狀態（Windows 環境）
                import ctypes
                user32 = ctypes.windll.user32
                
                # 虛擬鍵代碼
                VK_CONTROL = 0x11
                VK_MENU = 0x12  # Alt
                VK_P = 0x50
                
                # 檢查 Ctrl + Alt + P 是否同時按下
                ctrl_pressed = user32.GetAsyncKeyState(VK_CONTROL) < 0
                alt_pressed = user32.GetAsyncKeyState(VK_MENU) < 0
                p_pressed = user32.GetAsyncKeyState(VK_P) < 0
                
                if ctrl_pressed and alt_pressed and p_pressed:
                    # 按鍵組合被觸發
                    current_state = getattr(self, 'screen_protection_enabled', True)
                    new_state = not current_state
                    
                    self.screen_protection_enabled = new_state
                    try:
                        self.save_settings()
                    except Exception:
                        pass
                    
                    logging.info(f"熱鍵觸發：防錄屏保護已 {'啟用' if new_state else '禁用'} (Ctrl+Alt+P)")
                    aimbot_signal.update_log.emit(f"⌨️ 防錄屏已 {'啟用' if new_state else '禁用'}")
                    
                    # 如果剛啟用防錄屏，立即檢查是否正在錄屏
                    if new_state:
                        if self.detect_screen_recording():
                            logging.info("熱鍵激活：發現正在錄屏，準備隱藏")
                            aimbot_signal.screen_recording_detected_signal.emit()
                        else:
                            logging.info("熱鍵激活：未發現錄屏，已啟用防錄屏監控")
                    else:
                        # 剛禁用防錄屏，嘗試恢復窗口
                        if getattr(self, 'hidden_by_protection', False):
                            logging.info("熱鍵禁用：防錄屏已關閉，嘗試恢復窗口")
                            try:
                                aimbot_signal.restore_windows_signal.emit()
                                logging.info("熱鍵：恢復窗口信號已發射")
                            except Exception as e:
                                logging.error(f"熱鍵：恢復信號發射失敗: {e}")
                    
                    # 防止熱鍵連續觸發，等待一段時間
                    time.sleep(1)
                else:
                    time.sleep(0.2)  # 优化：从0.1改为0.2，减少热键检查频率,降低CPU占用
            except Exception as e:
                logging.debug(f"熱鍵監控錯誤: {e}")
                time.sleep(0.5)
    def refresh_account_data(self):
    # 1. 取得你登入時存檔的路徑
        auth_file = os.path.join(os.environ['APPDATA'], "cyrog_auth.dat")
        
        # 預設顯示 (如果沒讀到檔案)
        username = "未登入"
        user_key = "N/A"
        expiry = "N/A"
        key_status = "未驗證"
        remaining_days = "0"

        # 2. 嘗試讀取數據
        if os.path.exists(auth_file):
            try:
                with open(auth_file, "r") as f:
                    data = json.load(f)
                    username = data.get("account", "未知用戶")
                    user_key = data.get("key", "N/A")
                    expiry = data.get("expiry", "未知時間")
                    
                    # 3. 計算剩餘天數
                    exp_dt = datetime.strptime(expiry, "%Y-%m-%d %H:%M:%S")
                    now = datetime.now()
                    delta = exp_dt - now
                    remaining_days = max(0, delta.days)
                    
                    if delta.total_seconds() > 0:
                        key_status = f"<span style='color: #10b981;'>運作中 ({remaining_days} 天)</span>"
                    else:
                        key_status = "<span style='color: #f43f5e;'>已過期</span>"
            except Exception as e:
                logging.error(f"讀取授權檔失敗: {e}")
        
        latency = self.get_server_latency()
        
        # 根據延遲設定顏色
        if latency == -1:
            ping_color = "#f43f5e" # 紅色 (斷線)
            ping_text = "伺服器超時，請檢查網路連線"
        elif latency < 100:
            ping_color = "#10b981" # 綠色 (優良)
            ping_text = f"{latency} ms"
        else:
            ping_color = "#fbbf24" # 黃色 (一般)
            ping_text = f"{latency} ms"

        # 4. 返回動態 HTML
        return (
            f"<div style='line-height: 180%; font-family: \"Microsoft JhengHei\";'>"
            f"<b style='color: #38bdf8; font-size: 20px;'>💎 帳號資訊</b><br>"
            f"<hr style='border: 0.5px solid rgba(56, 189, 248, 0.3);'><br>"
            f"📡 <b>伺服器延遲：</b> <span style='color: {ping_color}; font-weight: bold;'>{ping_text}</span><br><br>"
            f"👤 <b>使用者：</b> {username}<br>"
            f"🔑 <b>卡密：</b> {user_key}<br>"
            f"🔑 <b>授權狀態：</b> {key_status}<br>"
            f"📅 <b>到期時間：</b> {expiry}<br>"
            f"🆔 <b>機器碼：</b> <span style='font-size: 11px; color: #94a3b8;'>{self.my_hwid[:16]}...</span>"
            f"<div style='background: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 8px; border-left: 4px solid #38bdf8;'>"
            f"🛡️ <b>安全提示：</b> 環境安全，防封已生效，核心已注入。"
            f"</div>"
        )
           
    def system_monitor_loop(self):
        """系統監測迴圈，定期更新 CPU/RAM 狀態。"""
        while True:
            if getattr(self, '_stop_threads', False):
                logging.info("system_monitor_loop 停止標誌已設置，退出線程")
                break
            try:
                cpu_usage = psutil.cpu_percent(interval=0.5)  #  改用 interval 參數而非依賴全局狀態
                msg = f"系統負載: CPU {cpu_usage}% | 引擎狀態: {'運作中' if self.aimbot_active else '待機'}"
                aimbot_signal.update_status.emit(msg)
            except Exception as e:
                logging.debug(f"系統監測異常: {e}")
            #  每 2 秒更新一次
            time.sleep(2)

    def load_background(self, image_path):
        """加載背景圖片"""
        if os.path.exists(image_path):
            self.bg_pixmap = QPixmap(image_path)
            if not self.bg_pixmap.isNull():
                logging.info(f"✅ 主菜單背景圖片已加載: {image_path}")
            else:
                self.bg_pixmap = None
        else:
            logging.info(f"ℹ️ 主菜單背景圖片不存在: {image_path} (使用默認顏色)")

    def initUI(self):
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(1100, 380)
        self.current_fps = 0
        
        # ⭐ 加载初始背景图片
        self.load_background("login_bg.png")

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer) # 強制使用高精度計時器
        self.timer.timeout.connect(self.update_announcement_time)
        self.timer.start(100)  # 优化：从1ms改为100ms，减少不必要的频繁更新
        
        # ⭐ 定义深色主题样式表
        self.stylesheet_dark = """
            /* 主容器 - 高级毛玻璃效果 */
            QFrame#MainContainer {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(20, 28, 42, 230),
                    stop:0.5 rgba(15, 24, 38, 230),
                    stop:1 rgba(10, 18, 30, 230));
                border: 2px solid rgba(56, 189, 248, 0.35);
                border-radius: 14px;
            }
            
            /* Logo 標題 - 优雅发光 */
            QLabel#LogoLabel {
                font-family: 'Microsoft YaHei UI';
                font-size: 16px;
                font-weight: 900;
                color: #38bdf8;
                letter-spacing: 3px;
                margin: 12px 0px;
                background: transparent;
            }
            
            /* 菜單按鈕 - 原始狀態 */
            QPushButton#MenuBtn {
                background-color: transparent;
                color: #94a3b8;
                border: none;
                padding: 10px 12px;
                text-align: left;
                font-family: 'Microsoft YaHei UI';
                font-size: 12px;
                font-weight: 600;
                border-radius: 6px;
                margin: 3px 6px;
                border-left: 3px solid transparent;
            }
            
            /* 菜單按鈕 - 懸停效果 */
            QPushButton#MenuBtn:hover {
                background-color: rgba(56, 189, 248, 0.15);
                color: #cbd5e1;
                border-left: 3px solid #38bdf8;
            }
            
            /* 菜單按鈕 - 按下效果 */
            QPushButton#MenuBtn:pressed {
                background-color: rgba(56, 189, 248, 0.25);
            }
            
            /* 活躍菜單按鈕 */
            QPushButton#ActiveBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(56, 189, 248, 0.2),
                    stop:1 rgba(56, 189, 248, 0.08));
                color: #0cf;
                font-family: 'Microsoft YaHei UI';
                font-size: 12px;
                font-weight: 700;
                border-left: 3px solid #38bdf8;
                border-radius: 6px;
                margin: 3px 6px;
                padding: 10px 12px;
                text-align: left;
            }
            
            /* 活躍按鈕 - 懸停 */
            QPushButton#ActiveBtn:hover {
                background: rgba(56, 189, 248, 0.2);
                color: #0cf;
                border-left: 3px solid #0ea5e9;
            }
            
            /* 頁面標題 */
            QLabel#PageTitle {
                font-family: 'Microsoft YaHei UI';
                font-size: 22px;
                font-weight: 900;
                color: #e0f2fe;
                margin-top: 5px;
                margin-bottom: 8px;
                background: transparent;
            }
            
            /* 分隔線 */
            QFrame#Separator {
                background: rgba(56, 189, 248, 0.4);
                max-height: 2px;
            }
            
            /* 普通文字 */
            QLabel#NormalText {
                color: #cbd5e1;
                font-family: 'Microsoft YaHei UI';
                font-size: 12px;
                background: transparent;
            }
            
            /* 複選框 */
            QCheckBox {
                color: #cbd5e1;
                font-family: 'Microsoft YaHei UI';
                spacing: 8px;
                background: transparent;
                margin: 2px 0px;
                font-size: 12px;
            }
            
            QCheckBox:hover {
                color: #e0f2fe;
            }
            
            /* 複選框指示器 */
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 2px solid #475569;
                background-color: rgba(15, 20, 30, 0.3);
            }
            
            QCheckBox::indicator:hover {
                border: 2px solid #38bdf8;
                background-color: rgba(56, 189, 248, 0.12);
            }
            
            /* 複選框選中狀態 */
            QCheckBox::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0284c7, stop:1 #38bdf8);
                border: 2px solid #38bdf8;
            }
            
            /* 水平滑塊 - 槽 */
            QSlider::groove:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(30, 41, 59, 0.6),
                    stop:1 rgba(15, 20, 30, 0.6));
                height: 5px;
                border-radius: 3px;
                border: 1px solid rgba(56, 189, 248, 0.15);
            }
            
            /* 水平滑塊 - 把手 */
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0284c7, stop:1 #38bdf8);
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
                border: 1px solid #0284c7;
            }
            
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #06b6d4, stop:1 #0cf);
                border: 1px solid #06b6d4;
            }
            
            /* 日誌輸出框 */
            QTextEdit#LogOutput {
                background-color: rgba(15, 20, 30, 0.7);
                color: #00ffd5;
                border: 1px solid rgba(56, 189, 248, 0.25);
                border-radius: 6px;
                padding: 8px;
                font-family: 'Consolas';
                font-size: 11px;
                selection-background-color: rgba(56, 189, 248, 0.3);
            }
            
            QTextEdit#LogOutput:focus {
                border: 1px solid #38bdf8;
                background-color: rgba(15, 20, 30, 0.85);
            }
            
            /* 狀態標籤 */
            QLabel#StatusLabel {
                color: #e0f2fe;
                font-family: 'Microsoft YaHei UI';
                font-size: 12px;
                font-weight: bold;
                padding: 6px 10px;
                background: rgba(56, 189, 248, 0.12);
                border: 1px solid rgba(56, 189, 248, 0.3);
                border-radius: 4px;
                margin: 2px 0px;
            }
            
            /* 單選按鈕 */
            QRadioButton {
                color: #cbd5e1;
                font-family: 'Microsoft YaHei UI';
                spacing: 8px;
                background: transparent;
                font-size: 12px;
                margin: 2px 0px;
            }
            
            QRadioButton:hover {
                color: #e0f2fe;
            }
            
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border-radius: 8px;
                border: 2px solid #475569;
                background-color: rgba(15, 20, 30, 0.3);
            }
            
            QRadioButton::indicator:hover {
                border: 2px solid #38bdf8;
                background-color: rgba(56, 189, 248, 0.12);
            }
            
            QRadioButton::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0284c7, stop:1 #38bdf8);
                border: 2px solid #38bdf8;
            }
        """
        
        self.setStyleSheet(self.stylesheet_dark)
        
        # ⭐ 定义浅色主题样式表
        self.stylesheet_light = """
            /* 主容器 - 浅色主题 */
            QFrame#MainContainer {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(241, 245, 249, 230),
                    stop:0.5 rgba(226, 232, 240, 230),
                    stop:1 rgba(219, 226, 236, 230));
                border: 2px solid rgba(14, 165, 233, 0.35);
                border-radius: 14px;
            }
            QLabel#LogoLabel {
                font-family: 'Microsoft YaHei UI';
                font-size: 16px;
                font-weight: 900;
                color: #0ea5e9;
                letter-spacing: 3px;
                margin: 12px 0px;
                background: transparent;
            }
            QLabel#PageTitle {
                font-family: 'Microsoft YaHei UI';
                font-size: 22px;
                font-weight: 900;
                color: #1e293b;
                margin-top: 5px;
                margin-bottom: 8px;
                background: transparent;
            }
            QLabel#NormalText {
                color: #475569;
                font-family: 'Microsoft YaHei UI';
                font-size: 12px;
                background: transparent;
            }
            QPushButton#MenuBtn {
                background-color: transparent;
                color: #64748b;
                border: none;
                padding: 10px 12px;
                text-align: left;
                font-family: 'Microsoft YaHei UI';
                font-size: 12px;
                font-weight: 600;
                border-radius: 6px;
                margin: 3px 6px;
                border-left: 3px solid transparent;
            }
            QPushButton#MenuBtn:hover {
                background-color: rgba(14, 165, 233, 0.15);
                color: #334155;
                border-left: 3px solid #0ea5e9;
            }
            QPushButton#ActiveBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(14, 165, 233, 0.2),
                    stop:1 rgba(14, 165, 233, 0.08));
                color: #0284c7;
                font-family: 'Microsoft YaHei UI';
                font-size: 12px;
                font-weight: 700;
                border-left: 3px solid #0ea5e9;
                border-radius: 6px;
                margin: 3px 6px;
                padding: 10px 12px;
                text-align: left;
            }
        """

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.container = QFrame(); self.container.setObjectName("MainContainer")
        self.main_layout.addWidget(self.container)
        self.content_hlayout = QHBoxLayout(self.container)

        self.side_bar = QVBoxLayout()
        self.logo = QLabel("XUANS"); self.logo.setObjectName("LogoLabel"); self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.side_bar.addWidget(self.logo)

        self.nav_buttons = []
        # ⭐ 菜单图标定义
        self.menu_icons = ["📊", "📢", "👤", "👁️", "🎯",  "⚡", "🎵", "⚙️", ]
        self.menu_shortcuts = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",]
        self.menus = ["系統狀況", "系统公告","帳號", "视觉辅助", "自动瞄准", "內核優化", "音樂中心", "设置中心", ]
        
        for i, text in enumerate(self.menus):
            # ⭐ 按钮文本添加图标和快捷键提示
            icon = self.menu_icons[i] if i < len(self.menu_icons) else "•"
            shortcut = self.menu_shortcuts[i] if i < len(self.menu_shortcuts) else ""
            btn = QPushButton(f"{icon} {text}")
            btn.setToolTip(f"快捷键: {shortcut}")
            btn.setObjectName("ActiveBtn" if i == 0 else "MenuBtn")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, idx=i: self.switch_page(idx))
            self.side_bar.addWidget(btn)
            self.nav_buttons.append(btn)
        
        self.side_bar.addStretch()
        exit_btn = QPushButton("🚪 安全退出")
        exit_btn.setStyleSheet("color: #f43f5e; font-weight: 900; background: transparent; padding: 10px;")
        exit_btn.clicked.connect(self.close)
        self.side_bar.addWidget(exit_btn)
        self.content_hlayout.addLayout(self.side_bar, 1)
        
        # ⭐ 右侧内容区域（主窗口 + 状态栏）
        right_layout = QVBoxLayout()
        self.stack = QStackedWidget()
        right_layout.addWidget(self.stack, 1)
        
        # ⭐ 添加状态栏
        status_bar = QFrame()
        status_bar.setObjectName("StatusBar")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(8, 4, 8, 4)
        
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setObjectName("NormalText")
        self.fps_label.setStyleSheet("color: #38bdf8; font-weight: 600; font-size: 11px;")
        
        self.ping_label = QLabel("🌐 Ping: -- ms")
        self.ping_label.setObjectName("NormalText")
        self.ping_label.setStyleSheet("color: #38bdf8; font-weight: 600; font-size: 11px;")
        
        self.cpu_label = QLabel("💻 CPU: --%")
        self.cpu_label.setObjectName("NormalText")
        self.cpu_label.setStyleSheet("color: #38bdf8; font-weight: 600; font-size: 11px;")
        
        self.memory_label = QLabel("🧠 RAM: --%")
        self.memory_label.setObjectName("NormalText")
        self.memory_label.setStyleSheet("color: #38bdf8; font-weight: 600; font-size: 11px;")
        
        status_layout.addWidget(self.fps_label)
        status_layout.addWidget(QFrame())  # 分隔符
        status_layout.addWidget(self.ping_label)
        status_layout.addWidget(QFrame())  # 分隔符
        status_layout.addWidget(self.cpu_label)
        status_layout.addWidget(QFrame())  # 分隔符
        status_layout.addWidget(self.memory_label)
        status_layout.addStretch()
        
        # ⭐ 主题切换和控制按钮
        self.theme_btn = QPushButton("🌙")
        self.theme_btn.setMaximumWidth(30)
        self.theme_btn.setToolTip("切换主题 (亮/暗)")
        self.theme_btn.clicked.connect(self.toggle_theme)
        self.theme_btn.setStyleSheet("background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 4px;")
        
        minimize_btn = QPushButton("−")
        minimize_btn.setMaximumWidth(30)
        minimize_btn.setToolTip("最小化")
        minimize_btn.clicked.connect(self.showMinimized)
        minimize_btn.setStyleSheet("background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 4px;")
        
        close_btn = QPushButton("✕")
        close_btn.setMaximumWidth(30)
        close_btn.setToolTip("关闭")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 4px;")
        
        status_layout.addWidget(self.theme_btn)
        status_layout.addWidget(minimize_btn)
        status_layout.addWidget(close_btn)
        
        right_layout.addWidget(status_bar)
        self.content_hlayout.addLayout(right_layout, 4)
        
        self.init_pages()
        
        # ⭐ 启动状态栏更新定时器
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status_bar)
        self.status_timer.start(1000)  # 优化：改为1000ms，减少CPU占用和刷新频率
        
        self.sizegrip = QSizeGrip(self)
        # --- 計時器設定---
        self.timer = QTimer(self) 
        self.timer.timeout.connect(self.update_announcement_time) 
        self.timer.start(100)  # 优化：从8ms改为100ms（约10fps），减少CPU占用
        self.latency_label = QLabel("Ping: -- ms", self)
        self.latency_label.setStyleSheet("color: rgba(255,255,255,0.5); font-size: 10px;")
        self.latency_label.move(880, 590) # 放在右下角
    
    def update_ping(self):
        ping = self.get_server_latency()
        self.latency_label.setText(f"Ping: {ping} ms")

    def update_announcement_time(self):
        # 1. 取得最新時間
        current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        
       # 2. 計算 UI 刷新 FPS
        if not hasattr(self, '_last_ui_time'):
            self._last_ui_time = time.time()
            self.ui_fps = 0
        else:
            now = time.time()
            delta = now - self._last_ui_time
            if delta > 0:
                # 計算的是 GUI 刷新的頻率
                self.ui_fps = 1.0 / delta
            self._last_ui_time = now
            # 根據 FPS 數值改變顏色
        if self.ui_fps >= 120:
            fps_color = "#a855f7"  # 紫色 (極佳)
        elif self.ui_fps >= 60:
            fps_color = "#22c55e" # 黃色 (普通)
        else:
            fps_color = "#f43f5e" # 紅色 (低幀)
        # 4. 只有當目前分頁是「系统公告」時才更新
        if self.stack.currentIndex() == 0:
            msg_text = (   f"<b>[偵測幀率]</b>：<span style='color: {fps_color};'>{self.ui_fps:.1f} FPS</span><br><br>"
                           f"<b>[目前時間]</b>：<span style='color: #38bdf8;'>{current_time}</span><br>"
                            "<b>[更新]</b>：更新時間:2026-1-15 22:10:58 <br>"
                            "<b>[更新]</b>：目前版本 Beta//v1.0.0<br>"
                            "<b>[狀態]</b>：核心引擎正常<br><br>"
                            "<b>[更新]</b>：新增了LOG日志<br>"
                            "<b>[更新]</b>：修復了少部分人開啟自瞄閃退的問題<br>"
                            "<b>[提示]</b>：如遇異常請重啟程式或聯繫客服支援 <br><br>"
                            "************************************************************************<br>"
                            "<b>[使用教程]</b>：先關閉所有自瞄功能在開懸浮窗，在開所有自瞄功能<br><br>"
                            "<b>[使用教程]</b>：直播前請開啟防錄屏並重啟輔助。<br><br>"
                            "如有任何問題請聯繫客服TG：cy-support<br>")
            if hasattr(self, 'announcement_label'):
                self.announcement_label.setText(msg_text)
    def init_pages(self):
        for i, text in enumerate(self.menus):
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(40, 20, 50, 20)
            layout.setSpacing(10)

            # 標題與分割線
            title = QLabel(text)
            title.setObjectName("PageTitle")
            sep = QFrame()
            sep.setObjectName("Separator")
            sep.setFixedWidth(200)
            layout.addWidget(title)
            layout.addWidget(sep)
            # 根據不同的頁面名稱，呼叫對應的設定函式
            if text == "系統狀況":
                self.setup_system_ui(layout)
            elif text == "自动瞄准":
                self.setup_aimbot_ui(layout)
            elif text == "视觉辅助":
                self.setup_ESP_ui(layout)
            elif text == "音樂中心":
                self.setup_music_ui(layout)
            elif text == "设置中心":
                self.setup_Setting_ui(layout)
            elif text == "內核優化":
                self.setup_kernel_optimization_ui(layout)
            elif text == "帳號":
                # ---直接定義---
                self.account_info_label = QLabel("正在加載帳號資訊請稍候...\n 正在加載防封模組... "); 
                self.account_info_label.setObjectName("NormalText")
                self.account_info_label.setWordWrap(True)
                self.account_info_label.setStyleSheet("background: transparent;")
                layout.addWidget(self.account_info_label)
            elif text == "系统公告":
                # 添加初始化按鈕
                self.init_decrypt_btn = QPushButton("初始化")
                self.init_decrypt_btn.setFixedHeight(60)
                self.init_decrypt_btn.setFixedWidth(250)
                self.init_decrypt_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                self.init_decrypt_btn.setStyleSheet("""
                    QPushButton {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #10B981, stop:1 #059669);
                        color: white;
                        font-weight: bold;
                        font-size: 16px;
                        border: none;
                        border-radius: 8px;
                        padding: 10px;
                    }
                    QPushButton:hover {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #059669, stop:1 #047857);
                    }
                    QPushButton:pressed {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #047857, stop:1 #065F46);
                    }
                """)
                self.decrypted = False
                self.has_initialized = False  # ✅ 添加初始化狀態追踪
                self.init_decrypt_btn.clicked.connect(self.on_decrypt_clicked)
                layout.addWidget(self.init_decrypt_btn)
                
                current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
                msg_text = (f"<b>[目前時間]</b>：{current_time}<br>"
                            "<b>[更新]</b>：更新時間:2026-1-15 22:10:58 <br>"
                            "<b>[更新]</b>：目前版本 Beta//v1.0.0<br>"
                            "<b>[狀態]</b>：核心引擎正常<br><br>"
                            "<b>[更新]</b>：修復了4K屏幕下字體顯示太小和信號與DPI報錯的問題<br>"
                            "<b>[更新]</b>：修復了少部分人開啟自瞄閃退的問題<br>"
                            "<b>[警告]</b>：直播前請開啟防錄屏並重啟輔助。<br><br>"
                            "<b>[提示]</b>：如遇異常請重啟程式或聯繫客服支援 <br><br>"
                            "************************************************************************<br>"
                            "<b>[使用教程]</b>：先關閉所有自瞄功能在開懸浮窗，在開所有自瞄功能<br><br>"
                            "如有任何問題請聯繫客服TG：cy-support<br>"


                            "<b>[聲明]</b>：任何沒看公告的使用者將自行承擔風險。<br><br>"
                            "使用者需自行承擔使用本軟體所帶來的風險，開發團隊不對任何因使用本軟體而導致的損失負責。<br><br>"
                            "<b>感謝您的支持與理解！</b><br>"
                            )
                self.announcement_label = QLabel(msg_text) 
                self.announcement_label.setObjectName("NormalText")
                self.announcement_label.setWordWrap(True)
                layout.addWidget(self.announcement_label)
            elif text == "关于我们":
                msg_text = ("<b>[關於我們]</b>：cy 團隊致力於打造高效穩定的遊戲輔助工具。<br><br>"
                            "電子郵件：cyrog120hz@gmail.com<br>"
                            "官方網站：www.cy-beta.com<br><br>"
                            "作者TG：fackAB<br>"
                            "官方TG：cy-TRACK_beta<br>"
                            "如有任何問題請聯繫客服TG：cy-support<br>"
                            "************************************************************************<br>"
                            "<b>[聲明]</b>：本軟體僅供學術研究與技術交流使用，嚴禁用於任何商業用途或違反遊戲規則的行為。<br><br>"
                            "使用者需自行承擔使用本軟體所帶來的風險，開發團隊不對任何因使用本軟體而導致的損失負責。<br><br>"
                            "<b>感謝您的支持與理解！</b><br>"
                            
                            )
                msg = QLabel(msg_text); msg.setObjectName("NormalText"); msg.setWordWrap(True)
                layout.addWidget(msg)
            
            layout.addStretch()
            self.stack.addWidget(page)

    def setup_kernel_optimization_ui(self, layout):
        """內核優化設置UI - 性能、檢測、追蹤、內存、硬件"""
        # 主滾動區域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; } QScrollBar { width: 8px; background: rgba(0,0,0,0); } QScrollBar::handle { background: rgba(56, 189, 248, 0.35); border-radius: 4px; }")
        
        scroll_widget = QWidget()
        scroll_widget.setObjectName("kernel_scroll_widget")
        scroll_widget.setStyleSheet("#kernel_scroll_widget { background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(6,10,18,255), stop:1 rgba(12,18,28,255)); }")
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(18)
        scroll_layout.setContentsMargins(12, 12, 12, 12)
        # 统一样式定义
        section_style = (
            "QGroupBox { color: #cfeefe; font-weight: bold; "
            "background: rgba(8, 15, 24, 180); border: 1px solid rgba(56,189,248,0.12); "
            "border-radius: 10px; padding-top: 12px; margin-top: 10px; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; font-size: 12px; }"
            "QLabel { color: #9fd7ff; }"
        )
        label_style = "color: #9fd7ff; font-weight: 600;"
        desc_style = "color: #93C5FD; font-size: 11px; font-style: italic; margin-left:8px;"

        # Presets 下拉（示例實例）
        presets_row = QHBoxLayout()
        presets_label = QLabel("預設配置:")
        presets_label.setStyleSheet(label_style)
        presets_combo = QComboBox()
        presets_combo.addItems(["低端設備 (Low)", "中端設備 (Medium)", "高端設備 (High)"])
        presets_combo.setCurrentIndex(1)
        presets_combo.setStyleSheet("QComboBox { background: rgba(20,28,38,220); color: #9fd7ff; border: 1px solid rgba(56,189,248,0.12); border-radius: 6px; padding: 6px; }")
        presets_row.addWidget(presets_label)
        presets_row.addWidget(presets_combo)
        presets_row.addStretch()
        scroll_layout.addLayout(presets_row)
        
        # ============ 1️⃣ 性能優化部分 ============
        perf_group = QGroupBox("⚙️ 性能優化")
        perf_group.setStyleSheet(section_style)
        perf_group.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        perf_layout = QVBoxLayout()
        perf_layout.setSpacing(12)
        
        # CPU 優先級
        cpu_prio_layout = QHBoxLayout()
        cpu_prio_label = QLabel("CPU 優先級:")
        cpu_prio_label.setStyleSheet(label_style)
        cpu_prio_label.setFixedWidth(130)
        cpu_prio_combo = QComboBox()
        cpu_prio_combo.addItems(["🔴 低", "🟡 中", "🟢 高"])
        cpu_prio_combo.setCurrentIndex(1)
        cpu_prio_combo.setStyleSheet("QComboBox { background: rgba(30, 41, 59, 200); color: #0cf; border: 1px solid rgba(56, 189, 248, 0.5); border-radius: 5px; padding: 5px; font-weight: bold; } QComboBox::drop-down { border: none; }")
        cpu_prio_layout.addWidget(cpu_prio_label)
        cpu_prio_layout.addWidget(cpu_prio_combo)
        cpu_prio_layout.addStretch()
        perf_layout.addLayout(cpu_prio_layout)
        
        # GPU 顯存優化
        gpu_opt_layout = QHBoxLayout()
        gpu_opt_label = QLabel("GPU 顯存優化:")
        gpu_opt_label.setStyleSheet(label_style)
        gpu_opt_label.setFixedWidth(130)
        gpu_opt_cb = QCheckBox("啟用極限優化")
        gpu_opt_cb.setChecked(True)
        gpu_opt_cb.setStyleSheet("QCheckBox { color: #0cf; font-weight: bold; }")
        gpu_opt_layout.addWidget(gpu_opt_label)
        gpu_opt_layout.addWidget(gpu_opt_cb)
        gpu_opt_layout.addStretch()
        perf_layout.addLayout(gpu_opt_layout)
        
        # 線程數設置
        thread_layout = QHBoxLayout()
        thread_label = QLabel("工作線程數:")
        thread_label.setStyleSheet(label_style)
        thread_label.setFixedWidth(130)
        thread_slider = QSlider(Qt.Orientation.Horizontal)
        thread_slider.setRange(1, 16)
        thread_slider.setValue(8)
        thread_slider.setStyleSheet("QSlider::groove:horizontal { background: #111827; height: 8px; border-radius: 4px; } QSlider::handle:horizontal { background: #F97316; width: 16px; margin: -6px 0; border-radius: 8px; border: 2px solid #D97706; } QSlider::handle:horizontal:hover { border: 2px solid rgba(217,119,6,0.9); }")
        thread_value = QLabel("8")
        thread_value.setStyleSheet("color: #F97316; font-weight: bold; min-width: 30px; qproperty-alignment: AlignCenter;")
        thread_slider.valueChanged.connect(lambda v: thread_value.setText(str(v)))
        thread_layout.addWidget(thread_label)
        thread_layout.addWidget(thread_slider)
        thread_layout.addWidget(thread_value)
        perf_layout.addLayout(thread_layout)
        
        # 緩存大小
        cache_layout = QHBoxLayout()
        cache_label = QLabel("緩存大小:")
        cache_label.setStyleSheet(label_style)
        cache_label.setFixedWidth(130)
        cache_slider = QSlider(Qt.Orientation.Horizontal)
        cache_slider.setRange(128, 2048)
        cache_slider.setValue(512)
        cache_slider.setSingleStep(128)
        cache_slider.setStyleSheet("QSlider::groove:horizontal { background: #111827; height: 8px; border-radius: 4px; } QSlider::handle:horizontal { background: #EC4899; width: 16px; margin: -6px 0; border-radius: 8px; border: 2px solid #BE185D; } QSlider::handle:horizontal:hover { border: 2px solid rgba(190,24,93,0.9); }")
        cache_value = QLabel("512 MB")
        cache_value.setStyleSheet("color: #EC4899; font-weight: bold; min-width: 60px; qproperty-alignment: AlignCenter;")
        cache_slider.valueChanged.connect(lambda v: cache_value.setText(f"{v} MB"))
        cache_layout.addWidget(cache_label)
        cache_layout.addWidget(cache_slider)
        cache_layout.addWidget(cache_value)
        perf_layout.addLayout(cache_layout)
        perf_group.setLayout(perf_layout)
        # 簡短說明與陰影效果提升可視層次
        perf_desc = QLabel("調整性能參數以平衡流暢度與精度。")
        perf_desc.setStyleSheet(desc_style)
        perf_layout.insertWidget(0, perf_desc)
        perf_group.setLayout(perf_layout)
        perf_group.setToolTip("性能相關設置：CPU / 線程 / 緩存等，影響整體響應與FPS。")
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(14)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(14, 165, 233, 70))
        perf_group.setGraphicsEffect(shadow)
        # 控件提示
        cpu_prio_combo.setToolTip("選擇CPU調度優先級 — 高會提升性能但可能影響系統響應。")
        gpu_opt_cb.setToolTip("啟用顯存優化以減少RAM使用。")
        thread_slider.setToolTip("設置工作線程數（越多越耗資源，但可提速）。")
        cache_slider.setToolTip("緩存大小會影響內存佔用與響應速度，單位MB。")
        scroll_layout.addWidget(perf_group)

        # 保存控件參考到 self 以便回調使用
        self.k_cpu_prio = cpu_prio_combo
        self.k_gpu_opt = gpu_opt_cb
        self.k_thread = thread_slider
        self.k_cache = cache_slider
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setFixedHeight(10)
        sep1.setStyleSheet("background: transparent;")
        scroll_layout.addWidget(sep1)
        
        # ============ 2️⃣ 檢測模式部分 ============
        detect_group = QGroupBox("🔍 檢測模式")
        detect_group.setStyleSheet(section_style)
        detect_group.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        detect_layout = QVBoxLayout()
        detect_layout.setSpacing(12)
        
        target_layout = QHBoxLayout()
        target_label = QLabel("目標檢測:")
        target_label.setFixedWidth(130)
        target_rb1 = QRadioButton("🎯 單目標 (快速)")
        target_rb2 = QRadioButton("👥 多目標 (精準)")
        target_rb1.setChecked(True)
        target_rb1.setStyleSheet("QRadioButton { color: #0cf; font-weight: bold; }")
        target_rb2.setStyleSheet("QRadioButton { color: #0cf; font-weight: bold; }")
        target_layout.addWidget(target_label)
        target_layout.addWidget(target_rb1)
        target_layout.addWidget(target_rb2)
        target_layout.addStretch()
        detect_layout.addLayout(target_layout)
        
        precision_layout = QHBoxLayout()
        precision_label = QLabel("檢測精度:")
        precision_label.setFixedWidth(130)
        precision_rb1 = QRadioButton("⚡ 快速檢測")
        precision_rb2 = QRadioButton("🎯 標準模式")
        precision_rb3 = QRadioButton("🔬 精準檢測")
        precision_rb2.setChecked(True)
        for rb in [precision_rb1, precision_rb2, precision_rb3]:
            rb.setStyleSheet("QRadioButton { color: #0cf; font-weight: bold; }")
        precision_layout.addWidget(precision_label)
        precision_layout.addWidget(precision_rb1)
        precision_layout.addWidget(precision_rb2)
        precision_layout.addWidget(precision_rb3)
        precision_layout.addStretch()
        detect_layout.addLayout(precision_layout)
        detect_group.setLayout(detect_layout)
        detect_desc = QLabel("選擇檢測目標與精度，根據場景權衡速度與準確率。")
        detect_desc.setStyleSheet(desc_style)
        detect_layout.insertWidget(0, detect_desc)
        detect_group.setToolTip("檢測模式：單目標快速或多目標精準。")
        shadow2 = QGraphicsDropShadowEffect()
        shadow2.setBlurRadius(12)
        shadow2.setOffset(0, 3)
        shadow2.setColor(QColor(56, 189, 248, 60))
        detect_group.setGraphicsEffect(shadow2)
        scroll_layout.addWidget(detect_group)

        self.k_target_single = target_rb1
        self.k_target_multi = target_rb2
        self.k_precision_fast = precision_rb1
        self.k_precision_standard = precision_rb2
        self.k_precision_precise = precision_rb3
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setFixedHeight(8)
        sep2.setStyleSheet("background: transparent;")
        scroll_layout.addWidget(sep2)
        
        # ============ 3️⃣ 追蹤算法部分 ============
        track_group = QGroupBox("🔄 追蹤算法")
        track_group.setStyleSheet(section_style)
        track_group.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        track_layout = QVBoxLayout()
        track_layout.setSpacing(12)
        
        algo_layout = QHBoxLayout()
        algo_label = QLabel("算法選擇:")
        algo_label.setFixedWidth(130)
        algo_rb1 = QRadioButton("Kalman 濾波")
        algo_rb2 = QRadioButton("光流法")
        algo_rb3 = QRadioButton("混合算法")
        algo_rb1.setChecked(True)
        for rb in [algo_rb1, algo_rb2, algo_rb3]:
            rb.setStyleSheet("QRadioButton { color: #0cf; font-weight: bold; }")
        algo_layout.addWidget(algo_label)
        algo_layout.addWidget(algo_rb1)
        algo_layout.addWidget(algo_rb2)
        algo_layout.addWidget(algo_rb3)
        algo_layout.addStretch()
        track_layout.addLayout(algo_layout)
        
        sensitive_layout = QHBoxLayout()
        sensitive_label = QLabel("追蹤敏感度:")
        sensitive_label.setFixedWidth(130)
        sensitive_slider = QSlider(Qt.Orientation.Horizontal)
        sensitive_slider.setRange(1, 100)
        sensitive_slider.setValue(50)
        sensitive_slider.setStyleSheet("QSlider::groove:horizontal { background: #111827; height: 8px; border-radius: 4px; } QSlider::handle:horizontal { background: #8B5CF6; width: 16px; margin: -6px 0; border-radius: 8px; border: 2px solid #7C3AED; } QSlider::handle:horizontal:hover { border: 2px solid rgba(124,58,237,0.9); }")
        sensitive_value = QLabel("50%")
        sensitive_value.setStyleSheet("color: #8B5CF6; font-weight: bold; min-width: 48px; qproperty-alignment: AlignCenter;")
        sensitive_slider.valueChanged.connect(lambda v: sensitive_value.setText(f"{v}%"))
        sensitive_layout.addWidget(sensitive_label)
        sensitive_layout.addWidget(sensitive_slider)
        sensitive_layout.addWidget(sensitive_value)
        track_layout.addLayout(sensitive_layout)
        track_group.setLayout(track_layout)
        track_desc = QLabel("選擇追蹤算法並調整敏感度，適配不同運動模式。")
        track_desc.setStyleSheet(desc_style)
        track_layout.insertWidget(0, track_desc)
        track_group.setToolTip("追蹤算法：Kalman（穩定）、光流（敏捷）、混合（兼顧）。")
        shadow3 = QGraphicsDropShadowEffect()
        shadow3.setBlurRadius(12)
        shadow3.setOffset(0, 3)
        shadow3.setColor(QColor(139, 92, 246, 60))
        track_group.setGraphicsEffect(shadow3)
        scroll_layout.addWidget(track_group)

        self.k_algo_kalman = algo_rb1
        self.k_algo_of = algo_rb2
        self.k_algo_hybrid = algo_rb3
        self.k_sensitivity = sensitive_slider
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.HLine)
        sep3.setFixedHeight(8)
        sep3.setStyleSheet("background: transparent;")
        scroll_layout.addWidget(sep3)
        
        # ============ 4️⃣ 內存管理部分 ============
        mem_group = QGroupBox("🧠 內存管理")
        mem_group.setStyleSheet(section_style)
        mem_group.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        mem_layout = QVBoxLayout()
        mem_layout.setSpacing(12)
        
        preload_layout = QHBoxLayout()
        preload_label = QLabel("內存預加載:")
        preload_label.setFixedWidth(130)
        preload_cb = QCheckBox("啟用預加載")
        preload_cb.setChecked(True)
        preload_cb.setStyleSheet("QCheckBox { color: #0cf; font-weight: bold; }")
        preload_layout.addWidget(preload_label)
        preload_layout.addWidget(preload_cb)
        preload_layout.addStretch()
        mem_layout.addLayout(preload_layout)
        
        precache_layout = QHBoxLayout()
        precache_label = QLabel("預緩存大小:")
        precache_label.setFixedWidth(130)
        precache_slider = QSlider(Qt.Orientation.Horizontal)
        precache_slider.setRange(256, 4096)
        precache_slider.setValue(1024)
        precache_slider.setSingleStep(256)
        precache_slider.setStyleSheet("QSlider::groove:horizontal { background: #111827; height: 8px; border-radius: 4px; } QSlider::handle:horizontal { background: #06B6D4; width: 16px; margin: -6px 0; border-radius: 8px; border: 2px solid #0891B2; } QSlider::handle:horizontal:hover { border: 2px solid rgba(8,145,178,0.9); }")
        precache_value = QLabel("1024 MB")
        precache_value.setStyleSheet("color: #06B6D4; font-weight: bold; min-width: 68px; qproperty-alignment: AlignCenter;")
        precache_slider.valueChanged.connect(lambda v: precache_value.setText(f"{v} MB"))
        precache_layout.addWidget(precache_label)
        precache_layout.addWidget(precache_slider)
        precache_layout.addWidget(precache_value)
        mem_layout.addLayout(precache_layout)
        
        gc_layout = QHBoxLayout()
        gc_label = QLabel("GC 間隔:")
        gc_label.setFixedWidth(130)
        gc_slider = QSlider(Qt.Orientation.Horizontal)
        gc_slider.setRange(100, 10000)
        gc_slider.setValue(2000)
        gc_slider.setSingleStep(100)
        gc_slider.setStyleSheet("QSlider::groove:horizontal { background: #111827; height: 8px; border-radius: 4px; } QSlider::handle:horizontal { background: #10B981; width: 16px; margin: -6px 0; border-radius: 8px; border: 2px solid #059669; } QSlider::handle:horizontal:hover { border: 2px solid rgba(16,185,129,0.9); }")
        gc_value = QLabel("2000 ms")
        gc_value.setStyleSheet("color: #10B981; font-weight: bold; min-width: 68px; qproperty-alignment: AlignCenter;")
        gc_slider.valueChanged.connect(lambda v: gc_value.setText(f"{v} ms"))
        gc_layout.addWidget(gc_label)
        gc_layout.addWidget(gc_slider)
        gc_layout.addWidget(gc_value)
        mem_layout.addLayout(gc_layout)
        mem_group.setLayout(mem_layout)
        mem_desc = QLabel("內存預加載與GC頻率控制，可以在性能與內存占用間取捨。")
        mem_desc.setStyleSheet(desc_style)
        mem_layout.insertWidget(0, mem_desc)
        mem_group.setToolTip("內存管理：預加載與預緩存影響啟動與持續占用。")
        shadow4 = QGraphicsDropShadowEffect()
        shadow4.setBlurRadius(12)
        shadow4.setOffset(0, 3)
        shadow4.setColor(QColor(6, 182, 212, 60))
        mem_group.setGraphicsEffect(shadow4)
        scroll_layout.addWidget(mem_group)

        self.k_preload = preload_cb
        self.k_precache = precache_slider
        self.k_gc = gc_slider
        sep4 = QFrame()
        sep4.setFrameShape(QFrame.HLine)
        sep4.setFixedHeight(8)
        sep4.setStyleSheet("background: transparent;")
        scroll_layout.addWidget(sep4)
        
        # ============ 5️⃣ 硬件配置部分 ============
        hw_group = QGroupBox("🖥️ 硬件配置")
        hw_group.setStyleSheet(section_style)
        hw_group.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        hw_layout = QVBoxLayout()
        hw_layout.setSpacing(12)
        
        gpu_layout = QHBoxLayout()
        gpu_label = QLabel("GPU 選擇:")
        gpu_label.setFixedWidth(130)
        gpu_combo = QComboBox()
        gpu_combo.addItems(["🟢 GPU 0 (推薦)", "🔵 GPU 1", "🟡 CPU"])
        gpu_combo.setCurrentIndex(0)
        gpu_combo.setStyleSheet("QComboBox { background: rgba(30, 41, 59, 200); color: #0cf; border: 1px solid rgba(56, 189, 248, 0.5); border-radius: 5px; padding: 5px; font-weight: bold; } QComboBox::drop-down { border: none; }")
        gpu_layout.addWidget(gpu_label)
        gpu_layout.addWidget(gpu_combo)
        gpu_layout.addStretch()
        hw_layout.addLayout(gpu_layout)
        
        cuda_layout = QHBoxLayout()
        cuda_label = QLabel("CUDA 核心數:")
        cuda_label.setFixedWidth(130)
        cuda_spinbox = QSpinBox()
        cuda_spinbox.setRange(128, 5120)
        cuda_spinbox.setValue(2560)
        cuda_spinbox.setSingleStep(128)
        cuda_spinbox.setStyleSheet("QSpinBox { background: rgba(30, 41, 59, 200); color: #0cf; border: 1px solid rgba(56, 189, 248, 0.5); border-radius: 5px; padding: 5px; }")
        cuda_layout.addWidget(cuda_label)
        cuda_layout.addWidget(cuda_spinbox)
        cuda_layout.addStretch()
        hw_layout.addLayout(cuda_layout)
        
        vram_layout = QHBoxLayout()
        vram_label = QLabel("顯存類型:")
        vram_label.setFixedWidth(130)
        vram_combo = QComboBox()
        vram_combo.addItems(["GDDR6 (推薦)", "GDDR6X", "HBM2", "HBM2e"])
        vram_combo.setCurrentIndex(0)
        vram_combo.setStyleSheet(gpu_combo.styleSheet())
        vram_layout.addWidget(vram_label)
        vram_layout.addWidget(vram_combo)
        vram_layout.addStretch()
        hw_layout.addLayout(vram_layout)
        hw_group.setLayout(hw_layout)
        hw_desc = QLabel("硬件選擇與CUDA配置，按實際GPU能力設置。")
        hw_desc.setStyleSheet(desc_style)
        hw_layout.insertWidget(0, hw_desc)
        hw_group.setToolTip("硬件配置：指定使用哪塊GPU及CUDA核心數。")
        shadow5 = QGraphicsDropShadowEffect()
        shadow5.setBlurRadius(12)
        shadow5.setOffset(0, 3)
        shadow5.setColor(QColor(16, 185, 129, 60))
        hw_group.setGraphicsEffect(shadow5)
        scroll_layout.addWidget(hw_group)

        self.k_gpu_select = gpu_combo
        self.k_cuda = cuda_spinbox
        self.k_vram = vram_combo
        self.k_presets = presets_combo
        
        # ============ 操作按鈕 ============
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        try:
            btn_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        except Exception:
            btn_layout.setAlignment(Qt.AlignHCenter)
        
        apply_btn = QPushButton("✅ 應用設置")
        apply_btn.setMinimumHeight(40)
        apply_btn.setFixedWidth(160)
        apply_btn.setStyleSheet("QPushButton { background-color: #10B981; color: white; border: none; border-radius: 8px; font-weight: bold; font-size: 13px; padding: 8px 16px; } QPushButton:hover { background-color: #059669; border: 2px solid rgba(16, 185, 129, 0.6); }")
        
        reset_btn = QPushButton("🔄 重置默認")
        reset_btn.setMinimumHeight(40)
        reset_btn.setFixedWidth(140)
        reset_btn.setStyleSheet("QPushButton { background-color: #F97316; color: white; border: none; border-radius: 8px; font-weight: bold; font-size: 13px; padding: 8px 16px; } QPushButton:hover { background-color: #EA580C; border: 2px solid rgba(249, 115, 22, 0.6); }")
        
        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(reset_btn)
        # 按鈕陰影與提示
        apply_desc = QLabel("")
        apply_btn.setToolTip("應用當前設置並保存到配置文件。")
        reset_btn.setToolTip("恢復所有內核優化參數到默認值。")
        apply_effect = QGraphicsDropShadowEffect()
        apply_effect.setBlurRadius(20)
        apply_effect.setOffset(0, 6)
        apply_effect.setColor(QColor(16, 185, 129, 100))
        apply_btn.setGraphicsEffect(apply_effect)
        reset_effect = QGraphicsDropShadowEffect()
        reset_effect.setBlurRadius(18)
        reset_effect.setOffset(0, 6)
        reset_effect.setColor(QColor(249, 115, 22, 100))
        reset_btn.setGraphicsEffect(reset_effect)
        btn_layout.addStretch()
        scroll_layout.addLayout(btn_layout)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)


        # 連接按鈕回調
        def apply_kernel():
            ks = {
                "cpu_priority": self.k_cpu_prio.currentText(),
                "gpu_memory_optimize": bool(self.k_gpu_opt.isChecked()),
                "worker_threads": int(self.k_thread.value()),
                "cache_size_mb": int(self.k_cache.value()),
                "target_mode": "multi" if self.k_target_multi.isChecked() else "single",
                "detection_precision": ("fast" if self.k_precision_fast.isChecked() else ("precise" if self.k_precision_precise.isChecked() else "standard")),
                "tracking_algorithm": ("kalman" if self.k_algo_kalman.isChecked() else ("optflow" if self.k_algo_of.isChecked() else "hybrid")),
                "tracking_sensitivity": int(self.k_sensitivity.value()),
                "memory_preload": bool(self.k_preload.isChecked()),
                "precache_size_mb": int(self.k_precache.value()),
                "gc_interval_ms": int(self.k_gc.value()),
                "gpu_selection": self.k_gpu_select.currentText(),
                "cuda_cores": int(self.k_cuda.value()),
                "vram_type": self.k_vram.currentText()
            }
            self.save_kernel_settings(ks)
            self.update_log_display("✅ 內核優化設定已應用並保存")

        def reset_kernel():
            self.reset_kernel_defaults()
            self.update_log_display("🔄 內核優化已恢復為默認值")

        apply_btn.clicked.connect(apply_kernel)
        reset_btn.clicked.connect(reset_kernel)

        # 預設下拉改變時應用示例配置到UI
        def on_preset_change(idx):
            presets = self.get_kernel_presets()
            key = list(presets.keys())[idx]
            self.apply_kernel_settings(presets[key])

        try:
            self.k_presets.currentIndexChanged.connect(on_preset_change)
        except Exception:
            pass

        # 載入已保存的 kernel 設定
        try:
            saved = self.load_kernel_settings()
            if saved:
                self.apply_kernel_settings(saved)
        except Exception:
            pass

    # ================= Kernel settings persistence =================
    def get_kernel_presets(self):
        """返回示例預設配置字典"""
        return {
            "Low": {
                "cpu_priority": "🔴 低",
                "gpu_memory_optimize": False,
                "worker_threads": 4,
                "cache_size_mb": 128,
                "target_mode": "single",
                "detection_precision": "fast",
                "tracking_algorithm": "kalman",
                "tracking_sensitivity": 30,
                "memory_preload": False,
                "precache_size_mb": 256,
                "gc_interval_ms": 1000,
                "gpu_selection": "🟡 CPU",
                "cuda_cores": 128,
                "vram_type": "GDDR6"
            },
            "Medium": {
                "cpu_priority": "🟡 中",
                "gpu_memory_optimize": True,
                "worker_threads": 8,
                "cache_size_mb": 512,
                "target_mode": "multi",
                "detection_precision": "standard",
                "tracking_algorithm": "kalman",
                "tracking_sensitivity": 50,
                "memory_preload": True,
                "precache_size_mb": 1024,
                "gc_interval_ms": 2000,
                "gpu_selection": "🟢 GPU 0 (推薦)",
                "cuda_cores": 2560,
                "vram_type": "GDDR6"
            },
            "High": {
                "cpu_priority": "🟢 高",
                "gpu_memory_optimize": True,
                "worker_threads": 16,
                "cache_size_mb": 2048,
                "target_mode": "multi",
                "detection_precision": "precise",
                "tracking_algorithm": "hybrid",
                "tracking_sensitivity": 75,
                "memory_preload": True,
                "precache_size_mb": 4096,
                "gc_interval_ms": 3000,
                "gpu_selection": "🟢 GPU 0 (推薦)",
                "cuda_cores": 4096,
                "vram_type": "GDDR6X"
            }
        }

    def save_kernel_settings(self, cfg: dict):
        """將 kernel 設定寫入 settings.json 的 kernel_optimization 鍵，保留其他設定"""
        try:
            settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
            data = {}
            if os.path.exists(settings_path):
                with open(settings_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except Exception:
                        data = {}
            data['kernel_optimization'] = cfg
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logging.warning(f"儲存 kernel 設定失敗: {e}")
            return False

    def load_kernel_settings(self):
        """從 settings.json 讀取 kernel_optimization 設定，如果沒有返回 None"""
        try:
            settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
            if not os.path.exists(settings_path):
                return None
            with open(settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get('kernel_optimization', None)
        except Exception as e:
            logging.warning(f"讀取 kernel 設定失敗: {e}")
            return None

    def apply_kernel_settings(self, cfg: dict):
        """將設定應用到 UI 控件上（不保存）"""
        if not cfg:
            return
        try:
            # CPU 優先級
            cpu_val = cfg.get('cpu_priority', '🟡 中')
            try:
                idx = self.k_cpu_prio.findText(cpu_val)
            except Exception:
                idx = -1
            if idx >= 0:
                try:
                    self.k_cpu_prio.setCurrentIndex(idx)
                except Exception:
                    pass

            self.k_gpu_opt.setChecked(bool(cfg.get('gpu_memory_optimize', False)))
            self.k_thread.setValue(int(cfg.get('worker_threads', 8)))
            self.k_cache.setValue(int(cfg.get('cache_size_mb', 512)))

            if cfg.get('target_mode', 'single') == 'multi':
                self.k_target_multi.setChecked(True)
            else:
                self.k_target_single.setChecked(True)

            prec = cfg.get('detection_precision', 'standard')
            if prec == 'fast':
                self.k_precision_fast.setChecked(True)
            elif prec == 'precise':
                self.k_precision_precise.setChecked(True)
            else:
                self.k_precision_standard.setChecked(True)

            alg = cfg.get('tracking_algorithm', 'kalman')
            if alg == 'optflow':
                self.k_algo_of.setChecked(True)
            elif alg == 'hybrid':
                self.k_algo_hybrid.setChecked(True)
            else:
                self.k_algo_kalman.setChecked(True)

            self.k_sensitivity.setValue(int(cfg.get('tracking_sensitivity', 50)))
            self.k_preload.setChecked(bool(cfg.get('memory_preload', True)))
            self.k_precache.setValue(int(cfg.get('precache_size_mb', 1024)))
            self.k_gc.setValue(int(cfg.get('gc_interval_ms', 2000)))

            # GPU / CUDA / VRAM
            try:
                idx_gpu = self.k_gpu_select.findText(cfg.get('gpu_selection', '🟢 GPU 0 (推薦)'))
                if idx_gpu >= 0:
                    self.k_gpu_select.setCurrentIndex(idx_gpu)
            except Exception:
                pass
            try:
                self.k_cuda.setValue(int(cfg.get('cuda_cores', 2560)))
            except Exception:
                pass
            try:
                idx_vram = self.k_vram.findText(cfg.get('vram_type', 'GDDR6'))
                if idx_vram >= 0:
                    self.k_vram.setCurrentIndex(idx_vram)
            except Exception:
                pass
        except Exception as e:
            logging.warning(f"應用 kernel 設定到 UI 失敗: {e}")

    def reset_kernel_defaults(self):
        """將內核面板恢復到 Medium（默認）並保存"""
        presets = self.get_kernel_presets()
        med = presets.get('Medium')
        if med:
            self.apply_kernel_settings(med)
            self.save_kernel_settings(med)

    def setup_aimbot_ui(self, layout):
        self.cb_cpu = QCheckBox(" 啟用CPU AI自瞄引擎")
        self.cb_gpu = QCheckBox(" 啟用追鎖")
        self.check_boxes = [self.cb_cpu, self.cb_gpu]
        for cb in self.check_boxes:
            cb.stateChanged.connect(self.on_aimbot_toggle)
            layout.addWidget(cb)
        
        # FOV 顯示開關（控制 overlay 上是否繪製 FOV 圈）自瞄功能開啟後才有意義
        self.fov_checkbox = QCheckBox(" 顯示 FOV 圈(自瞄)")
        self.fov_checkbox.setChecked(True)
        self.fov_checkbox.stateChanged.connect(self.toggle_fov_enabled)
        layout.addWidget(self.fov_checkbox)
        
        # FOV 顯示開關（控制 overlay 上是否繪製 FOV 圈）追鎖功能開啟後才有意義
        self.fov_track_checkbox = QCheckBox(" 顯示 FOV 圈(追鎖)")
        self.fov_track_checkbox.setChecked(True)
        self.fov_track_checkbox.stateChanged.connect(self.toggle_fov_enabled)
        layout.addWidget(self.fov_track_checkbox)

        # ========== 1. 平滑度調整 ==========
        self.smooth_label = QLabel("瞄准平滑度: 50", objectName="NormalText")
        self.smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self.smooth_slider.setRange(1, 100)
        self.smooth_slider.setValue(int(self.aim_smoothing * 100))
        self.smooth_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #1E293B;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #38BDF8;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
                border: 2px solid #0EA5E9;
            }
            QSlider::handle:horizontal:hover {
                background: #0EA5E9;
                border: 2px solid rgba(14, 165, 233, 0.6);
            }
        """)
        self.smooth_slider.valueChanged.connect(self.on_smoothing_changed)
        layout.addWidget(self.smooth_label)
        layout.addWidget(self.smooth_slider)

        # ========== 2. FOV範圍調整 ==========
        self.fov_label = QLabel("瞄准范围 (FOV): 0", objectName="NormalText")
        self.fov_slider = QSlider(Qt.Orientation.Horizontal)
        self.fov_slider.setRange(0, 800)
        self.fov_slider.setValue(self.fov_radius)
        self.fov_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #1E293B;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #8B5CF6;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
                border: 2px solid #7C3AED;
            }
            QSlider::handle:horizontal:hover {
                background: #7C3AED;
                border: 2px solid rgba(124, 58, 237, 0.6);
            }
        """)
        self.fov_slider.valueChanged.connect(self.sync_fov_value)
        layout.addWidget(self.fov_label)
        layout.addWidget(self.fov_slider)

        # ========== 3. 推理間隔調整 ==========
        self.infer_interval_label = QLabel(f"推理间隔: {self.aim_infer_interval:.3f}s", objectName="NormalText")
        self.infer_interval_slider = QSlider(Qt.Orientation.Horizontal)
        self.infer_interval_slider.setRange(10, 200)  # 0.01s 到 0.2s
        self.infer_interval_slider.setValue(int(self.aim_infer_interval * 1000))
        self.infer_interval_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #1E293B;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #EC4899;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
                border: 2px solid #BE185D;
            }
            QSlider::handle:horizontal:hover {
                background: #BE185D;
                border: 2px solid rgba(190, 24, 93, 0.6);
            }
        """)
        self.infer_interval_slider.valueChanged.connect(self.on_infer_interval_changed)
        layout.addWidget(self.infer_interval_label)
        layout.addWidget(self.infer_interval_slider)

        # ========== 4. 最大移動調整 ==========
        self.max_move_label = QLabel(f"死區: {self.aim_max_move}px", objectName="NormalText")
        self.max_move_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_move_slider.setRange(10, 200)
        self.max_move_slider.setValue(self.aim_max_move)
        self.max_move_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #1E293B;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #F59E0B;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
                border: 2px solid #D97706;
            }
            QSlider::handle:horizontal:hover {
                background: #D97706;
                border: 2px solid rgba(217, 119, 6, 0.6);
            }
        """)
        self.max_move_slider.valueChanged.connect(self.on_max_move_changed)
        layout.addWidget(self.max_move_label)
        layout.addWidget(self.max_move_slider)

        # ========== 5. 置信度閾值調整 ==========
        self.confidence_label = QLabel(f"置信度: {self.cfg.get('aimbot.confidence_threshold', 0.35):.2f}", objectName="NormalText")
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(10, 95)  # 0.1 到 0.95
        self.confidence_slider.setValue(int(self.cfg.get('aimbot.confidence_threshold', 0.35) * 100))
        self.confidence_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #1E293B;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #06B6D4;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
                border: 2px solid #0891B2;
            }
            QSlider::handle:horizontal:hover {
                background: #0891B2;
                border: 2px solid rgba(8, 145, 178, 0.6);
            }
        """)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        layout.addWidget(self.confidence_label)
        layout.addWidget(self.confidence_slider)
        # ========== 配置按鈕 ==========
        config_btn_layout = QHBoxLayout()
        config_btn_layout.setSpacing(10)
        
        # 導出配置按鈕（綠色）
        export_btn = QPushButton("📤 導出配置")
        export_btn.setMinimumHeight(40)
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #059669;
                border: 2px solid rgba(16, 185, 129, 0.6);
            }
            QPushButton:pressed {
                background-color: #047857;
                padding-top: 10px;
                padding-bottom: 6px;
            }
        """)
        export_btn.clicked.connect(lambda: self.export_config_to_file())
        config_btn_layout.addWidget(export_btn)
        
        # 導入配置按鈕（藍色）
        import_btn = QPushButton("📥 導入配置")
        import_btn.setMinimumHeight(40)
        import_btn.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #2563EB;
                border: 2px solid rgba(59, 130, 246, 0.6);
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
                padding-top: 10px;
                padding-bottom: 6px;
            }
        """)
        import_btn.clicked.connect(self.on_import_config_clicked)
        config_btn_layout.addWidget(import_btn)
        
        # 重置默認按鈕（橙紅色）
        reset_btn = QPushButton("🔄 重置默認")
        reset_btn.setMinimumHeight(40)
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #F97316;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #EA580C;
                border: 2px solid rgba(249, 115, 22, 0.6);
            }
            QPushButton:pressed {
                background-color: #C2410C;
                padding-top: 10px;
                padding-bottom: 6px;
            }
        """)
        reset_btn.clicked.connect(lambda: self.reset_config_to_default())
        config_btn_layout.addWidget(reset_btn)
        
        layout.addLayout(config_btn_layout)

        self.status_label = QLabel("YOLO 狀態: 等待指令", objectName="StatusLabel")
        layout.addWidget(self.status_label)
        
        self.log_output = QTextEdit()
        self.log_output.setObjectName("LogOutput")
        self.log_output.setReadOnly(True)
        self.log_output.setFixedHeight(150)
        layout.addWidget(self.log_output)
    
    # ========== 自瞄參數調整回調方法 ==========
    
    def on_smoothing_changed(self, value):
        """平滑度滑块回调"""
        if not getattr(self, 'is_program_initialized', False):
            aimbot_signal.update_log.emit("❌ 請先點擊「系統公告」頁面的「初始化」按鈕來啟動程序！")
            return
        smooth_value = value / 100.0
        self.aim_smoothing = smooth_value
        self.smooth_label.setText(f"瞄准平滑度: {value}")
        self.sync_aimbot_to_config('smoothing', smooth_value)
        aimbot_signal.update_log.emit(f"✓ 平滑度已调整: {smooth_value:.2f}")
    
    def on_infer_interval_changed(self, value):
        """推理间隔滑块回调"""
        if not getattr(self, 'is_program_initialized', False):
            aimbot_signal.update_log.emit("❌ 請先點擊「系統公告」頁面的「初始化」按鈕來啟動程序！")
            return
        interval_value = value / 1000.0  # 转换为秒
        self.aim_infer_interval = interval_value
        self.infer_interval_label.setText(f"推理间隔: {interval_value:.3f}s")
        self.sync_aimbot_to_config('infer_interval', interval_value)
        aimbot_signal.update_log.emit(f"✓ 推理间隔已调整: {interval_value:.3f}s")
    
    def on_max_move_changed(self, value):
        """最大移動滑块回调"""
        if not getattr(self, 'is_program_initialized', False):
            aimbot_signal.update_log.emit("❌ 請先點擊「系統公告」頁面的「初始化」按鈕來啟動程序！")
            return
        self.aim_max_move = value
        self.max_move_label.setText(f"最大移動: {value}px")
        self.sync_aimbot_to_config('max_move', value)
        aimbot_signal.update_log.emit(f"✓ 最大移動已调整: {value}px")
    
    def on_confidence_changed(self, value):
        """置信度滑块回调"""
        if not getattr(self, 'is_program_initialized', False):
            aimbot_signal.update_log.emit("❌ 請先點擊「系統公告」頁面的「初始化」按鈕來啟動程序！")
            return
        confidence_value = value / 100.0
        self.confidence_label.setText(f"置信度: {confidence_value:.2f}")
        self.cfg.set('aimbot.confidence_threshold', confidence_value)
        self.cfg.save_config()
        aimbot_signal.update_log.emit(f"✓ 置信度已调整: {confidence_value:.2f}")
    
    def on_import_config_clicked(self):
        """導入配置按鈕回調"""
        # ✅ 檢查程序是否已初始化
        if not getattr(self, 'is_program_initialized', False):
            aimbot_signal.update_log.emit("❌ 請先點擊「系統公告」頁面的「初始化」按鈕來啟動程序！")
            logging.warning("❌ 配置導入功能被禁用：程序未初始化")
            return
        
        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇配置文件", "", "JSON Configuration (*.json)"
        )
        if file_path:
            if self.import_config_from_file(file_path):
                # 重新加載UI值
                self.smooth_slider.setValue(int(self.aim_smoothing * 100))
                self.infer_interval_slider.setValue(int(self.aim_infer_interval * 1000))
                self.max_move_slider.setValue(self.aim_max_move)
                self.confidence_slider.setValue(int(self.cfg.get('aimbot.confidence_threshold', 0.35) * 100))
                aimbot_signal.update_log.emit("✅ 配置已成功導入")


 
    def on_aimbot_toggle(self, state):
        # ✅ 檢查程序是否已初始化
        if not getattr(self, 'is_program_initialized', False):
            aimbot_signal.update_log.emit("❌ 請先點擊「系統公告」頁面的「初始化」按鈕來啟動程序！")
            logging.warning("❌ 自瞄功能被禁用：程序未初始化")
            # 取消複選框狀態
            for cb in self.check_boxes:
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)
            return
        
        active_list = [cb for cb in self.check_boxes if cb.isChecked()]
        if active_list:
            self.target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.aimbot_active = True
            logging.info(f"🎯 自瞄已啟動，設備: {self.target_device}")
            if self.yolo_model is None:
                logging.info("⏳ 後台線程正在加載模型...")
                # 后台加载模型使用 QThread（保存引用以防止被回收）
                self.model_loader_thread = FunctionThread(self.load_model_thread)
                self.model_loader_thread.start()
            else:
                self.start_aim_thread()
        else:
            self.aimbot_active = False
            logging.info("⏹️ 自瞄已停用")
            aimbot_signal.update_log.emit("自瞄循環已停用。")

    def load_model_thread(self):
        """後台線程加載模型，使用預設路徑避免重複轉換"""
        if not self.model_path:
            aimbot_signal.update_log.emit("❌ 模型路徑未設置")
            return
            
        aimbot_signal.update_log.emit(f"正在加載引擎 ({self.target_device})...")
        try:
            aimbot_signal.update_log.emit(f"📦 使用模型: {self.model_path}")
            self.yolo_model = YOLO(self.model_path)
            self.yolo_model = YOLO(self.model_path, task='detect')
            
            # GPU 預熱 - 在集成GPU上運行一次推理以初始化GPU
            if torch.cuda.is_available():
                # 顯示GPU預熱按鈕
                self.gpu_warmup_btn.setText("🔥 GPU 預熱中")
                self.gpu_warmup_btn.setEnabled(True)
                self.gpu_warmup_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #EF4444;
                        color: white;
                        border: none;
                        border-radius: 8px;
                        font-weight: bold;
                        font-size: 13px;
                        padding: 8px 16px;
                    }
                """)
                aimbot_signal.update_log.emit("🔥 GPU 預熱中...")
                try:
                    # 創建一個小的假輸入進行預熱
                    dummy_input = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
                    _ = self.yolo_model.predict(dummy_input, conf=0.5, imgsz=320, verbose=False, device=0)
                    aimbot_signal.update_log.emit("✅ GPU 預熱完成！")
                    
                    # 預熱完成後隱藏按鈕
                    self.gpu_warmup_btn.setText("✅ GPU 預熱完成")
                    self.gpu_warmup_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #10B981;
                            color: white;
                            border: none;
                            border-radius: 8px;
                            font-weight: bold;
                            font-size: 13px;
                            padding: 8px 16px;
                        }
                    """)
                except Exception as e:
                    aimbot_signal.update_log.emit(f"⚠️ GPU 預熱失敗: {e}，繼續運行")
                    self.gpu_warmup_btn.setText("⚠️ GPU 預熱失敗")
                    self.gpu_warmup_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #F59E0B;
                            color: white;
                            border: none;
                            border-radius: 8px;
                            font-weight: bold;
                            font-size: 13px;
                            padding: 8px 16px;
                        }
                    """)
            
            aimbot_signal.update_log.emit("✅ 模型部署完成！")
            self.start_aim_thread()
        except Exception as e:
            aimbot_signal.update_log.emit(f"❌ 載入失敗: {e}")

    def start_aim_thread(self):
        # 兼容 threading.Thread 與 QThread 的運行檢查
        alive = False
        if self.aimbot_thread is not None:
            if hasattr(self.aimbot_thread, 'is_alive'):
                try:
                    alive = self.aimbot_thread.is_alive()
                except Exception:
                    alive = False
            elif hasattr(self.aimbot_thread, 'isRunning'):
                try:
                    alive = self.aimbot_thread.isRunning()
                except Exception:
                    alive = False

        if not alive:
            logging.info("🔄 啟動自瞄循環線程...")
            self.aimbot_thread = FunctionThread(self.aimbot_loop)
            self.aimbot_thread.start()
            logging.info("✅ 自瞄循環線程已啟動")
        else:
            logging.warning("⚠️ 自瞄循環線程已在運行中")

    def aimbot_loop(self):
        # 載入 YOLO11 模型
        if self.yolo_model is None:
            logging.info(f"📦 加載模型: {self.model_path}")
            self.yolo_model = YOLO(self.model_path)
            logging.info("✅ 模型加載完成")

        # 偵測區域設置 
        d_size = 320 
        center = d_size / 2
        logging.debug(f"偵測區域大小: {d_size}x{d_size}")
        
        with mss.mss() as sct:
            mon = sct.monitors[1]
            sw, sh = mon["width"], mon["height"]
            logging.info(f"📺 螢幕解析度: {sw}x{sh}")
            
            # 計算螢幕中心偏移量
            offset_x = (sw - d_size) // 2
            offset_y = (sh - d_size) // 2
            region = {"top": offset_y, "left": offset_x, "width": d_size, "height": d_size}

            while self.aimbot_active and not getattr(self, '_stop_threads', False):
                loop_start = time.time()
                
                # 1. 高速截圖 - 優化：使用 asarray 替代 array 減少內存複製
                screenshot = sct.grab(region)
                frame = np.asarray(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # 2. 模型推理控制 - 優化：降低推理頻率至 10Hz (0.1s), 保持滑鼠移動在 60Hz
                now = time.time()
                new_targets = []
                best_target = None
                min_score = -1.0
                fov_limit = self.fov_slider.value()
                if fov_limit == 0:
                    fov_limit = d_size

                should_infer = now - self._last_infer >= self.aim_infer_interval
                if should_infer:
                    self._last_infer = now
                    # 传入 device 让模型使用指定硬件
                    try:
                        device_arg = getattr(self, 'target_device', 'cpu')
                    except:
                        device_arg = 'cpu'
                    
                    # 获取当前的置信度阈值
                    conf_threshold = self.cfg.get('aimbot.confidence_threshold', 0.35)
                    
                    results = self.yolo_model.predict(frame, conf=conf_threshold, imgsz=d_size, verbose=False, device=device_arg)

                    for r in results:
                        # 获取 boxes 的 array 表示
                        try:
                            xywh = r.boxes.xywh.cpu().numpy()
                        except Exception:
                            # 备用方式：从 xyxy 转换
                            for box in r.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                                w, h = x2 - x1, y2 - y1
                                dist = np.hypot(cx - center, cy - center)
                                abs_x = x1 + offset_x
                                abs_y = y1 + offset_y
                                conf = float(getattr(box, 'conf', 1.0)) if hasattr(box, 'conf') else 1.0
                                score = conf / (1.0 + dist)
                                if dist < fov_limit and score > min_score:
                                    min_score = score
                                    best_target = (cx - center, cy - center)
                                new_targets.append((abs_x, abs_y, w, h, False))
                            continue

                        confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else np.ones(len(xywh))
                        for box, conf in zip(xywh, confs):
                            cx, cy = box[0], box[1]
                            w, h = box[2], box[3]
                            dist = np.hypot(cx - center, cy - center)
                            abs_x = cx - (w / 2) + offset_x
                            abs_y = cy - (h / 2) + offset_y
                            score = float(conf) / (1.0 + dist)
                            locked = False
                            if dist < fov_limit and score > min_score:
                                min_score = score
                                best_target = (cx - center, cy - center)
                                locked = True
                            new_targets.append((abs_x, abs_y, w, h, locked))

                # 3. 滑鼠移動控制 - 保持高頻 (60Hz) 確保流暢追蹤
                if best_target is not None:
                    # 从 UI 读取平滑值
                    s = self.smooth_slider.value() / 100.0 if hasattr(self, 'smooth_slider') else self.aim_smoothing
                    # EMA 平滑
                    self._target_offset_x = (1 - s) * self._target_offset_x + s * best_target[0]
                    self._target_offset_y = (1 - s) * self._target_offset_y + s * best_target[1]
                    move_x = int(np.clip(self._target_offset_x * self.aim_sensitivity, -self.aim_max_move, self.aim_max_move))
                    move_y = int(np.clip(self._target_offset_y * self.aim_sensitivity, -self.aim_max_move, self.aim_max_move))
                    if move_x != 0 or move_y != 0:
                        try:
                            self.mouse.move(move_x, move_y)
                        except Exception:
                            pass

                # 4. 優化：僅在目標改變時才更新 overlay，減少不必要的繪製
                if new_targets != getattr(self, 'last_aimbot_targets', None):
                    self.overlay.targets = new_targets
                    self.overlay.fov = fov_limit
                    self.overlay.update()
                    self.last_aimbot_targets = new_targets

                # 5. 精確帧時間控制 - 目標 60fps (16.67ms per frame)
                loop_elapsed = time.time() - loop_start
                target_frametime = 1.0 / 60.0
                sleep_time = max(0, target_frametime - loop_elapsed)
                
                if sleep_time > 0.0001:
                    time.sleep(sleep_time)
                else:
                    # 当循环处理时间超过目标帧时间时，短暂让出 CPU（避免 100% 占用）
                    time.sleep(0.002)
                # 更新并平滑当前 FPS（移动平均）
                try:
                    inst_fps = 1.0 / max(1e-6, loop_elapsed)
                    # 指数移动平均
                    self.current_fps = (0.9 * getattr(self, 'current_fps', 0.0)) + (0.1 * inst_fps)
                except Exception:
                    self.current_fps = getattr(self, 'current_fps', 0.0)

                # 自适应：如果 FPS 过低则提高推理间隔，降低负载；恢复时逐步还原
                try:
                    lower_threshold = float(self.cfg.get('aimbot.adapt_fps_lower', 40))
                    upper_threshold = float(self.cfg.get('aimbot.adapt_fps_upper', 55))
                    if self.current_fps < lower_threshold:
                        # 增加推理间隔，最多到 _max_aim_infer_interval
                        new_interval = min(self._max_aim_infer_interval, self.aim_infer_interval * 1.5)
                        if new_interval != self.aim_infer_interval:
                            logging.info(f"自适应降频：FPS={self.current_fps:.1f} < {lower_threshold}，将 infer_interval {self.aim_infer_interval:.3f} -> {new_interval:.3f}")
                            self.aim_infer_interval = new_interval
                            # 同时减慢 overlay 更新频率
                            try:
                                self.overlay_update_timer.setInterval(200)
                            except Exception:
                                pass
                    elif self.current_fps > upper_threshold and self.aim_infer_interval > self._base_aim_infer_interval:
                        # 逐步恢复到基线
                        new_interval = max(self._base_aim_infer_interval, self.aim_infer_interval * 0.9)
                        if new_interval != self.aim_infer_interval:
                            logging.info(f"自适应恢复：FPS={self.current_fps:.1f} > {upper_threshold}，將 infer_interval {self.aim_infer_interval:.3f} -> {new_interval:.3f}")
                            self.aim_infer_interval = new_interval
                            try:
                                self.overlay_update_timer.setInterval(50)
                            except Exception:
                                pass
                except Exception as e:
                    logging.debug(f"自适应調整失敗: {e}")


    def on_fov_slider_changed(self, value):
        """
        當使用者拉動滑桿時，這個函式會收到最新的數值 (value)
        """
        # ✅ 檢查程序是否已初始化
        if not getattr(self, 'is_program_initialized', False):
            aimbot_signal.update_log.emit("❌ 請先點擊「系統公告」頁面的「初始化」按鈕來啟動程序！")
            return
        
        # 1. 更新標籤文字
        self.fov_label.setText(f"瞄准范围 (FOV): {value}")
        
        # 2. 更新繪圖用的變數
        self.fov_radius = value 
        # 觸發 paintEvent 重新執行，畫出新大小的圓圈
        self.update()


    def setup_ESP_ui(self, layout):
        layout.addWidget(QLabel("視覺輔助設定：", objectName="NormalText"))

        self.cb_box = QCheckBox(" 顯示方框 (Box ESP)")
        self.cb_box.setChecked(self.esp_config.get('box', False))
        self.cb_box.stateChanged.connect(lambda v: self._update_esp_config('box', v))
        layout.addWidget(self.cb_box)

        self.cb_line = QCheckBox(" 顯示射線 (Line ESP)")
        self.cb_line.setChecked(self.esp_config.get('line', False))
        self.cb_line.stateChanged.connect(lambda v: self._update_esp_config('line', v))
        layout.addWidget(self.cb_line)

        self.cb_Skeleton = QCheckBox(" 顯示骨架 (Skeleton ESP)")
        self.cb_Skeleton.setChecked(self.esp_config.get('skeleton', False))
        self.cb_Skeleton.stateChanged.connect(lambda v: self._update_esp_config('skeleton', v))
        layout.addWidget(self.cb_Skeleton)

        self.cb_boxes = QCheckBox(" 顯示盒子 (Boxes ESP)")
        self.cb_boxes.setChecked(self.esp_config.get('boxes', False))
        self.cb_boxes.stateChanged.connect(lambda v: self._update_esp_config('boxes', v))
        layout.addWidget(self.cb_boxes)

        self.cb_Distance = QCheckBox(" 顯示距離 (Distance ESP)")
        self.cb_Distance.setChecked(self.esp_config.get('distance', False))
        self.cb_Distance.stateChanged.connect(lambda v: self._update_esp_config('distance', v))
        layout.addWidget(self.cb_Distance)

        self.cb_Threat = QCheckBox(" 顯示受攻擊預警 (Threat ESP)")
        self.cb_Threat.setChecked(self.esp_config.get('threat', False))
        self.cb_Threat.stateChanged.connect(lambda v: self._update_esp_config('threat', v))
        layout.addWidget(self.cb_Threat)
        
        self.cb_Grenade = QCheckBox(" 顯示投擲物 (Grenade ESP)")
        self.cb_Grenade.setChecked(self.esp_config.get('grenade', False))
        self.cb_Grenade.stateChanged.connect(lambda v: self._update_esp_config('grenade', v))
        layout.addWidget(self.cb_Grenade)
        
        self.cb_Caesar = QCheckBox(" 顯示 CAESAR UI")
        self.cb_Caesar.setChecked(self.esp_config.get('caesar', True))
        self.cb_Caesar.stateChanged.connect(lambda v: self._update_esp_config('caesar', v))
        layout.addWidget(self.cb_Caesar)

    def _update_esp_config(self, key: str, state: int):
        """更新 ESP 配置並通知 overlay 重繪"""
        # 如果程式尚未初始化，僅記錄警告，但仍允許更新 ESP 配置以便預覽
        if not getattr(self, 'is_program_initialized', False):
            aimbot_signal.update_log.emit("⚠️ 程式尚未初始化：ESP 設定將被暫存並在初始化後生效（亦可用於預覽）。")
            logging.warning(f"⚠️ ESP 設定在未初始化時被修改: {key}")
        
        try:
            self.esp_config[key] = bool(state > 0)
            logging.debug(f"ESP 配置更新: {key} = {self.esp_config[key]}")
            # 觸發 overlay 重繪或檢查邏輯
            if hasattr(self, 'overlay') and self.overlay:
                self.overlay.update()
        except Exception as e:
            logging.warning(f"ESP 配置更新失敗: {e}")



    def setup_music_ui(self, layout):
        # --- 1. 初始化播放組件 ---
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        
        # 設定初始音量 
        self.audio_output.setVolume(0.5) 

        # --- 2. 播放狀態標籤 ---
        self.current_track_label = QLabel("🎵 目前狀態: 停止播放")
        self.current_track_label.setStyleSheet("""
            color: #38BDF8; font-weight: bold; background-color: #1E293B; 
            padding: 8px; border-radius: 5px; margin-bottom: 2px;
        """)
        layout.addWidget(self.current_track_label)

        # --- 3. 音樂按鈕清單 ---
        self.playlist = {
            "Drop Tower": "music/droptower.mp3",
            "On My Way": "music/onmyway.mp3",
            "Summerdays": "music/summerdays.mp3",
            "Rooftops": "music/rooftops.mp3"
        }

        for name, path in self.playlist.items():
            rb = QRadioButton(name)
            rb.setStyleSheet("color: white; padding: 2px;")
            rb.toggled.connect(lambda checked, p=path, n=name: self.play_music(checked, p, n))
            layout.addWidget(rb)

        # 消除空隙
        layout.addStretch(1)

        # --- 4. 音量控制區 ---
        self.vol_label = QLabel("音量: 0")
        self.vol_label.setStyleSheet("color: #38BDF8; margin-top: 20px;")
        layout.addWidget(self.vol_label)

        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(0) # 預設音量為 0
        # ⚡ 這裡連接到函數
        self.vol_slider.valueChanged.connect(self.change_volume)
        layout.addWidget(self.vol_slider)

    # --- 播放函數 ---
    def play_music(self, checked, path, name):
        if checked and hasattr(self, 'player'):
            if os.path.exists(path):
                self.current_track_label.setText(f"🎵 正在播放: {name}")
                self.player.setSource(QUrl.fromLocalFile(os.path.abspath(path)))
                self.player.play()
            else:
                self.current_track_label.setText(f"❌ 找不到檔案: {name}")

    # --- 音量函數 ---
    def change_volume(self, value):
        if hasattr(self, 'audio_output'):
            # 更新標籤
            self.vol_label.setText(f"音量: {value}")
            self.audio_output.setVolume(value / 100.0)

    def setup_Setting_ui(self, layout):
        layout.addWidget(QLabel("系統設置：", objectName="NormalText"))
        self.cb_save = QCheckBox(" 自動保存設定"); layout.addWidget(self.cb_save)

        self.cb_Smoothandhighdefinitiondrawing = QCheckBox(" 流暢高清繪圖 "); layout.addWidget(self.cb_Smoothandhighdefinitiondrawing)

        self.cb_Blurayhighdefinitiongraphics = QCheckBox(" 藍光高清繪圖 "); layout.addWidget(self.cb_Blurayhighdefinitiongraphics)

        self.cb_PerformanceMode = QCheckBox(" 性能模式 "); layout.addWidget(self.cb_PerformanceMode)

        self.cb_Normalmode = QCheckBox(" 普通模式 "); layout.addWidget(self.cb_Normalmode)

        self.cb_Quietmode = QCheckBox(" 安靜模式 "); layout.addWidget(self.cb_Quietmode)

        self.cb_Screenrecordingprotection = QCheckBox(" 防錄屏 "); layout.addWidget(self.cb_Screenrecordingprotection)
        # 連接防錄屏開關
        try:
            self.cb_Screenrecordingprotection.stateChanged.connect(self.set_screen_protection_enabled)
        except Exception:
            pass
    
    def setup_system_ui(self, layout):
        """設置系統狀況UI"""
        layout.addWidget(QLabel("系統狀況監控：", objectName="NormalText"))
        self.cpu_label = QLabel("CPU 使用率: -- %", objectName="NormalText")
        self.memory_label = QLabel("記憶體使用率: -- %", objectName="NormalText")
        self.ui_fps_label = QLabel("UI FPS: -- /s", objectName="NormalText")
        layout.addWidget(self.cpu_label)
        layout.addWidget(self.memory_label)
        layout.addWidget(self.ui_fps_label)
        # 啟動計時器定期更新系統狀況
        self.sys_timer = QTimer(self)
        self.sys_timer.timeout.connect(self.update_system_status)
        self.sys_timer.start(1000)  # 每秒更新一次

    def update_system_status(self):
     cpu = psutil.cpu_percent()
     ram = psutil.virtual_memory().percent
     self.cpu_label.setText(f"CPU 使用率: {cpu}%")
     self.memory_label.setText(f"記憶體使用率: {ram}%")
     # 更新 UI FPS 顯示
     try:
         if hasattr(self, 'ui_fps') and hasattr(self, 'ui_fps_label'):
             self.ui_fps_label.setText(f"UI FPS: {self.ui_fps:.1f} /s")
     except Exception:
         pass
    
     # 更新右下角的 Ping
     self.update_ping()

    def update_status_display(self, msg):
        try:
            if self.status_label: self.status_label.setText(msg)
        except RuntimeError: pass

    def on_decrypt_clicked(self):
        """處理初始化/解密按鈕點擊事件"""
        if not self.has_initialized:
            # 第一次點擊：初始化 -> 解密
            self.init_decrypt_btn.setText("解密")
            self.has_initialized = True
            self.init_decrypt_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #059669, stop:1 #047857);
                    color: white;
                    font-weight: bold;
                    font-size: 16px;
                    border: none;
                    border-radius: 8px;
                    padding: 10px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #047857, stop:1 #065F46);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #065F46, stop:1 #064E3B);
                }
            """)
            logging.info("🔄 系統公告已初始化，請再次點擊進行解密...")
            aimbot_signal.update_log.emit("⏳ 系統初始化完成，請按下解密按鈕")
        
        elif not self.decrypted:
            # 第二次點擊：解密 -> 解密完成（禁用按鈕）
            self.init_decrypt_btn.setText("✅ 解密完成")
            self.decrypted = True
            self.is_program_initialized = True  # ✅ 設置程序已初始化，保持功能啟用
            
            # 禁用按鈕
            self.init_decrypt_btn.setEnabled(False)
            
            # 更新按鈕樣式（禁用狀態）
            self.init_decrypt_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #059669, stop:1 #047857);
                    color: white;
                    font-weight: bold;
                    font-size: 16px;
                    border: none;
                    border-radius: 8px;
                    padding: 10px;
                }
                QPushButton:disabled {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #065F46, stop:1 #054E3B);
                    color: #E0F2EF;
                    opacity: 0.7;
                }
            """)
            logging.info("🔐 系統公告已解密，程序已初始化，所有功能已啟用！")
            aimbot_signal.update_log.emit("✅ 程序已初始化，所有功能已啟用！")

    def changeEvent(self, event):
        """監聽窗口狀態變化以優化CPU占用"""
        super().changeEvent(event)
        if event.type() == event.Type.WindowStateChange:
            # 如果窗口被最小化，停止刷新定時器
            if self.isMinimized():
                if hasattr(self, 'repaint_timer'):
                    self.repaint_timer.stop()
                if hasattr(self, 'overlay_update_timer'):
                    self.overlay_update_timer.stop()
                if hasattr(self, 'bg_fade_timer'):
                    self.bg_fade_timer.stop()
                logging.debug("⏸️ 菜單最小化：刷新定時器已停止，降低CPU占用")
            else:
                # 窗口恢復時，重新啟動定時器
                if hasattr(self, 'repaint_timer'):
                    self.repaint_timer.start()
                if hasattr(self, 'overlay_update_timer'):
                    self.overlay_update_timer.start()
                if hasattr(self, 'bg_fade_timer'):
                    self.bg_fade_timer.start()
                logging.debug("▶️ 菜單恢復：刷新定時器已重啟")

    def switch_page(self, index):
        """正確的分頁切換邏輯"""
        self.stack.setCurrentIndex(index)
        
        # 更新按鈕外觀
        for i, btn in enumerate(self.nav_buttons):
            btn.setObjectName("ActiveBtn" if i == index else "MenuBtn")
            btn.setStyle(btn.style())
        
        # ⭐ 根據頁面切換背景
        bg_images = [
            "login_bg.png",  # 系統狀況
            "login_bg.png",  # 系統公告
            "login_bg.png",  # 帳號
            "login_bg.png",  # 視覺輔助
            "login_bg.png",  # 自動瞄準
            "login_bg.png",  # 內存功能
            "login_bg.png",  # 音樂中心
            "login_bg.png",  # 設置中心
        ]
        
        if index < len(bg_images):
            self.load_background(bg_images[index])

        # 切換到帳號分頁時觸發打字機
        if self.menus[index] == "帳號":
            if hasattr(self, 'refresh_account_data'):
                raw_data = self.refresh_account_data() 
                self.start_typing_effect(raw_data)
    
    def load_background(self, image_path):
        """加载背景图片并开始淡入淡出过渡"""
        if not os.path.exists(image_path):
            logging.warning(f"背景图片不存在: {image_path}")
            return
        
        # 加载新的背景图片
        new_pixmap = QPixmap(image_path)
        if new_pixmap.isNull():
            logging.error(f"无法加载背景图片: {image_path}")
            return
        
        # 设置目标背景
        self.bg_pixmap_target = new_pixmap
        if self.bg_pixmap_current is None:
            # 如果是第一次加载，直接设置
            self.bg_pixmap_current = new_pixmap
            self.bg_opacity = 1.0
        else:
            # 否则开始淡入淡出过渡
            self.bg_opacity = 0.0
        
        self.update()
    
    def _update_bg_fade(self):
        """更新背景淡入淡出效果"""
        if self.bg_pixmap_target is None:
            return
        
        if self.bg_opacity < 1.0:
            # 淡入过渡 (大约0.8秒完全过渡)
            self.bg_opacity += 0.08
            if self.bg_opacity >= 1.0:
                self.bg_opacity = 1.0
                self.bg_pixmap_current = self.bg_pixmap_target
            self.update()

    def mousePressEvent(self, event):
        """處理視窗點擊（拖拽準備）"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        """處理視窗移動"""
        if event.buttons() == Qt.MouseButton.LeftButton and self.drag_pos:
            self.move(event.globalPosition().toPoint() - self.drag_pos)
            event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'sizegrip'): self.sizegrip.move(self.width() - 20, self.height() - 20)
    
    def keyPressEvent(self, event):
        """⭐ 快捷键处理 - F1-F10 快速切换菜单"""
        key = event.key()
        # F1-F10 对应菜单索引 0-9
        if key >= 16777264 and key <= 16777273:  # F1-F10 的键码
            menu_idx = key - 16777264
            if menu_idx < len(self.nav_buttons):
                self.switch_page(menu_idx)
                self.nav_buttons[menu_idx].click()
        else:
            super().keyPressEvent(event)
    
    def update_status_bar(self):
        """⭐ 更新状态栏信息（FPS、Ping、CPU、内存）"""
        import psutil
        import threading
        
        def get_stats():
            try:
                # 获取当前帧率
                fps = getattr(self, 'current_fps', 0)
                self.fps_label.setText(f"FPS: {fps:.1f}")
                
                # 获取CPU使用率
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_label.setText(f"💻 CPU: {cpu_percent:.1f}%")
                
                # 获取内存使用率
                memory = psutil.virtual_memory()
                self.memory_label.setText(f"🧠 RAM: {memory.percent:.1f}%")
                
                # Ping模拟
                ping = 0
                self.ping_label.setText(f"🌐 Ping: {ping} ms")
            except: pass
        
        # 使用 QThread 运行获取统计的短期任务（保存引用以防止被回收）
        self.stats_thread = FunctionThread(get_stats)
        self.stats_thread.start()
    
    def toggle_theme(self):
        """⭐ 切换亮/暗主题"""
        self.is_dark_theme = not self.is_dark_theme
        
        if self.is_dark_theme:
            primary_color = "#38bdf8"
            text_color = "#ffffff"
            icon = "🌙"
            self.setStyleSheet(self.stylesheet_dark)
        else:
            primary_color = "#0ea5e9"
            text_color = "#1e293b"
            icon = "☀️"
            self.setStyleSheet(self.stylesheet_light)
        
        # 更新主题按钮图标
        self.theme_btn.setText(icon)
        
        # 更新状态栏标签颜色
        for label in [self.fps_label, self.ping_label, self.cpu_label, self.memory_label]:
            label.setStyleSheet(f"color: {primary_color}; font-weight: 600; font-size: 11px;")
    
    def paintEvent(self, event):
        """菜单绘制 - 绘制背景图片和毛玻璃效果"""
        # 绘制背景图片（带淡入淡出过渡）
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self.bg_pixmap_current and not self.bg_pixmap_current.isNull():
            # 缩放背景图片以适应窗口
            scaled_bg = self.bg_pixmap_current.scaled(
                self.size(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # 绘制当前背景（当前不透明度）
            painter.setOpacity(self.bg_opacity)
            painter.drawPixmap(0, 0, scaled_bg)
            painter.setOpacity(1.0)
            
            # 如果正在过渡中，也绘制目标背景
            if self.bg_pixmap_target and self.bg_opacity < 1.0:
                scaled_target = self.bg_pixmap_target.scaled(
                    self.size(),
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                painter.setOpacity(1.0 - self.bg_opacity)
                painter.drawPixmap(0, 0, scaled_target)
                painter.setOpacity(1.0)
        
        painter.end()
        super().paintEvent(event)

# --- 5. 主控制器 ---
class MainController(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(120, 100)
        self.is_on = False
        self.model_path = None  # 儲存模型路徑
        self.drag_start_pos = None
        self.mouse_press_pos = None
        self.has_moved = False
        self.switch_rect = QRect(30, 34, 60, 32) # 開關矩形區域
        self.security_timer = QTimer(self)       
        #呼叫載入邏輯
        self.load_model()

        # 將模型路徑傳遞給菜單
        self.menu = ModernModMenu(model_path=self.model_path)

    def load_model(self):
        """初始化模型路徑，只在程式啟動時執行一次"""
        global ONNX_CONVERSION_DONE, CACHED_MODEL_PATH, ONNX_CONVERSION_LOCK
        
        # 如果已經轉換過，直接使用緩存
        if CACHED_MODEL_PATH:
            logging.info(f"📦 使用緩存的模型路徑: {CACHED_MODEL_PATH}")
            self.model_path = CACHED_MODEL_PATH
            return
        
        with ONNX_CONVERSION_LOCK:
            # 雙重檢查：鎖定後再檢查一次
            if CACHED_MODEL_PATH:
                self.model_path = CACHED_MODEL_PATH
                return
            
            import os
            from ultralytics import YOLO
            
            pt_model_path = "yolo11n.pt"
            onnx_model_path = "yolo11n.onnx"
            
            logging.info(f"🔍 檢查模型文件...")

            # 1. 檢查 ONNX 是否已存在
            if os.path.exists(onnx_model_path):
                logging.info(f"✅ ONNX 模型已存在: {onnx_model_path}")
                CACHED_MODEL_PATH = onnx_model_path
                self.model_path = onnx_model_path
                logging.info(f"📦 模型路徑設置: {self.model_path}")
                return

            # 2. 如果 ONNX 不存在，檢查 PT 模型是否存在
            if not os.path.exists(pt_model_path):
                logging.error(f"❌ 找不到任何模型文件 ({pt_model_path})")
                self.model_path = None
                return

            # 3. 進行一次性的 ONNX 轉換（使用全局標誌防止重複）
            if ONNX_CONVERSION_DONE:
                logging.info("⏭️ ONNX 轉換已在其他線程進行中，跳過...")
                return
                
            ONNX_CONVERSION_DONE = True
            logging.info(f"⏳ 未發現 ONNX 模型，正在從 PT 進行轉換...")
            try:
                temp_model = YOLO(pt_model_path)
                logging.info("⏳ 正在進行 ONNX 轉換 (首次，需時約 3-5 秒)...")
                # 注意: 新版 ultralytics 的 export() 可能不支援 slim=True 參數。
                # 為兼容性起見，在此使用通用參數，不傳入未知的關鍵字參數。
                temp_model.export(format="onnx", imgsz=320)
                
                if os.path.exists(onnx_model_path):
                    logging.info(f"✅ ONNX 轉換完成: {onnx_model_path}")
                    CACHED_MODEL_PATH = onnx_model_path
                    self.model_path = onnx_model_path
                else:
                    raise FileNotFoundError(f"轉換失敗：輸出文件不存在")
                    
            except Exception as e:
                logging.warning(f"⚠️ ONNX 轉換失敗，降級使用 PT 模型: {e}")
                CACHED_MODEL_PATH = pt_model_path
                self.model_path = pt_model_path
                ONNX_CONVERSION_DONE = False  # 重置標誌以便重試
                
    def paintEvent(self, event):
        p = QPainter()
        p.begin(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # 1. 獲取當前組件的寬高
            w = float(self.width())
            h = float(self.height())
            
            # 2. 繪製背景
            p.setBrush(QColor(255, 255, 255, 1))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(0.0, 0.0, w, h)
            
            # 3. 獲取開關矩形的座標與尺寸
            sw_x = float(self.switch_rect.x())
            sw_y = float(self.switch_rect.y())
            sw_w = float(self.switch_rect.width())
            sw_h = float(self.switch_rect.height())

            # 4. 繪製開關主體
            bg_color = QColor("#34C759") if self.is_on else QColor("#39393D")
            p.setBrush(bg_color)
            p.drawRoundedRect(sw_x, sw_y, sw_w, sw_h, 16.0, 16.0)
            
            # 5. 繪製圓形旋鈕
            p.setBrush(QColor("white"))
            knob_x = sw_x + (30.0 if self.is_on else 2.0)
            knob_y = sw_y + 2.0
            p.drawEllipse(knob_x, knob_y, 28.0, 28.0)
            
        finally:
            p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_press_pos = event.position().toPoint()
            self.drag_start_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self.has_moved = False

    def mouseMoveEvent(self, event):
        if self.drag_start_pos:
            if (event.position().toPoint() - self.mouse_press_pos).manhattanLength() > 5:
                self.has_moved = True
                new_pos = event.globalPosition().toPoint() - self.drag_start_pos
                self.move(new_pos)
                if self.is_on: self.menu.move(new_pos.x() - 165, new_pos.y() + 80)

    def mouseReleaseEvent(self, event):
        # 1. 確保有按下且沒有大幅度拖動
        if not hasattr(self, 'has_moved'): self.has_moved = False
        
        if not self.has_moved and self.mouse_press_pos:
            # 2. 取得點擊座標
            px = self.mouse_press_pos.x()
            py = self.mouse_press_pos.y()
            
            # 3. 取得開關矩形範圍
            rx = self.switch_rect.x()
            ry = self.switch_rect.y()
            rw = self.switch_rect.width()
            rh = self.switch_rect.height()
            
            # 4. 手動數值比對點擊是否在開關區域內
            if rx <= px <= rx + rw and ry <= py <= ry + rh:
                self.is_on = not self.is_on
                
                # 處理選單顯示邏輯
                if hasattr(self, 'menu') and self.menu:
                    if self.is_on:
                        self.menu.show()
                        self.menu.move(self.x() - 165, self.y() + 80)
                    else:
                        self.menu.hide()
                
                self.update() # 觸發 paintEvent 重繪開關外觀
                
        # 5. 清理狀態
        self.drag_start_pos = None
        self.mouse_press_pos = None
        self.has_moved = False
    def stop_all_threads(self):
        """在程序退出前嘗試優雅停止/等待所有與 UI 相關的 QThread 實例，
        避免出現 "QThread: Destroyed while thread '' is still running" 的警告。
        """
        candidates = []
        try:
            if hasattr(self, 'menu') and self.menu:
                # model loader
                ml = getattr(self.menu, 'model_loader_thread', None)
                if isinstance(ml, QThread):
                    candidates.append(ml)
                # aimbot loop thread
                at = getattr(self.menu, 'aimbot_thread', None)
                if isinstance(at, QThread):
                    candidates.append(at)
        except Exception:
            pass

        for t in candidates:
            try:
                if t.isRunning():
                    try:
                        t.requestInterruption()
                    except Exception:
                        pass
                    try:
                        t.quit()
                    except Exception:
                        pass
                    t.wait(2000)
                    if t.isRunning():
                        try:
                            t.terminate()
                        except Exception:
                            pass
            except Exception:
                pass

# --- 程式入口 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        app.aboutToQuit.connect(stop_all_function_threads)
    except Exception:
        pass
    
    # 1. 顯示登入窗口
    login = LoginWindow()
    login.show()
    app.exec() 
    
    # 2. 驗證成功後進入主程式
    if hasattr(login, 'is_authenticated') and login.is_authenticated:
        logging.info("✅ 用戶驗證成功")
        
        # 4. 啟動主選單控制器
        try:
            logging.info("🚀 啟動主選單...")
            ctrl = MainController()
            ctrl.show()
            # 在應用退出前嘗試停止所有子線程，避免 QThread 錯誤
            try:
                app.aboutToQuit.connect(ctrl.stop_all_threads)
            except Exception:
                pass
            sys.exit(app.exec())
        except Exception as e:
            logging.error(f"❌ 主程式啟動失敗: {e}", exc_info=True)
            sys.exit(1)
            
    else:
        logging.info("❌ 用戶驗證失敗或取消登入")
        sys.exit()     