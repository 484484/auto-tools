import cv2
import numpy as np
import pyautogui
import time
import sys
import os
import argparse
from datetime import datetime

pyautogui.FAILSAFE = True

CONFIDENCE_THRESHOLD = 0.8                  #置信度阈值
SCAN_INTERVAL = 0.5                         #扫描间隔
CLICK_COOLDOWN = 2.0                        #点击后冷却时间


def load_templates(image_paths: list):
    templates = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"[错误] 找不到图片: {image_path}")
            sys.exit(1)

        template = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if template is None:
            print(f"[错误] 无法读取图片: {image_path}")
            sys.exit(1)

        templates.append({"path": image_path, "image": template})
    return templates


def capture_screen():
    screenshot = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return frame


def find_and_click(screen, templates, threshold: float) -> bool:
    # 遍历所有目标图片进行匹配（列表顺序即为优先级）
    for t_data in templates:
        template = t_data["image"]
        img_path = t_data["path"]

        result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            h, w = template.shape[:2]
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2

            now = datetime.now().strftime("%H:%M:%S")
            # 打印中包含匹配成功的具体图片名称，方便调试
            print(
                f"[{now}] 匹配成功 [{os.path.basename(img_path)}] (置信度: {max_val:.3f}) -> 点击坐标 ({center_x}, {center_y})")

            pyautogui.moveTo(center_x, center_y, duration=0.15)
            pyautogui.click()
            return True

    return False


def run(image_paths: list, threshold: float, interval: float, cooldown: float, once: bool):
    templates = load_templates(image_paths)

    print("=" * 50)
    print(f"  共加载了 {len(templates)} 个目标图片 (按顺序优先匹配):")
    for t in templates:
        h, w = t["image"].shape[:2]
        print(f"    - {t['path']} ({w}x{h})")
    print(f"  匹配阈值: {threshold}")
    print(f"  扫描间隔: {interval}s")
    print(f"  点击冷却: {cooldown}s")
    print(f"  单次模式: {'是（点击任意一个即退出）' if once else '否（Ctrl+C 停止）'}")
    print(f"  安全机制: 鼠标移到屏幕左上角可紧急停止")
    print("=" * 50)
    print("监测中...\n")

    try:
        while True:
            screen = capture_screen()
            clicked = find_and_click(screen, templates, threshold)

            if clicked:
                if once:
                    print("单次模式，已完成点击，退出。")
                    break
                time.sleep(cooldown)
            else:
                time.sleep(interval)

    except KeyboardInterrupt:
        print("\n已手动停止监测。")
    except pyautogui.FailSafeException:
        print("\n触发安全机制（鼠标移至左上角），已停止。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="屏幕图像识别自动点击工具 (支持多图)")
    # 使用 nargs='+' 允许传入一个或多个图片路径
    parser.add_argument("images", nargs='+', help="目标图片路径 (png/jpg)，多张图片用空格隔开")
    parser.add_argument("-t", "--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"匹配置信度阈值 0~1 (默认 {CONFIDENCE_THRESHOLD})")
    parser.add_argument("-i", "--interval", type=float, default=SCAN_INTERVAL,
                        help=f"扫描间隔秒数 (默认 {SCAN_INTERVAL})")
    parser.add_argument("-c", "--cooldown", type=float, default=CLICK_COOLDOWN,
                        help=f"点击后冷却秒数 (默认 {CLICK_COOLDOWN})")
    parser.add_argument("--once", action="store_true",
                        help="只要点击了任意一张图片就退出")

    args = parser.parse_args()
    run(args.images, args.threshold, args.interval, args.cooldown, args.once)
