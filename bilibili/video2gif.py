# -*- coding: utf-8 -*-
"""
视频转 GIF 脚本（调用 ffmpeg，两遍调色板法，高质量小体积）。

用法示例：
    # 整段转换，默认 12fps、宽度 480
    python video2gif.py input.mp4

    # 指定输出
    python video2gif.py input.mp4 -o out.gif

    # 截取第 3 秒开始、时长 5 秒
    python video2gif.py input.mp4 -ss 3 -t 5

    # 自定义帧率与宽度（高度按比例自动）
    python video2gif.py input.mp4 --fps 15 --width 600

    # 批量：把某文件夹下所有视频各转成同名 gif
    python video2gif.py D:\\videos --batch
"""
import argparse
import os
import subprocess
import sys

# 你的 ffmpeg 路径（D:\ffmped\ffmpeg\bin\ffmpeg.exe）
FFMPEG = r"D:\ffmped\ffmpeg\bin\ffmpeg.exe"

VIDEO_EXTS = (".mp4", ".flv", ".mkv", ".avi", ".mov", ".webm", ".ts", ".m4v")


def convert(src, dst, fps=12, width=480, ss=None, t=None, loop=0):
    """用两遍调色板法把单个视频转成 GIF。"""
    if not os.path.isfile(FFMPEG):
        sys.exit(f"找不到 ffmpeg: {FFMPEG}")
    if not os.path.isfile(src):
        sys.exit(f"找不到输入文件: {src}")

    # 滤镜：抽帧 + 缩放（width=-1 表示保持原宽，-2 表示按比例且为偶数）
    fps_scale = f"fps={fps},scale={width}:-1:flags=lanczos"
    palette = dst + ".palette.png"

    # 公共的时间裁剪参数（放在 -i 前面更快，按关键帧定位）
    trim = []
    if ss is not None:
        trim += ["-ss", str(ss)]
    if t is not None:
        trim += ["-t", str(t)]

    # 第一遍：生成调色板
    cmd1 = [FFMPEG, "-y", *trim, "-i", src,
            "-vf", f"{fps_scale},palettegen=stats_mode=diff",
            "-update", "1", palette]
    # 第二遍：用调色板生成 GIF
    cmd2 = [FFMPEG, "-y", *trim, "-i", src, "-i", palette,
            "-lavfi", f"{fps_scale} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5",
            "-loop", str(loop), dst]

    print(f"[1/2] 生成调色板 ...")
    subprocess.run(cmd1, check=True)
    print(f"[2/2] 生成 GIF ...")
    subprocess.run(cmd2, check=True)

    if os.path.isfile(palette):
        os.remove(palette)

    size_mb = os.path.getsize(dst) / 1024 / 1024
    print(f"完成: {dst}  ({size_mb:.2f} MB)")


def main():
    p = argparse.ArgumentParser(description="视频转 GIF（基于 ffmpeg）")
    p.add_argument("input", help="输入视频文件，或 --batch 时为文件夹")
    p.add_argument("-o", "--output", help="输出 gif 路径（单文件时有效）")
    p.add_argument("--fps", type=int, default=12, help="帧率，默认 12")
    p.add_argument("--width", type=int, default=480, help="输出宽度像素，默认 480，高度按比例")
    p.add_argument("-ss", "--start", help="起始时间，如 3 或 00:00:03")
    p.add_argument("-t", "--duration", help="持续时长（秒）")
    p.add_argument("--loop", type=int, default=0, help="循环次数，0=无限")
    p.add_argument("--batch", action="store_true", help="把文件夹内所有视频各转一份")
    args = p.parse_args()

    if args.batch:
        folder = args.input
        if not os.path.isdir(folder):
            sys.exit(f"--batch 需要文件夹: {folder}")
        vids = [f for f in os.listdir(folder)
                if f.lower().endswith(VIDEO_EXTS)]
        if not vids:
            sys.exit("文件夹里没有找到视频文件")
        for name in vids:
            src = os.path.join(folder, name)
            dst = os.path.splitext(src)[0] + ".gif"
            print(f"\n=== {name} ===")
            convert(src, dst, args.fps, args.width, args.start, args.duration, args.loop)
    else:
        src = args.input
        dst = args.output or (os.path.splitext(src)[0] + ".gif")
        convert(src, dst, args.fps, args.width, args.start, args.duration, args.loop)


if __name__ == "__main__":
    main()
