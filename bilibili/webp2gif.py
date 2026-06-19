# -*- coding: utf-8 -*-
"""
WebP 转 GIF（支持动态 webp，保留帧时长；静态 webp 转单帧 gif）。
透明背景会填成指定颜色（GIF 只支持 1 位透明，默认白底）。

用法：
    python webp2gif.py 图片.webp                 # 输出同名 .gif
    python webp2gif.py 图片.webp -o out.gif
    python webp2gif.py 文件夹 --batch            # 批量转文件夹内所有 webp
    python webp2gif.py a.webp --bg "#000000"     # 透明区域用黑色填充
"""
import argparse
import os
import sys
from PIL import Image, ImageSequence


def _hex2rgb(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def convert(src, dst, bg=(255, 255, 255)):
    im = Image.open(src)
    frames = []
    durations = []
    for frame in ImageSequence.Iterator(im):
        rgba = frame.convert("RGBA")
        # 把透明区域贴到纯色背景上，再转成 GIF 调色板
        canvas = Image.new("RGBA", rgba.size, bg + (255,))
        canvas.alpha_composite(rgba)
        frames.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=256))
        durations.append(frame.info.get("duration", 100))

    if not frames:
        sys.exit(f"读不到帧: {src}")

    frames[0].save(
        dst,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=True,
    )
    n = len(frames)
    size_kb = os.path.getsize(dst) / 1024
    kind = "动态" if n > 1 else "静态"
    print(f"完成: {dst}  ({kind} {n}帧, {size_kb:.1f} KB)")


def main():
    p = argparse.ArgumentParser(description="WebP 转 GIF")
    p.add_argument("input", help="webp 文件，或 --batch 时为文件夹")
    p.add_argument("-o", "--output", help="输出 gif 路径（单文件时有效）")
    p.add_argument("--bg", default="#FFFFFF", help="透明区域填充色，默认白色 #FFFFFF")
    p.add_argument("--batch", action="store_true", help="转换文件夹内所有 webp")
    args = p.parse_args()
    bg = _hex2rgb(args.bg)

    if args.batch:
        folder = args.input
        if not os.path.isdir(folder):
            sys.exit(f"--batch 需要文件夹: {folder}")
        webps = [f for f in os.listdir(folder) if f.lower().endswith(".webp")]
        if not webps:
            sys.exit("文件夹里没有 webp 文件")
        for name in webps:
            src = os.path.join(folder, name)
            dst = os.path.splitext(src)[0] + ".gif"
            convert(src, dst, bg)
    else:
        src = args.input
        dst = args.output or (os.path.splitext(src)[0] + ".gif")
        convert(src, dst, bg)


if __name__ == "__main__":
    main()
