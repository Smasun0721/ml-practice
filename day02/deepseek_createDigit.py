import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os

def generate_digit_28x28(digit=7, save_path="./resource/demo.png"):
    """
    生成一张真正的 28x28 像素手写数字图（黑底白字）
    保存为 PNG，像素值 0~255
    """
    # 1. 创建 100x100 大画布
    img = Image.new("L", (100, 100), color=0)   # 黑色背景
    draw = ImageDraw.Draw(img)

    # 2. 加载字体
    try:
        font_path = "C:/Windows/Fonts/arial.ttf"
        if not os.path.exists(font_path):
            font_path = "/System/Library/Fonts/Helvetica.ttc"
        font = ImageFont.truetype(font_path, 85)
    except:
        font = ImageFont.load_default()

    # 3. 白色写数字
    draw.text((15, 5), str(digit), fill=255, font=font)

    # 4. 缩放到 28x28（高质量抗锯齿）
    try:
        img_28 = img.resize((28, 28), Image.Resampling.LANCZOS)
    except AttributeError:
        img_28 = img.resize((28, 28), Image.LANCZOS)

    # 5. 转为数组（用于显示）
    arr = np.array(img_28)

    # 6. 显示（仅查看）
    plt.imshow(arr, cmap='gray')
    plt.axis('off')
    plt.show()

    # 7. ★ 直接保存 PIL 图像，确保尺寸为 28x28
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img_28.save(save_path)   # 不加任何参数，保存为 PNG

    print(f"✅ 图片已生成并保存至: {save_path}")
    print(f"📊 数组形状: {arr.shape}, 像素值范围: {arr.min()} ~ {arr.max()}")
    return arr

# 生成数字 7
generate_digit_28x28(digit=3)