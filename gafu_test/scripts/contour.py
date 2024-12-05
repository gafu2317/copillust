import cv2
import numpy as np

def contour(image):
    """画像を処理してエッジと輪郭を検出"""
    # OpenCVのBGR形式からRGB形式に変換
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 画像のぼかし
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # エッジを検出
    edges = cv2.Canny(blurred, 50, 150)

    # 輪郭を検出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 元の画像に輪郭を描画
    output = image.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

    return output
