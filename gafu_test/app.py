import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np

# モデルの定義とロード
class KeypointCNN(nn.Module):
    def __init__(self):
        super(KeypointCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 61 * 61, 128)  # 244/4 = 61
        self.fc2 = nn.Linear(128, 42)  # 21 keypoints * 2 (x, y)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 61 * 61)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# モデルのインスタンス作成と重みのロード
model = KeypointCNN()
model.load_state_dict(torch.load('models/model.pth'))
model.eval()  # 推論モードに切り替え

# 画像の変換
transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
])

# 推論関数
def predict(image_path):
    # 画像をオープンして変換
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # 元のサイズを取得
    image_tensor = transform(image).unsqueeze(0)  # バッチ次元を追加

    with torch.no_grad():
        output = model(image_tensor)
    
    keypoints = output.numpy().flatten()  # 出力をリストに変換
    return keypoints, original_size  # キーポイントと元のサイズを返す

# キーポイントを画像に描画する関数
def draw_skeleton(keypoints, original_size):
    skeleton_image = Image.new('RGB', original_size, (255, 255, 255))
    draw = ImageDraw.Draw(skeleton_image)
    keypoints = keypoints.reshape(-1, 2)

    for i, (x, y) in enumerate(keypoints):
        draw.ellipse((x-3, y-3, x+3, y+3), fill='red')  # キーポイントを赤い円で描画
        draw.text((x, y), str(i), fill='black')  # インデックスを表示

    # 骨格を描画（接続関係を確認）
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # 右手
        (0, 5), (5, 6), (6, 7), (7, 8),  # 左手
        (0, 9), (9, 10), (10, 11),  # 体幹
        (11, 12), (12, 13),  # 足
        (10, 14), (14, 15),  # 足
        (0, 16), (16, 17),  # 頭
    ]

    for start, end in connections:
        x1, y1 = keypoints[start]
        x2, y2 = keypoints[end]
        draw.line((x1, y1, x2, y2), fill='blue', width=2)

    return skeleton_image

# 画像ファイルパスを指定して推論を実行
image_path = '/Users/fukutomi/Library/CloudStorage/OneDrive-NITech/programming/c0de/pixivhackthon2/gafu_test/samlpe_images/sample2.png'
keypoints, original_size = predict(image_path)

# 描画して表示
skeleton_image = draw_skeleton(keypoints, original_size)
skeleton_image.show()  # または skeleton_image.save('output.jpg') で保存