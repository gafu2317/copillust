import os
import json
import torch
import gradio as gr
from torchvision import transforms
from PIL import Image, ImageDraw  # ImageDrawをインポート
from scripts.model import SimplePoseModel  # model.pyからモデルをインポート
import cv2
from scripts.model import train_pose_model  # モデルのトレーニング関数のインポート

# モデルをトレーニング
try:
    image_dir = '/Users/fukutomi/Library/CloudStorage/OneDrive-NITech/programming/c0de/pixivhackthon2/gafu_test/data/resized/'
    annotation_dir = '/Users/fukutomi/Library/CloudStorage/OneDrive-NITech/programming/c0de/pixivhackthon2/gafu_test/data/annotations/'

    images = [os.path.join(image_dir, img_file) for img_file in os.listdir(image_dir) if img_file.endswith('.png') or img_file.endswith('.jpg')]
    annotation_files = [os.path.join(annotation_dir, annotation_file) for annotation_file in os.listdir(annotation_dir) if annotation_file.endswith('.json')]

    annotations = []
    for annotation_file in annotation_files:
        with open(annotation_file, 'r') as f:
            annotation = json.load(f)
        annotations.append(annotation['points'])

    # 画像をモデルが期待する形式に変換
    transform = transforms.Compose([
        transforms.Resize((244, 244)),  # モデルに合わせたサイズにリサイズ
        transforms.ToTensor(),  # テンソルに変換
    ])

    processed_images = []
    for img_path in images:
        img = Image.open(img_path).convert("RGB")  # 画像を読み込む
        img = transform(img)  # 前処理を適用
        processed_images.append(img)

    # バッチにする
    processed_images = torch.stack(processed_images)  # バッチ化

    # モデルをトレーニング
    train_pose_model(processed_images, annotations, batch_size=32, num_epochs=10, learning_rate=0.001)

except Exception as e:
    print(f"Error during training: {e}")

# # モデルのロード
# model = SimplePoseModel()
# model.load_state_dict(torch.load('pose_model.pth'))
# model.eval()

# # 画像の前処理
# transform = transforms.Compose([
#     transforms.Resize((244, 244)),
#     transforms.ToTensor(),
# ])

# # 骨格を描画する関数
# def draw_keypoints(image, keypoints):
#     draw = ImageDraw.Draw(image)
#     for i in range(0, len(keypoints), 2):  # x, y座標が交互に格納されていると仮定
#         x = keypoints[i]
#         y = keypoints[i + 1]
#         draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill='red')  # 骨格ポイントを赤い円で描画
#     return image

# # 骨格抽出を行う関数
# def predict_pose(image):
#     original_image = image.copy()  # 元の画像を保持
#     image = transform(image).unsqueeze(0)  # バッチサイズの次元を追加
#     with torch.no_grad():
#         output = model(image)
    
#     keypoints = output.numpy().flatten()  # NumPy配列として出力
#     result_image = draw_keypoints(original_image, keypoints)  # 骨格を描画
#     return result_image  # 描画した画像を返す

# # Gradioインターフェースの定義
# iface = gr.Interface(
#     fn=predict_pose,
#     inputs=gr.Image(type="pil"),  # PIL形式で画像を受け取る
#     outputs=gr.Image(type="pil"),  # 出力もPIL形式
#     title="骨格抽出アプリ",
#     description="画像をアップロードすると、骨格のキーポイントを抽出して描画します。",
# )

# # アプリの起動
# if __name__ == "__main__":
#     iface.launch()
