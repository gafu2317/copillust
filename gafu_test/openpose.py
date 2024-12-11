import cv2
import torch
from torchvision import transforms
from PIL import Image
from controlnet import ControlNetModel

# モデルの読み込み
model = ControlNetModel.from_pretrained("https://huggingface.co/gsdf/Counterfeit-V3.0/resolve/main/Counterfeit-V3.0_fix_fp16.safetensors")
model.eval()

# 画像の読み込み
image_path = "/Users/fukutomi/Library/CloudStorage/OneDrive-NITech/programming/c0de/pixivhackthon2/gafu_test/samlpe_images/sample1.png"  # イラストのパス
input_image = Image.open(image_path).convert("RGB")

# 画像の前処理
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),  # モデルの入力サイズに合わせてリサイズ
    transforms.ToTensor(),
])
input_tensor = preprocess(input_image).unsqueeze(0)  # バッチ次元を追加

# 画像をモデルに入力
with torch.no_grad():
    output = model(input_tensor)

# 出力の後処理（骨格を画像に描画）
skeleton_image = output.squeeze().permute(1, 2, 0).numpy()  # チャンネル順を変更
skeleton_image = (skeleton_image * 255).astype('uint8')  # 画像に変換

# 結果の表示
cv2.imshow('Skeleton', skeleton_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
