import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import glob

class KeypointDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        self.img_dir = img_dir
        self.annotations = []
        self.transform = transform
                
        # アノテーションファイルを読み込む
        with open(annotations_file) as f:
            data = json.load(f)
            for annotation in data:
                img_path = annotation['file_name']
                
                # 画像ファイルが存在する場合のみ追加
                if os.path.exists(img_path):
                    self.annotations.append(annotation)

        # データセットが空でないことを確認
        if len(self.annotations) == 0:
            print("Error: No valid images found in the dataset.")
            raise ValueError("No valid images found in the dataset.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_name = os.path.join(self.img_dir, annotation['file_name'])
        image = Image.open(img_name).convert("RGB")
        
        # キーポイントをリストに変換
        keypoints = []
        for point in annotation['points'].values():
            keypoints.extend(point)
        
        if self.transform:
            image = self.transform(image)

        return image, np.array(keypoints).astype(np.float32)

# 使用例
transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
])

# データセットのインスタンス作成
dataset = KeypointDataset('./', './data/data.json', transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# モデルの定義
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

model = KeypointCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習ループ
num_epochs = 10
for epoch in range(num_epochs):
    for images, keypoints in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, keypoints.view(-1, 42))  # 18はキーポイントの数
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# モデルの保存
torch.save(model.state_dict(), 'models/model.pth')