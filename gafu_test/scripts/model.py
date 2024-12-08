import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# データセットクラスの定義
class PoseDataset(Dataset):
    def __init__(self, image_paths, annotations, transform=None):
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)

        # アノテーションを取得
        annotation = self.annotations[idx]
        keypoints = annotation['points']

        # キーポイントをフラットなリストに変換
        flat_keypoints = []
        for point in keypoints.values():
            flat_keypoints.extend(point)  # x, yのペアを追加

        return image, flat_keypoints  # フラットなキーポイントを返す

# モデルの定義
class SimplePoseModel(nn.Module):
    def __init__(self):
        super(SimplePoseModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 122 * 122, 100)
        self.fc2 = nn.Linear(100, 42)  # 例: 21キーポイント（x,y）

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = x.view(-1, 16 * 122 * 122)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# モデルのトレーニングを行う関数
def train_pose_model(image_paths, annotations, batch_size=32, num_epochs=10, learning_rate=0.001):
    # 画像の前処理
    transform = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
    ])

    # データセットとデータローダーを作成
    dataset = PoseDataset(image_paths, annotations, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # モデルのインスタンス化
    model = SimplePoseModel()
    
    # 損失関数とオプティマイザの設定
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # トレーニングループ
    for epoch in range(num_epochs):
        print("Training model...5.1")
        for images, keypoints in dataloader:
            print("Training model...5.2")
            optimizer.zero_grad()
            print("Training model...5.3")
            # keypointsをTensorに変換
            keypoints = torch.tensor(keypoints, dtype=torch.float32)
            print("Training model...5.4")
            outputs = model(images)
            print("Training model...5.5")
            loss = criterion(outputs, keypoints)
            print("Training model...5.6")
            loss.backward()
            print("Training model...5.7")
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training model...6")
    # モデルの保存先ディレクトリを指定
    model_dir = '/Users/fukutomi/Library/CloudStorage/OneDrive-NITech/programming/c0de/pixivhackthon2/gafu_test/models'
    os.makedirs(model_dir, exist_ok=True)

    print("Training model...7")
    # モデルの保存
    torch.save(model.state_dict(), os.path.join(model_dir, 'pose_model.pth'))
    print("Model saved as model/pose_model.pth")
    print("Model saved as pose_model.pth")
