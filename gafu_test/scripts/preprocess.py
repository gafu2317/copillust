import cv2
import numpy as np
import json
import os

def resize_and_pad_images(image_data_list, target_size=224):
    """
    画像とアノテーションをリサイズし、アノテーションを調整する関数。

    Args:
        image_data_list (list): 画像データとアノテーションの辞書のリスト。
        target_size (int): リサイズ後の画像のサイズ。

    Returns:
        list: リサイズされた画像と調整されたアノテーションの辞書のリスト。
    """
    results = []

    for image_data in image_data_list:
        img = cv2.imread(image_data['file_name'])  # 画像を読み込む

        # 元のサイズを取得
        original_height, original_width = img.shape[:2]

        # アスペクト比を計算
        aspect_ratio = original_width / original_height

        # 新しいサイズを計算
        if original_width > original_height:
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(target_size * aspect_ratio)

        # リサイズする
        img_resized = cv2.resize(img, (new_width, new_height))

        # パディングを計算
        top = (target_size - new_height) // 2
        bottom = target_size - new_height - top
        left = (target_size - new_width) // 2
        right = target_size - new_width - left

        # パディングを追加
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # アノテーションのスケーリング
        scale_x = new_width / original_width
        scale_y = new_height / original_height

        adjusted_points = {}
        for key, point in image_data['points'].items():
            adjusted_x = point[0] * scale_x + left  # パディングを考慮
            adjusted_y = point[1] * scale_y + top    # パディングを考慮
            adjusted_points[key] = [adjusted_x, adjusted_y]

        # 新しいアノテーションを作成
        adjusted_data = {
            "file_name": image_data['file_name'],
            "points": adjusted_points,
            "width": target_size,
            "height": target_size
        }

        # 結果をリストに追加
        results.append((img_padded, adjusted_data))

    return results

# データを読み込むディレクトリを指定
image_data_file = '/Users/fukutomi/Library/CloudStorage/OneDrive-NITech/programming/c0de/pixivhackthon2/gafu_test/data/train.json'

# JSONファイルを読み込む
with open(image_data_file, 'r') as f:
    image_data_list = json.load(f)

# 出力ディレクトリの指定
img_output_dir = 'data/resized'
os.makedirs(img_output_dir, exist_ok=True)
annotation_output_dir = 'data/annotations'
os.makedirs(annotation_output_dir, exist_ok=True)

# リサイズとパディングを実行
results = []
for image_data in image_data_list:
    file_path = image_data['file_name']

    # 画像を読み込む
    img = cv2.imread(file_path)

    # 画像が読み込めなかった場合の処理
    if img is None:
        print(f"Error: Image {file_path} could not be loaded. Skipping...")
        continue  # 画像が読み込めなかった場合はスキップ

    # ここでリサイズとパディングを実行
    adjusted_results = resize_and_pad_images([image_data], target_size=224)  # 1つの画像データでリサイズを実行

    # リサイズした画像とアノテーションを保存
    for i, (padded_image, adjusted_data) in enumerate(adjusted_results):
        # 画像を保存
        cv2.imwrite(os.path.join(img_output_dir, f'resized_image_{len(results) + i}.png'), padded_image)

        # 調整されたアノテーションを保存
        with open(os.path.join(annotation_output_dir, f'adjusted_annotations_{len(results) + i}.json'), 'w') as f:
            json.dump(adjusted_data, f, indent=4)

    results.extend(adjusted_results)  # 結果を追加

print("リサイズ、パディング、アノテーションの調整が完了しました。")