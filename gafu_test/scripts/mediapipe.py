import cv2
import mediapipe as mp

# Mediapipeの初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def detect_pose(image):
    """
    入力画像から人物の骨格を検出し、結果を描画して返す関数。

    Parameters:
    image (numpy.ndarray): 入力画像 (BGR形式)

    Returns:
    output (numpy.ndarray): 骨格を描画した画像
    """
    # 画像の高さと幅を取得
    height, width, _ = image.shape

    # BGRからRGBに変換（Mediapipeが必要とするフォーマット）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Mediapipeで骨格検出を実行
    result = pose.process(image_rgb)

    # 検出結果を描画
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        # 各関節の座標を取得して出力
        for i, landmark in enumerate(result.pose_landmarks.landmark):
            x = landmark.x * width    # x座標をピクセル単位に変換
            y = landmark.y * height   # y座標をピクセル単位に変換
            z = landmark.z            # z座標（深度情報）は正規化されている
            print(f"関節 {i}: x={x:.2f}, y={y:.2f}, z={z:.2f}")

    return image  # 骨格を描画した画像を返す

# # 画像を読み込む
# image_path = "input_image.jpg"  # ここに画像ファイルのパスを指定
# image = cv2.imread(image_path)

# if image is None:
#     print("Error: 画像を開けませんでした。")
#     exit()

# # 骨格を検出
# output_image = detect_pose(image)

# # 結果を表示
# cv2.imshow('Pose Detection', output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
