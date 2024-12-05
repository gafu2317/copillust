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
        # 骨格を描画した画像を作成
        mp_drawing.draw_landmarks(
            image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    return image  # 骨格を描画した画像を返す