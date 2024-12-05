import gradio as gr
from scripts.contour import contour # 輪郭抽出関数のインポート
from scripts.mediapipe import detect_pose # ファイル監視関数のインポート

# Gradioインターフェースの定義
iface = gr.Interface(
    fn=detect_pose,  
    inputs=gr.Image(type="numpy"),  # 入力タイプはNumPy配列
    outputs=gr.Image(type="numpy"),  # 出力はNumPy配列
    title="Contour Extraction",
    description="Upload an image to detect edges and draw contours."
)

# アプリの起動
iface.launch()