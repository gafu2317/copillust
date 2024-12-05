# ディレクトリ構造

gafu_test/
│
├── data/
│   ├── images/          # アニメ画像が保存されるディレクトリ
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...          # その他の画像ファイル
│
├── models/              # トレーニング済みモデルを保存するディレクトリ
│   └── 
│
├── scripts/             # スクリプトファイルを保存するディレクトリ
│   ├── dataset.py       # データセットクラスの定義
│   ├── model.py         # モデルの定義
│   ├── train.py         # トレーニングループの実装
│   └── evaluate.py      # モデル評価のスクリプト
│
├── venv/                # 仮想環境
│
├── app.py               # 実行ファイル
├── requirements.txt     # 必要なパッケージをリストしたファイル
└── README.md            # プロジェクトの説明
