import os

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import img_to_array, load_img

from training_common import (
	ENTITY_CONFIGS,
	dataset_dir,
	denormalize_label_values,
	normalize_label_values,
	model_path,
	preprocess_image_array,
)


ENTITY_TYPE = 'line'
DATASET_SUFFIX = '_n1.1'
MODEL_PATH = model_path(ENTITY_TYPE, DATASET_SUFFIX)
IMAGE_DIR = dataset_dir(ENTITY_TYPE, DATASET_SUFFIX)

# 予測したい画像の指定方法:
# 1. 同じデータセットフォルダ内の画像を使う場合
#    例: IMG_PATH = os.path.join(IMAGE_DIR, 'p00001.jpg')
#
# 2. データセット外の新しい画像を直接指定する場合
#    例: IMG_PATH = r'C:\Users\ab21055\Desktop\my_line.jpg'
#
# 研究用に未学習画像で試したいときは、2 のように外部画像を指定するのが確実。
# ただし、その場合は labels.csv に正解がないので、下の「本物の属性値」は表示されない。
IMG_PATH = os.path.join(IMAGE_DIR, 'p00001.jpg')
LABEL_PATH = os.path.join(IMAGE_DIR, 'labels.csv')
TARGET_COLUMNS = ENTITY_CONFIGS[ENTITY_TYPE]['target_columns']

# モデルの読み込み
model = load_model(MODEL_PATH, compile=False)
# 必要なら以下で再コンパイル
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 画像の前処理
img = load_img(IMG_PATH, target_size=(224, 224))
img = preprocess_image_array(img_to_array(img))

img = np.expand_dims(img, axis=0)
print("--- 画像配列の内容（先頭10要素）---")
print(img.flatten()[:10])
print(f"mean: {np.mean(img):.4f}, max: {np.max(img):.4f}, min: {np.min(img):.4f}")

# 予測
pred = model.predict(img)[0]
pred_raw = denormalize_label_values(ENTITY_TYPE, pred)
print("--- 予測結果 ---")
for name, value in zip(TARGET_COLUMNS, pred_raw):
    print(f"{name:12}: {value:8.2f}")

# 本物の属性値をラベルCSVから取得
labels_df = pd.read_csv(LABEL_PATH)
image_name = os.path.basename(IMG_PATH)
row = labels_df.loc[labels_df['image'] == image_name]

print("--- 本物の属性値 ---")
if row.empty:
	print("label not found")
else:
	true_values = row.iloc[0][TARGET_COLUMNS].to_numpy(dtype=float)
	for name, value in zip(TARGET_COLUMNS, true_values):
		print(f"{name:12}: {value:8.2f}")

	true_values_normalized = normalize_label_values(ENTITY_TYPE, true_values)
	print("--- 正規化後の比較 ---")
	for name, pred_value, true_value in zip(TARGET_COLUMNS, pred, true_values_normalized):
		print(f"{name:12}: pred={pred_value:8.4f}, true={true_value:8.4f}")
