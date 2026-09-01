import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import img_to_array, load_img

from training_common import (
    ENTITY_CONFIGS,
    dataset_dir,
    denormalize_label_values,
    model_path,
    preprocess_image_array,
)

# --- 設定 ---
ENTITY_TYPE = os.environ.get('ENTITY_TYPE', 'arc')
DEFAULT_MODEL_DATASET_SUFFIX = '_n1.1' if ENTITY_TYPE == 'line' else '_n1'
MODEL_DATASET_SUFFIX = os.environ.get('MODEL_DATASET_SUFFIX', DEFAULT_MODEL_DATASET_SUFFIX)
EVAL_DATASET_SUFFIX = os.environ.get('EVAL_DATASET_SUFFIX', '_n2')
SHOW_PLOTS = os.environ.get('SHOW_PLOTS', '0') == '1'
MODEL_PATH = model_path(ENTITY_TYPE, MODEL_DATASET_SUFFIX)
MODEL_STEM = os.path.splitext(os.path.basename(MODEL_PATH))[0]
IMG_SIZE = (224, 224)
LABEL_CSV = 'labels.csv'
IMG_DIR = dataset_dir(ENTITY_TYPE, EVAL_DATASET_SUFFIX)
TARGET_COLUMNS = ENTITY_CONFIGS[ENTITY_TYPE]['target_columns']
SAMPLE_SIZE = 100
RANDOM_SEED = 42
RESULT_CSV_PATH = f'prediction_results_{MODEL_STEM}_eval{EVAL_DATASET_SUFFIX}.csv'
HISTOGRAM_PATH_PREFIX = f'error_hist_{MODEL_STEM}_eval{EVAL_DATASET_SUFFIX}'


def circular_error_degrees(pred_values, true_values):
    """度数法の角度差を、360 度周期を考慮して計算する。"""
    difference = np.mod(np.abs(pred_values - true_values), 360.0)
    return np.minimum(difference, 360.0 - difference)

print('--- 評価設定 ---')
print(f'entity_type: {ENTITY_TYPE}')
print(f'model_dataset_suffix: {MODEL_DATASET_SUFFIX}')
print(f'eval_dataset_suffix: {EVAL_DATASET_SUFFIX}')
print(f'model_path: {MODEL_PATH}')
print(f'image_dir: {IMG_DIR}')
print(f'show_plots: {SHOW_PLOTS}')

# --- モデル読み込み ---
model = load_model(MODEL_PATH, compile=False)
labels_df = pd.read_csv(os.path.join(IMG_DIR, LABEL_CSV))

# 画像ファイルリストを取得し、ランダムに100枚だけ選択
all_image_files = glob.glob(os.path.join(IMG_DIR, '*.png')) + glob.glob(os.path.join(IMG_DIR, '*.jpg'))
all_image_files = sorted(all_image_files)  # ファイル名順でソート
if len(all_image_files) > SAMPLE_SIZE:
    import random
    random.seed(RANDOM_SEED)
    image_files = random.sample(all_image_files, SAMPLE_SIZE)
    image_files = sorted(image_files)  # サンプル後も順序を固定
else:
    image_files = all_image_files

# ラベル数と画像数が一致している前提で、ファイル名順で対応
preds = []
labels = []
img_names = []

for idx, img_path in enumerate(image_files):
    img_name = os.path.basename(img_path)
    row = labels_df.loc[labels_df['image'] == img_name]
    if row.empty:
        continue
    label = row.iloc[0][TARGET_COLUMNS].to_numpy(dtype=float)
    # 画像前処理
    img = load_img(img_path, target_size=IMG_SIZE)
    img = preprocess_image_array(img_to_array(img))
    pred = model.predict(img[np.newaxis, ...])[0]
    preds.append(denormalize_label_values(ENTITY_TYPE, pred))
    labels.append(label)
    img_names.append(img_name)

preds = np.array(preds)
labels = np.array(labels)

# --- 誤差計算と統計 ---
errors = np.abs(preds - labels)
if ENTITY_TYPE == 'arc':
    for angle_name in ('start_angle', 'cw_sweep_angle'):
        angle_index = TARGET_COLUMNS.index(angle_name)
        errors[:, angle_index] = circular_error_degrees(preds[:, angle_index], labels[:, angle_index])

# --- 画像ごとの予測結果を表にまとめてCSV保存 ---
rows = []
for i, img_name in enumerate(img_names):
    row_data = {"image": img_name}
    total_abs_error = 0.0
    for j, attr_name in enumerate(TARGET_COLUMNS):
        pred_value = float(preds[i, j])
        true_value = float(labels[i, j])
        abs_error = float(errors[i, j])
        row_data[f"pred_{attr_name}"] = pred_value
        row_data[f"true_{attr_name}"] = true_value
        row_data[f"abs_error_{attr_name}"] = abs_error
        total_abs_error += abs_error
    row_data["mean_abs_error"] = total_abs_error / len(TARGET_COLUMNS)
    row_data["sum_abs_error"] = total_abs_error
    rows.append(row_data)

results_df = pd.DataFrame(rows)
results_df.to_csv(RESULT_CSV_PATH, index=False, encoding='utf-8-sig')
print(f'--- 画像ごとの予測結果を保存 ---')
print(f'保存先: {RESULT_CSV_PATH}')
print(results_df.head(10).to_string(index=False))


attr_names = TARGET_COLUMNS
print('--- 属性ごとの誤差統計 ---')
for i in range(errors.shape[1]):
    attr_errors = errors[:, i]
    name = attr_names[i] if i < len(attr_names) else f'属性{i+1}'
    print(f'{name}: 平均={attr_errors.mean():.3f}, 標準偏差={attr_errors.std():.3f}, 最大={attr_errors.max():.3f}, 最小={attr_errors.min():.3f}')
    plt.hist(attr_errors, bins=30, alpha=0.5)
    plt.title(f'{name} 誤差分布')
    plt.xlabel('誤差')
    plt.ylabel('件数')
    histogram_path = f'{HISTOGRAM_PATH_PREFIX}_{name}.png'
    plt.savefig(histogram_path, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

print(f'全体平均誤差: {errors.mean():.3f}')
print(f'全体標準偏差: {errors.std():.3f}')

print('--- 誤差上位10件 ---')
top10_df = results_df.sort_values('sum_abs_error', ascending=False).head(10)
display_columns = ['image', 'mean_abs_error', 'sum_abs_error']
for attr_name in TARGET_COLUMNS:
    display_columns.append(f'abs_error_{attr_name}')
print(top10_df[display_columns].to_string(index=False))
