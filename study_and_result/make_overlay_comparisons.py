#!/usr/bin/env python3
"""
make_overlay_comparisons.py の使い方

概要:
 - 予測結果の CSV と対応するテスト画像フォルダを読み込み、真値（緑）と予測（赤）を重ねたオーバーレイ画像を生成します。
 - 出力は `output/overlays/<entity>/` に `<image>_overlay.png` の形で保存されます。

設定方法:
 - CSV ファイル名はスクリプト内で定義されたファイル名を使うか、または CSV 名を変更して実行してください。
 - 1 エンティティあたりの出力枚数は、環境変数 `OVERLAY_K` で上書きできます（例: `OVERLAY_K=20`）。デフォルトはスクリプト内の設定値です。
 - 背景を白紙にするか元画像を使うかは `BLANK_BACKGROUND` の値を切り替えてください（True=白背景、False=元画像）。

注意:
 - CSV の列名は `image`（画像ファイル名）、真値列は `true_...`、予測列は `pred_...` の形式である必要があります。
 - 画像は CSV の `image` 列に書かれたファイル名で `test_*` フォルダから読み込みます。
"""

import os
import math
import random
import pandas as pd
from PIL import Image, ImageDraw


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def draw_circle(draw, cx, cy, r, color, width=3):
    bbox = [cx - r, cy - r, cx + r, cy + r]
    draw.ellipse(bbox, outline=color, width=width)


def draw_arc(draw, cx, cy, r, start_deg, cw_sweep_deg, color, width=3):
    # PIL draws arcs CCW from start to end. cw_sweep_deg is clockwise sweep.
    # Convert to CCW end angle
    start = start_deg % 360
    end = (start - cw_sweep_deg) % 360
    bbox = [cx - r, cy - r, cx + r, cy + r]
    draw.arc(bbox, start=start, end=end, fill=color, width=width)


def draw_line_from_center(draw, cx, cy, half_length, cos2, sin2, color, width=3):
    # recover theta from cos(2θ), sin(2θ)
    two_theta = math.atan2(sin2, cos2)
    theta = 0.5 * two_theta
    dx = math.cos(theta) * half_length
    dy = math.sin(theta) * half_length
    x1 = cx - dx
    y1 = cy - dy
    x2 = cx + dx
    y2 = cy + dy
    draw.line([x1, y1, x2, y2], fill=color, width=width)


BLANK_BACKGROUND = True


# ------------------
TASKS = {
    # example entries; change filenames/folders as needed
    'circle': ('prediction_results_circle_model_n1_eval_n2.csv', 'test_circle_data_n2', None),
    'arc': ('prediction_results_model_arc_circularloss_data_n1_eval_n2.csv', 'test_arc_data_n2', None),
    'line': ('prediction_results_model_line_data_n1.1_eval_n2.csv', 'test_line_data_n2', None),
}

SELECT_K = 10
OUT_ROOT = os.path.join('output', 'overlays')
#
# ------------------


def process_entity(entity_name, pred_csv, test_folder, out_folder, draw_fn, select_k=10):
    df = pd.read_csv(pred_csv)
    ensure_dir(out_folder)
    # choose images: random sample if enough rows
    rows = df.to_dict(orient='records')
    if len(rows) == 0:
        print('No rows in', pred_csv)
        return
    sample = rows[:select_k] if len(rows) <= select_k else random.sample(rows, select_k)
    for row in sample:
        img_name = row.get('image')
        img_path = os.path.join(test_folder, img_name)
        if BLANK_BACKGROUND:
            # assume images are 224x224; if not, fall back to opening file
            try:
                im = Image.new('RGB', (224, 224), 'white')
            except Exception:
                if not os.path.exists(img_path):
                    print('Image not found:', img_path)
                    continue
                im = Image.open(img_path).convert('RGB')
        else:
            if not os.path.exists(img_path):
                print('Image not found:', img_path)
                continue
            im = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(im)
        # draw ground truth in green
        try:
            draw_fn(draw, row, pred=False)
        except Exception as e:
            print('draw true failed', img_name, e)
        # draw prediction in red
        try:
            draw_fn(draw, row, pred=True)
        except Exception as e:
            print('draw pred failed', img_name, e)
        out_path = os.path.join(out_folder, img_name.replace('.jpg', '') + '_overlay.png')
        im.save(out_path)
    print('Saved overlays to', out_folder)


def circle_draw_fn(draw, row, pred=False):
    if pred:
        cx = float(row['pred_center_x'])
        cy = float(row['pred_center_y'])
        r = float(row['pred_radius'])
        color = (255, 0, 0)
    else:
        cx = float(row['true_center_x'])
        cy = float(row['true_center_y'])
        r = float(row['true_radius'])
        color = (0, 255, 0)
    draw_circle(draw, cx, cy, r, color, width=3)


def arc_draw_fn(draw, row, pred=False):
    if pred:
        cx = float(row['pred_center_x'])
        cy = float(row['pred_center_y'])
        r = float(row['pred_radius'])
        start = float(row['pred_start_angle'])
        sweep = float(row['pred_cw_sweep_angle'])
        color = (255, 0, 0)
    else:
        cx = float(row['true_center_x'])
        cy = float(row['true_center_y'])
        r = float(row['true_radius'])
        start = float(row['true_start_angle'])
        sweep = float(row['true_cw_sweep_angle'])
        color = (0, 255, 0)
    draw_arc(draw, cx, cy, r, start, sweep, color, width=3)


def line_draw_fn(draw, row, pred=False):
    if pred:
        cx = float(row['pred_center_x'])
        cy = float(row['pred_center_y'])
        half_length = float(row['pred_half_length'])
        cos2 = float(row['pred_cos2_theta'])
        sin2 = float(row['pred_sin2_theta'])
        color = (255, 0, 0)
    else:
        cx = float(row['true_center_x'])
        cy = float(row['true_center_y'])
        half_length = float(row['true_half_length'])
        cos2 = float(row['true_cos2_theta'])
        sin2 = float(row['true_sin2_theta'])
        color = (0, 255, 0)
    draw_line_from_center(draw, cx, cy, half_length, cos2, sin2, color, width=3)


if __name__ == '__main__':
    random.seed(1)
    # mapping: entity -> (prediction CSV, test folder, draw fn)
    tasks = {
        'circle': ('prediction_results_circle_model_n1_eval_n2.csv', 'test_circle_data_n2', circle_draw_fn),
        'arc': ('prediction_results_model_arc_circularloss_data_n1_eval_n2.csv', 'test_arc_data_n2', arc_draw_fn),
        'line': ('prediction_results_model_line_data_n1.1_eval_n2.csv', 'test_line_data_n2', line_draw_fn),
    }
    out_root = os.path.join('output', 'overlays')
    ensure_dir(out_root)
    # number of overlays per entity can be set with env OVERLAY_K (default 10)
    try:
        select_k = int(os.environ.get('OVERLAY_K', '10'))
    except Exception:
        select_k = 10
    for name, (csvf, testdir, fn) in tasks.items():
        csv_path = os.path.join('.', csvf)
        test_folder = os.path.join('.', testdir)
        out_folder = os.path.join(out_root, name)
        if not os.path.exists(csv_path):
            print('Prediction CSV not found for', name, csv_path)
            continue
        if not os.path.isdir(test_folder):
            print('Test folder not found for', name, test_folder)
            continue
        process_entity(name, csv_path, test_folder, out_folder, fn, select_k=select_k)
