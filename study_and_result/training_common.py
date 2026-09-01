"""
学習処理の共通部品をまとめたモジュール。

今回追加した内容:
1. 円・線・弧を別々に学習するための設定を ENTITY_CONFIGS に分離
2. labels.csv から各エンティティに必要な列だけを読む処理を共通化
3. モデル保存名と loss 曲線保存名を図形ごとに分離
4. VGG16 向け前処理と回帰 head の構成を 1 か所に集約
5. 生のラベル値をコード内で正規化し、予測時は逆正規化できるように変更
6. val_loss が停滞したときに学習率を下げ、改善が止まれば早期終了する callback を追加

このファイル自体は直接実行せず、train_circle.py / train_line.py / train_arc.py
から呼び出して使う。

ラベルの前提:
- CSV には正規化前の生の値を入れる
- 座標と半径は 0 から 224 の範囲を想定
- 角度と sweep 角は度数法で 0 から 360 の範囲を想定

このファイルの役割:
- train_circle.py / train_line.py / train_arc.py から共通で使う処理をまとめる
- 画像読み込み、ラベル読み込み、ラベル正規化、モデル作成、学習、保存を一括で担当する
- 図形ごとの差分は ENTITY_CONFIGS に閉じ込め、学習本体は同じ処理で回せるようにする

処理の流れ:
1. ENTITY_CONFIGS で図形ごとのラベル列を定義する
2. load_entity_dataset() で画像と CSV を読み込む
3. 画像は VGG16 の preprocess_input に合わせて前処理する
4. ラベルは 224 / 360 を使って正規化する
5. build_regression_model() で VGG16 ベースの回帰モデルを作る
6. train_entity_model() で学習し、loss 曲線とモデルを保存する
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import ops
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import img_to_array, load_img
from sklearn.model_selection import train_test_split


# 入力画像はすべてこのサイズにリサイズして学習へ渡す。
IMG_SIZE = (224, 224)
# 座標と半径の正規化に使う上限値。CSV の生ラベルはこの範囲を想定する。
COORDINATE_MAX = 224.0
# 角度の正規化に使う上限値。度数法の角度を想定する。
ANGLE_MAX = 360.0
# loss 曲線の縦軸範囲。見づらいときはこの値を調整する。
LOSS_Y_AXIS_RANGE = (0.0, 0.5)
# val_loss が止まったときの学習率調整設定。
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_MIN = 1e-6
# val_loss の改善が止まったときに学習を打ち切る設定。
EARLY_STOPPING_PATIENCE = 8
# 回帰 head の設定。Flatten を使わず GAP で圧縮してから全結合へ渡す。
HEAD_DENSE_UNITS = 256
HEAD_DROPOUT_RATE = 0.3

# 図形ごとの違いをまとめた設定表。
# ここを見れば、各図形がどのフォルダを使い、どの列を教師値にするか分かる。
ENTITY_CONFIGS = {
    "circle": {
        "dir_prefix": "test_circle_data",
        "target_columns": ["center_x", "center_y", "radius"],
        "normalization_scales": [COORDINATE_MAX, COORDINATE_MAX, COORDINATE_MAX],
    },
    "line": {
        "dir_prefix": "test_line_data",
        "target_columns": ["center_x", "center_y", "half_length", "cos2_theta", "sin2_theta"],
        "normalization_scales": [COORDINATE_MAX, COORDINATE_MAX, COORDINATE_MAX, 1.0, 1.0],
    },
    "arc": {
        "dir_prefix": "test_arc_data",
        "target_columns": ["center_x", "center_y", "radius", "start_angle", "cw_sweep_angle"],
        "normalization_scales": [COORDINATE_MAX, COORDINATE_MAX, COORDINATE_MAX, ANGLE_MAX, ANGLE_MAX],
    },
}

# 角度誤差の重み係数（環境変数で上書き可能）。
# 意味: arc/line の角度誤差に対して加算する係数。デフォルトは 4.0
# 0 にすると角度への追加重みは無効（従来の振る舞いに近い）。
try:
    ANGLE_WEIGHT_COEFF = float(os.environ.get('ANGLE_WEIGHT_COEFF', '4.0'))
except Exception:
    ANGLE_WEIGHT_COEFF = 4.0


def dataset_tag(dataset_suffix: str) -> str:
    """保存ファイル名に使うデータセット識別子を返す。"""
    return f"data{dataset_suffix}" if dataset_suffix else "base"


def dataset_dir(entity_type: str, dataset_suffix: str) -> str:
    """図形種別と接尾辞からデータセットのフォルダ名を組み立てる。"""
    return f"{ENTITY_CONFIGS[entity_type]['dir_prefix']}{dataset_suffix}"


def model_path(entity_type: str, dataset_suffix: str) -> str:
    """学習済みモデルの保存先ファイル名を返す。"""
    # これまで arc の追加重みでサフィックスを付けていたが、
    # arc 側の角度重み付けは廃止するためサフィックスは付けない。
    return f"model_{entity_type}_{dataset_tag(dataset_suffix)}.keras"


def loss_curve_path(entity_type: str, dataset_suffix: str) -> str:
    """loss 曲線画像の保存先ファイル名を返す。"""
    # arc の角度重み付けは廃止済みのため、サフィックスは付けない。
    return f"loss_curve_{entity_type}_{dataset_tag(dataset_suffix)}.png"


def normalization_scales(entity_type: str) -> np.ndarray:
    """指定した図形の各属性を何で割るかを配列で返す。"""
    return np.array(ENTITY_CONFIGS[entity_type]["normalization_scales"], dtype=np.float32)


def normalize_label_values(entity_type: str, values) -> np.ndarray:
    """CSV の生ラベルを学習用の 0 から 1 付近の値へ変換する。"""
    values_array = np.asarray(values, dtype=np.float32)
    return values_array / normalization_scales(entity_type)


def denormalize_label_values(entity_type: str, values) -> np.ndarray:
    """正規化済みの予測値を、人が読める元の座標・角度へ戻す。"""
    values_array = np.asarray(values, dtype=np.float32)
    return values_array * normalization_scales(entity_type)


def preprocess_image_array(image_array) -> np.ndarray:
    """VGG16 の学習済み重みが想定する形式へ画像配列を変換する。"""
    return preprocess_input(np.asarray(image_array, dtype=np.float32))


def load_entity_dataset(entity_type: str, dataset_suffix: str, img_size=IMG_SIZE, data_root: str = None):
    """
    1種類の図形データセットを読み込む。

    やっていること:
    - labels.csv を開く
    - 必要な列があるか確認する
    - 各画像を読み込んで 224x224 にそろえる
    - 画像を VGG16 の preprocess_input で前処理する
    - ラベルを図形ごとの尺度で正規化する

    弧データについては、終点角そのものではなく
    start_angle と cw_sweep_angle を教師値として使う。
    """
    config = ENTITY_CONFIGS[entity_type]

    # Resolve base directory for dataset lookup in the following priority:
    # 1) explicit `data_root` argument
    # 2) environment variable `DATA_ROOT`
    # 3) module directory (this package location)
    if data_root:
        base_dir = os.path.abspath(data_root)
    elif os.environ.get("DATA_ROOT"):
        base_dir = os.path.abspath(os.environ.get("DATA_ROOT"))
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    image_dir = os.path.join(base_dir, dataset_dir(entity_type, dataset_suffix))
    label_path = os.path.join(image_dir, "labels.csv")

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    labels_df = pd.read_csv(label_path)
    missing_columns = [column for column in config["target_columns"] if column not in labels_df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in {label_path}: {missing_columns}")

    images = []
    labels = []
    image_names = []

    for _, row in labels_df.iterrows():
        img_path = os.path.join(image_dir, row["image"])
        if not os.path.exists(img_path):
            continue

        # 画像側は VGG16 の学習済み重みに合わせた前処理を適用する。
        image = load_img(img_path, target_size=img_size)
        image = preprocess_image_array(img_to_array(image))

        # ラベル側の正規化。座標や角度は別の基準で割る。
        label = row[config["target_columns"]].to_numpy(dtype=np.float32)
        label = normalize_label_values(entity_type, label)

        images.append(image)
        labels.append(label)
        image_names.append(row["image"])

    if not images:
        raise ValueError(f"No images were loaded from {image_dir}")

    return np.array(images), np.array(labels), image_names


def build_regression_model(entity_type: str, num_outputs: int, img_size=IMG_SIZE):
    """
    VGG16 を特徴抽出器として使う回帰モデルを作る。

    num_outputs は図形ごとの出力次元数。
    - circle: 3
    - line: 5
    - arc: 5
    """
    base_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(*img_size, 3)))
    features = GlobalAveragePooling2D()(base_model.output)
    features = Dense(HEAD_DENSE_UNITS, activation="relu")(features)
    features = Dropout(HEAD_DROPOUT_RATE)(features)
    output = Dense(num_outputs, activation="linear")(features)
    model = Model(inputs=base_model.input, outputs=output)

    # VGG16 の事前学習済み部分は固定し、最後の回帰層だけ学習する。
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(), loss=loss_for_entity(entity_type), metrics=["mae"])
    return model


def circular_distance_normalized(y_true, y_pred):
    """0 から 1 に正規化した角度同士の円周距離を返す。"""
    difference = ops.abs(y_pred - y_true)
    one = ops.cast(1.0, difference.dtype)
    difference = ops.mod(difference, one)
    return ops.minimum(difference, one - difference)


def arc_loss(y_true, y_pred):
    """arc の 5 属性誤差を平均し、角度だけ円周距離で計算する。"""
    center_x_loss = ops.square(y_pred[:, 0] - y_true[:, 0])
    center_y_loss = ops.square(y_pred[:, 1] - y_true[:, 1])
    radius_loss = ops.square(y_pred[:, 2] - y_true[:, 2])
    # base normalized angle losses (0..1 range squared)
    start_angle_loss = ops.square(circular_distance_normalized(y_true[:, 3], y_pred[:, 3]))
    sweep_angle_loss = ops.square(circular_distance_normalized(y_true[:, 4], y_pred[:, 4]))
    stacked_losses = ops.stack(
        [center_x_loss, center_y_loss, radius_loss, start_angle_loss, sweep_angle_loss],
        axis=-1,
    )
    return ops.mean(stacked_losses, axis=-1)


def loss_for_entity(entity_type: str):
    """図形ごとに使う loss を返す。"""
    if entity_type == "arc":
        return arc_loss

    if entity_type == "line":
        # custom loss for lines: center + half_length + angle vector (cos2,sin2)
        def line_loss(y_true, y_pred):
            cx = ops.square(y_pred[:, 0] - y_true[:, 0])
            cy = ops.square(y_pred[:, 1] - y_true[:, 1])
            half = ops.square(y_pred[:, 2] - y_true[:, 2])
            cos2 = ops.square(y_pred[:, 3] - y_true[:, 3])
            sin2 = ops.square(y_pred[:, 4] - y_true[:, 4])
            # angle vector L2 error
            ang = cos2 + sin2
            # weight angle by half_length (normalized). longer lines -> angle matters more
            ang = ang * (1.0 + ANGLE_WEIGHT_COEFF * y_true[:, 2])
            stacked = ops.stack([cx, cy, half, ang], axis=-1)
            return ops.mean(stacked, axis=-1)

        return line_loss

    # default: keep mse for circle (simple) but user can switch to custom if desired
    return "mse"


def save_loss_curve(history, output_path: str, title: str):
    """学習中の train loss / val loss を画像として保存する。"""
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylim(*LOSS_Y_AXIS_RANGE)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.title(title)
    plt.savefig(output_path)
    plt.close()


def build_training_callbacks():
    """学習停滞時の学習率調整と早期終了に使う callback 群を返す。"""
    return [
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


def train_entity_model(entity_type: str, dataset_suffix: str = "_n1", epochs: int = 10, batch_size: int = 32, data_root: str = None):
    """
    1種類の図形だけを学習する共通入口。

    この関数を train_circle.py / train_line.py / train_arc.py から呼び出すことで、
    図形ごとに別モデルを作って学習できる。
    """
    images, labels, _ = load_entity_dataset(entity_type, dataset_suffix, img_size=IMG_SIZE, data_root=data_root)
    print(f"[{entity_type}] loaded images: {len(images)}")

    # データを訓練用とテスト用に分割する。
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    model = build_regression_model(entity_type, labels.shape[1])
    callbacks = build_training_callbacks()

    # fit() で学習を行い、その途中経過として validation loss も記録する。
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    curve_output_path = loss_curve_path(entity_type, dataset_suffix)
    save_loss_curve(history, curve_output_path, f"{entity_type} training loss")
    print(f"[{entity_type}] saved loss curve: {curve_output_path}")

    # 学習に使っていないテストデータで最終評価する。
    loss, mae = model.evaluate(x_test, y_test)
    print(f"[{entity_type}] test loss={loss:.6f}, test mae={mae:.6f}")

    saved_model_path = model_path(entity_type, dataset_suffix)
    # 学習後のモデルを保存し、あとで predict.py などから再利用できるようにする。
    model.save(saved_model_path)
    print(f"[{entity_type}] saved model: {saved_model_path}")

    return saved_model_path, curve_output_path
