##　主な構成

 含まれるファイル:
 - `training_common.py` : 学習／データ読み込み／モデル構築の共通関数群
 - `train_circle.py`, `train_line.py`, `train_arc.py` : 図形別の学習エントリスクリプト
 - `predict_and_error_stats.py` : 学習済モデルによる一括評価（CSV出力・誤差ヒストグラム生成）
 - `predict.py` : 単一画像の予測確認用ヘルパー
 - `test_circle_data_n1...`:学習に使うデータセット
 - `make_overlay_comparisons.py` : 定性的評価用のオーバーレイ作成スクリプト
 - `requirements.txt` : 依存パッケージ一覧


 ```

 学習の実行例（直線）:
 ```powershell
 # 1) カレントディレクトリをパッケージ内にして実行
 cd study_and_result
 python train_line.py

 # 2) または、どこから実行してもデータパスを明示する
 python train_line.py --data-root "C:\path\to\study_and_result" --dataset-suffix "_n1"
```

 一括評価の実行例:
 ```powershell
 $env:ENTITY_TYPE='line'
 $env:MODEL_DATASET_SUFFIX='_n1.1'
 $env:EVAL_DATASET_SUFFIX='_n2'
 python predict_and_error_stats.py
 ```

 （
## 各スクリプトの詳細（必要なデータ・実行手順・生成ファイル）

 動かすには `--data-root` 引数か環境変数 `DATA_ROOT` を使ってデータフォルダのルートを指定。

 例:
 ```powershell
 # 環境変数で指定する方法
 $env:DATA_ROOT = "C:\path\to\study_and_result"
 python train_line.py
```

-〇`training_common.py`
	- 必要なデータ: 各 `test_*` ディレクトリ（例: `test_line_data_n1.1/`）内の `labels.csv` と画像群（`image` 列にファイル名）
	- 役割: 画像読み込み、ラベル正規化（`COORDINATE_MAX=224.0`, `ANGLE_MAX=360.0`）、VGG16 ベースの回帰モデル構築、学習ルーチン（`train_entity_model`）を提供
	- 実行方法: 他の `train_*.py` から呼び出す
	- 生成ファイル（呼び出し先）: 学習済モデル（`model_{entity}_{tag}.keras`）、損失曲線（`loss_curve_{entity}_{tag}.png`）
	- 備考: `ENTITY_CONFIGS` に各図形の `target_columns` と `dir_prefix` が定義されています。

- 〇`train_circle.py`, `train_line.py`, `train_arc.py`
	- 必要なデータ: 対応するデータフォルダ（例: `test_circle_data_n1/`, `test_line_data_n1.1/`, `test_arc_data_n1/`）内の `labels.csv` と画像
	- 実行コマンド:
		- Windows PowerShell:
			```powershell
			python train_line.py
			```
		- （必要なら `epochs` / `batch_size` をスクリプト内で変更）
	- 環境変数: 特に必須はなし（スクリプトはハードコードされた `dataset_suffix` を使う）
	- 生成ファイル: `model_{entity}_{dataTag}.keras`, `loss_curve_{entity}_{dataTag}.png`（カレントディレクトリ）
	- 備考: 学習時に VGG16 の conv 部分は凍結され、回帰ヘッドのみ学習されます。学習ログ内に `test loss` / `test mae` が表示されます。

-〇 `predict_and_error_stats.py`
	- 必要なデータ: 学習済モデルファイル（例: `model_line_data_n1.1.keras`）、評価用データフォルダ（例: `test_line_data_n2/`）の `labels.csv` と画像
	- 実行コマンド（PowerShell 例）:
		```powershell
		$env:ENTITY_TYPE='line'
		$env:MODEL_DATASET_SUFFIX='_n1.1'
		$env:EVAL_DATASET_SUFFIX='_n2'
		python predict_and_error_stats.py
		```
	- 主な設定: `SAMPLE_SIZE`（デフォルト100）、`SHOW_PLOTS`（環境変数 `SHOW_PLOTS=1` で表示）
	- 生成ファイル:
		- `prediction_results_{MODEL_STEM}_eval{EVAL_DATASET_SUFFIX}.csv`（1画像ごとの予測/真値/絶対誤差）
		- `error_hist_{MODEL_STEM}_eval{EVAL_DATASET_SUFFIX}_{属性名}.png`（属性ごとの誤差ヒストグラム）
	- 備考: arc の角度誤差は周期（360度）を考慮して計算されます。

- 〇`predict.py`
	- 必要なデータ: 学習済モデル（`model_{entity}_{tag}.keras`）と、予測したい単一画像（スクリプト内 `IMG_PATH` を編集）
	- 実行コマンド:
		```powershell
		python predict.py
		```
	- 生成ファイル: なし（標準出力に予測値と、もし `labels.csv` に該当行があれば真値も表示）
	- 備考: 単体確認用。複数画像の評価や CSV 出力は `predict_and_error_stats.py` を使ってください。



-〇 `make_overlay_comparisons.py`
	- 必要なデータ: `prediction_results_*.csv`（`predict_and_error_stats.py` で生成）および対応するテストデータフォルダ（`test_*_n2/`）
	- 実行コマンド例:
		```powershell
		python make_overlay_comparisons.py
		
		```
	- 生成ファイル: `output/overlays/<entity>/*_overlay.png`, `output/annotated/*`（オーバーレイ画像や注釈付き画像、サマリー画像）
	
---
## 実験の実行順序（推奨ワークフロー）
以下は実験を再現する際の推奨順序です。各ステップごとに実行するスクリプト、必要データ、生成されるファイルを示します。

1) 環境準備
	 - 何をするか: 仮想環境作成と依存インストール
	 - コマンド (PowerShell):
		 ```powershell
		 cd study_and_result
		 python -m venv .venv
		 .\.venv\Scripts\Activate.ps1
		 pip install -r requirements.txt
		 ```
	 
2) データ確認
 	 各 `test_*` フォルダに `labels.csv` と画像が揃っているか確認。`labels.csv` は学習で使う形式（`center_x,center_y,half_length` など）である必要があります。もし別形式のラベル（例: エンドポイント形式）しか無い場合は、配布先で事前に所定の CSV 形式へ変換してください
3) 学習（必要なモデルごとに実行）
	 `train_circle.py` / `train_arc.py` / `train_line.py` を順に（または必要なものだけ）実行
	 - コマンド例:
		 ```powershell
		 python train_circle.py
		 python train_arc.py
		 python train_line.py
		 ```
	 - 必要データ: 各 `test_*_n1` フォルダ（学習用）の `labels.csv` と画像
	 - 生成物: `model_{entity}_{tag}.keras`, `loss_curve_{entity}_{tag}.png`

4) 定量評価（評価データで）
学習済モデルを読み `predict_and_error_stats.py` でランダムサンプルを予測・誤差集計
	 - コマンド例:
		 ```powershell
		 $env:ENTITY_TYPE='line'
		 $env:MODEL_DATASET_SUFFIX='_n1.1'
		 $env:EVAL_DATASET_SUFFIX='_n2'
		 python predict_and_error_stats.py
		 ```
	 - 必要データ: 学習済モデルファイル、評価用 `test_*_n2` の `labels.csv` と画像
	 - 生成物: `prediction_results_{MODEL_STEM}_eval{EVAL_DATASET_SUFFIX}.csv`, `error_hist_*.png`

5) 定性的評価（オーバーレイ）
 	`predict_and_error_stats.py` で作成した CSV を使って `make_overlay_comparisons.py` を実行し、真値／予測の重ね描き画像を作成します。
 	- コマンド例:
 		 ```powershell
 		 python make_overlay_comparisons.py
 		 ```
 	- 必要データ: `prediction_results_*.csv`, 対応する `test_*_n2` フォルダ
 	- 生成物: `output/overlays/...`（オーバーレイ画像）

6) 単体確認（任意）
	 特定の画像1枚を `predict.py` で確認
	 - コマンド例:
		 ```powershell
		 # predict.py 内の IMG_PATH を確認・編集してから実行
		 python predict.py
		 ```
	 - 生成物: 標準出力に予測値（必要なら真値）
