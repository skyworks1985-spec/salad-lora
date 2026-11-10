# SaladCloud LoRA 学習パッケージ

SaladCloud のコンテナ（RTX 4090）上で Kohya sd-scripts による SDXL LoRA 学習を回すための一式。
ローカル PC（RTX 4060）で実績のある学習レシピをそのままクラウドに移すことが目的で、対象は org `skyworks1985` / project `default`。

SaladCloud は「SSH なし・永続ボリュームなし・ノードは家庭用 PC で割り込みあり」という前提なので、設計は次の3点に寄せてある。

- 起動時に必要なもの（ベースモデル＋データセット）を URL からダウンロードする
- エポックごとのチェックポイントが生成された瞬間にアップロードする（ノードが落ちても最大で当該エポック分だけの損失）
- 進捗はすべて標準出力に出す（Salad のログ API / ポータルの Container Logs で追える）

## ファイル構成

| ファイル | 役割 |
|---|---|
| `Dockerfile` | 学習イメージの定義。PyTorch 2.6.0 + CUDA 12.4 の公式ランタイムに、タグ固定（v0.12.3）した kohya-ss/sd-scripts と依存関係を入れたもの |
| `entrypoint.sh` | コンテナの実行本体。環境変数で駆動し、ダウンロード→学習→チェックポイント即時アップロード→最終アップロードまでを行う |
| `salad_group_template.json` | コンテナグループ作成 API（`POST /organizations/skyworks1985/projects/default/containers`）に渡すリクエストボディの雛形 |
| `README.md` | 本書 |

## データフロー

```
[ファイルホスト] ── MODEL_URL ────────▶ base_model.safetensors (約6.5GB) を起動時にダウンロード
[ファイルホスト] ── DATASET_ZIP_URL ──▶ dataset.zip を展開し、"10_<trigger>" 形式の
                                          リピートフォルダ群を train_data_dir として検出
        │
        ▼  学習（sdxl_train_network.py、エポック毎に保存）
output/<OUTPUT_NAME>-000001.safetensors, -000002..., 最終 <OUTPUT_NAME>.safetensors
        │
        └── 生成され次第バックグラウンドでアップロード ──▶ [ファイルホスト]
終了時: 残りの出力ファイルと train.log をアップロードして終了（成功時 exit 0）
```

コンテナが停止するとローカルディスクは消える。残したいものはすべてアップロード経由で外に出す、という割り切り。

## 学習レシピ（ローカル実績との対応）

| 項目 | 値 | 備考 |
|---|---|---|
| ベースモデル | WAI-Nsfw-Illustrious-17（SDXL、約6.5GB） | `MODEL_URL` から取得 |
| network_dim / alpha | 32 / 16 | |
| 学習率 | 1e-4、cosine、warmup 100 steps | |
| 解像度 | 1024×1024、バケット有効 | |
| バッチサイズ | 1 | |
| その他 | gradient checkpointing、cache_latents、adamw8bit（bitsandbytes）、mixed precision bf16、保存 fp16、sdpa、seed 42 | |
| 保存 | エポック毎（`save_every_n_epochs=1`） | チェックポイント即時アップロードの単位 |
| `--network_train_unet_only` | **必須** | テキストエンコーダ側を学習しない指定 |
| `--cache_text_encoder_outputs` | **付けない** | 上記と併用すると競合するため、ローカルレシピ同様に除外 |

## 環境変数（entrypoint.sh）

必須は `MODEL_URL` と `DATASET_ZIP_URL` の2つ。残りは既定値がローカルレシピ通りに入っている。

- 学習系: `OUTPUT_NAME`、`EPOCHS`（既定5）、`MAX_TRAIN_STEPS`（指定時はエポック数を上書き。スモークテスト用）、`NETWORK_DIM`、`NETWORK_ALPHA`、`LEARNING_RATE`、`LR_SCHEDULER`、`LR_WARMUP_STEPS`、`RESOLUTION`、`TRAIN_BATCH_SIZE`、`SEED`、`CACHE_LATENTS_TO_DISK`（1で潜在キャッシュもディスク保存）
- アップロード系: `UPLOAD_MODE`（`put` / `rclone` / `none`、既定 `none`）、`UPLOAD_PUT_BASE_URL`、`UPLOAD_AUTH_HEADER`、`RCLONE_REMOTE`
- 運用系: `WATCH_INTERVAL`（チェックポイント監視間隔、既定15秒）

## 未決定事項（2つ）

1. **コンテナレジストリのアカウント**
   ビルドしたイメージの push 先。Docker Hub / GHCR など。`salad_group_template.json` の `container.image` は現在プレースホルダ（`REGISTRY_PLACEHOLDER/salad-lora-train:0.1.0`）。プライベートレジストリにする場合は、Salad 側にレジストリ認証情報の登録も別途必要。
2. **ファイルホスト**
   ベースモデルとデータセット zip の配布元、および学習結果（チェックポイント・ログ）の受け取り先。`entrypoint.sh` の `upload_file()` は `curl PUT` と `rclone` の2方式に対応する差し込み口だけ用意してあり、対象ホストが決まったら中身を確定させる（スクリプト内に TODO 明記）。ホストが決まるまでは `UPLOAD_MODE=none` で「アップロードするつもりのファイル名」がログに出るだけになる。

## スモークテスト計画（100ステップ）

本番投入の前に、同じイメージ・同じ経路で小さく1回回して全経路を検証する。

- 設定: `MAX_TRAIN_STEPS=100`、`EPOCHS` は不問。データセットは本番と同じものを使う（DL 経路とフォルダ検出も含めて検証するため）
- 所要見込み: 学習100ステップが数分、起動・イメージプル・モデルDLに約10分、合計15分前後
- 費用見込み: 4090 medium（$0.347/h）で **約 $0.09（$0.1 未満）**
- 確認項目:
  - イメージが Salad ノードにプルされ、GPU（4090）が認識される
  - モデル／データセットのダウンロードとデータセット構造の自動検出が通る
  - 学習が回り、loss がログに出る
  - チェックポイントのアップロード経路が動く（100ステップが1エポック未満の場合は最終ファイル＋ログのアップロードで検証し、エポック途中チェックポイントの即時アップロードは本番初回で確認する）
  - 終了コード 0 で終わり、コンテナグループが再起動せず停止すること（`restart_policy: never` の確認も兼ねる）

## 本番1ランあたりの想定コスト

前提: 約2,000ステップ（5〜7エポック級）の学習が 4090 で約45〜75分、起動・ダウンロードに約10分。

| 優先度 | 単価 | 1ランの目安 |
|---|---|---|
| medium | $0.347/h | **約 $0.32〜0.49（$0.5 以内）** |
| （参考）batch | $0.16/h | 約 $0.15〜0.23。ただし割り込みリスクは上がる |
| （参考）high | $0.44/h | 約 $0.40〜0.62 |

ローカル 4060 では同条件で約9時間かかる計算なので、時間短縮効果に対してコストは誤差レベル。

## Salad API スキーマについて（確認できたこと・できなかったこと）

公式 API リファレンス（Create Container Group）と公式クイックスタートをもとに `salad_group_template.json` を組んだ。

確認できたこと:

- パスは `POST https://api.salad.com/api/public/organizations/{org}/projects/{project}/containers`、認証は `Salad-Api-Key` ヘッダ
- `name` / `display_name` / `container.image` / `container.resources{cpu, memory, gpu_classes[], storage_amount}` / `container.environment_variables` / `container.image_caching` / `autostart_policy` / `restart_policy` / `replicas` / `priority` の各フィールドが存在する
- API 経由の作成では `autostart_policy` の既定が false なので、雛形では `true` を明示した
- RTX 4090 の GPU クラス ID `ed563892-aacd-40f5-80b7-90c9be6c759b` はこの org で実測確認済みのもの

未確認・要検証（初回作成時に確認する）:

- `restart_policy` の列挙値。リファレンスの例示は `"always"` のみ。雛形の `"never"` は「完了したジョブを再起動しない」ための指定で、SDK／ポータルの選択肢命名に基づく。受理されなければ `"on_failure"` 等への変更が必要
- `resources` の単位。`memory` はクイックスタート例（小さなデモで 2048）から MiB と解釈し 16384（16 GiB）を指定。`storage_amount` は bytes と解釈し 100 GiB 分（107374182400）を指定した
- `container.logging` は Axiom / HTTP / Datadog 等への**外部ログ送信**を指定する形で、送信先の情報が必須。組み込みのコンテナログ（ポータル Container Logs と Logs API）は指定なしで有効なため、雛形では `logging` を省略した。外部送信先を決めたらここに追加する
- `priority` はリファレンスのオブジェクト例（`"high"`）に基づき `"medium"` を指定した
