# RETFound-Green 眼底影像分類前測

以凍結的 RETFound-Green v0.1 抽取 384 維特徵，用 APTOS 標註影像做 cosine 5-NN 等權投票，評估 DR 0–4 級。測試包含 APTOS 內部保留集與 IDRiD 官方測試集。

## 專案目錄

| 目錄 | 內容 |
|---|---|
| `scripts/` | 資料查核、特徵抽取及分類評分 |
| `reports/` | [簡版報告](reports/RETFound_Green_Pilot_Brief.md) 與 [完整報告](reports/RETFound_Green_Pilot_Report.md) |
| `models/` | 本機模型權重 |
| `data/archives/`、`data/raw/` | 原始壓縮檔與解壓資料 |
| `artifacts/` | 腳本產生的清單、特徵、逐張預測與指標 |

Git 忽略模型、原始資料、環境及執行產物。已下載與解壓的資料可直接沿用。

## 重現步驟

以下指令從 `pilot/retfound-green-aptos-idrid` 分支的工作目錄在 Windows PowerShell 執行。需要 Python 3.12。前測使用 RTX 4070 與 CUDA 12.8 版 PyTorch。

```powershell
py -3.12 -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
& $python -m pip install timm==1.0.31 numpy==2.5.4 Pillow==13.0.0
```

使用 CPU 時，把 PyTorch 索引最後的 `cu128` 改為 `cpu`，並在抽取特徵指令加上 `--device cpu`。[PyTorch 安裝版本](https://pytorch.org/get-started/previous-versions/)

1. 將 [APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) 的 `aptos2019-blindness-detection.zip` 和 [IDRiD Disease Grading](https://zenodo.org/records/17219542) 的 `B. Disease Grading.zip` 存入 `data/archives/`。
2. 將 [RETFound-Green v0.1 權重](https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth) 存為 `models/retfoundgreen_statedict.pth`。前測權重的本機 SHA-256 為 `431de5dbc1bebbb32f60e2c0bcf8daa4f8bcbf06f7cb1e1dc97ec589713942e1`。
3. 設定解壓資料目錄。已解壓時可跳過兩行解壓指令。本機既有的 `.worktrees/retfound-green` 可把 `$raw` 設為 `../../data/raw`，沿用共用的已解壓資料。

```powershell
$raw = 'data/raw'
& $python -m zipfile -e 'data/archives/aptos2019-blindness-detection.zip' "$raw/aptos2019"
& $python -m zipfile -e 'data/archives/B. Disease Grading.zip' "$raw/idrid"
```

4. 依序執行資料查核、特徵抽取、80/20 與 70/30 評分。各輸出目錄須為新目錄。重跑時可用 `--output` 改名，並透過 `--stage3`、`--stage4` 指向對應輸出。

```powershell
$idrid = "$raw/idrid/B. Disease Grading"
& $python scripts/prepare_stage3.py `
  --aptos-labels "$raw/aptos2019/train.csv" `
  --aptos-images "$raw/aptos2019/train_images" `
  --idrid-testing-labels "$idrid/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv" `
  --idrid-testing-images "$idrid/1. Original Images/b. Testing Set" `
  --output (Join-Path (Get-Location).Path 'artifacts/stage3')
& $python scripts/extract_stage4.py
& $python scripts/evaluate_stage5.py
& $python scripts/evaluate_stage5.py --aptos-test-fraction 0.3 --output 'artifacts/stage5_aptos_70_30'
```

## 核對結果

在 `artifacts/stage5/` 與 `artifacts/stage5_aptos_70_30/` 查閱 `metrics.json` 及兩份逐張預測 CSV。70/30 保留原 APTOS 700 張測試影像，分層從參考集移入 351 張。IDRiD 兩次皆使用同一批官方 103 張影像。

| APTOS 切分 | 參考張數 | APTOS 測試張數 | APTOS Macro-F1 | IDRiD Macro-F1 |
|---|---:|---:|---:|---:|
| 80/20 | 2,804 | 700 | 55.50% | 27.36% |
| 70/30 | 2,453 | 1,051 | 55.75% | 26.71% |

70/30 用縮小的參考集重測原 APTOS 700 張影像時，Macro-F1 為 54.72%。
