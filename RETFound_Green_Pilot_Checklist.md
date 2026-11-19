# RETFound-Green k-NN 前測 checklist

2026-11-19 修訂。搭配 [前測計畫](RETFound_Green_Pilot_Plan.md)。完成後逐項填寫證據路徑、雜湊或摘要，再打勾。各階段保留使用者回覆、日期與核准範圍。

## 階段 0–2　已完成的工作

- [x] 在空目錄初始化 Git（`8fef696`），先建立 pilot 分支與 worktree，再放入文件。階段 1 已經使用者驗收。
- [x] 查核官方 v0.1 原始權重的 `num_classes=0` 設定與特徵輸出。直接分級的逐張分數、Accuracy 與 Macro-F1 在兩個資料集均為 N/A。查核紀錄見 `ce59849`，此階段採文件查核。
- [x] 使用者於 2026-11-19 選擇第二項，即固定 APTOS 參考集的 k-NN。已更新計畫與 checklist。
- [x] **使用者確認**　2026-11-19 已授權開始階段 3（含兩個測試集的五級覆蓋限制）。

## 階段 3　資料與評估規則

**目前進度（2026-11-19）**　正式資料已完成解壓、對帳、逐張解碼及切分。APTOS 保留測試與 IDRiD 官方測試皆符合 DR 五級覆蓋。階段 3 已獲使用者驗收。

- [x] 建立 [階段 3 查核程式](scripts/prepare_stage3.py)。輸出三份正式清單、audit.json 與 exclusions.csv。任一必要清單缺級時，輸出稽核與排除紀錄供查明原因。
- [x] 確認發布來源。[APTOS 2019 Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) 的有標註 train.csv 與 train_images。[IDRiD 官方頁面](https://idrid.grand-challenge.org/Home/) 指向 IEEE DataPort，需 Disease Grading 的官方 Testing Images 與 Testing Labels。另有 [Zenodo 的 Disease Grading 封存](https://zenodo.org/records/17219542)。
- [x] 在 pilot worktree 建立 .venv。已驗證 Python 3.9、Pillow 11.3.0、PNG 解碼及查核程式入口。套件站無法連線時，將本機既有的 Pillow 複製至虛擬環境。
- [x] 取得並解壓使用者提供的兩個 ZIP。APTOS 解出 5,593 個檔案，IDRiD 解出 520 個檔案。檔名與大小均與 ZIP 清單一致。

從 pilot worktree 執行下方指令。資料共用專案根目錄的 `data/raw/`。清單中的影像路徑以 pilot worktree 為基準。若重跑，請以另一個未建立的輸出目錄保留本次結果。

```powershell
$stageOutput = Join-Path (Get-Location).Path 'artifacts/stage3'
& .\.venv\Scripts\python.exe scripts/prepare_stage3.py --aptos-labels ../../data/raw/aptos2019/train.csv --aptos-images ../../data/raw/aptos2019/train_images --idrid-testing-labels '../../data/raw/idrid/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv' --idrid-testing-images '../../data/raw/idrid/B. Disease Grading/1. Original Images/b. Testing Set' --output $stageOutput
```



- [x] 登錄 APTOS 2019 Kaggle 官方有標註 train 與 IDRiD Disease Grading 官方 testing。兩份標註路徑與 SHA-256 見 [audit.json](artifacts/stage3/audit.json)。
- [x] 四個官方分割的影像 ID 均與 CSV 逐筆吻合。前測使用的影像皆成功解碼。APTOS 排除 96 張同級重複影像，另隔離 62 張像素相同但等級有衝突的影像。排除清單見 [exclusions.csv](artifacts/stage3/exclusions.csv)，三份清單各三筆抽查見 `audit.json`。
- [x] 以種子 42 分層切分 APTOS 的 3,504 張有效影像。參考集 2,804 張，測試集 700 張（19.98%），兩組皆涵蓋五級，ID 與精確 RGB 像素雜湊無交集。官方 CSV 無病人 ID 可驗證病人層級隔離。
- [x] 固定 IDRiD 官方 testing 的 103 張有效影像，與 APTOS 參考集無精確 RGB 像素重複。此檢查以 EXIF 方向校正後的 RGB 像素為準。
- [x] **測試集五級覆蓋**　APTOS 保留測試與 IDRiD 官方測試兩組皆通過。

| 階段 3 清單 | DR 0 | DR 1 | DR 2 | DR 3 | DR 4 | 合計 |
|---|---:|---:|---:|---:|---:|---:|
| APTOS 參考 | 1,437 | 270 | 738 | 142 | 217 | 2,804 |
| APTOS 保留測試 | 359 | 68 | 184 | 35 | 54 | 700 |
| IDRiD 官方測試 | 34 | 5 | 32 | 19 | 13 | 103 |
- [x] 核對三份清單的 SHA-256、張數、各級分布與影像路徑。雜湊及 k=5、cosine、等權投票、平手規則與評分設定記錄於 `audit.json`。
- [x] **使用者確認**　2026-11-19 已驗收階段 3，並授權進入階段 4。

## 階段 4　抽取官方權重的特徵

**目前進度（2026-11-19）**　已在 RTX 4070 使用官方 v0.1 權重，完成三份清單共 3,607 張影像的特徵抽取，使用者已驗收。

官方權重位於工作目錄的 `models/retfoundgreen_statedict.pth`。重現步驟請看 [README](README.md)。

```powershell
$labPython = Join-Path (Resolve-Path '../../../../lab3/.venv').Path 'Scripts/python.exe'
& $labPython -B -u scripts/extract_stage4.py
```

- [x] 登錄使用者下載的官方 v0.1 `retfoundgreen_statedict.pth`、本機 SHA-256、環境與 RGB 前處理設定。來源及完整設定見 [audit.json](artifacts/stage4/audit.json)。
- [x] 6 張實際影像的小批驗證通過。RGB 輸入 `[6, 3, 392, 392]`，輸出 `[6, 384]`，數值全為有限值，各向量的 L2 範數均大於零。
- [x] 抽取 APTOS 參考集 `[2804, 384]`、APTOS 保留測試集 `[700, 384]`、IDRiD 官方測試集 `[103, 384]`。重新讀取三份 `.npz`，確認影像 ID 與階段 3 的清單逐筆同序、無重複、特徵為有限的非零 float32 向量，並核對三份特徵檔與清單的 SHA-256。
- [x] **使用者確認**　2026-11-19 已驗收第四階段並授權進入第五階段。

## 階段 5　分類及評分

- [x] 使用 [evaluate_stage5.py](scripts/evaluate_stage5.py)，兩組共用 2,804 張 APTOS 參考影像。固定 cosine、k=5 和等權投票。
- [x] 兩份逐張預測 CSV 共 803 筆，已獨立重算五個鄰居、相似度、五級票數與預測等級，並確認影像 ID 與測試清單逐筆一致。
- [x] 依影像 ID 加入真值，用 scikit-learn 重算 Accuracy、Macro-F1、各級 Recall 和混淆矩陣。矩陣與樣本數吻合，兩組五級真值皆有樣本。
- [x] 已提交 [結果摘要](artifacts/stage5/summary.md)、兩份逐張預測及 [指標與雜湊](artifacts/stage5/metrics.json)。APTOS Macro-F1 為 55.50%，IDRiD 為 27.36%，相差 28.14 個百分點。
- [x] 依使用者要求另跑 APTOS 70/30 補充實驗。參考集 2,453 張，測試集 1,051 張，包含原 700 張測試影像。IDRiD 沿用官方 103 張。詳見 [70/30 結果](artifacts/stage5_aptos_70_30/summary.md)。
- [x] **使用者確認**　已確認簡版報告並要求整理專案與撰寫重現說明。

## 交接紀錄

| 日期（臺灣時間） | 進度 | 待確認 |
|---|---|---|
| 2026-11-19 | 階段 1 驗收，階段 2 完成原始權重介面查核 | 詳見 `ce59849` |
| 2026-11-19 | 使用者選擇 k-NN，兩份文件已修訂 | 新版計畫驗收與階段 3 啟動 |
| 2026-11-19 | 階段 3 已啟動，資料查核與切分程式通過合成測試 | 等待正式 APTOS 與 IDRiD 檔案 |
| 2026-11-19 | 階段 3 正式資料與清單查核完成，兩測試集五級覆蓋通過 | 等待階段 3 驗收 |
| 2026-11-19 | 使用者驗收階段 3 並核准開始階段 4，抽取流程通過隨機權重測試 | 等待官方 v0.1 checkpoint |
| 2026-11-19 | 階段 4 官方權重載入與 3,607 張特徵抽取完成 | 等待階段 4 驗收 |
| 2026-11-19 | 階段 4 已驗收，階段 5 分類與獨立評分查核完成 | 已確認簡版報告 |
