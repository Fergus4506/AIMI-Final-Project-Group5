# RETFound-Green k-NN 前測 checklist

2026-11-19 修訂。搭配 [前測計畫](RETFound_Green_Pilot_Plan.md)。完成後逐項填寫證據路徑、雜湊或摘要，再打勾。各階段保留使用者回覆、日期與核准範圍。

## 階段 0–2　已完成的工作

- [x] 在空目錄初始化 Git（`8fef696`），先建立 pilot 分支與 worktree，再放入文件。階段 1 已經使用者驗收。
- [x] 查核官方 v0.1 原始權重的 `num_classes=0` 設定與特徵輸出。直接分級的逐張分數、Accuracy 與 Macro-F1 在兩個資料集均為 N/A。查核紀錄見 `ce59849`，此階段採文件查核。
- [x] 使用者於 2026-11-19 選擇第二項，即固定 APTOS 參考集的 k-NN。已更新計畫與 checklist。
- [x] **使用者確認**　2026-11-19 已授權開始階段 3（含兩個測試集的五級覆蓋限制）。

## 階段 3　資料與評估規則

**目前進度（2026-11-19）**　資料查核與切分程式已實作，等待正式資料。以合成影像驗證約 80/20 切分、五級覆蓋、重跑雜湊一致，以及排除跨資料集重複後缺級仍會中止。實際張數、雜湊與缺件待正式資料查核。

- [x] 建立 [階段 3 查核程式](scripts/prepare_stage3.py)。輸出三份正式清單、audit.json 與 exclusions.csv。任一必要清單缺級時，輸出稽核與排除紀錄供查明原因。
- [x] 確認發布來源。[APTOS 2019 Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) 的有標註 train.csv 與 train_images。[IDRiD 官方頁面](https://idrid.grand-challenge.org/Home/) 指向 IEEE DataPort，需 Disease Grading 的官方 Testing Images 與 Testing Labels。另有 [Zenodo 的 Disease Grading 封存](https://zenodo.org/records/17219542)。
- [x] 在 pilot worktree 建立 .venv。已驗證 Python 3.9、Pillow 11.3.0、PNG 解碼及查核程式入口。套件站無法連線時，將本機既有的 Pillow 複製至虛擬環境。
- [ ] 取得實際檔案。APTOS 官方 Kaggle 頁要求同意競賽規則。IDRiD 的 Zenodo ZIP 下載遭瀏覽器封鎖，IEEE DataPort 官方入口要求登入。下載完成後再查核三份清單。

執行指令由 pilot worktree 啟動。以下路徑為範例，填入實際解壓位置。需要 Python 3.9 與 Pillow。清單的影像路徑以 pilot worktree 為基準。

```powershell
python scripts/prepare_stage3.py --aptos-labels artifacts/aptos/train.csv --aptos-images artifacts/aptos/train_images --idrid-testing-labels "artifacts/idrid/Testing Labels.csv" --idrid-testing-images "artifacts/idrid/Testing Set" --output artifacts/stage3
```



- [ ] 登錄 APTOS、IDRiD 來源、版本、官方 split、影像與標註位置。
- [ ] 逐筆對帳影像與 DR 標註，每份資料抽查至少三筆 ID、相對路徑與等級。列缺件、標註衝突、解碼失敗與重複影像的處置。
- [ ] 按種子 42 分層建立 APTOS 約 80% 參考與 20% 測試清單。核對兩者無 ID 或已知病人與重複群組重疊，參考集及保留測試集各自涵蓋五級，並記錄分組與實際比例。
- [ ] 固定 IDRiD 官方 testing 的有效影像清單。檢查與 APTOS 參考集的確定重複，逐筆記錄排除理由及重疊檢查限制。
- [ ] **測試集五級覆蓋**　排除與切分完成後，分別填寫 APTOS 與 IDRiD 測試集的 DR 0、1、2、3、4 級有效張數，各級須至少 1 張。APTOS 缺級依計畫調整切分後重核。IDRiD 官方 testing 核對後仍缺級，附各級張數及排除原因，暫緩階段 4 並請使用者確認後續評估範圍。
- [ ] 為三份清單記錄 SHA-256、張數、各級分布。保存 k=5、cosine、等權投票、平手規則與評分設定。
- [ ] **使用者確認**　提交資料查核、切分證據及兩測試集各自的五級張數。五級覆蓋通過並取得同意後進入階段 4。

## 階段 4　抽取官方權重的特徵

- [ ] 登錄官方 checkpoint 的名稱、版本、來源、SHA-256、執行環境與前處理設定。
- [ ] 用小批影像確認輸入為 RGB 且形狀為 `[N,3,392,392]`，輸出為 `[N,384]`、數值有限、各向量非零。記錄可重現的推論指令與小批輸出。
- [ ] 對三份清單抽出特徵。核對檔案筆數、ID 集合、順序、形狀、有限性與非零向量，記錄特徵檔和清單的雜湊。
- [ ] **使用者確認**　提交特徵查核結果與檔案位置，取得同意後進入階段 5。

## 階段 5　分類及評分

- [ ] 固定同一份 APTOS 參考庫和預先確認的規則，對 APTOS 測試與 IDRiD 測試各跑一次 k-NN。
- [ ] 逐張保存預測、五級票數、鄰居 ID、參考等級與相似度。核對每筆五票、鄰居皆來自參考集、平手規則一致以及預測 ID 與測試清單一一對應。
- [ ] 依測試 ID 加入真值，分別重算 Macro-F1、Accuracy、各級 Recall、support 和混淆矩陣。核對各資料集的評分筆數、矩陣總數及五級真值張數均大於 0。
- [ ] 提交兩份預測、指標表和摘要。標示使用的 checkpoint、切分、主要錯誤和資料重疊查核範圍。
- [ ] **使用者確認**　記錄結案驗收。

## 交接紀錄

| 日期（臺灣時間） | 進度 | 待確認 |
|---|---|---|
| 2026-11-19 | 階段 1 驗收，階段 2 完成原始權重介面查核 | 詳見 `ce59849` |
| 2026-11-19 | 使用者選擇 k-NN，兩份文件已修訂 | 新版計畫驗收與階段 3 啟動 |
| 2026-11-19 | 階段 3 已啟動，資料查核與切分程式通過合成測試 | 等待正式 APTOS 與 IDRiD 檔案 |
