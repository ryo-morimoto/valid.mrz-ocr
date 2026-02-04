# YOLO MRZ Detection 実験ログ

## 目標

**mAP@0.5 > 95%** (MRZ 検出精度、Roboflow test set)

## 制約

- Train: 42 images, Val: 10, Test: 5 (Roboflow MRZ_Passport_dataset, CC BY 4.0)
- アーキテクチャ: YOLOv8-nano (COCO pretrained)
- Classes: MRZ_P (single_cls mode)
- Export: ONNX (imgsz=320)

---

## 実験結果一覧

| Version | Best mAP@0.5 | Delta | 主要変更 | 判定 |
|---------|-------------|-------|---------|------|
| v5 | (要確認) | — | 100ep, freeze=10, degrees=90 | baseline |

---

## 詳細ログ

### v5: YOLOv8-nano Finetuning Baseline
- **仮説**: COCO pretrained の YOLOv8-nano を MRZ 検出に fine-tune
- **変更**: 100 epochs, freeze=10, single_cls=True, degrees=90, mosaic=1.0, mixup=0.1, copy_paste=0.1
- **結果**: (要確認 — results.csv から抽出)
- **教訓**: (要確認)

---

## パターンと教訓 (横断的)

(実験蓄積後に更新)

---

## 更新ルール

新バージョンの結果が出るたびに以下を追記:
1. 詳細ログセクションにエントリを追加
2. 結果一覧テーブルを更新
3. パターンと教訓を該当する場合に更新
