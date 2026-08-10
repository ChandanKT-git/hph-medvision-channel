# (base) chanduu@ChandanKT:~/OSS/hiperhealth/hph-medvision-channel$ python -m training.train --data-dir ./training/data

# DINOv2 ViT-S/14 — ISIC Skin Lesion Training

Device: cuda
GPU: NVIDIA GeForce RTX 2050
VRAM: 4.0 GB
Mode: 5-fold CV
Batch size: 32
Phase 1: 15 epochs (LR=0.001)
Phase 2: 25 epochs (LR=1e-05, unfreeze last 4 blocks)

Config saved to outputs/training_config.json
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /home/chanduu/.netrc.
wandb: Currently logged in as: chandankt (chandankt-polaris-school-of-technology) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.28.1
wandb: Run data is saved locally in /home/chanduu/OSS/hiperhealth/hph-medvision-channel/wandb/run-20260809_111118-ykbt4k1u
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run dinov2-isic-5fold
wandb: ⭐️ View project at https://wandb.ai/chandankt-polaris-school-of-technology/medvision-skin
wandb: 🚀 View run at https://wandb.ai/chandankt-polaris-school-of-technology/medvision-skin/runs/ykbt4k1u

Loading dataset...
[data] Loaded 10015 images from training/data/HAM10000_metadata
[data] 7470 unique lesions
[data] Image dirs: [PosixPath('training/data/HAM10000_images_part_1'), PosixPath('training/data/HAM10000_images_part_2')]
[0] mel : 1113 ( 11.1%)
[1] nv : 6705 ( 66.9%)
[2] bcc : 514 ( 5.1%)
[3] akiec : 327 ( 3.3%)
[4] bkl : 1099 ( 11.0%)
[5] df : 115 ( 1.1%)
[6] vasc : 142 ( 1.4%)
Total: 10015 images
[split] Fold 0: train=8036, val=1979
[split] Fold 1: train=8002, val=2013
[split] Fold 2: train=8013, val=2002
[split] Fold 3: train=8004, val=2011
[split] Fold 4: train=8005, val=2010

============================================================
FOLD 0 / 4
============================================================
Loading DINOv2 ViT-S/14...
Using cache found in /home/chanduu/.cache/torch/hub/facebookresearch_dinov2_main
/home/chanduu/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:51: UserWarning: xFormers is not available (SwiGLU)
warnings.warn("xFormers is not available (SwiGLU)")
/home/chanduu/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:33: UserWarning: xFormers is not available (Attention)
warnings.warn("xFormers is not available (Attention)")
/home/chanduu/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:40: UserWarning: xFormers is not available (Block)
warnings.warn("xFormers is not available (Block)")
Trainable: 50,951 / 22,107,527 (0.23%)

── Fold 0 | Phase 1: Linear Probing ──
[phase1] Epoch 1/15 | Train Loss: 2.4035 | Val Loss: 1.3926 | Val Bal Acc: 45.3% | LR: 0.000333 | 79s
✓ New best: 45.3% → saved to best_model.pth
[phase1] Epoch 2/15 | Train Loss: 2.0358 | Val Loss: 1.2749 | Val Bal Acc: 56.6% | LR: 0.000667 | 80s
✓ New best: 56.6% → saved to best_model.pth
[phase1] Epoch 3/15 | Train Loss: 1.9406 | Val Loss: 1.2655 | Val Bal Acc: 56.4% | LR: 0.001000 | 79s
[phase1] Epoch 4/15 | Train Loss: 1.8637 | Val Loss: 1.1330 | Val Bal Acc: 61.6% | LR: 0.001000 | 79s
✓ New best: 61.6% → saved to best_model.pth
[phase1] Epoch 5/15 | Train Loss: 1.7433 | Val Loss: 1.1351 | Val Bal Acc: 62.0% | LR: 0.000983 | 80s
✓ New best: 62.0% → saved to best_model.pth
[phase1] Epoch 6/15 | Train Loss: 1.7649 | Val Loss: 1.1788 | Val Bal Acc: 61.8% | LR: 0.000933 | 80s
[phase1] Epoch 7/15 | Train Loss: 1.7091 | Val Loss: 1.0857 | Val Bal Acc: 64.8% | LR: 0.000854 | 80s
✓ New best: 64.8% → saved to best_model.pth
[phase1] Epoch 8/15 | Train Loss: 1.7411 | Val Loss: 1.1036 | Val Bal Acc: 66.4% | LR: 0.000750 | 80s
✓ New best: 66.4% → saved to best_model.pth
[phase1] Epoch 9/15 | Train Loss: 1.6339 | Val Loss: 1.0671 | Val Bal Acc: 64.7% | LR: 0.000629 | 80s
[phase1] Epoch 10/15 | Train Loss: 1.6488 | Val Loss: 1.1150 | Val Bal Acc: 63.9% | LR: 0.000500 | 81s
[phase1] Epoch 11/15 | Train Loss: 1.5784 | Val Loss: 1.0258 | Val Bal Acc: 66.6% | LR: 0.000371 | 80s
✓ New best: 66.6% → saved to best_model.pth
[phase1] Epoch 12/15 | Train Loss: 1.5871 | Val Loss: 1.0688 | Val Bal Acc: 63.7% | LR: 0.000250 | 80s
[phase1] Epoch 13/15 | Train Loss: 1.6426 | Val Loss: 1.0535 | Val Bal Acc: 66.2% | LR: 0.000146 | 80s
[phase1] Epoch 14/15 | Train Loss: 1.5496 | Val Loss: 1.0603 | Val Bal Acc: 65.0% | LR: 0.000067 | 80s
[phase1] Epoch 15/15 | Train Loss: 1.5945 | Val Loss: 1.0474 | Val Bal Acc: 65.6% | LR: 0.000017 | 80s

── Fold 0 | Phase 2: Fine-tuning (last 4 blocks) ──
[phase2] Epoch 1/25 | Train Loss: 1.6202 | Val Loss: 0.9999 | Val Bal Acc: 69.2% | LR: 0.000003 | 113s
✓ New best: 69.2% → saved to best_model.pth
[phase2] Epoch 2/25 | Train Loss: 1.5812 | Val Loss: 0.9635 | Val Bal Acc: 71.7% | LR: 0.000007 | 114s
✓ New best: 71.7% → saved to best_model.pth
[phase2] Epoch 3/25 | Train Loss: 1.5762 | Val Loss: 0.9505 | Val Bal Acc: 72.2% | LR: 0.000010 | 113s
✓ New best: 72.2% → saved to best_model.pth
[phase2] Epoch 4/25 | Train Loss: 1.4830 | Val Loss: 0.9069 | Val Bal Acc: 75.4% | LR: 0.000010 | 113s
✓ New best: 75.4% → saved to best_model.pth
[phase2] Epoch 5/25 | Train Loss: 1.4420 | Val Loss: 0.9359 | Val Bal Acc: 71.4% | LR: 0.000010 | 114s
[phase2] Epoch 6/25 | Train Loss: 1.3947 | Val Loss: 0.9193 | Val Bal Acc: 73.0% | LR: 0.000010 | 114s
[phase2] Epoch 7/25 | Train Loss: 1.4417 | Val Loss: 0.9080 | Val Bal Acc: 69.1% | LR: 0.000010 | 113s
[phase2] Epoch 8/25 | Train Loss: 1.3294 | Val Loss: 0.8846 | Val Bal Acc: 73.4% | LR: 0.000009 | 113s
[phase2] Epoch 9/25 | Train Loss: 1.3799 | Val Loss: 0.8706 | Val Bal Acc: 74.5% | LR: 0.000009 | 114s
[phase2] Epoch 10/25 | Train Loss: 1.3597 | Val Loss: 0.9180 | Val Bal Acc: 70.2% | LR: 0.000008 | 113s
[phase2] Epoch 11/25 | Train Loss: 1.3210 | Val Loss: 0.9259 | Val Bal Acc: 73.7% | LR: 0.000008 | 114s
[phase2] Epoch 12/25 | Train Loss: 1.2611 | Val Loss: 0.8889 | Val Bal Acc: 71.8% | LR: 0.000007 | 115s
[phase2] Epoch 13/25 | Train Loss: 1.2778 | Val Loss: 0.8616 | Val Bal Acc: 75.2% | LR: 0.000006 | 123s
[phase2] Epoch 14/25 | Train Loss: 1.3329 | Val Loss: 0.8990 | Val Bal Acc: 73.5% | LR: 0.000006 | 125s
⏹ Early stopping after 10 epochs without improvement

✅ Fold 0 complete — Best Bal Acc: 75.4%

Fold 0 evaluation:
Balanced Accuracy: 74.2%
Cohen's Kappa: 0.468
Macro F1: 0.557
AUC-ROC (OVR): 0.934

============================================================
FOLD 1 / 4
============================================================
Loading DINOv2 ViT-S/14...
Using cache found in /home/chanduu/.cache/torch/hub/facebookresearch_dinov2_main
Trainable: 50,951 / 22,107,527 (0.23%)

── Fold 1 | Phase 1: Linear Probing ──
[phase1] Epoch 1/15 | Train Loss: 2.5029 | Val Loss: 1.4126 | Val Bal Acc: 48.4% | LR: 0.000333 | 102s
✓ New best: 48.4% → saved to best_model.pth
[phase1] Epoch 2/15 | Train Loss: 2.0652 | Val Loss: 1.2084 | Val Bal Acc: 57.8% | LR: 0.000667 | 86s
✓ New best: 57.8% → saved to best_model.pth
[phase1] Epoch 3/15 | Train Loss: 1.9442 | Val Loss: 1.1699 | Val Bal Acc: 58.3% | LR: 0.001000 | 101s
✓ New best: 58.3% → saved to best_model.pth
[phase1] Epoch 4/15 | Train Loss: 1.8474 | Val Loss: 1.2079 | Val Bal Acc: 58.9% | LR: 0.001000 | 92s
✓ New best: 58.9% → saved to best_model.pth
[phase1] Epoch 5/15 | Train Loss: 1.8168 | Val Loss: 1.1099 | Val Bal Acc: 61.4% | LR: 0.000983 | 95s
✓ New best: 61.4% → saved to best_model.pth
[phase1] Epoch 6/15 | Train Loss: 1.7701 | Val Loss: 1.1293 | Val Bal Acc: 61.9% | LR: 0.000933 | 80s
✓ New best: 61.9% → saved to best_model.pth
[phase1] Epoch 7/15 | Train Loss: 1.7283 | Val Loss: 1.1101 | Val Bal Acc: 62.4% | LR: 0.000854 | 94s
✓ New best: 62.4% → saved to best_model.pth
[phase1] Epoch 8/15 | Train Loss: 1.6577 | Val Loss: 1.0966 | Val Bal Acc: 65.5% | LR: 0.000750 | 94s
✓ New best: 65.5% → saved to best_model.pth
[phase1] Epoch 9/15 | Train Loss: 1.6965 | Val Loss: 1.0754 | Val Bal Acc: 64.9% | LR: 0.000629 | 92s
[phase1] Epoch 10/15 | Train Loss: 1.6744 | Val Loss: 1.0337 | Val Bal Acc: 65.2% | LR: 0.000500 | 94s
[phase1] Epoch 11/15 | Train Loss: 1.6187 | Val Loss: 1.0507 | Val Bal Acc: 64.2% | LR: 0.000371 | 106s
[phase1] Epoch 12/15 | Train Loss: 1.6225 | Val Loss: 1.0438 | Val Bal Acc: 65.1% | LR: 0.000250 | 80s
[phase1] Epoch 13/15 | Train Loss: 1.5952 | Val Loss: 1.0383 | Val Bal Acc: 64.5% | LR: 0.000146 | 95s
[phase1] Epoch 14/15 | Train Loss: 1.6343 | Val Loss: 1.0474 | Val Bal Acc: 65.4% | LR: 0.000067 | 89s
[phase1] Epoch 15/15 | Train Loss: 1.6445 | Val Loss: 1.0361 | Val Bal Acc: 65.9% | LR: 0.000017 | 86s
✓ New best: 65.9% → saved to best_model.pth

── Fold 1 | Phase 2: Fine-tuning (last 4 blocks) ──
[phase2] Epoch 1/25 | Train Loss: 1.5901 | Val Loss: 0.9760 | Val Bal Acc: 71.6% | LR: 0.000003 | 115s
✓ New best: 71.6% → saved to best_model.pth
[phase2] Epoch 2/25 | Train Loss: 1.5677 | Val Loss: 0.9552 | Val Bal Acc: 72.1% | LR: 0.000007 | 115s
✓ New best: 72.1% → saved to best_model.pth
[phase2] Epoch 3/25 | Train Loss: 1.5575 | Val Loss: 1.0229 | Val Bal Acc: 68.8% | LR: 0.000010 | 123s
[phase2] Epoch 4/25 | Train Loss: 1.4561 | Val Loss: 0.9569 | Val Bal Acc: 71.1% | LR: 0.000010 | 122s
[phase2] Epoch 5/25 | Train Loss: 1.4747 | Val Loss: 0.9092 | Val Bal Acc: 72.8% | LR: 0.000010 | 117s
✓ New best: 72.8% → saved to best_model.pth
[phase2] Epoch 6/25 | Train Loss: 1.4563 | Val Loss: 0.8913 | Val Bal Acc: 74.0% | LR: 0.000010 | 114s
✓ New best: 74.0% → saved to best_model.pth
[phase2] Epoch 7/25 | Train Loss: 1.4342 | Val Loss: 0.9030 | Val Bal Acc: 74.0% | LR: 0.000010 | 120s
✓ New best: 74.0% → saved to best_model.pth
[phase2] Epoch 8/25 | Train Loss: 1.3678 | Val Loss: 0.8894 | Val Bal Acc: 74.2% | LR: 0.000009 | 136s
✓ New best: 74.2% → saved to best_model.pth
[phase2] Epoch 9/25 | Train Loss: 1.3148 | Val Loss: 0.8911 | Val Bal Acc: 75.4% | LR: 0.000009 | 111s
✓ New best: 75.4% → saved to best_model.pth
[phase2] Epoch 10/25 | Train Loss: 1.2703 | Val Loss: 0.9242 | Val Bal Acc: 74.8% | LR: 0.000008 | 130s
[phase2] Epoch 11/25 | Train Loss: 1.3099 | Val Loss: 0.8680 | Val Bal Acc: 75.5% | LR: 0.000008 | 110s
✓ New best: 75.5% → saved to best_model.pth
[phase2] Epoch 12/25 | Train Loss: 1.3485 | Val Loss: 0.9193 | Val Bal Acc: 74.8% | LR: 0.000007 | 122s
[phase2] Epoch 13/25 | Train Loss: 1.2322 | Val Loss: 0.8645 | Val Bal Acc: 76.0% | LR: 0.000006 | 123s
✓ New best: 76.0% → saved to best_model.pth
[phase2] Epoch 14/25 | Train Loss: 1.2952 | Val Loss: 0.8792 | Val Bal Acc: 76.2% | LR: 0.000006 | 123s
✓ New best: 76.2% → saved to best_model.pth
[phase2] Epoch 15/25 | Train Loss: 1.3052 | Val Loss: 0.8464 | Val Bal Acc: 75.8% | LR: 0.000005 | 122s
[phase2] Epoch 16/25 | Train Loss: 1.2908 | Val Loss: 0.8402 | Val Bal Acc: 76.0% | LR: 0.000004 | 135s
[phase2] Epoch 17/25 | Train Loss: 1.2370 | Val Loss: 0.8567 | Val Bal Acc: 75.2% | LR: 0.000004 | 108s
[phase2] Epoch 18/25 | Train Loss: 1.2641 | Val Loss: 0.8620 | Val Bal Acc: 75.7% | LR: 0.000003 | 116s
[phase2] Epoch 19/25 | Train Loss: 1.2036 | Val Loss: 0.8483 | Val Bal Acc: 76.2% | LR: 0.000002 | 122s
[phase2] Epoch 20/25 | Train Loss: 1.2489 | Val Loss: 0.8627 | Val Bal Acc: 75.4% | LR: 0.000002 | 123s
[phase2] Epoch 21/25 | Train Loss: 1.2111 | Val Loss: 0.8429 | Val Bal Acc: 75.5% | LR: 0.000001 | 119s
[phase2] Epoch 22/25 | Train Loss: 1.2306 | Val Loss: 0.8473 | Val Bal Acc: 77.0% | LR: 0.000001 | 119s
✓ New best: 77.0% → saved to best_model.pth
[phase2] Epoch 23/25 | Train Loss: 1.2183 | Val Loss: 0.8460 | Val Bal Acc: 76.9% | LR: 0.000000 | 120s
[phase2] Epoch 24/25 | Train Loss: 1.1252 | Val Loss: 0.8420 | Val Bal Acc: 77.1% | LR: 0.000000 | 119s
✓ New best: 77.1% → saved to best_model.pth
[phase2] Epoch 25/25 | Train Loss: 1.2369 | Val Loss: 0.8454 | Val Bal Acc: 76.8% | LR: 0.000000 | 126s

✅ Fold 1 complete — Best Bal Acc: 77.1%

Fold 1 evaluation:
Balanced Accuracy: 76.2%
Cohen's Kappa: 0.525
Macro F1: 0.575
AUC-ROC (OVR): 0.943

============================================================
FOLD 2 / 4
============================================================
Loading DINOv2 ViT-S/14...
Using cache found in /home/chanduu/.cache/torch/hub/facebookresearch_dinov2_main
Trainable: 50,951 / 22,107,527 (0.23%)

── Fold 2 | Phase 1: Linear Probing ──
[phase1] Epoch 1/15 | Train Loss: 2.4173 | Val Loss: 1.3766 | Val Bal Acc: 49.9% | LR: 0.000333 | 81s
✓ New best: 49.9% → saved to best_model.pth
[phase1] Epoch 2/15 | Train Loss: 2.0263 | Val Loss: 1.3689 | Val Bal Acc: 50.8% | LR: 0.000667 | 81s
✓ New best: 50.8% → saved to best_model.pth
[phase1] Epoch 3/15 | Train Loss: 1.9244 | Val Loss: 1.2326 | Val Bal Acc: 55.2% | LR: 0.001000 | 93s
✓ New best: 55.2% → saved to best_model.pth
[phase1] Epoch 4/15 | Train Loss: 1.9108 | Val Loss: 1.2158 | Val Bal Acc: 57.2% | LR: 0.001000 | 68s
✓ New best: 57.2% → saved to best_model.pth
[phase1] Epoch 5/15 | Train Loss: 1.7327 | Val Loss: 1.1037 | Val Bal Acc: 60.6% | LR: 0.000983 | 81s
✓ New best: 60.6% → saved to best_model.pth
[phase1] Epoch 6/15 | Train Loss: 1.7214 | Val Loss: 1.1941 | Val Bal Acc: 61.1% | LR: 0.000933 | 81s
✓ New best: 61.1% → saved to best_model.pth
[phase1] Epoch 7/15 | Train Loss: 1.7401 | Val Loss: 1.1240 | Val Bal Acc: 60.3% | LR: 0.000854 | 81s
[phase1] Epoch 8/15 | Train Loss: 1.6436 | Val Loss: 1.0612 | Val Bal Acc: 64.2% | LR: 0.000750 | 81s
✓ New best: 64.2% → saved to best_model.pth
[phase1] Epoch 9/15 | Train Loss: 1.7031 | Val Loss: 1.1308 | Val Bal Acc: 63.3% | LR: 0.000629 | 81s
[phase1] Epoch 10/15 | Train Loss: 1.6420 | Val Loss: 1.0732 | Val Bal Acc: 64.9% | LR: 0.000500 | 93s
✓ New best: 64.9% → saved to best_model.pth
[phase1] Epoch 11/15 | Train Loss: 1.6122 | Val Loss: 1.1150 | Val Bal Acc: 62.6% | LR: 0.000371 | 81s
[phase1] Epoch 12/15 | Train Loss: 1.5493 | Val Loss: 1.0665 | Val Bal Acc: 65.2% | LR: 0.000250 | 82s
✓ New best: 65.2% → saved to best_model.pth
[phase1] Epoch 13/15 | Train Loss: 1.5444 | Val Loss: 1.0604 | Val Bal Acc: 65.6% | LR: 0.000146 | 86s
✓ New best: 65.6% → saved to best_model.pth
[phase1] Epoch 14/15 | Train Loss: 1.6290 | Val Loss: 1.0642 | Val Bal Acc: 65.7% | LR: 0.000067 | 99s
✓ New best: 65.7% → saved to best_model.pth
[phase1] Epoch 15/15 | Train Loss: 1.5927 | Val Loss: 1.0600 | Val Bal Acc: 65.0% | LR: 0.000017 | 82s

── Fold 2 | Phase 2: Fine-tuning (last 4 blocks) ──
[phase2] Epoch 1/25 | Train Loss: 1.5726 | Val Loss: 1.0173 | Val Bal Acc: 66.5% | LR: 0.000003 | 120s
✓ New best: 66.5% → saved to best_model.pth
[phase2] Epoch 2/25 | Train Loss: 1.6402 | Val Loss: 0.9999 | Val Bal Acc: 67.0% | LR: 0.000007 | 121s
✓ New best: 67.0% → saved to best_model.pth
[phase2] Epoch 3/25 | Train Loss: 1.5562 | Val Loss: 1.0402 | Val Bal Acc: 65.9% | LR: 0.000010 | 122s
[phase2] Epoch 4/25 | Train Loss: 1.4938 | Val Loss: 0.9662 | Val Bal Acc: 67.0% | LR: 0.000010 | 122s
✓ New best: 67.0% → saved to best_model.pth
[phase2] Epoch 5/25 | Train Loss: 1.4211 | Val Loss: 0.8897 | Val Bal Acc: 71.0% | LR: 0.000010 | 120s
✓ New best: 71.0% → saved to best_model.pth
[phase2] Epoch 6/25 | Train Loss: 1.4915 | Val Loss: 1.0897 | Val Bal Acc: 65.6% | LR: 0.000010 | 123s
[phase2] Epoch 7/25 | Train Loss: 1.4149 | Val Loss: 0.9299 | Val Bal Acc: 71.2% | LR: 0.000010 | 122s
✓ New best: 71.2% → saved to best_model.pth
[phase2] Epoch 8/25 | Train Loss: 1.3873 | Val Loss: 1.0238 | Val Bal Acc: 67.3% | LR: 0.000009 | 122s
[phase2] Epoch 9/25 | Train Loss: 1.3589 | Val Loss: 0.9043 | Val Bal Acc: 69.1% | LR: 0.000009 | 135s
[phase2] Epoch 10/25 | Train Loss: 1.3890 | Val Loss: 0.9173 | Val Bal Acc: 71.0% | LR: 0.000008 | 109s
[phase2] Epoch 11/25 | Train Loss: 1.2978 | Val Loss: 0.8524 | Val Bal Acc: 75.2% | LR: 0.000008 | 122s
✓ New best: 75.2% → saved to best_model.pth
[phase2] Epoch 12/25 | Train Loss: 1.2194 | Val Loss: 0.9386 | Val Bal Acc: 72.3% | LR: 0.000007 | 122s
[phase2] Epoch 13/25 | Train Loss: 1.2576 | Val Loss: 0.8645 | Val Bal Acc: 75.1% | LR: 0.000006 | 119s
[phase2] Epoch 14/25 | Train Loss: 1.2545 | Val Loss: 0.8831 | Val Bal Acc: 72.2% | LR: 0.000006 | 113s
[phase2] Epoch 15/25 | Train Loss: 1.2481 | Val Loss: 0.8805 | Val Bal Acc: 74.1% | LR: 0.000005 | 113s
[phase2] Epoch 16/25 | Train Loss: 1.2574 | Val Loss: 0.8493 | Val Bal Acc: 73.2% | LR: 0.000004 | 113s
[phase2] Epoch 17/25 | Train Loss: 1.2196 | Val Loss: 0.8449 | Val Bal Acc: 74.2% | LR: 0.000004 | 113s
[phase2] Epoch 18/25 | Train Loss: 1.2236 | Val Loss: 0.8568 | Val Bal Acc: 73.9% | LR: 0.000003 | 113s
[phase2] Epoch 19/25 | Train Loss: 1.2820 | Val Loss: 0.8586 | Val Bal Acc: 75.3% | LR: 0.000002 | 113s
✓ New best: 75.3% → saved to best_model.pth
[phase2] Epoch 20/25 | Train Loss: 1.2653 | Val Loss: 0.8581 | Val Bal Acc: 73.0% | LR: 0.000002 | 113s
[phase2] Epoch 21/25 | Train Loss: 1.1940 | Val Loss: 0.8370 | Val Bal Acc: 75.2% | LR: 0.000001 | 113s
[phase2] Epoch 22/25 | Train Loss: 1.2172 | Val Loss: 0.8534 | Val Bal Acc: 74.2% | LR: 0.000001 | 113s
[phase2] Epoch 23/25 | Train Loss: 1.1853 | Val Loss: 0.8564 | Val Bal Acc: 75.3% | LR: 0.000000 | 113s
✓ New best: 75.3% → saved to best_model.pth
[phase2] Epoch 24/25 | Train Loss: 1.2172 | Val Loss: 0.8477 | Val Bal Acc: 75.6% | LR: 0.000000 | 113s
✓ New best: 75.6% → saved to best_model.pth
[phase2] Epoch 25/25 | Train Loss: 1.1704 | Val Loss: 0.8481 | Val Bal Acc: 74.9% | LR: 0.000000 | 113s

✅ Fold 2 complete — Best Bal Acc: 75.6%

Fold 2 evaluation:
Balanced Accuracy: 74.9%
Cohen's Kappa: 0.498
Macro F1: 0.565
AUC-ROC (OVR): 0.936

============================================================
FOLD 3 / 4
============================================================
Loading DINOv2 ViT-S/14...
Using cache found in /home/chanduu/.cache/torch/hub/facebookresearch_dinov2_main
Trainable: 50,951 / 22,107,527 (0.23%)

── Fold 3 | Phase 1: Linear Probing ──
[phase1] Epoch 1/15 | Train Loss: 2.4509 | Val Loss: 1.4122 | Val Bal Acc: 46.2% | LR: 0.000333 | 80s
✓ New best: 46.2% → saved to best_model.pth
[phase1] Epoch 2/15 | Train Loss: 2.0020 | Val Loss: 1.2707 | Val Bal Acc: 51.3% | LR: 0.000667 | 80s
✓ New best: 51.3% → saved to best_model.pth
[phase1] Epoch 3/15 | Train Loss: 1.8904 | Val Loss: 1.2157 | Val Bal Acc: 61.3% | LR: 0.001000 | 80s
✓ New best: 61.3% → saved to best_model.pth
[phase1] Epoch 4/15 | Train Loss: 1.7989 | Val Loss: 1.2995 | Val Bal Acc: 54.8% | LR: 0.001000 | 80s
[phase1] Epoch 5/15 | Train Loss: 1.8495 | Val Loss: 1.1866 | Val Bal Acc: 58.4% | LR: 0.000983 | 80s
[phase1] Epoch 6/15 | Train Loss: 1.6910 | Val Loss: 1.1189 | Val Bal Acc: 60.0% | LR: 0.000933 | 81s
[phase1] Epoch 7/15 | Train Loss: 1.7486 | Val Loss: 1.1333 | Val Bal Acc: 60.6% | LR: 0.000854 | 82s
[phase1] Epoch 8/15 | Train Loss: 1.6670 | Val Loss: 1.0473 | Val Bal Acc: 64.3% | LR: 0.000750 | 80s
✓ New best: 64.3% → saved to best_model.pth
[phase1] Epoch 9/15 | Train Loss: 1.7155 | Val Loss: 1.0985 | Val Bal Acc: 62.4% | LR: 0.000629 | 80s
[phase1] Epoch 10/15 | Train Loss: 1.6152 | Val Loss: 1.0978 | Val Bal Acc: 61.0% | LR: 0.000500 | 80s
[phase1] Epoch 11/15 | Train Loss: 1.6534 | Val Loss: 1.0603 | Val Bal Acc: 63.6% | LR: 0.000371 | 80s
[phase1] Epoch 12/15 | Train Loss: 1.6313 | Val Loss: 1.0717 | Val Bal Acc: 64.2% | LR: 0.000250 | 80s
[phase1] Epoch 13/15 | Train Loss: 1.6376 | Val Loss: 1.0571 | Val Bal Acc: 63.4% | LR: 0.000146 | 81s
[phase1] Epoch 14/15 | Train Loss: 1.6916 | Val Loss: 1.0570 | Val Bal Acc: 63.7% | LR: 0.000067 | 82s
[phase1] Epoch 15/15 | Train Loss: 1.5282 | Val Loss: 1.0458 | Val Bal Acc: 64.7% | LR: 0.000017 | 81s
✓ New best: 64.7% → saved to best_model.pth

── Fold 3 | Phase 2: Fine-tuning (last 4 blocks) ──
[phase2] Epoch 1/25 | Train Loss: 1.5762 | Val Loss: 1.0048 | Val Bal Acc: 67.4% | LR: 0.000003 | 125s
✓ New best: 67.4% → saved to best_model.pth
[phase2] Epoch 2/25 | Train Loss: 1.5181 | Val Loss: 1.0325 | Val Bal Acc: 68.1% | LR: 0.000007 | 100s
✓ New best: 68.1% → saved to best_model.pth
[phase2] Epoch 3/25 | Train Loss: 1.5514 | Val Loss: 0.9870 | Val Bal Acc: 68.5% | LR: 0.000010 | 113s
✓ New best: 68.5% → saved to best_model.pth
[phase2] Epoch 4/25 | Train Loss: 1.5124 | Val Loss: 0.9531 | Val Bal Acc: 72.1% | LR: 0.000010 | 113s
✓ New best: 72.1% → saved to best_model.pth
[phase2] Epoch 5/25 | Train Loss: 1.4583 | Val Loss: 0.9813 | Val Bal Acc: 70.6% | LR: 0.000010 | 113s
[phase2] Epoch 6/25 | Train Loss: 1.4068 | Val Loss: 0.9072 | Val Bal Acc: 73.8% | LR: 0.000010 | 113s
✓ New best: 73.8% → saved to best_model.pth
[phase2] Epoch 7/25 | Train Loss: 1.4539 | Val Loss: 0.9236 | Val Bal Acc: 72.5% | LR: 0.000010 | 125s
[phase2] Epoch 8/25 | Train Loss: 1.3983 | Val Loss: 0.9532 | Val Bal Acc: 71.9% | LR: 0.000009 | 113s
[phase2] Epoch 9/25 | Train Loss: 1.3563 | Val Loss: 0.9525 | Val Bal Acc: 72.4% | LR: 0.000009 | 113s
[phase2] Epoch 10/25 | Train Loss: 1.3532 | Val Loss: 0.9432 | Val Bal Acc: 74.3% | LR: 0.000008 | 100s
✓ New best: 74.3% → saved to best_model.pth
[phase2] Epoch 11/25 | Train Loss: 1.3521 | Val Loss: 0.8731 | Val Bal Acc: 74.7% | LR: 0.000008 | 113s
✓ New best: 74.7% → saved to best_model.pth
[phase2] Epoch 12/25 | Train Loss: 1.3360 | Val Loss: 0.8491 | Val Bal Acc: 75.8% | LR: 0.000007 | 113s
✓ New best: 75.8% → saved to best_model.pth
[phase2] Epoch 13/25 | Train Loss: 1.2918 | Val Loss: 0.8839 | Val Bal Acc: 73.7% | LR: 0.000006 | 113s
[phase2] Epoch 14/25 | Train Loss: 1.3072 | Val Loss: 0.9067 | Val Bal Acc: 73.8% | LR: 0.000006 | 113s
[phase2] Epoch 15/25 | Train Loss: 1.2758 | Val Loss: 0.8468 | Val Bal Acc: 73.8% | LR: 0.000005 | 113s
[phase2] Epoch 16/25 | Train Loss: 1.1985 | Val Loss: 0.8774 | Val Bal Acc: 73.6% | LR: 0.000004 | 113s
[phase2] Epoch 17/25 | Train Loss: 1.2092 | Val Loss: 0.8736 | Val Bal Acc: 74.2% | LR: 0.000004 | 113s
[phase2] Epoch 18/25 | Train Loss: 1.2202 | Val Loss: 0.8579 | Val Bal Acc: 74.9% | LR: 0.000003 | 113s
[phase2] Epoch 19/25 | Train Loss: 1.2084 | Val Loss: 0.8742 | Val Bal Acc: 75.9% | LR: 0.000002 | 113s
✓ New best: 75.9% → saved to best_model.pth
[phase2] Epoch 20/25 | Train Loss: 1.2180 | Val Loss: 0.8433 | Val Bal Acc: 74.2% | LR: 0.000002 | 113s
[phase2] Epoch 21/25 | Train Loss: 1.1339 | Val Loss: 0.8364 | Val Bal Acc: 76.1% | LR: 0.000001 | 113s
✓ New best: 76.1% → saved to best_model.pth
[phase2] Epoch 22/25 | Train Loss: 1.2212 | Val Loss: 0.8454 | Val Bal Acc: 75.1% | LR: 0.000001 | 113s
[phase2] Epoch 23/25 | Train Loss: 1.1771 | Val Loss: 0.8463 | Val Bal Acc: 75.3% | LR: 0.000000 | 113s
[phase2] Epoch 24/25 | Train Loss: 1.2036 | Val Loss: 0.8486 | Val Bal Acc: 75.3% | LR: 0.000000 | 113s
[phase2] Epoch 25/25 | Train Loss: 1.2232 | Val Loss: 0.8475 | Val Bal Acc: 75.5% | LR: 0.000000 | 113s

✅ Fold 3 complete — Best Bal Acc: 76.1%

Fold 3 evaluation:
Balanced Accuracy: 76.8%
Cohen's Kappa: 0.486
Macro F1: 0.574
AUC-ROC (OVR): 0.943

============================================================
FOLD 4 / 4
============================================================
Loading DINOv2 ViT-S/14...
Using cache found in /home/chanduu/.cache/torch/hub/facebookresearch_dinov2_main
Trainable: 50,951 / 22,107,527 (0.23%)

── Fold 4 | Phase 1: Linear Probing ──
[phase1] Epoch 1/15 | Train Loss: 2.4351 | Val Loss: 1.3301 | Val Bal Acc: 51.4% | LR: 0.000333 | 81s
✓ New best: 51.4% → saved to best_model.pth
[phase1] Epoch 2/15 | Train Loss: 2.0289 | Val Loss: 1.2255 | Val Bal Acc: 56.9% | LR: 0.000667 | 81s
✓ New best: 56.9% → saved to best_model.pth
[phase1] Epoch 3/15 | Train Loss: 1.9337 | Val Loss: 1.0931 | Val Bal Acc: 64.4% | LR: 0.001000 | 82s
✓ New best: 64.4% → saved to best_model.pth
[phase1] Epoch 4/15 | Train Loss: 1.8731 | Val Loss: 1.2386 | Val Bal Acc: 61.1% | LR: 0.001000 | 83s
[phase1] Epoch 5/15 | Train Loss: 1.7845 | Val Loss: 1.0761 | Val Bal Acc: 65.2% | LR: 0.000983 | 97s
✓ New best: 65.2% → saved to best_model.pth
[phase1] Epoch 6/15 | Train Loss: 1.7351 | Val Loss: 1.1944 | Val Bal Acc: 61.2% | LR: 0.000933 | 68s
[phase1] Epoch 7/15 | Train Loss: 1.7138 | Val Loss: 1.1212 | Val Bal Acc: 65.8% | LR: 0.000854 | 83s
✓ New best: 65.8% → saved to best_model.pth
[phase1] Epoch 8/15 | Train Loss: 1.7390 | Val Loss: 1.1307 | Val Bal Acc: 64.4% | LR: 0.000750 | 94s
[phase1] Epoch 9/15 | Train Loss: 1.7131 | Val Loss: 1.0631 | Val Bal Acc: 67.5% | LR: 0.000629 | 70s
✓ New best: 67.5% → saved to best_model.pth
[phase1] Epoch 10/15 | Train Loss: 1.6577 | Val Loss: 1.0879 | Val Bal Acc: 66.6% | LR: 0.000500 | 82s
[phase1] Epoch 11/15 | Train Loss: 1.6248 | Val Loss: 1.1258 | Val Bal Acc: 63.7% | LR: 0.000371 | 82s
[phase1] Epoch 12/15 | Train Loss: 1.5657 | Val Loss: 1.0686 | Val Bal Acc: 68.0% | LR: 0.000250 | 83s
✓ New best: 68.0% → saved to best_model.pth
[phase1] Epoch 13/15 | Train Loss: 1.5727 | Val Loss: 1.0388 | Val Bal Acc: 68.0% | LR: 0.000146 | 88s
[phase1] Epoch 14/15 | Train Loss: 1.5425 | Val Loss: 1.0331 | Val Bal Acc: 68.3% | LR: 0.000067 | 87s
✓ New best: 68.3% → saved to best_model.pth
[phase1] Epoch 15/15 | Train Loss: 1.6347 | Val Loss: 1.0399 | Val Bal Acc: 68.3% | LR: 0.000017 | 88s
✓ New best: 68.3% → saved to best_model.pth

── Fold 4 | Phase 2: Fine-tuning (last 4 blocks) ──
[phase2] Epoch 1/25 | Train Loss: 1.5868 | Val Loss: 1.0576 | Val Bal Acc: 68.2% | LR: 0.000003 | 124s
[phase2] Epoch 2/25 | Train Loss: 1.6512 | Val Loss: 0.9795 | Val Bal Acc: 70.2% | LR: 0.000007 | 118s
✓ New best: 70.2% → saved to best_model.pth
[phase2] Epoch 3/25 | Train Loss: 1.5128 | Val Loss: 0.9099 | Val Bal Acc: 74.6% | LR: 0.000010 | 163s
✓ New best: 74.6% → saved to best_model.pth
[phase2] Epoch 4/25 | Train Loss: 1.5574 | Val Loss: 0.9564 | Val Bal Acc: 74.1% | LR: 0.000010 | 215s
[phase2] Epoch 5/25 | Train Loss: 1.4457 | Val Loss: 0.8993 | Val Bal Acc: 77.2% | LR: 0.000010 | 113s
✓ New best: 77.2% → saved to best_model.pth
[phase2] Epoch 6/25 | Train Loss: 1.4821 | Val Loss: 0.8904 | Val Bal Acc: 76.8% | LR: 0.000010 | 127s
[phase2] Epoch 7/25 | Train Loss: 1.3629 | Val Loss: 0.8734 | Val Bal Acc: 77.8% | LR: 0.000010 | 102s
✓ New best: 77.8% → saved to best_model.pth
[phase2] Epoch 8/25 | Train Loss: 1.3963 | Val Loss: 0.8972 | Val Bal Acc: 77.6% | LR: 0.000009 | 115s
[phase2] Epoch 9/25 | Train Loss: 1.4020 | Val Loss: 0.9503 | Val Bal Acc: 73.4% | LR: 0.000009 | 114s
[phase2] Epoch 10/25 | Train Loss: 1.3460 | Val Loss: 0.8842 | Val Bal Acc: 76.6% | LR: 0.000008 | 115s
[phase2] Epoch 11/25 | Train Loss: 1.3884 | Val Loss: 0.8538 | Val Bal Acc: 80.1% | LR: 0.000008 | 115s
✓ New best: 80.1% → saved to best_model.pth
[phase2] Epoch 12/25 | Train Loss: 1.3249 | Val Loss: 0.8386 | Val Bal Acc: 78.5% | LR: 0.000007 | 126s
[phase2] Epoch 13/25 | Train Loss: 1.3055 | Val Loss: 0.8646 | Val Bal Acc: 78.6% | LR: 0.000006 | 115s
[phase2] Epoch 14/25 | Train Loss: 1.2565 | Val Loss: 0.8340 | Val Bal Acc: 80.8% | LR: 0.000006 | 115s
✓ New best: 80.8% → saved to best_model.pth
[phase2] Epoch 15/25 | Train Loss: 1.2361 | Val Loss: 0.8708 | Val Bal Acc: 78.2% | LR: 0.000005 | 102s
[phase2] Epoch 16/25 | Train Loss: 1.3224 | Val Loss: 0.8577 | Val Bal Acc: 78.1% | LR: 0.000004 | 115s
[phase2] Epoch 17/25 | Train Loss: 1.2613 | Val Loss: 0.8872 | Val Bal Acc: 77.2% | LR: 0.000004 | 114s
[phase2] Epoch 18/25 | Train Loss: 1.2561 | Val Loss: 0.8578 | Val Bal Acc: 78.3% | LR: 0.000003 | 114s
[phase2] Epoch 19/25 | Train Loss: 1.2243 | Val Loss: 0.8178 | Val Bal Acc: 79.4% | LR: 0.000002 | 115s
[phase2] Epoch 20/25 | Train Loss: 1.2382 | Val Loss: 0.8360 | Val Bal Acc: 79.2% | LR: 0.000002 | 115s
[phase2] Epoch 21/25 | Train Loss: 1.2072 | Val Loss: 0.8266 | Val Bal Acc: 80.9% | LR: 0.000001 | 115s
✓ New best: 80.9% → saved to best_model.pth
[phase2] Epoch 22/25 | Train Loss: 1.1440 | Val Loss: 0.8320 | Val Bal Acc: 80.5% | LR: 0.000001 | 126s
[phase2] Epoch 23/25 | Train Loss: 1.2386 | Val Loss: 0.8311 | Val Bal Acc: 79.9% | LR: 0.000000 | 101s
[phase2] Epoch 24/25 | Train Loss: 1.2424 | Val Loss: 0.8220 | Val Bal Acc: 80.5% | LR: 0.000000 | 114s
[phase2] Epoch 25/25 | Train Loss: 1.1822 | Val Loss: 0.8224 | Val Bal Acc: 80.2% | LR: 0.000000 | 114s

✅ Fold 4 complete — Best Bal Acc: 80.9%

Fold 4 evaluation:
Balanced Accuracy: 81.1%
Cohen's Kappa: 0.536
Macro F1: 0.631
AUC-ROC (OVR): 0.955
[eval] Report saved to outputs/evaluation_report.json

============================================================
EVALUATION SUMMARY
============================================================
Balanced Accuracy: 76.6% +/- 2.4%
Cohen's Kappa: 0.503 +/- 0.025
Macro F1: 0.580 +/- 0.026
AUC-ROC (OVR): 0.942 +/- 0.007

Per-class metrics:
Class Precision Recall F1

---

       mel  0.341+/-0.025  0.729+/-0.025  0.464+/-0.026
        nv  0.986+/-0.004  0.632+/-0.030  0.770+/-0.022
       bcc  0.617+/-0.064  0.834+/-0.053  0.706+/-0.040
     akiec  0.463+/-0.095  0.668+/-0.104  0.537+/-0.082
       bkl  0.540+/-0.018  0.708+/-0.060  0.611+/-0.012
        df  0.324+/-0.103  0.814+/-0.074  0.457+/-0.111
      vasc  0.353+/-0.035  0.978+/-0.028  0.518+/-0.038

Best model (fold 4): outputs/best_model.pth
Best balanced accuracy: 80.9%
Total training time: 334m 46s
wandb:
wandb: Run history:
wandb: fold_0/phase1/lr ▃▆████▇▆▅▄▄▃▂▁▁
wandb: fold_0/phase1/train_loss █▅▄▄▃▃▂▃▂▂▁▁▂▁▁
wandb: fold_0/phase1/val_bal_acc ▁▅▅▆▆▆▇█▇▇█▇█▇█
wandb: fold_0/phase1/val_loss █▆▆▃▃▄▂▂▂▃▁▂▂▂▁
wandb: fold_0/phase2/lr ▁▅█████▇▇▆▆▅▄▃
wandb: fold_0/phase2/train_loss █▇▇▅▅▄▅▂▃▃▂▁▁▂
wandb: fold_0/phase2/val_bal_acc ▁▄▄█▄▅▁▆▇▂▆▄█▆
wandb: fold_0/phase2/val_loss █▆▅▃▅▄▃▂▁▄▄▂▁▃
wandb: fold_1/phase1/lr ▃▆████▇▆▅▄▄▃▂▁▁
wandb: fold_1/phase1/train_loss █▅▄▃▃▂▂▁▂▂▁▁▁▁▁
wandb: +30 ...
wandb:
wandb: Run summary:
wandb: best_bal_acc_mean 0.76619
wandb: best_bal_acc_std 0.02403
wandb: best_fold 4
wandb: fold_0/phase1/lr 2e-05
wandb: fold_0/phase1/train_loss 1.59449
wandb: fold_0/phase1/val_bal_acc 0.65611
wandb: fold_0/phase1/val_loss 1.04738
wandb: fold_0/phase2/lr 1e-05
wandb: fold_0/phase2/train_loss 1.33291
wandb: fold_0/phase2/val_bal_acc 0.7354
wandb: +34 ...
wandb:
wandb: 🚀 View run dinov2-isic-5fold at: https://wandb.ai/chandankt-polaris-school-of-technology/medvision-skin/runs/ykbt4k1u
wandb: ⭐️ View project at: https://wandb.ai/chandankt-polaris-school-of-technology/medvision-skin
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260809_111118-ykbt4k1u/logs

✅ Training complete!
(base) chanduu@ChandanKT:~/OSS/hiperhealth/hph-medvision-channel$
