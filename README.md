# Reducing Annotation Dependency for Underwater Fish Detection Through Open-Vocabulary Auto-Annotation

Code and results for my BSc Honours thesis (Rhodes University, 2026), supervised by Prof Dane Brown.

The project tests whether underwater fish detectors trained on boxes produced automatically by Grounding DINO can stand in for detectors trained on human-drawn boxes. Three detectors (YOLOv8s, YOLOv11x, RF-DETR-L) are each trained on three label sources (human, Grounding DINO tiny, Grounding DINO base) and evaluated on human labels, per sub-collection, and on the external DeepFish dataset.

![Pipeline](pipeline.png)

## Key results

| | Value |
|---|---|
| Pseudo-labelling time (12,466 images, GD-tiny, RTX 4070) | 33.1 min (vs ≈1,270 h estimated manual) |
| Fish found by Grounding DINO, relative to human labels | 76% |
| mAP50:95 on human test labels, human-trained | 0.63–0.66 |
| mAP50:95 on human test labels, GD-trained | 0.43–0.44 |
| Retention (GD-trained ÷ human-trained), mAP50:95 / mAP50 | 67–70% / 84–86% |
| Retention by sub-collection | 49% (Aldabra) to 76% (Pondoland) |

Full results are in `results/` and in the thesis.

## Repository layout

```
scripts/     all code used in the thesis (see below)
results/     result CSVs, evaluation outputs and training logs
pipeline.png methodology overview (thesis Figure 3.1)
```

### Scripts by stage

| Stage | Scripts |
|---|---|
| Data preparation | `merge_datasets.py`, `prepare_deepfish.py` |
| Pseudo-labelling | `label_gd_tiny.py`, `label_gd_base.py` (prompt `"fish."`, threshold 0.3) |
| Threshold selection | `threshold_sweep.py` |
| Training (9 models) | `training/`, run all with `run_all_training.sh` |
| DeepFish training | `train_v11x_deepfish_retrain.py`, `train_v11x_finetune.py`, `train_v11x_finetune_gdtiny.py` |
| Evaluation, YOLO | `run_cross_eval.py`, `run_subcollection_eval.py`, `run_deepfish_benchmark.py` |
| Evaluation, RF-DETR | `rfdetr_reeval.py`, `rfdetr_deepfish_chunked.py`, `run_deepfish_chunked.sh` |
| Out-of-domain models on SAIAB | `eval_ood_on_saiab.py` |
| CLAHE probe | `run_clahe_eval.py` |
| Figures | `plot_training_curves.py`, `make_qualitative.py`, `make_slide_figs.py`, `visualise_labels.py`, `visualise_models.py` |

RF-DETR mAP reported in the thesis comes from `rfdetr_reeval.py` (predictions kept above confidence 0.001, matching Ultralytics `val`); it supersedes the RF-DETR numbers produced by the earlier evaluation scripts.

## Environment

Python 3.13, single NVIDIA RTX 4070 (12 GB). Main packages are pinned in `requirements.txt` (Ultralytics 8.4.37, rfdetr 1.7.1, transformers 5.5.3, supervision 0.28.0).

```bash
python -m venv fish_env && source fish_env/bin/activate
pip install -r requirements.txt
```

Scripts use absolute paths from the original server (`/mnt/LinuxBackup/jo/...`). Edit the path constants at the top of each script before running.

## Data

The datasets are **not** included.

- **SAIAB**: underwater imagery from the South African Institute for Aquatic Biodiversity (Aldabra, Lake Tanganyika, Pondoland, Protea Banks), with annotations by Daanyaal Salie. Contact SAIAB for access.
- **DeepFish**: public dataset by Saleh et al. (2020).

Expected layout: one YOLO-format dataset per label source (`combined_hbb`, `combined_gd_tiny`, `combined_gd_base`), each with `train/`, `valid/` and `test/` folders containing `images/` and `labels/`, plus one folder per sub-collection (`aldabra_hbb`, `tanganyika_hbb`, `pondoland_hbb`, `protea_hbb`) for per-sub-collection evaluation.

## Citation

J. H. Bahar, *Reducing Annotation Dependency for Underwater Fish Detection Through Open-Vocabulary Auto-Annotation*, BSc Honours thesis, Rhodes University, 2026.
