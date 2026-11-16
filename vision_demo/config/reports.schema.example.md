# Measured report rows

The shipped reports.example.json is EMPTY. Do not replace missing results with
invented numbers. In notebook REPORTS, add a curated row for an actual completed
run, for example using values read from your own evaluation/compute JSON:

```python
REPORTS = [{
    'id': 'unique-run-id',
    'label': 'YOLOv10 S6',
    'source': 'self-trained',  # Distinguish paper / released / self-trained.
    'degradation': 'low_light',
    'architecture': 'v10',
    'dataset': 'VOC_Dark_Val',
    'metric': 'VOC 11points AP50 / standard difficult policy',
    'epochs': ACTUAL_COMPLETED_EPOCHS,
    'ap50': MEASURED_AP50_AS_FRACTION,  # 0..1, not 0..100.
    'train_hours': MEASURED_TRAINING_SECONDS / 3600,
    'vram_gb': MEASURED_PEAK_BYTES / (1024 ** 3),
    'vram_scope': 'train-only profile',  # Or 'whole train/validation session'.
    'budget_group': 'v10-lowlight-same-source-seed42-matched-epochs'
}]
```

Omit unknown numeric fields; UI shows an em dash. A chart is shown only for rows
with both AP50/train_hours and an explicit shared budget_group. Assign the group
only when dataset, evaluation protocol, actual budget and source provenance make
comparison defensible. Selective training does not itself accelerate inference.
