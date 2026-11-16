# LR-YOLO Vision Lab

React/TypeScript frontend + FastAPI GPU inference, served on one Kaggle/Ngrok
origin. No OpenAI API. See [runbook](../docs/vision_demo_runbook.md).

Start with [Kaggle notebook](../notebooks/demo/01_kaggle_lryolo_demo.ipynb).
Author VOC_Dark/v10 downloads by default; three other official VOC releases
are selectable. Your own checkpoints/config/sidecars can be added later. The
tested MMCV wheel is an input. VOC dataset
preparation and random gallery follow immutable native manifests. Missing model
or reports stay unavailable/empty; no fabricated research results.

Frontend source: `web/src/`. Backend: `server/`. Registry examples: `config/`.
Notebook packaging: `../notebooks/demo/build_notebook.py`. No original training
configs/models are edited by this demo.
