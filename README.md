# Restoring Reality: Image Deblurring and Denoising from Real-Life Scenes

Restoring sharp, noise-free content from handheld captures is the main goal of this UCSD ECE253 project by Group 6657. The repository is organized into two complementary research tracks:

- **Deblurring** — A PyTorch implementation of MPRNet for motion-blur removal from real-life sequences.
- **Denoising** — A FFDNet-based model for recovering clean signals from high-ISO and low-light captures (structure stubbed out for future work).

Use this document as the single source of truth for assets, environment setup, training/evaluation commands, and project status.

## Repository Layout

| Path | Description |
| ---- | ----------- |
| `Deblurring/` | Source for the MPRNet-based deblurring pipeline (`train.py`, `test.py`, `MPRNet.py`, `utils/`, etc.). |
| `Deblurring/Datasets/` | Placeholder directory; download datasets via the linked Drive folder before training/testing. |
| `Deblurring/pretrained_models/` | Folder for pretrained checkpoints (`model_deblurring.pth`) retrieved from Drive. |
| `Deblurring/results/` | Stores inference outputs; curated qualitative comparisons live in the linked Drive folder. |
| `Denoising/` | Source for the FFDNet-based denoising pipeline |
| `Deblurring/Datasets/` | Placeholder directory; download datasets via the link or manually before training/testing. |
| `README.md` | High-level documentation (this file). |

## Getting Started

### 1. Clone and Environment

> Any recent PyTorch build (1.11+) with CUDA is fine; adapt the install command to your GPU/OS combo. The notebooks under `Deblurring/*.ipynb` and `Denoising/*.ipynb` can be run on the Colab environment.

### 2. Data, Models, and Results

| Asset | Location | Notes |
| ----- | -------- | ----- |
| Deblurring datasets | [Drive link](https://drive.google.com/drive/folders/1y9eipStrhnKyPAxF32KZZXoUMjO_8q7_?usp=share_link) | Contains GoPro, HIDE, RealBlur (_train/test split expected as `Datasets/<dataset>/<split>/<input|target>`). |
| Deblurring pretrained weights | [Drive link](https://drive.google.com/drive/folders/1HsB4PL_HtB6sKZvvXu4v3qBWcNwLbXt6?usp=share_link) | Download into `Deblurring/pretrained_models/`. Default `train.py` and `test.py` point to `model_deblurring.pth`. |
| Deblurring results | [Drive link](https://drive.google.com/drive/folders/1QL4aco21Hs_-uVKbkkQ3DXqcRDu7d0pZ?usp=share_link) | Curated comparisons of inputs vs. outputs across datasets. |
| Denoising datasets (STL10) | [Drive link](https://drive.google.com/drive/folders/1zBB8amtPaqe8kAPxAZdjoJ3FfCvg8wfu?usp=drive_link) | Low-res dataset |
| Denoising datasets (DIV2K) | [Drive link](https://drive.google.com/drive/folders/19l2M5Efx1zy5lh-HEsfO2rIyumN7xeMq?usp=drive_link) | How-res real-life dataset |


## Deblurring Workflow (MPRNet)

### Configure Training

All runtime options live in `Deblurring/training.yml`. Key entries:

- `MODEL.MODE` / `MODEL.SESSION`: Tag checkpoints and tensorboard runs.
- `TRAINING.TRAIN_DIR` / `TRAINING.VAL_DIR`: Point to the dataset split (defaults assume RealBlur).
- `TRAINING.TRAIN_PS` / `VAL_PS`: Patch sizes for random crops.
- `OPTIM.*`: Controls epochs, batch size (default 16), and cosine LR schedule.

Adjust GPU IDs via `GPU: [0,1,2,3]`. The script automatically applies `CUDA_VISIBLE_DEVICES`.

### Train

```bash
cd Deblurring
python train.py \
  --config training.yml    # implicit via Config class; edit the file beforehand
```

Training initializes from `pretrained_models/model_deblurring.pth`, applies the Charbonnier + edge loss combo across MPRNet stages, and writes logs/checkpoints under `./checkpoints/Deblurring/{results,models}/<SESSION>/`.

Resume training by setting `TRAINING.RESUME: True`; the script will load the latest checkpoint from the session folder and continue the LR schedule.

### Evaluate / Inference

```bash
cd Deblurring
python test.py \
  --input_dir ./Datasets \
  --result_dir ./results \
  --weights ./pretrained_models/model_deblurring.pth \
  --dataset GoPro \
  --gpus 0
```

- Valid `--dataset` values: `GoPro`, `RealBlur`
- Outputs are written to `Deblurring/results/<dataset>/`. RealBlur datasets trigger automatic padding so images need not be multiples of 8.
- For reproducible benchmarks, keep the directory layout `Datasets/<dataset>/test/input` and `.../target`.

### Experiment Tracking

- Use `evaluate_RealBlur.py` or `evaluate_GOPRO_HIDE.m` for dataset-specific metrics once predictions are generated.
- Store curated figures or processed `.npz` logs in the shared Drive so collaborators can review qualitative differences without pulling large assets.

## Denoising Workflow (FFDNet)

 - Use 'FFDNet.ipynb' for running the pretrained model or perform testing.
 - Make sure to download the corresponding dataset before running the notebooks.

## Contributing

1. Create a feature branch per experiment (`feature/denoising-baseline`, `feat/gopro-ablations`, etc.).
2. Document new configs or scripts in a short module-level README (follow the style in `Deblurring/`).
3. Open a PR with dataset/model notes so others can reproduce the findings.

## References

- S. W. Zamir *et al.* “Multi-Stage Progressive Image Restoration,” CVPR 2021. [arXiv:2102.02808](https://arxiv.org/abs/2102.02808).
