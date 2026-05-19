# Transparent Glassware Affordance Reasoning and Primitive Recovery

This repository is a **research prototype for transparent laboratory glassware manipulation**, built **on top of the original FastSAM project**.

The current work extends the original FastSAM codebase toward two connected goals:

1. **Task-conditioned affordance reasoning**
   - Given an image and a task instruction, predict an affordance heatmap indicating where the object should be contacted.

2. **Primitive / compound primitive recovery**
   - Recover a structured geometric representation for transparent glassware from RGB-D observations.
   - Use the recovered geometry as a stable intermediate representation for downstream contact or grasp proposal generation.

This repository is **not** the original FastSAM release anymore. It is now a project workspace containing:

- the original FastSAM segmentation codebase as a base dependency
- a lightweight affordance learning pipeline
- a chem-lab HOVA-style affordance dataset
- primitive fitting prototypes for glassware

---

## 1. Project Motivation

Transparent laboratory glassware is difficult for robot perception because:

- raw RGB-D depth is often noisy, missing, or aligned to the background
- object boundaries are unstable due to transparency and reflections
- generic segmentation or grasp detection is usually not enough for manipulation

At the same time, glassware has **strong geometric priors**:

- beaker ≈ open cylinder
- glass rod ≈ cylinder / capsule
- flask ≈ frustum + neck cylinder
- dropper bottle ≈ body cylinder + neck + cap

This project treats glassware manipulation as a **structured perception problem**:

```text
RGB / RGB-D + instruction
    -> task-conditioned affordance heatmap
    -> primitive / compound primitive recovery
    -> part-level affordance surface
    -> contact / grasp proposal
```

---

## 2. Relationship to the Original FastSAM

This repository is **based on the original FastSAM project** from CASIA-IVA-Lab.

FastSAM is still used here as a practical segmentation component for:

- quick object masking
- ROI extraction
- dataset bootstrapping
- debugging and prototyping

However, the main research target of this repository is no longer “segment anything”.

The focus has shifted to:

- transparent glassware affordance reasoning
- HOVA-style supervision generation
- lightweight instruction-conditioned heatmap prediction
- structured primitive fitting and geometric recovery

So if you are looking for the original upstream FastSAM README, model description, or benchmarks, this repository is no longer the right reference. This README documents **the current research project state** instead.

---

## 3. Main Components

### 3.1 `affordance/`

This package contains the current affordance learning pipeline.

Main functionality:

- lightweight text-conditioned affordance prediction
- synthetic smoke dataset support
- chem-HOVA full-image training
- inference and batch inference utilities

Important files:

- `affordance/train_chem_hova.py`
  - full-image affordance training on `chem_hova_dataset`
- `affordance/infer_chem_hova.py`
  - single-image inference for the full-image affordance model
- `affordance/run_batch_infer_chem_hova.py`
  - batch inference over the full test split
- `affordance/models/text_encoder.py`
  - MiniLM-based text encoder
- `affordance/models/glover_lite_fullimg.py`
  - GLOVER-lite full-image affordance model

### 3.2 `chem_hova_dataset/`

This is a HOVA-style chemistry affordance dataset derived from an existing RGB dataset.

It contains:

- `images/{train,val,test}`
- `GT_gaussian/{train,val,test}`
- `annotations/{train,val,test}/chem_lab_<split>.json`
- `previews/{train,val,test}`
- `tools/generate_chem_hova_from_yolo.py`

The affordance supervision uses:

- object/action-specific point selection
- hand-written geometry rules
- Gaussian heatmap generation

This dataset is currently used to verify that the affordance training pipeline is runnable and convergent.

### 3.3 `primitive_recovery/`

This package contains early primitive fitting code.

Current status:

- a simple MVP for **beaker fitting**
- parameter initialization from mask + depth
- mask/depth/contour/prior losses
- rendered mask/depth visualization

Important files:

- `primitive_recovery/fit_beaker.py`
  - fit a simple beaker primitive from mask and depth
- `primitive_recovery/generate_beaker_masks_with_fastsam.py`
  - generate batch beaker masks for a RealSense sequence with FastSAM
- `primitive_recovery/templates.py`
  - primitive parameter structure
- `primitive_recovery/render_beaker.py`
  - simple beaker rendering utilities
- `primitive_recovery/losses.py`
  - fitting losses

This part is still an MVP and should be treated as an evolving prototype rather than a finished primitive recovery system.

### 3.4 `real_dataset/`

This contains captured real RGB-D sequences, including:

- `real_dataset/realsense_dataset/<id>/rgb`
- `real_dataset/realsense_dataset/<id>/depth`
- `real_dataset/realsense_dataset/<id>/depth_completed`

These data are used for:

- mask extraction
- primitive fitting experiments
- qualitative structured perception debugging

---

## 4. Environment

Per project convention:

> Always use the `fastsam-annot` conda environment as the Python environment for this project.

Typical usage:

```bash
conda activate fastsam-annot
cd /home/dsj/FastSAM
```

---

## 5. Current Workflows

## 5.1 Train the full-image affordance baseline

Example:

```bash
HF_ENDPOINT=https://hf-mirror.com python -m affordance.train_chem_hova \
  --dataset-root /home/dsj/FastSAM/chem_hova_dataset \
  --save-dir affordance_runs/chem_hova_fullimg \
  --epochs 1 \
  --batch-size 4 \
  --image-size 384 \
  --device cuda
```

If CUDA is unavailable, replace `--device cuda` with `--device cpu`.

## 5.2 Run single-image affordance inference

```bash
HF_ENDPOINT=https://hf-mirror.com python -m affordance.infer_chem_hova \
  --checkpoint /home/dsj/FastSAM/affordance_runs/chem_hova_fullimg/best.pt \
  --rgb /home/dsj/FastSAM/chem_hova_dataset/images/test/ce_000006.jpg \
  --instruction "Where should I interact with the Alcohol lamp to hold it?" \
  --output-dir /home/dsj/FastSAM/affordance_outputs/chem_hova_test \
  --device cuda
```

## 5.3 Run batch affordance inference on the test set

```bash
HF_ENDPOINT=https://hf-mirror.com python -m affordance.run_batch_infer_chem_hova \
  --dataset-root /home/dsj/FastSAM/chem_hova_dataset \
  --checkpoint /home/dsj/FastSAM/affordance_runs/chem_hova_fullimg_e5/best.pt \
  --output-root /home/dsj/FastSAM/affordance_outputs/chem_hova_test_all \
  --device cuda
```

## 5.4 Generate beaker masks for a RealSense sequence using FastSAM

```bash
python segpredict.py
```

By default this processes:

- input:
  - `/home/dsj/FastSAM/real_dataset/realsense_dataset/4/rgb`
- output masks:
  - `/home/dsj/FastSAM/real_dataset/realsense_dataset/4/masks_fastsam_beaker`
- output previews:
  - `/home/dsj/FastSAM/real_dataset/realsense_dataset/4/previews_fastsam_beaker`

## 5.5 Fit a simple beaker primitive from mask + depth

```bash
python -m primitive_recovery.fit_beaker \
  --mask /home/dsj/FastSAM/real_dataset/realsense_dataset/4/masks_fastsam_beaker/000000.png \
  --depth /home/dsj/FastSAM/real_dataset/realsense_dataset/4/depth_completed/000000.png \
  --output-dir /home/dsj/FastSAM/primitive_outputs/beaker_fit_000000 \
  --fx 605.74 \
  --fy 605.42 \
  --cx 335.28 \
  --cy 249.04 \
  --maxiter 120
```

This is currently a simplified MVP:

- the primitive model is still rough
- the optimization is not yet a full 3D transparent glassware solution
- results should be interpreted as a structural debugging prototype

---

## 6. Current Research Status

### Affordance side

What is already working:

- chem-HOVA style dataset construction
- full-image instruction-conditioned affordance training
- stable loss decrease on the current baseline
- test-time batch visualization

What it currently means:

- the data format is usable
- the training pipeline is functional
- the GLOVER-inspired affordance direction is technically feasible in this codebase

What it does **not** yet mean:

- transparent glassware manipulation is solved
- final affordance quality is already sufficient
- the current baseline is the final model

### Primitive side

What is already working:

- basic parameterized beaker model
- initialization from mask and depth
- optimization loop with several geometric losses
- rendered mask/depth outputs for debugging

What is not yet finished:

- reliable object masks from real sequences
- true category-conditioned primitive library
- compound primitive fitting
- robust use of real aligned camera intrinsics
- integration with affordance surface reasoning
- 3D contact/grasp proposal synthesis

---

## 7. Important Limitations

This repository currently contains several **MVP / prototype** modules.

Please keep the following in mind:

- `primitive_recovery/` is not a final glassware fitting system
- some scripts still use simplified assumptions or coarse masks
- RealSense intrinsics and depth alignment should be verified before reporting geometry results
- transparent-object depth quality remains a key bottleneck
- the affordance baseline is a proof-of-pipeline result, not a final benchmark claim

---

## 8. Suggested Next Steps

The most reasonable near-term progression is:

1. improve real glassware masks
2. validate primitive fitting on real completed depth
3. upgrade beaker primitive to better express rim / taper / base structure
4. extend to more glassware categories:
   - glass rod
   - flask
   - dropper bottle
5. project affordance heatmaps to primitive surfaces
6. synthesize task-conditioned 3D contact or grasp proposals

---

## 9. Acknowledgement

This project is **based on the original FastSAM codebase** and keeps that code as an important underlying component for segmentation and prototyping.

We acknowledge:

- the original **FastSAM** project by CASIA-IVA-Lab
- the **Ultralytics** integration used inside the repository
- the conceptual inspiration from **GLOVER** for task-conditioned affordance reasoning

If you want to cite the original segmentation backbone or compare with the upstream project, please cite the original FastSAM work separately.
