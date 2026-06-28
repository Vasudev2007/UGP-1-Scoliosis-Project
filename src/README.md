# Pipeline - Codes for Pipeline-Version-1

Short README for the pipeline code bundle used in the end-sem project. This repository contains helper scripts and modules for alignment, denoising, mesh processing, visualization and video-to-frame conversion used by the pipeline driver `main_script.py`.

## Overview

- **Purpose:** Collection of scripts used to convert video input into processed 3D-friendly data, clean/align meshes, estimate ground plane, and export meshes.
- **Entry point:** `main_script.py`

## Repository structure

- `main_script.py` - Top-level driver that orchestrates steps in the pipeline (video import, processing, alignment, mesh export). Run this from the repository root.
- `alignment/` - Tools for manual or semi-automatic alignment of point clouds / meshes.
  - `enhanced_point_picker.py` - interactive point picking utilities
  - `manual_alignment_v2.py` - manual alignment helpers
  - `requirements.txt` - Python deps for this module
- `denoising/` - Denoising utilities and GPU experiments.
  - `denoising.py`, `gpu_try.py`, `orginal.py`
  - `requirements.txt`
- `final_visualization/` - Visualization helpers.
  - `ground_plane_estimator_simple.py` - simple ground plane estimation used for visualization/alignment
- `glb_to_ply/` - Converter(s) for GLB -> PLY or other mesh formats.
  - `glb_to_ply.py`
  - `requirements.txt`
- `gnd_estimate/` - More advanced ground estimation / point picking.
  - `enhanced_point_picker.py`
  - `ground_plane_estimator_v2.py`
  - `requirements.txt`
- `meshlab/` - Mesh processing routines that rely on Meshlab/PyMeshLab style operations.
  - `mesh_algo.py`
  - `requirements.txt`
- `vid_to_frame/` - Utilities for extracting frames from video.
  - `vid_to_frame.py`, `original.py`

## Requirements

Each subfolder that requires Python packages includes a `requirements.txt`. To create a virtual environment and install all dependencies, from the repository root run (Windows example):

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r alignment/requirements.txt
pip install -r denoising/requirements.txt
pip install -r glb_to_ply/requirements.txt
pip install -r gnd_estimate/requirements.txt
pip install -r meshlab/requirements.txt
```

Note: Some modules may rely on system libraries (Meshlab, GPU drivers, Open3D, CUDA). See the per-folder `requirements.txt` files for specifics and install system packages as needed.

## Usage

- Run the main pipeline driver from the repository root:

```bash
python main_script.py
```

- Many scripts are intended to be run individually for development or debugging. Example:

```bash
python vid_to_frame/vid_to_frame.py --input path/to/video.mp4 --outdir frames/
python glb_to_ply/glb_to_ply.py --input model.glb --output model.ply
```

Check each script header for command-line options; some utilities are interactive (point pickers, manual alignment), so run them in an environment that supports GUI windows.

## Development notes

- The repository contains experimental scripts (GPU trials, original copies). Use the `orginal.py` and `original.py` files for reference; production-ready logic is usually in the other similarly named files.
- If you need to add reproducible steps, update `main_script.py` to expose CLI flags for each stage (extract frames, reconstruct, denoise, align, export).

## Troubleshooting

- If visualization windows do not appear, ensure you run the scripts from a desktop session (not headless) or use offscreen renderers if available.
- For CUDA/GPU issues, make sure the correct driver + CUDA toolkit matching installed packages is present.

## Contact

For questions about the pipeline logic or to report issues, contact the project author or maintainers in the project repository (local group / course instructors).

---
Generated README for quick onboarding and developer reference.
