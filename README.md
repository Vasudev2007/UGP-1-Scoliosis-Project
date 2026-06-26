# 3D Reconstruction Pipeline for Scoliosis Assessment

## Overview

This repository contains the implementation of an end-to-end computer vision pipeline developed as part of my Undergraduate Project (UGP) under **Prof. Tushar Sandhan**, Department of Electrical Engineering, IIT Kanpur.

The objective of the project is to reconstruct anatomically aligned 3D patient meshes from multi-view video recordings and automatically quantify scoliosis severity using clinically inspired geometric metrics.

---

# Pipeline Overview

The complete pipeline consists of four major stages.

```
Multi-view Videos
        │
        ▼
Pre-processing
        │
        ▼
3D Point Cloud Generation (VGGT)
        │
        ▼
Mesh Construction & Refinement
        │
        ▼
Anatomical Axis Alignment
        │
        ▼
Asymmetry Index (AIX) Estimation
```

---

# Stage 1 — Video Pre-processing

This stage is common for both **Near View** and **Far View** recordings.

```
Videos
   │
   ▼
Frame Extraction
   │
   ▼
Frame Pre-processing
   │
   ▼
VGGT
   │
   ▼
.glb → .ply Conversion
   │
   ▼
Point Cloud
```

---

# Stage 2 — Mesh Formation

Near-view and far-view meshes are processed independently.

## Far View

```
Point Cloud
      │
      ▼
SOR + ROR Denoising
      │
      ▼
Mesh Generation
      │
      ▼
Mesh Post-processing
```

Since far-view reconstruction contains significantly more noise, adaptive denoising is applied before mesh generation.

---

## Near View

```
Point Cloud
      │
      ▼
(Optional Denoising)
      │
      ▼
Mesh Generation
      │
      ▼
Mesh Post-processing
```

Near-view reconstructions generally contain less noise, making denoising optional.

---

# Stage 3 — Anatomical Axis Alignment

## Ground Plane Estimation

```
Far Mesh
    │
    ▼
Ground Plane Estimation
    │
    ▼
Y-axis Alignment
```

The ground plane is estimated using a RANSAC-based approach, aligning the global Y-axis with the ground normal.

---

## X-axis Alignment

```
Foot Detection
      │
      ▼
3D Skeletonization
      │
      ▼
X-axis Alignment
```

Foot-based seeding together with skeletonization is used to estimate anatomical orientation.

---

## Near–Far Mesh Registration

```
Aligned Far Mesh
          +
Near Mesh
      │
      ▼
Manual Alignment
      │
      ▼
Aligned Near Mesh
```

---

# Stage 4 — Asymmetry Index (AIX)

The aligned near-view mesh is used for automated scoliosis quantification.

Current metrics include:

* Hip Center Offset
* Neck Center Offset
* Coronal Offset
* Rib Hump Angle
* Spine Asymmetry

Pipeline:

```
Aligned Mesh
      │
      ▼
Spine Estimation
      │
      ▼
Mid-Sagittal Plane
      │
      ▼
AIX Computation
```

---

# Repository Structure

## Stage 1 — Pre-processing

| Purpose              | Script            | Environment |
| -------------------- | ----------------- | ----------- |
| Video → Frames       | `vid_to_frame.py` | Default     |
| Frame Pre-processing | `aniket`          | aniket      |
| VGGT Inference       | `demo_gradio.py`  | `vv_vggt`   |
| GLB → PLY            | `glb_to_ply.py`   | `vv_glb`    |

---

## Stage 2 — Mesh Processing

| Purpose               | Script         | Environment  |
| --------------------- | -------------- | ------------ |
| Point Cloud Denoising | `denoising.py` | `vv_denoise` |
| Mesh Generation       | `mesh_algo.py` | `vv_meshlab` |
| Mesh Post-processing  | *(Planned)*    | —            |

---

## Stage 3 — Registration

| Purpose                 | Script                  | Environment       |
| ----------------------- | ----------------------- | ----------------- |
| Ground Plane Estimation | `auto_gnd_estimate.py`  | `vv_gnd_estimate` |
| X-axis Alignment        | `skeleton_gradio.py`    | `vv_x_axis`       |
| Near–Far Registration   | `manual_alignmentv2.py` | `vv_mesh_align`   |

---

## Stage 4 — AIX Estimation

| Purpose             | Script         | Environment |
| ------------------- | -------------- | ----------- |
| Hip Offset          | `hip_aix.py`   | `vv_aix`    |
| Spine Estimation    | `spine_aix.py` | `vv_aix`    |
| Rib Hump Estimation | `hump_aix.py`  | `vv_aix`    |

---

# Work in Progress

The following components are currently under development:

* Frame pre-processing and relighting
* Automated mesh post-processing
* Automated near–far mesh registration
* Improved X-axis estimation
* Mesh quality enhancement

---

# Future Improvements

* Complete automation of the entire reconstruction pipeline
* Automatic mesh registration
* GPU acceleration
* Interactive visualization interface
* Clinical validation against expert annotations

---

# Acknowledgements

This work was carried out as part of the Undergraduate Research Project (UGP) under **Prof. Tushar Sandhan**, Department of Electrical Engineering, IIT Kanpur.

This work makes use of several outstanding open-source projects.

- **VGGT**
  - Paper: https://arxiv.org/abs/2503.11651
  - Repository: https://github.com/facebookresearch/vggt

- **Open3D**
  - https://github.com/isl-org/Open3D

- **PyMeshLab**
  - https://github.com/cnr-isti-vclab/PyMeshLab

- **OpenCV**
  - https://opencv.org/
