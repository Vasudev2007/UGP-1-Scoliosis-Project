# 3D Scoliosis Diagnosis Pipeline

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-active-success.svg)

Welcome to the **Scoliosis Project** repository. This project aims to provide a robust, non-invasive, deep learning and computer vision assisted screening and diagnosis pipeline for Scoliosis using 3D point cloud and mesh processing.

By leveraging a combination of Video-to-3D models (like VGGT) and specialized geometric processing (Meshlab algorithms, Point Cloud denoising, and alignment), this pipeline can extract critical metrics such as the **Hump Angle** and **Spine AIX** from bareback video inputs.

---

## 🌟 Key Features

1. **Video to Point Cloud Conversion**: Automated extraction of frames from patient videos, pre-processing, and passing through a VGGT model to construct 3D Point Clouds.
2. **Mesh Generation & Denoising**: Robust SOR/ROR denoising techniques applied on far meshes, and Meshlab-based reconstruction for detailed 3D surface meshes.
3. **Automated Alignment**: Ground plane estimation, X-axis alignment using foot-based 3D skeletonization, and accurate co-registration of near and far meshes.
4. **Clinical Metric Estimation (AIX)**: Estimation of spine centroids, mid-sagittal planes, hip-neck offsets, and hump angles directly from the processed 3D meshes.

---

## 🏗️ Pipeline Architecture
This is the overview of the pipeline: 

The pipeline is split into two phases and can be viewed as that.
Phase-1: Video to mesh 
Phase-2: mesh to AIX 

Phase-1 has 3 parts and Phase-2 has two parts I am giving names to the two phases

### Phase-1: 
<img width="882" height="405" alt="image" src="https://github.com/user-attachments/assets/bcef242f-717a-4d1e-893c-7ea34e14cdd8" />
This is the overall pipeline Architecutre for phase-1 
### Part 1: Pre-processing

Input: Videos 
Sample Input: 
![Uploading image.png…]()

**Video Shooting Technique**
* The person is kept in **center** with respect to the background.
* The size of the person in comparison to the background should be large for VGGT to reconstruct properly.
* Camera is rotated in circle about the person while filming to gather all views for 3D Reconstruction until one complete circle is not made.
* Here we have done the setup with tshirt-on. But same can be done with bare-back.
  
- **Workflow**: `Videos -> Frames -> Pre-processed Frames -> VGGT -> PLY Conversion -> Point Cloud`
- Step-1: Videos to Frames:
- 45 frames both for Near and Far video is chosen. This particular number ‘45’ is completely due to GPU Capabilities. 
Higher the frames, higher the quality of mesh and higher the accuracy. 
Produces very decent result in 30-45 frames.

- Extracts frames from near and far videos.
- Constructs initial 3D `.glb` models and converts them to `.ply` point clouds.

### Part 2: Mesh Formation
- **Far Mesh**: Handles heavily noisy data via SOR-ROR denoising before creating the mesh using MeshLab algorithms.
- **Near Mesh**: Optional denoising is applied, followed by mesh formation and mesh post-processing.

### Part 3: Axis Alignment
- **Ground Plane Estimation**: Calculates the ground plane from the far mesh and aligns the -Y axis to the ground normal.
- **X-axis Alignment**: Employs foot-based seeding and 3D skeletonization to align the X-axis parallel to the feet.
- **Mesh Registration**: Semi-manual/automatic alignment of the near mesh with the aligned far mesh.

### Part 4: AIX (Asymmetry Index) Estimation
Calculated directly on the aligned near mesh:
- **Hip AIX**: Offset calculation between hip-center and neck-center.
- **Spine Estimation**: Mid-sagittal plane estimation via centroids.
- **Hump Angle**: AIX calculation using the optimized mid-sagittal plane.

---

## 📂 Repository Structure

```text
├── src/                  # Main Pipeline Source Code (After running organize_repo.py)
│   ├── alignment/        # Point picking and manual/automatic alignment helpers
│   ├── denoising/        # SOR-ROR denoising scripts and GPU experiments
│   ├── gnd_estimate/     # Ground plane estimation algorithms
│   ├── meshlab/          # Meshlab based mesh formation logic
│   ├── vid_to_frame/     # Video frame extraction modules
│   └── main_script.py    # Pipeline Driver Script
├── archive/              # Past progress, reports, and legacy evaluations
├── docs/                 # Official documentation, presentations, and literature
├── README.md             # This file
├── .gitignore            # Git ignore file
└── LICENSE               # MIT License
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with `conda` or `venv` for environment management. You will also need **Meshlab** installed on your system for some of the mesh processing scripts.

### Installation

Navigate into the `src/` directory (once reorganized):

```bash
cd src

# Create a virtual environment
python -m venv .venv

# Activate the environment (Windows)
.\.venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies for each module
pip install -r alignment/requirements.txt
pip install -r denoising/requirements.txt
pip install -r meshlab/requirements.txt
# (See individual folders for more specific environment needs)
```

### Usage

You can run the full pipeline driver:

```bash
python main_script.py
```

Or run individual modules for testing and visualization:

```bash
# Example: Extract frames
python vid_to_frame/vid_to_frame.py --input path/to/video.mp4 --outdir frames/
```

> **Note**: Some scripts utilize interactive GUIs for point-picking (e.g., `enhanced_point_picker.py`). Ensure you run these in an environment that supports window rendering.

---

## 🤝 Contributing

Contributions to the pipeline are welcome. If you are adding reproducible steps, please expose them as CLI flags in `main_script.py` (e.g., `--extract-frames`, `--denoise`, `--align`). 

For bug reports or questions, please open an issue or contact the repository maintainers.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
