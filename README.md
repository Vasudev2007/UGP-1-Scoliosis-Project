# 3D Scoliosis Diagnosis Pipeline

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-active-success.svg)


Welcome to the **Scoliosis Project** repository. This project aims to provide a robust, non-invasive, deep learning and computer vision assisted screening and diagnosis pipeline for Scoliosis using 3D point cloud and mesh processing.

By leveraging a combination of Video-to-3D models (like VGGT) and specialized geometric processing (Meshlab algorithms, Point Cloud denoising, and alignment), this pipeline can extract critical metrics such as the **Hump Angle** and **Spine AIX** from bareback video inputs.

---

## 🌟 Overview of the Pipeline Architecture 

The pipeline can be divided into two phases. 

- **Phase 1**: Pre-processing for AIX estimation 
- **Phase 2**: AIX estimation

I have described the pipeline in four parts for the ease of understanding. 
Phase 1 comprises parts 1 to 3. 
Phase 2 comprises part 4 only. 

1. **Video to Point Cloud Conversion**: Automated extraction of frames from patient videos, pre-processing, and passing through a VGGT model to construct 3D Point Clouds.
2. **Mesh Generation & Denoising**: Robust SOR/ROR denoising techniques applied on far meshes, and Meshlab-based reconstruction for detailed 3D surface meshes.
3. **Automated Alignment**: Ground plane estimation, X-axis alignment using foot-based 3D skeletonization, and accurate co-registration of near and far meshes.
4. **Clinical Metric Estimation (AIX)**: Estimation of spine centroids, mid-sagittal planes, hip-neck offsets, and hump angles directly from the processed 3D meshes.

---

<div align="center">
  <img width="881" height="457" alt="image" src="https://github.com/user-attachments/assets/b49db86c-4647-46ba-a0df-a3f01369901f" />
</div>

## 🏗️ Pipeline Architecture

### Phase 1: Pre-processing for AIX estimation

<div align="center">
  <img width="882" height="405" alt="image" src="https://github.com/user-attachments/assets/bcef242f-717a-4d1e-893c-7ea34e14cdd8" />
  <br>
  <em>This is the overall pipeline Architecture for phase-1</em>
</div>

#### Part 1: Video to Point Cloud conversion: 

<div align="center">
  <img width="531" height="380" alt="image" src="https://github.com/user-attachments/assets/d5ddb247-8d5b-4657-adde-5fea2893ed39" />
</div>

- **Workflow**: `Videos -> Frames -> Pre-processed Frames -> VGGT -> PLY Conversion -> Point Cloud`


**Step 0: Far and Near View video input**
 
**Video Shooting Technique:**
* The person is kept in **center** with respect to the background.
* The size of the person in comparison to the background should be large for VGGT to reconstruct properly.
* The camera is rotated in a circle about the person while filming to gather all views for 3D Reconstruction until one complete circle is made.
* Here we have done the setup with a t-shirt on, but the same can be done with bare-back.
  
**Step 1: Videos to Frames**

**Key points:** 
* 45 frames for both Near and Far video are chosen. This particular number ‘45’ is completely due to GPU Capabilities. 
* Higher the frames, higher the quality of mesh and higher the accuracy. Produces very decent results in 30-45 frames.

<div align="center">
  <img width="752" height="371" alt="image" src="https://github.com/user-attachments/assets/cd83cc84-8871-4e05-ad76-76ab1e274854" />
</div>

- Extracts frames from near and far videos.
- Constructs initial 3D `.glb` models and converts them to `.ply` point clouds.

**Step 2: Frames to 3D Point cloud**

<div align="center">
  <img width="850" height="421" alt="image" src="https://github.com/user-attachments/assets/589eed2e-e782-470b-832d-620cf9fe4df3" />
</div>

**Key points:**
We utilized VGGT for 3D reconstruction. We experimented with a lot of other open-source 3D reconstruction softwares which can be found in the experiments folder. We also checked for polycam reconstruction and colmap as well. 
We experimented with VGGT’s confidence thresholds. We found the following settings via repetitive trials and errors to be the best: 
* **Near Reconstruction:** 85 Confidence threshold
* **Far Reconstruction:** 50 Confidence threshold (default) 

The output is in `.glb` format which is converted to `.ply` format using Python Scripts (which is not shown in the pipeline diagram). Keeping the threshold for near reconstruction as 85 removed the need for denoising it.

<div align="center">
  <img width="853" height="380" alt="image" src="https://github.com/user-attachments/assets/0e64b18c-0c39-469d-bc40-e3b0c1727c69" />
</div>

---

#### Part 2: Mesh Generation and Denoising

**Overview of what we have done till now:**

<div align="center">
  <img width="746" height="395" alt="image" src="https://github.com/user-attachments/assets/a9f7b43f-f6e9-4605-bd63-879cecd5d7ef" />
</div>

**Far Point Cloud Denoising:** 
<div align="center">
  <img width="862" height="377" alt="image" src="https://github.com/user-attachments/assets/fca769e7-9101-4315-8235-0a3802f2a874" />
</div>

**Types of noise in Far-point cloud:** 
<div align="center">
  <img width="880" height="536" alt="image" src="https://github.com/user-attachments/assets/f805f2bf-e99f-4581-a28b-fdb8b45c2a9a" />
</div>

**Denoising Method for Far-point cloud:** 
<div align="center">
  <img width="322" height="456" alt="image" src="https://github.com/user-attachments/assets/39b531bf-d02f-4238-bc15-6bc0445e84fc" />
</div>
Thanks to this open-source implementation, we utilized this method: https://www.open3d.org/docs/latest/tutorial/geometry/pointcloud_outlier_removal.html

**Module 1: SOR Denoising** 
Statistical Outlier Removal removes points that are further away from their neighbors compared to the average for the point cloud. It takes two input parameters:
* `nb_neighbors`, which specifies how many neighbors are taken into account in order to calculate the average distance for a given point.
* `std_ratio`, which allows setting the threshold level based on the standard deviation of the average distances across the point cloud. The lower this number the more aggressive the filter will be.

**SOR Technique:** 
<div align="center">
  <img width="243" height="351" alt="image" src="https://github.com/user-attachments/assets/a9542969-334c-44d6-bd36-ba637b2e5888" />
</div>


**Module 2: ROR Denoising:** 
Radius Outlier Removal removes points that have few neighbors in a given sphere around them. Two parameters can be used to tune the filter to your data:
* `nb_points`, which lets you pick the minimum amount of points that the sphere should contain.
* `radius`, which defines the radius of the sphere that will be used for counting the neighbors.

<div align="center">
  <img width="221" height="176" alt="image" src="https://github.com/user-attachments/assets/803fe063-cc9e-4076-8627-8c963cedd612" />
</div>

**For near mesh denoising:** 
VGGT's internal confidence threshold setting was already giving good to use results, hence we didn't bother with denoising it much. I explored different denoising adaptive settings for that too. But our trials were done in a similar indoor environment so noise was of a single type. When we tried to do the same experiment in outdoor settings, the video contained other types of noise for which VGGT's internal setting and a fixed threshold can't help us anymore. Hence if anyone is working on this in the future, please use some denoising for near-mesh as well which is adaptive to the noise present in the point cloud (same SOR and ROR denoising was found ineffective for near mesh). 

- **Far Mesh**: Handles heavily noisy data via SOR-ROR denoising before creating the mesh using MeshLab algorithms.
- **Near Mesh**: Optional denoising is applied, followed by mesh formation and mesh post-processing.

**Mesh reconstruction:**
<div align="center">
  <img width="870" height="441" alt="image" src="https://github.com/user-attachments/assets/a9f8ecee-c7e8-42d5-9382-183cf4daf7c2" />
</div>

For mesh reconstruction we utilized meshlab (or equivalently in Python `pymeshlab` which is an excellent open-source implementation). We tried other mesh reconstruction softwares and libraries such as Trimesh but we didn't explore it much as meshlab did our work. 

<div align="center">
  <img width="841" height="457" alt="image" src="https://github.com/user-attachments/assets/6303f04c-3aba-4bb8-9842-fa6f9e98eb25" />
</div>

---

#### Part 3: Axis Alignment

<div align="center">
  <img width="877" height="455" alt="image" src="https://github.com/user-attachments/assets/3fcb6c3c-9bb1-410f-9e62-4f68d204ab26" />
</div>


This was the second most difficult part of our project after the AIX. We tried different approaches here. 

**Approach 1: User Point-Prompts** 
Ground plane estimation on far view mesh via user point-prompts and then manual alignment of near mesh with far-mesh with the help of user-given similar points on far and near mesh. User had to give point prompts. 

This is the video where you see the concept:-
https://github.com/user-attachments/assets/dd799ff7-8163-4b09-94db-104bff3c622e

**Approach 2: Automating Ground-Plane estimation:** 
**Logic:** 
1. Mesh -> converted to point cloud 
2. Then find the largest 3 planes via iteration 
3. Evaluate candidate ground plane via the following definition 
   `Score = ortho_score*2 + spread_ratio` 

   a. **Ortho_score**: 
   `dot_pc3 = abs(np.dot(normal, pc3)  ; ortho_score = 1.0 - dot_pc3`
   Convention : `pc1` (Major axis) in our case would be the spinal axis roughly. `Pc3` (minor axis) in our case would be the left-right axis. Ortho-score reduces the chance of walls being selected as the candidate as `pc3` for the wall-normal would be parallel. 
   
   b. **Spread_ratio**: 
   `spread = np.max(dists) - np.min(dists) ; spread_ratio = spread / (max_extent + 1e-6)`
   Points from the mesh are projected into the candidate plane and then spread is calculated. Assumption that torso proj. spread < leg proj. spread from the true ground plane perspective. 

[gnd_auto_estimated.webm](https://github.com/user-attachments/assets/975ca648-88a7-4014-af14-830066b95adb)
I automated the first part of ground plane estimation using a RANSAC based algorithm. 


**Automating X-axis alignment:** 
[X_axis auto estimation.webm](https://github.com/user-attachments/assets/31a0f464-ed6c-4057-a08e-d89a44b575ff)

**Logic of the code:**
* The Y-axis is aligned with the negative normal of the ground plane.
* A Region of Interest (ROI) is taken from 5% to 25% of the mesh's max spread above the ground plane.
* The ROI and full mesh are voxelized.
* The ROI is skeletonized using 3D Skeletonization.
* Foot-seeds are identified by finding the largest holes in a slab from the ground plane to 1% of the max spread, exploiting that the surface mesh will show gaps near the feet when viewed from below. Two holes indicate two seeds; one large hole indicates one.
* Geodesic distance is computed on the ROI, using the foot-seeds as the source.
* Parts with larger geodesic distances (e.g., hands) are pruned from the skeletonized ROI.
* The pruned ROI points are projected onto the ground plane, and the centroids of the two leg clusters are found.
* The X-axis is aligned parallel to the line connecting the two leg centroids.

<div align="center">
  <img width="972" height="557" alt="image" src="https://github.com/user-attachments/assets/6695bad8-b3c9-463e-b420-f87766ebf4ca" />
</div>


<div align="center">
  <img width="997" height="556" alt="image" src="https://github.com/user-attachments/assets/593f1302-c416-46a6-91f3-dbb7af8d9489" />
</div>

[roi.webm](https://github.com/user-attachments/assets/f4081337-44a0-43db-b2be-19cf7b26b025)
[roi_skeleton.webm](https://github.com/user-attachments/assets/9c2516b7-32a7-486c-88bd-b89ece589942)


<div align="center">
  <img width="960" height="425" alt="image" src="https://github.com/user-attachments/assets/0d166d42-e0ea-4300-b5bb-a457eed522b9" />
</div>
[idea_of_hole.webm](https://github.com/user-attachments/assets/d2e5c39f-0fb2-41cb-b402-df457ee46ca7)

[prunned_roi_skeleton.webm](https://github.com/user-attachments/assets/603bc2f5-71b5-42a4-84a8-55f834989d65)

<div align="center">
  <img width="1013" height="485" alt="image" src="https://github.com/user-attachments/assets/f4214bdd-4627-4ae9-836c-771bc58949cf" />
</div>

Since any pipeline which depends on user prompts tend to be inefficient and incorporates user-prompt errors and other things. We wanted to eliminate that so we tried different ways and explored different things, but we are not able to make it into a concrete algorithm that worked. In case any person wanted to work on this, here is the idea. 


**PCA alignment of near and far-mesh:** 
- **X-axis Alignment**: Employs foot-based seeding and 3D skeletonization to align the X-axis parallel to the feet.
- **Mesh Registration**: Semi-manual/automatic alignment of the near mesh with the aligned far mesh.

---

### Phase 2: AIX Estimation

#### Part 4: AIX (Asymmetry Index) Estimation
Calculated directly on the aligned near mesh:


- **Hip AIX**: Offset calculation between hip-center and neck-center.
<div align="center">
  <img width="1013" height="485" alt="image" src="https://github.com/user-attachments/assets/d3bf59f4-3b95-4d68-a0af-8a20b1b8aec8" />
</div>

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
