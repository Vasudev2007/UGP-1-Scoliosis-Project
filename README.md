# 3D Scoliosis Diagnosis Pipeline

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-active-success.svg)

Welcome to the **Scoliosis Project** repository. This project aims to provide a robust, non-invasive, deep learning, and computer vision-assisted screening and diagnosis pipeline for Scoliosis using 3D point cloud and mesh processing.

By leveraging a combination of Video-to-3D models (like VGGT) and specialized geometric processing (Meshlab algorithms, Point Cloud denoising, and alignment), this pipeline can extract critical metrics such as the **Hump Angle** and **Spine AIX** from bareback video inputs.

---

## 📑 Table of Contents
1. [Overview of the Pipeline Architecture](#-overview-of-the-pipeline-architecture)
2. [Phase 1: Pre-processing for AIX Estimation](#-phase-1-pre-processing-for-aix-estimation)
    * [Part 1: Video to Point Cloud Conversion](#part-1-video-to-point-cloud-conversion)
    * [Part 2: Mesh Generation and Denoising](#part-2-mesh-generation-and-denoising)
    * [Part 3: Axis Alignment](#part-3-axis-alignment)
3. [Phase 2: AIX Estimation](#-phase-2-aix-estimation)
    * [Part 4: AIX Calculation Sections](#part-4-aix-asymmetry-index-estimation)
4. [Future Phase Project](#-future-phase-project)
5. [Repository Structure](#-repository-structure)

---

## 🌟 Overview of the Pipeline Architecture 

The pipeline can be divided into two main phases:

- **Phase 1**: Pre-processing for AIX estimation (Comprises Parts 1 to 3)
- **Phase 2**: AIX estimation (Comprises Part 4 only)

I have described the pipeline in four parts for ease of understanding:

1. **Video to Point Cloud Conversion**: Automated extraction of frames from patient videos, pre-processing, and passing through a VGGT model to construct 3D Point Clouds.
2. **Mesh Generation & Denoising**: Robust SOR/ROR denoising techniques applied on far meshes, and Meshlab-based reconstruction for detailed 3D surface meshes.
3. **Automated Alignment**: Ground plane estimation, X-axis alignment using foot-based 3D skeletonization, and accurate co-registration of near and far meshes.
4. **Clinical Metric Estimation (AIX)**: Estimation of spine centroids, mid-sagittal planes, hip-neck offsets, and hump angles directly from the processed 3D meshes.

<div align="center">
  <img width="881" height="457" alt="image" src="https://github.com/user-attachments/assets/b49db86c-4647-46ba-a0df-a3f01369901f" />
</div>

---

## 🏗️ Phase 1: Pre-processing for AIX Estimation

<div align="center">
  <img width="882" height="405" alt="image" src="https://github.com/user-attachments/assets/bcef242f-717a-4d1e-893c-7ea34e14cdd8" />
  <br>
  <em>Overall Pipeline Architecture for Phase-1</em>
</div>

### Part 1: Video to Point Cloud Conversion

<div align="center">
  <img width="531" height="380" alt="image" src="https://github.com/user-attachments/assets/d5ddb247-8d5b-4657-adde-5fea2893ed39" />
</div>

**Workflow**: `Videos -> Frames -> Pre-processed Frames -> VGGT -> PLY Conversion -> Point Cloud`

#### Step 0: Far and Near View Video Input
**Video Shooting Technique:**
* The person is kept in the **center** with respect to the background.
* The size of the person in comparison to the background should be large for VGGT to reconstruct properly.
* The camera is rotated in a circle about the person while filming to gather all views for 3D Reconstruction until one complete circle is made.
* Here we have done the setup with a t-shirt on, but the same can be done bare-back.
  
#### Step 1: Videos to Frames
**Key points:** * 45 frames for both Near and Far videos are chosen. This particular number ‘45’ is strictly due to GPU capabilities. 
* Higher frame counts yield higher quality meshes and accuracy. The pipeline produces very decent results with 30-45 frames.

<div align="center">
  <img width="752" height="371" alt="image" src="https://github.com/user-attachments/assets/cd83cc84-8871-4e05-ad76-76ab1e274854" />
</div>

#### Step 2: Frames to 3D Point Cloud
<div align="center">
  <img width="850" height="421" alt="image" src="https://github.com/user-attachments/assets/589eed2e-e782-470b-832d-620cf9fe4df3" />
</div>

**Key points:**
* We utilized VGGT for 3D reconstruction. We experimented with various open-source 3D reconstruction software (found in the experiments folder), including Polycam and COLMAP. 
* Through repetitive trials and errors, we optimized VGGT’s confidence thresholds: 
  * **Near Reconstruction:** 85 Confidence threshold
  * **Far Reconstruction:** 50 Confidence threshold (default) 

The output is in `.glb` format, which is converted to `.ply` format using Python scripts. Setting the threshold for near reconstruction to 85 largely eliminated the need for subsequent denoising.

<div align="center">
  <img width="853" height="380" alt="image" src="https://github.com/user-attachments/assets/0e64b18c-0c39-469d-bc40-e3b0c1727c69" />
</div>

---

### Part 2: Mesh Generation and Denoising

**Overview of progress so far:**
<div align="center">
  <img width="746" height="395" alt="image" src="https://github.com/user-attachments/assets/a9f7b43f-f6e9-4605-bd63-879cecd5d7ef" />
</div>

#### Step 1: Far Point Cloud Denoising  
<div align="center">
  <img width="862" height="377" alt="image" src="https://github.com/user-attachments/assets/fca769e7-9101-4315-8235-0a3802f2a874" />
</div>

**Types of noise in the Far-point cloud:** <div align="center">
  <img width="880" height="536" alt="image" src="https://github.com/user-attachments/assets/f805f2bf-e99f-4581-a28b-fdb8b45c2a9a" />
</div>

**Denoising Method for Far-point cloud:** <div align="center">
  <img width="322" height="456" alt="image" src="https://github.com/user-attachments/assets/39b531bf-d02f-4238-bc15-6bc0445e84fc" />
</div>
*Reference for methodology: [Open3D Pointcloud Outlier Removal](https://www.open3d.org/docs/latest/tutorial/geometry/pointcloud_outlier_removal.html)*

* **Module 1: SOR Denoising:** Statistical Outlier Removal removes points that are further away from their neighbors compared to the average for the point cloud. 
  * `nb_neighbors`: specifies how many neighbors are considered to calculate the average distance.
  * `std_ratio`: sets the threshold level based on the standard deviation. A lower number means a more aggressive filter.
  
  <div align="center">
    <img width="243" height="351" alt="image" src="https://github.com/user-attachments/assets/a9542969-334c-44d6-bd36-ba637b2e5888" />
  </div>

* **Module 2: ROR Denoising:** Radius Outlier Removal removes points that have few neighbors within a given sphere around them.
  * `nb_points`: minimum amount of points the sphere should contain.
  * `radius`: defines the radius of the sphere used for counting neighbors.

  <div align="center">
    <img width="221" height="176" alt="image" src="https://github.com/user-attachments/assets/803fe063-cc9e-4076-8627-8c963cedd612" />
  </div>

#### Step 2: Near Point Cloud Denoising 
* VGGT's internal confidence threshold setting provided strong results, minimizing the need for denoising. However, tests in outdoor settings introduced new noise types where fixed thresholds failed. 
* Future iterations should explore adaptive denoising for near-meshes, as standard SOR/ROR proved ineffective here.
  * **Far Mesh**: Handles heavily noisy data via SOR-ROR denoising before creating the mesh using MeshLab algorithms.
  * **Near Mesh**: Optional denoising is applied, followed by mesh formation and mesh post-processing.

#### Step 3: Mesh Reconstruction
<div align="center">
  <img width="870" height="441" alt="image" src="https://github.com/user-attachments/assets/a9f8ecee-c7e8-42d5-9382-183cf4daf7c2" />
</div>

* We utilized MeshLab (or its Python equivalent, `pymeshlab`) for robust mesh reconstruction. 

<div align="center">
  <img width="841" height="457" alt="image" src="https://github.com/user-attachments/assets/6303f04c-3aba-4bb8-9842-fa6f9e98eb25" />
</div>

---

### Part 3: Axis Alignment

<div align="center">
  <img width="877" height="455" alt="image" src="https://github.com/user-attachments/assets/3fcb6c3c-9bb1-410f-9e62-4f68d204ab26" />
</div>

The convention used for axis alignment is as follows: 
* **Y-axis:** Negative normal of the ground plane 
* **X-axis:** Lateral axis (from left half to right half) 
* **Z-axis:** Spine axis (from head to pelvis) 

Part 3 is divided into four critical sections: 
1. Y-axis estimation of the far mesh 
2. X-axis estimation of the far mesh 
3. Z-axis estimation of the far mesh 
4. Alignment of near mesh with far mesh and transfer of axis 

#### Approach 1: User Point-Prompts (Manual Alignment)

* **Step 1: Ground plane estimation (Y-axis):** Ground plane estimation on the far view mesh via user point-prompts, applying RANSAC on the prompted points, and fitting a plane. 
  <div align="center">
    <video src="https://github.com/user-attachments/assets/dd799ff7-8163-4b09-94db-104bff3c622e" controls="controls" muted="muted" style="max-width: 100%;"></video>
  </div>

* **Step 2: Simultaneous execution of Sections 2, 3, and 4:** The user provides 4 point prompts on the far mesh and near mesh simultaneously (Left shoulder, Right shoulder, neck center, and pelvis center). 
  
  Using these 4 points and normalizing mesh size, we align the near and far mesh. The axes are defined as: 
  1. Left-shoulder and Right-shoulder points define the X vector direction. 
  2. Pelvis-center and Head-center points define the Z vector direction.

  <div align="center">
    <video src="https://github.com/user-attachments/assets/40c62192-7ce6-4550-8524-a7573c9b2c11" controls="controls" muted="muted" style="max-width: 100%;"></video>
  </div>

  *GUI Demonstration:* Below is the first draft of the end-to-end Phase 1 pipeline, built into a Gradio GUI: 
  <div align="center">
    <a href="https://www.youtube.com/watch?v=YLP0G1gguTc">
      <img src="https://img.youtube.com/vi/YLP0G1gguTc/maxresdefault.jpg" alt="Scoliosis Pipeline Demo" width="600">
    </a>
  </div>

#### Approach 2: Automating Everything

To eliminate user-prompt errors, we explored automated algorithms:

* **Step 1: Automating Ground-Plane Estimation:** 1. Convert Mesh to point cloud. 
  2. Find the largest 3 planes via iteration. 
  3. Evaluate candidate ground planes using: `Score = ortho_score*2 + spread_ratio`.
  
  <div align="center">
    <video src="https://github.com/user-attachments/assets/975ca648-88a7-4014-af14-830066b95adb" controls="controls" muted="muted" style="max-width: 100%;"></video>
  </div>

* **Step 2: Lateral Axis Estimation (X-axis) via 3D Skeletonization:** <div align="center">
    <video src="https://github.com/user-attachments/assets/31a0f464-ed6c-4057-a08e-d89a44b575ff" controls="controls" muted="muted" style="max-width: 100%;"></video>
  </div>

  **Logic of the code:**
  * Y-axis is aligned with the negative normal of the ground plane.
  * An ROI is taken from 5% to 25% of the mesh's max spread above the ground plane.
  * The ROI is skeletonized, and foot-seeds are identified by locating the largest holes in the mesh slab near the ground.
  * Geodesic distance is computed, pruning larger distances (e.g., hands) to isolate leg clusters.
  * The pruned ROI points project onto the ground plane to find leg centroids. The X-axis aligns parallel to the line connecting these centroids.

  <div align="center">
    <img width="972" height="557" alt="image" src="https://github.com/user-attachments/assets/6695bad8-b3c9-463e-b420-f87766ebf4ca" />
  </div>

  * **Step 2.1: ROI Extraction and Voxelization**
    <div align="center">
      <img width="997" height="556" alt="image" src="https://github.com/user-attachments/assets/593f1302-c416-46a6-91f3-dbb7af8d9489" />
      <em>Image showing ROI extraction followed by voxelization</em>
      <br><br>
      <video src="https://github.com/user-attachments/assets/f4081337-44a0-43db-b2be-19cf7b26b025" controls="controls" muted="muted" style="max-width: 100%;"></video>
      <em>Video of the voxelized ROI</em>
    </div>

  * **Step 2.2: Foot-based Geodesic Pruning**
    <div align="center">
      <img width="960" height="425" alt="image" src="https://github.com/user-attachments/assets/0d166d42-e0ea-4300-b5bb-a457eed522b9" />
      <em>Foot-based seeding on the 3D mesh</em>
      <br><br>
      <video src="https://github.com/user-attachments/assets/603bc2f5-71b5-42a4-84a8-55f834989d65" controls="controls" muted="muted" style="max-width: 100%;"></video>
      <em>Pruned ROI isolated to leg skeleton points</em>
    </div>

  * **Step 2.3: X-Axis Estimation via Centroids**
    <div align="center">
      <img width="1013" height="485" alt="image" src="https://github.com/user-attachments/assets/f4214bdd-4627-4ae9-836c-771bc58949cf" />
      <em>The X-axis perfectly aligned as the lateral axis</em>
    </div>

* **Step 3: Z-Axis Estimation:** The Z-axis is derived naturally as the cross product of the X-axis and Y-axis. 
* **Step 4: Alignment of Near and Far-Mesh:** Potential future solutions include PCA-based rough alignment followed by 3D-Rasterization transformer models (like [PREDATOR](https://arxiv.org/abs/2011.13005)) for precise co-registration.

---

## 🔬 Phase 2: AIX Estimation

<div align="center">
  <img width="327" height="365" alt="image" src="https://github.com/user-attachments/assets/2ea6e803-5e9e-400c-81c5-598a1b5d2c00" />
</div>

### Part 4: AIX (Asymmetry Index) Estimation

Once the mesh is aligned, we proceed to compute key medical parameters directly from the near mesh. Based on literature reviews, we focused on three critical measurements:

#### Section 1: AIX-1 (Digital Twin of Spine)
<div align="center">
  <img width="845" height="495" alt="image" src="https://github.com/user-attachments/assets/245315f2-4f67-4e57-8099-b39b223c08f6" />
  <em>Spine Estimation</em>
</div>

We extract two vital pieces of information: the mid-sagittal plane and the spine trajectory. This allows for area-based or volume-based AIX computation as suggested in clinical literature.

<div align="center">
  <img width="597" height="188" alt="image" src="https://github.com/user-attachments/assets/0b0f31dc-0507-4e8b-bc2c-2c2c06e7e543" />
  <em>Area-based AIX calculation methods.</em>
</div>

#### Section 2: AIX-2 (Coronal Offset & Rib-Hump Angle)
**Logic:** The 3D mesh back is partitioned into grids based on height measurements `h(i,j)` from the ground plane. A hump is identified when neighboring grids show disproportionate heights. We fit a normal plane to the humps (the "rib-hump plane") to compute critical parameters, effectively acting as a digital scoliometer.

<div align="center">
  <img width="307" height="353" alt="image" src="https://github.com/user-attachments/assets/b1de071e-956c-4947-bcec-3f7724ebcf57" />
</div>

*See the algorithm in action:* [Rib-Hump Angle Visualization (WebM)](https://github.com/user-attachments/assets/642424b0-de84-476d-bc56-2c14337f121a)

<div align="center">
  <img width="813" height="498" alt="image" src="https://github.com/user-attachments/assets/40efbdac-8511-41c9-bcb9-e33784686993" />
  <em>Parameters extracted from the rib-hump plane.</em>
</div>

#### Section 3: AIX-3 (Head and Pelvis Center Shift)
<div align="center">
  <img width="855" height="506" alt="image" src="https://github.com/user-attachments/assets/5b974007-91c8-485b-b79d-b5d84b198f41" />
</div>

Here is the video demonstration of the final AIX extraction algorithm: 
<div align="center">
  <a href="https://www.youtube.com/watch?v=6BdOaebS250">
    <img src="https://img.youtube.com/vi/6BdOaebS250/maxresdefault.jpg" alt="Demo Video" width="600">
  </a>
  <br>
  <em>Click on the thumbnail to watch on YouTube</em>
</div>

---

## 🚀 Future Phase Project

If anyone going through the repo is keen, we can use these extracted parameters to facilitate large-scale data collection on scoliosis and non-scoliosis patients. This data could train an ML classification model for automated screening. My scope concluded at the parameter extraction phase, and the broader project has since been transferred to Stanford. 

If someone wants an explanation of any part of the codebase or math, feel free to open an issue or email me. 

---

## 📂 Repository Structure

```text
├── src/                  # Main Pipeline Source Code
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
