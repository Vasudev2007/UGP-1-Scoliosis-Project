# Main Pipeline Idea: 

## Part 1: Pre-processing 
This is common for both near and far 

videos-> frames-> pre-processing-frames-> VGGT-> ply conversion ->Point cloud.



## Part2: Mesh formation ( Near and far)

This is seperate for near and far. 

### Far mesh 
Far mesh had lots of noise in it so we need denoising for it. 

Far Point Cloud -> Denoising(SOR-ROR)-> Mesh formation -> mesh_post-procesing( Meshlab algo).

### Near mesh 

The noise in near mesh was less. So we can still have an optional denoising there. 


Near Point Cloud-> Optional Denoising-> Mesh Formation( Meshlab algo)-> mesh-post-processing

## Part 3:  Axis alignment 

### Ground Plane estimation 
far mesh-> y_aligned_far_mesh ( Ground plane estimation and -Y axis aligned to ground plane normal )

### X axis alignment 
Foot based seeding and 3D skeletonisation -> X axis aligned parallel to line joining the feets 

### Manual Alignment of near and far mesh 

Near mesh and Aligned far mesh-> Manual mesh alignment algo -> Aligned_near_mesh 

## Part4: AIX Estimation on near aligned mesh 


### AIX-1 : Hip aix ( Hip-center and neck center offset)

### AIX-2: Hump angle estimation

1. Near_aligned_mesh-> mid-saggital-plane estimation and spine aix estimation-> Hip aix -> Hump aix 
 
2. 

3. near_aligned_mesh + mid-saggital-optimized-plane-> hump_aix estimation 


# Code Locations
## Part-1 Codes Summary 
| Purpose | Code Name | Path |  Conda Environment|
|----------|-----------|------|------|
| video->frames | vid_to_frame.py | Individual_models/models/vid_to_frame/vid_to_frame.py | No envirnoment as such ( can be run in vv_vggt , no such heavy libraries needed) |
frames->pre-processing | aniket | aniket | aniket|
frames->vggt | demo_gradio.py | Individual_models/vggt/demo_gradio.py | vv_vggt|
vggt's .glb->.ply | glb_to_ply.py | Individual_models/glb_to_ply|  vv_glb ( needs to be confirmed)||


## Part-2 Codes Summary 

| Purpose | Code Name | Path |  Conda Environment|
|----------|-----------|------|------|
|far_ply> denoised_far_ply | denoising.py | Individual_models/denoising/denoising.py | vv_denoise| 
denoised_far_ply-> far_mesh | mesh_algo.py | Individual_models/meshlab/mesh_algo.py | vv_meshlab |
far_mesh-> mesh_post-processing | no_script_written only idea| -| -|
near_ply-> Optional denoising| No script only idea | -|  -| 
near_denoised_ply-> near_mesh( same code as above) | mesh_algo.py | Individual_models/meshlab/mesh_algo.py| vv_meshlab |
near_mesh-> near-processed-mesh | no script written , Shikhar, Idea written in Aniket's Tab | - | -|near_mesh-> mesh_preprocessing | no_script_written only idea| Shikhar | Shikhar|
## Part-3 Codes Summary 

| Purpose | Code Name | Path |  Conda Environment|
|----------|-----------|------|------|
processed_far_mesh->ground_plane_estimation and alignment( y_aligned_far_mesh ) | auto_gnd_estimation.py | Changes/gnd_estimate/auto_gnd_estimate.py | vv_gnd_estimate| 
y_aligned_far_mesh-> x-axis estimation and alignment ( aligned_far_mesh) | skeleton_gradio.py | Changes/X_axis/foot_based_seeding/skeleton_gradio.py | vv_x_axis( not sure)|
aligned_far_mesh + near_processed_mesh -> aligned_near_mesh | manual_alignmentv2.py | Individual/models/mesh_alignment/manual_alignmentv2.py| vv_mesh_align ( check from readme, not sure)

## Part-4 AIX estimation ( I am not able to remember it. Refer reaadme in the folder Changes/ AIX )
| Purpose | Code Name | Path |  Conda Environment|
|----------|-----------|------|------|
near_aligned_mesh-> offset between hip center and neck center | hip_aix.py ( not sure of name of the file , there would be a similar file with hip_aix... something which is mostly I think the correct name ) | Changes/AIX/Hip_aix/hip_aix.py | vv_aix | 
near_aligned_mesh -> Spine estimation using Centroids and estimation of mid-saggital plane |spine_aix.py |Changes/AIX/Spine_aix/spine_aix.py |vv_aix | 
near_aligned_mesh + mid-saggital-plane | hump_aix.py | Changes/AIX/Hump_aix/hump_aix.py | vv_aix| 


# Inputs needed from you ( Aniket/Shikhar)
### 0. Creation of Pre-processing of frames and addition of relighting  ( Aniket )
### 1. skeleton_gradio.py (any )
### 2. auto_gnd_estimate.py ( any)
### 3. Optional -> Manual Alignment Automation ( Shikhar )
### 4. Post-processing of near and far mesh ( Shikhar )













