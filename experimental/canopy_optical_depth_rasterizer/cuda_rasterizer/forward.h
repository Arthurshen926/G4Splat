/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec2* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* transMat_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		// float* isovals,
		// float3* normals,
		float* transMats,
		float* colors,
		float4* normal_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		float focal_x, float focal_y,
		const float2* points_xy_image,
		const float* features,
		const float* transMats,
		const float* depths,
		const float4* normal_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* out_others);

	// Native mixed representation path. The two preprocess launches write
	// into one global screen/depth/radius/tile domain. mixed_render then
	// consumes the single sorted list and interleaves 2D surfels and 3D EWA
	// volumes during front-to-back alpha compositing.
	void mixed_preprocess_surfaces(
		int surface_count,
		const float* means3D,
		const glm::vec2* scales,
		float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* viewmatrix,
		const float* projmatrix,
		int W, int H,
		int* radii,
		float2* means2D,
		float* depths,
		float* transMats,
		float3* normals,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	void mixed_preprocess_volumes(
		int surface_count,
		int volume_count,
		const float* means3D,
		const glm::vec3* scales,
		float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* viewmatrix,
		const float* projmatrix,
		int W, int H,
		float focal_x, float focal_y,
		float tan_fovx, float tan_fovy,
		int* radii,
		float2* means2D,
		float* depths,
		float* cov3Ds,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	void mixed_render(
		const dim3 grid, dim3 block,
			const uint2* ranges,
			const uint32_t* point_list,
			const uint64_t* point_list_keys,
			int surface_count,
		int W, int H,
		const float2* means2D,
		const float* colors,
		const float* opacities,
		const int* surface_gate_indices,
		const float* surface_gate_atlas,
		int gate_count,
		int gate_size,
		const float* depth_query_bounds,
		bool has_depth_query,
		const float* surface_transMats,
		const float3* surface_normals,
		const float4* volume_conic,
		const float* depths,
		const float* audit_fields,
		int audit_field_count,
		float* primitive_responsibility,
		float* gate_responsibility,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* out_others);
}


#endif
