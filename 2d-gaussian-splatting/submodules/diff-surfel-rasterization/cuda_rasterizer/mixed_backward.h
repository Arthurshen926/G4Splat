/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 */

#ifndef CUDA_RASTERIZER_MIXED_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_MIXED_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace MIXED_BACKWARD
{
	void render(
		dim3 grid,
		dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int surface_count,
		int W,
		int H,
		const float* background,
		const float2* means2D,
		const float* colors,
		const float* opacities,
		const float* surface_transMats,
		const float3* surface_normals,
		const float4* volume_conic,
		const float* depths,
		const float* final_T,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* dL_dothers,
		float* dL_dsurface_transMat,
		float3* dL_dmean2D,
		float3* dL_dsurface_normal,
		float4* dL_dvolume_conic,
		float* dL_ddepth,
		float* dL_dopacity,
		float* dL_dcolors);

	void preprocessVolumes(
		int volume_count,
		const float3* means3D,
		const int* radii,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		float scale_modifier,
		const float* cov3Ds,
		const float* viewmatrix,
		const float* projmatrix,
		float focal_x,
		float focal_y,
		float tan_fovx,
		float tan_fovy,
		const float3* dL_dmean2D,
		const float4* dL_dconic,
		const float* dL_ddepth,
		glm::vec3* dL_dmean3D,
		float* dL_dcov3D,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot);
}

#endif
