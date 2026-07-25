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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* transMat_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* out_others,
			int* radii = nullptr,
			bool debug = false);

		static int mixedForward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int surface_count,
			const int volume_count,
			const float* background,
			const int width,
			const int height,
			const float* surface_means3D,
			const float* surface_scales,
			const float* surface_rotations,
			const float* volume_means3D,
			const float* volume_scales,
			const float* volume_rotations,
			const float* colors,
			const float* opacities,
			const int* surface_gate_indices,
			const float* surface_gate_atlas,
			const int gate_count,
			const int gate_size,
			const float scale_modifier,
			const float* viewmatrix,
			const float* projmatrix,
			const float tan_fovx,
			const float tan_fovy,
			const bool prefiltered,
			const float* audit_fields,
			const int audit_field_count,
			float* primitive_responsibility,
			float* gate_responsibility,
			float* out_color,
			float* out_others,
			int* radii,
			bool debug);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* transMat_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_depths,
			float* dL_dmean2D,
			float* dL_dnormal,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dtransMat,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			bool debug);

		static void mixedBackward(
			const int surface_count,
			const int volume_count,
			const int rendered_count,
			const float* background,
			const int width,
			const int height,
			const float* surface_means3D,
			const float* surface_scales,
			const float* surface_rotations,
			const float* volume_means3D,
			const float* volume_scales,
			const float* volume_rotations,
			const float* colors,
			const float* opacities,
			const int* surface_gate_indices,
			const float* surface_gate_atlas,
			const int gate_size,
			const float scale_modifier,
			const float* viewmatrix,
			const float* projmatrix,
			const float tan_fovx,
			const float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpixels,
			const float* dL_dothers,
			float* dL_dmean2D,
			float* dL_dsurface_normal,
			float* dL_dsurface_transMat,
			float* dL_dvolume_conic,
			float* dL_ddepth,
			float* dL_dopacity,
			float* dL_dsurface_gate_atlas,
			float* dL_dcolors,
			float* dL_dsurface_means3D,
			float* dL_dsurface_scales,
			float* dL_dsurface_rotations,
			float* dL_dvolume_cov3D,
			float* dL_dvolume_means3D,
			float* dL_dvolume_scales,
			float* dL_dvolume_rotations,
			bool debug);
	};
};

#endif
