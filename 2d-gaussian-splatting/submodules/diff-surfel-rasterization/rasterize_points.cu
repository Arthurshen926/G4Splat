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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

#define CHECK_INPUT(x)											\
	AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
	// AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
	AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(colors);
  CHECK_INPUT(opacity);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_others = torch::full({3+3+1, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
	  }

	  rendered = CudaRasterizer::Rasterizer::forward(
		geomFunc,
		binningFunc,
		imgFunc,
		P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		transMat_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_others.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, out_others, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeMixedGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& surface_means3D,
	const torch::Tensor& surface_scales,
	const torch::Tensor& surface_rotations,
	const torch::Tensor& volume_means3D,
	const torch::Tensor& volume_scales,
	const torch::Tensor& volume_rotations,
	const torch::Tensor& colors,
	const torch::Tensor& opacities,
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& audit_fields,
	const bool prefiltered,
	const bool debug)
{
	const int surface_count = surface_means3D.size(0);
	const int volume_count = volume_means3D.size(0);
	const int primitive_count = surface_count + volume_count;
	if (surface_means3D.ndimension() != 2 ||
		surface_means3D.size(1) != 3 ||
		volume_means3D.ndimension() != 2 ||
		volume_means3D.size(1) != 3)
	{
		AT_ERROR("surface_means3D and volume_means3D must have shape (N, 3)");
	}
	if (surface_scales.ndimension() != 2 || surface_scales.size(1) != 2)
		AT_ERROR("surface_scales must have shape (N_surface, 2)");
	if (volume_scales.ndimension() != 2 || volume_scales.size(1) != 3)
		AT_ERROR("volume_scales must have shape (N_volume, 3)");
	if (colors.ndimension() != 2 ||
		colors.size(0) != primitive_count ||
		colors.size(1) != NUM_CHANNELS)
		AT_ERROR("colors must have shape (N_surface + N_volume, 3)");
	if (opacities.numel() != primitive_count)
		AT_ERROR("opacities must contain N_surface + N_volume values");
	const int audit_field_count =
		audit_fields.numel() == 0 ? 0 : audit_fields.size(0);
	if (audit_field_count > 8)
		AT_ERROR("audit_fields supports at most 8 channels");
	if (audit_field_count > 0 &&
		(audit_fields.ndimension() != 3 ||
		 audit_fields.size(1) != image_height ||
		 audit_fields.size(2) != image_width))
		AT_ERROR("audit_fields must have shape (K, H, W)");

	CHECK_INPUT(background);
	CHECK_INPUT(surface_means3D);
	CHECK_INPUT(surface_scales);
	CHECK_INPUT(surface_rotations);
	CHECK_INPUT(volume_means3D);
	CHECK_INPUT(volume_scales);
	CHECK_INPUT(volume_rotations);
	CHECK_INPUT(colors);
	CHECK_INPUT(opacities);
	CHECK_INPUT(viewmatrix);
	CHECK_INPUT(projmatrix);
	CHECK_INPUT(audit_fields);

	const int H = image_height;
	const int W = image_width;
	const auto float_opts =
		colors.options().dtype(torch::kFloat32);
	torch::Tensor out_color =
		torch::zeros({NUM_CHANNELS, H, W}, float_opts);
	// Seven legacy auxiliary maps plus surface/volume alpha and depth.
	torch::Tensor out_others =
		torch::zeros({11, H, W}, float_opts);
	torch::Tensor radii = torch::zeros(
		{primitive_count},
		colors.options().dtype(torch::kInt32));
	torch::Tensor responsibility = audit_field_count > 0
		? torch::zeros(
			{primitive_count, audit_field_count + 1},
			float_opts)
		: torch::empty({0, 0}, float_opts);
	const torch::TensorOptions byte_options =
		torch::TensorOptions().dtype(torch::kByte).device(colors.device());
	torch::Tensor geomBuffer = torch::empty({0}, byte_options);
	torch::Tensor binningBuffer = torch::empty({0}, byte_options);
	torch::Tensor imgBuffer = torch::empty({0}, byte_options);
	auto geomFunc = resizeFunctional(geomBuffer);
	auto binningFunc = resizeFunctional(binningBuffer);
	auto imgFunc = resizeFunctional(imgBuffer);

	int rendered = 0;
	if (primitive_count > 0)
	{
		rendered = CudaRasterizer::Rasterizer::mixedForward(
			geomFunc,
			binningFunc,
			imgFunc,
			surface_count,
			volume_count,
			background.contiguous().data_ptr<float>(),
			W,
			H,
			surface_means3D.contiguous().data_ptr<float>(),
			surface_scales.contiguous().data_ptr<float>(),
			surface_rotations.contiguous().data_ptr<float>(),
			volume_means3D.contiguous().data_ptr<float>(),
			volume_scales.contiguous().data_ptr<float>(),
			volume_rotations.contiguous().data_ptr<float>(),
			colors.contiguous().data_ptr<float>(),
			opacities.contiguous().data_ptr<float>(),
			scale_modifier,
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			prefiltered,
			audit_fields.contiguous().data_ptr<float>(),
			audit_field_count,
			responsibility.contiguous().data_ptr<float>(),
			out_color.contiguous().data_ptr<float>(),
			out_others.contiguous().data_ptr<float>(),
			radii.contiguous().data_ptr<int>(),
			debug);
	}
	return std::make_tuple(
		rendered,
		out_color,
		out_others,
		radii,
		responsibility,
		geomBuffer,
		binningBuffer,
		imgBuffer);
}

std::tuple<
	torch::Tensor,
	torch::Tensor,
	torch::Tensor,
	torch::Tensor,
	torch::Tensor,
	torch::Tensor,
	torch::Tensor,
	torch::Tensor,
	torch::Tensor>
RasterizeMixedGaussiansBackwardCUDA(
	const torch::Tensor& background,
	const torch::Tensor& surface_means3D,
	const torch::Tensor& surface_scales,
	const torch::Tensor& surface_rotations,
	const torch::Tensor& volume_means3D,
	const torch::Tensor& volume_scales,
	const torch::Tensor& volume_rotations,
	const torch::Tensor& colors,
	const torch::Tensor& opacities,
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& radii,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_others,
	const torch::Tensor& geomBuffer,
	const int rendered_count,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug)
{
	CHECK_INPUT(background);
	CHECK_INPUT(surface_means3D);
	CHECK_INPUT(surface_scales);
	CHECK_INPUT(surface_rotations);
	CHECK_INPUT(volume_means3D);
	CHECK_INPUT(volume_scales);
	CHECK_INPUT(volume_rotations);
	CHECK_INPUT(colors);
	CHECK_INPUT(opacities);
	CHECK_INPUT(viewmatrix);
	CHECK_INPUT(projmatrix);
	CHECK_INPUT(radii);
	CHECK_INPUT(dL_dout_color);
	CHECK_INPUT(dL_dout_others);
	CHECK_INPUT(geomBuffer);
	CHECK_INPUT(binningBuffer);
	CHECK_INPUT(imageBuffer);
	const int surface_count = surface_means3D.size(0);
	const int volume_count = volume_means3D.size(0);
	const int primitive_count = surface_count + volume_count;
	const int H = dL_dout_color.size(1);
	const int W = dL_dout_color.size(2);

	torch::Tensor dL_dmean2D =
		torch::zeros({primitive_count, 3}, colors.options());
	torch::Tensor dL_dsurface_normal =
		torch::zeros({surface_count, 3}, colors.options());
	torch::Tensor dL_dsurface_transMat =
		torch::zeros({surface_count, 9}, colors.options());
	torch::Tensor dL_dvolume_conic =
		torch::zeros({volume_count, 4}, colors.options());
	torch::Tensor dL_ddepth =
		torch::zeros({primitive_count}, colors.options());
	torch::Tensor dL_dopacity =
		torch::zeros_like(opacities);
	torch::Tensor dL_dcolors =
		torch::zeros_like(colors);
	torch::Tensor dL_dsurface_means3D =
		torch::zeros_like(surface_means3D);
	torch::Tensor dL_dsurface_scales =
		torch::zeros_like(surface_scales);
	torch::Tensor dL_dsurface_rotations =
		torch::zeros_like(surface_rotations);
	torch::Tensor dL_dvolume_cov3D =
		torch::zeros({volume_count, 6}, colors.options());
	torch::Tensor dL_dvolume_means3D =
		torch::zeros_like(volume_means3D);
	torch::Tensor dL_dvolume_scales =
		torch::zeros_like(volume_scales);
	torch::Tensor dL_dvolume_rotations =
		torch::zeros_like(volume_rotations);

	if (primitive_count > 0)
	{
		CudaRasterizer::Rasterizer::mixedBackward(
			surface_count,
			volume_count,
			rendered_count,
			background.contiguous().data_ptr<float>(),
			W,
			H,
			surface_means3D.contiguous().data_ptr<float>(),
			surface_scales.contiguous().data_ptr<float>(),
			surface_rotations.contiguous().data_ptr<float>(),
			volume_means3D.contiguous().data_ptr<float>(),
			volume_scales.contiguous().data_ptr<float>(),
			volume_rotations.contiguous().data_ptr<float>(),
			colors.contiguous().data_ptr<float>(),
			opacities.contiguous().data_ptr<float>(),
			scale_modifier,
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			radii.contiguous().data_ptr<int>(),
			reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
			dL_dout_color.contiguous().data_ptr<float>(),
			dL_dout_others.contiguous().data_ptr<float>(),
			dL_dmean2D.contiguous().data_ptr<float>(),
			dL_dsurface_normal.contiguous().data_ptr<float>(),
			dL_dsurface_transMat.contiguous().data_ptr<float>(),
			dL_dvolume_conic.contiguous().data_ptr<float>(),
			dL_ddepth.contiguous().data_ptr<float>(),
			dL_dopacity.contiguous().data_ptr<float>(),
			dL_dcolors.contiguous().data_ptr<float>(),
			dL_dsurface_means3D.contiguous().data_ptr<float>(),
			dL_dsurface_scales.contiguous().data_ptr<float>(),
			dL_dsurface_rotations.contiguous().data_ptr<float>(),
			dL_dvolume_cov3D.contiguous().data_ptr<float>(),
			dL_dvolume_means3D.contiguous().data_ptr<float>(),
			dL_dvolume_scales.contiguous().data_ptr<float>(),
			dL_dvolume_rotations.contiguous().data_ptr<float>(),
			debug);
	}
	return std::make_tuple(
		dL_dsurface_means3D,
		dL_dsurface_scales,
		dL_dsurface_rotations,
		dL_dvolume_means3D,
		dL_dvolume_scales,
		dL_dvolume_rotations,
		dL_dcolors,
		dL_dopacity,
		dL_dmean2D);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
	 const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_others,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{

  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(radii);
  CHECK_INPUT(colors);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);
  CHECK_INPUT(binningBuffer);
  CHECK_INPUT(imageBuffer);
  CHECK_INPUT(geomBuffer);

  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dnormal = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dtransMat = torch::zeros({P, 9}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 2}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  transMat_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_others.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dnormal.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dtransMat.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dtransMat, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}
