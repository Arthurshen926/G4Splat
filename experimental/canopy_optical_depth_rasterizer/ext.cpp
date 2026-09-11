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

#include <torch/extension.h>
#include "rasterize_points.h"
std::vector<torch::Tensor> diagnosticVolumeSupport(torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, int, int, float, float);
std::vector<torch::Tensor> diagnosticSurfaceSupport(torch::Tensor,torch::Tensor,torch::Tensor,
    torch::Tensor,torch::Tensor,torch::Tensor,int,int);
std::vector<torch::Tensor> diagnosticSurfaceDepths(torch::Tensor,torch::Tensor,torch::Tensor,
    torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("diagnostic_volume_support", &diagnosticVolumeSupport);
  m.def("diagnostic_surface_support", &diagnosticSurfaceSupport);
  m.def("diagnostic_surface_depths", &diagnosticSurfaceDepths);
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("rasterize_mixed_gaussians", &RasterizeMixedGaussiansCUDA);
  m.def("rasterize_mixed_gaussians_backward", &RasterizeMixedGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
}
