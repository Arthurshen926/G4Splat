// Read-only projection export for conditional joint-ray diagnostics.
// Uses the exact native volume preprocessor; no alternative projection math.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "cuda_rasterizer/forward.h"
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/mixed_sort.cuh"

// Surface-only diagnostic exports. No renderer forward/backward is replaced.
std::vector<torch::Tensor> diagnosticSurfaceSupport(
    torch::Tensor xyz, torch::Tensor scales, torch::Tensor rotations,
    torch::Tensor opacity, torch::Tensor view, torch::Tensor projection,
    int width, int height)
{
    TORCH_CHECK(xyz.is_cuda() && xyz.scalar_type()==torch::kFloat32 && xyz.dim()==2 && xyz.size(1)==3);
    c10::cuda::CUDAGuard guard(xyz.device());
    for (auto t : {xyz,scales,rotations,opacity,view,projection}) {
        TORCH_CHECK(t.device()==xyz.device() && t.scalar_type()==torch::kFloat32 && t.is_contiguous());
        TORCH_CHECK(torch::isfinite(t).all().item<bool>());
    }
    const int n=xyz.size(0);
    TORCH_CHECK(scales.dim()==2 && scales.size(0)==n && scales.size(1)==2);
    TORCH_CHECK(rotations.dim()==2 && rotations.size(0)==n && rotations.size(1)==4 && opacity.numel()==n);
    TORCH_CHECK(view.numel()==16 && projection.numel()==16 && width>0 && height>0 && (scales>0).all().item<bool>());
    TORCH_CHECK((opacity>=0).all().item<bool>() && (opacity<=1).all().item<bool>());
    auto opts=xyz.options();
    auto xy=torch::zeros({n,2},opts),depth=torch::zeros({n},opts);
    auto transforms=torch::zeros({n,9},opts),normals=torch::zeros({n,3},opts);
    auto radii=torch::zeros({n},opts.dtype(torch::kInt32)),tiles=torch::zeros_like(radii);
    if(n) FORWARD::mixed_preprocess_surfaces(n,xyz.data_ptr<float>(),
        reinterpret_cast<const glm::vec2*>(scales.data_ptr<float>()),1.f,
        reinterpret_cast<const glm::vec4*>(rotations.data_ptr<float>()),opacity.data_ptr<float>(),
        view.data_ptr<float>(),projection.data_ptr<float>(),width,height,radii.data_ptr<int>(),
        reinterpret_cast<float2*>(xy.data_ptr<float>()),depth.data_ptr<float>(),transforms.data_ptr<float>(),
        reinterpret_cast<float3*>(normals.data_ptr<float>()),
        dim3((width+BLOCK_X-1)/BLOCK_X,(height+BLOCK_Y-1)/BLOCK_Y,1),
        reinterpret_cast<uint32_t*>(tiles.data_ptr<int>()),false);
    TORCH_CHECK(cudaDeviceSynchronize()==cudaSuccess,"Native surface projection failed");
    return {xy,depth,transforms,normals,radii};
}

__global__ void diagnosticSurfaceDepthKernel(int count,int surfaces,const int64_t* ids,
    const int* pixels,const float2* xy,const float* depth,const float* transforms,
    const float3* normals,const float* opacity,const int* gate_indices,float* out,bool* valid)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const uint2 pixel={static_cast<uint32_t>(pixels[2*i]),static_cast<uint32_t>(pixels[2*i+1])};
    const auto primitive=mixedLoadPrimitive(ids[i],surfaces,xy,opacity,transforms,normals,nullptr,depth);
    const auto candidate=mixedEvaluateCandidate(primitive,surfaces,pixel,
        make_float2(float(pixel.x),float(pixel.y)),gate_indices,nullptr,0,0);
    out[i]=candidate.depth;valid[i]=candidate.valid;
}

std::vector<torch::Tensor> diagnosticSurfaceDepths(
    torch::Tensor xy,torch::Tensor depth,torch::Tensor transforms,torch::Tensor normals,
    torch::Tensor opacity,torch::Tensor ids,torch::Tensor pixels)
{
    TORCH_CHECK(xy.is_cuda() && xy.dim()==2 && xy.size(1)==2);
    c10::cuda::CUDAGuard guard(xy.device());
    for(auto t:{xy,depth,transforms,normals,opacity})
        TORCH_CHECK(t.device()==xy.device() && t.scalar_type()==torch::kFloat32 && t.is_contiguous());
    const int n=xy.size(0),count=ids.numel();
    TORCH_CHECK(depth.numel()==n && opacity.numel()==n && transforms.dim()==2 && transforms.size(0)==n && transforms.size(1)==9);
    TORCH_CHECK(normals.dim()==2 && normals.size(0)==n && normals.size(1)==3);
    TORCH_CHECK(ids.device()==xy.device() && ids.scalar_type()==torch::kInt64 && ids.dim()==1 && ids.is_contiguous());
    TORCH_CHECK(pixels.device()==xy.device() && pixels.scalar_type()==torch::kInt32 && pixels.is_contiguous());
    TORCH_CHECK(pixels.dim()==2 && pixels.size(0)==count && pixels.size(1)==2);
    TORCH_CHECK((ids>=0).all().item<bool>() && (ids<n).all().item<bool>() && (pixels>=0).all().item<bool>());
    auto result=torch::zeros({count},xy.options()),valid=torch::zeros({count},xy.options().dtype(torch::kBool));
    auto gates=torch::full({n},-1,xy.options().dtype(torch::kInt32));
    if(count) diagnosticSurfaceDepthKernel<<<(count+255)/256,256>>>(count,n,ids.data_ptr<int64_t>(),pixels.data_ptr<int>(),
        reinterpret_cast<float2*>(xy.data_ptr<float>()),depth.data_ptr<float>(),transforms.data_ptr<float>(),
        reinterpret_cast<float3*>(normals.data_ptr<float>()),opacity.data_ptr<float>(),gates.data_ptr<int>(),
        result.data_ptr<float>(),valid.data_ptr<bool>());
    TORCH_CHECK(cudaDeviceSynchronize()==cudaSuccess,"Native surface pixel depth query failed");
    return {result,valid};
}

std::vector<torch::Tensor> diagnosticVolumeSupport(
    torch::Tensor xyz, torch::Tensor scales, torch::Tensor rotations,
    torch::Tensor view, torch::Tensor projection, int width, int height,
    float tanx, float tany)
{
    TORCH_CHECK(xyz.is_cuda() && xyz.scalar_type()==torch::kFloat32 && xyz.dim()==2 && xyz.size(1)==3);
    c10::cuda::CUDAGuard guard(xyz.device());
    for (auto t : {xyz, scales, rotations, view, projection}) {
        TORCH_CHECK(t.device()==xyz.device() && t.scalar_type()==torch::kFloat32 && t.is_contiguous());
        TORCH_CHECK(torch::isfinite(t).all().item<bool>());
    }
    const int n=xyz.size(0);
    TORCH_CHECK(scales.sizes()==xyz.sizes() && rotations.dim()==2 && rotations.size(0)==n && rotations.size(1)==4);
    TORCH_CHECK(view.numel()==16 && projection.numel()==16 && width>0 && height>0 && tanx>0 && tany>0);
    TORCH_CHECK((scales>0).all().item<bool>());
    auto opts=xyz.options();
    auto xy=torch::zeros({n,2},opts), depth=torch::zeros({n},opts);
    auto conic=torch::zeros({n,4},opts), cov=torch::zeros({n,6},opts), opacity=torch::ones({n},opts);
    auto radii=torch::zeros({n},opts.dtype(torch::kInt32)), tiles=torch::zeros_like(radii);
    if (n) FORWARD::mixed_preprocess_volumes(0,n,xyz.data_ptr<float>(),
        reinterpret_cast<const glm::vec3*>(scales.data_ptr<float>()),1.f,
        reinterpret_cast<const glm::vec4*>(rotations.data_ptr<float>()),opacity.data_ptr<float>(),
        view.data_ptr<float>(),projection.data_ptr<float>(),width,height,width/(2.f*tanx),height/(2.f*tany),tanx,tany,
        radii.data_ptr<int>(),reinterpret_cast<float2*>(xy.data_ptr<float>()),depth.data_ptr<float>(),
        cov.data_ptr<float>(),reinterpret_cast<float4*>(conic.data_ptr<float>()),
        dim3((width+BLOCK_X-1)/BLOCK_X,(height+BLOCK_Y-1)/BLOCK_Y,1),
        reinterpret_cast<uint32_t*>(tiles.data_ptr<int>()),false);
    TORCH_CHECK(cudaDeviceSynchronize()==cudaSuccess, "Native support projection failed");
    return {xy,depth,conic,radii};
}
