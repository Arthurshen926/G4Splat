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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ void compute_transmat(
	const float3& p_orig,
	const glm::vec2 scale,
	float mod,
	const glm::vec4 rot,
	const float* projmatrix,
	const float* viewmatrix,
	const int W,
	const int H, 
	glm::mat3 &T,
	float3 &normal
) {

	glm::mat3 R = quat_to_rotmat(rot);
	glm::mat3 S = scale_to_mat(scale, mod);
	glm::mat3 L = R * S;

	// center of Gaussians in the camera coordinate
	glm::mat3x4 splat2world = glm::mat3x4(
		glm::vec4(L[0], 0.0),
		glm::vec4(L[1], 0.0),
		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
	);

	glm::mat4 world2ndc = glm::mat4(
		projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
		projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
		projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
		projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
	);

	glm::mat3x4 ndc2pix = glm::mat3x4(
		glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
		glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
		glm::vec4(0.0, 0.0, 0.0, 1.0)
	);

	T = glm::transpose(splat2world) * world2ndc * ndc2pix;
	normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);

}

// Computing the bounding box of the 2D Gaussian and its center
// The center of the bounding box is used to create a low pass filter
__device__ bool compute_aabb(
	glm::mat3 T, 
	float cutoff, 
	float2& point_image,
	float2 & extent
) {
	float3 T0 = {T[0][0], T[0][1], T[0][2]};
	float3 T1 = {T[1][0], T[1][1], T[1][2]};
	float3 T3 = {T[2][0], T[2][1], T[2][2]};

	// Compute AABB
	float3 temp_point = {cutoff * cutoff, cutoff * cutoff, -1.0f};
	float distance = sumf3(T3 * T3 * temp_point);
	// The projected ellipse has a finite image-space AABB only when its
	// cutoff support does not intersect the projective horizon w(u,v)=0.
	// In this parameterization that condition is
	//
	//   cutoff^2 * (T3.x^2 + T3.y^2) - T3.z^2 < 0.
	//
	// The former equality-only check admitted the entire positive branch.
	// Those ellipses are mathematically unbounded, yet the algebra below
	// returned enormous finite extents (10^5--10^7 pixels in Cambridge).
	// Binning them across every tile creates exactly the view-wide paint
	// splats that UV refinement is intended to remove.  This is a domain
	// check, not a scene-dependent radius threshold: every finite projected
	// ellipse, however oblique, still follows the unchanged path.
	if (!isfinite(distance) || !(distance < 0.0f))
		return false;
	float3 f = (1 / distance) * temp_point;

	point_image = {
		sumf3(f * T0 * T3),
		sumf3(f * T1 * T3)
	};  
	
	float2 temp = {
		sumf3(f * T0 * T0),
		sumf3(f * T1 * T1)
	};
	float2 half_extend = point_image * point_image - temp;
	extent = sqrtf2(maxf2(1e-4, half_extend));
	return true;
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
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
	const float tan_fovx, const float tan_fovy,
	const float focal_x, const float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;
	
	// Compute transformation matrix
	glm::mat3 T;
	float3 normal;
	if (transMat_precomp == nullptr)
	{
		compute_transmat(((float3*)orig_points)[idx], scales[idx], scale_modifier, rotations[idx], projmatrix, viewmatrix, W, H, T, normal);
		float3 *T_ptr = (float3*)transMats;
		T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
		T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
		T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};
	} else {
		glm::vec3 *T_ptr = (glm::vec3*)transMat_precomp;
		T = glm::mat3(
			T_ptr[idx * 3 + 0], 
			T_ptr[idx * 3 + 1],
			T_ptr[idx * 3 + 2]
		);
		normal = make_float3(0.0, 0.0, 1.0);
	}

#if DUAL_VISIABLE
	float cos = -sumf3(p_view * normal);
	if (cos == 0) return;
	float multiplier = cos > 0 ? 1: -1;
	normal = multiplier * normal;
#endif

#if TIGHTBBOX // no use in the paper, but it indeed help speeds.
	// the effective extent is now depended on the opacity of gaussian.
	float cutoff = sqrtf(max(9.f + 2.f * logf(opacities[idx]), 0.000001));
#else
	float cutoff = 3.0f;
#endif

	// Compute center and radius
	float2 point_image;
	float radius;
	{
		float2 extent;
		bool ok = compute_aabb(T, cutoff, point_image, extent);
		if (!ok) return;
		radius = ceil(max(extent.x, extent.y));
	}

	uint2 rect_min, rect_max;
	getRect(point_image, radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// Compute colors 
	if (colors_precomp == nullptr) {
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	depths[idx] = p_view.z;
	radii[idx] = (int)radius;
	points_xy_image[idx] = point_image;
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ transMats,
	const float* __restrict__ depths,
	const float4* __restrict__ normal_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };


#if RENDER_AXUTILITY
	// render axutility ouput
	float N[3] = {0};
	float D = { 0 };
	float M1 = {0};
	float M2 = {0};
	float distortion = {0};
	float median_depth = {0};
	// float median_weight = {0};
	float median_contributor = {-1};

#endif

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			if (p.z == 0.0) continue;
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y); 
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 

			// compute intersection and depth
			float rho = min(rho3d, rho2d);
			float depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z; 
			if (depth < near_n) continue;
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, opa * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			float w = alpha * T;
#if RENDER_AXUTILITY
			// Render depth distortion map
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.
			float A = 1-T;
			float m = far_n / (far_n - near_n) * (1 - near_n / depth);
			distortion += (m * m * A + M2 - 2 * m * M1) * w;
			D  += depth * w;
			M1 += m * w;
			M2 += m * m * w;

			if (T > 0.5) {
				median_depth = depth;
				// median_weight = w;
				median_contributor = contributor;
			}
			// Render normal map
			for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * w;
#endif

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * w;
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

#if RENDER_AXUTILITY
		n_contrib[pix_id + H * W] = median_contributor;
		final_T[pix_id + H * W] = M1;
		final_T[pix_id + 2 * H * W] = M2;
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
		for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
		// out_others[pix_id + MEDIAN_WEIGHT_OFFSET * H * W] = median_weight;
#endif
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* means2D,
	const float* colors,
	const float* transMats,
	const float* depths,
	const float4* normal_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_others)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		means2D,
		colors,
		transMats,
		depths,
		normal_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_others);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
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
	const int W, const int H,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		transMats,
		rgb,
		normal_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}

// --- Native mixed 2D surfel + 3D EWA path -------------------------------

__device__ float3 mixedComputeCov2D(
	const float3& mean,
	float focal_x,
	float focal_y,
	float tan_fovx,
	float tan_fovy,
	const float* cov3D,
	const float* viewmatrix)
{
	float3 t = transformPoint4x3(mean, viewmatrix);
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0.0f, 0.0f, 0.0f);
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);
	glm::mat3 T = W * J;
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);
	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return {
		float(cov[0][0]),
		float(cov[0][1]),
		float(cov[1][1])
	};
}

__device__ void mixedComputeCov3D(
	const glm::vec3 scale,
	float mod,
	const glm::vec4 rot,
	float* cov3D)
{
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;
	const float r = rot.x;
	const float x = rot.y;
	const float y = rot.z;
	const float z = rot.w;
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z),
		2.f * (x * y - r * z),
		2.f * (x * z + r * y),
		2.f * (x * y + r * z),
		1.f - 2.f * (x * x + z * z),
		2.f * (y * z - r * x),
		2.f * (x * z - r * y),
		2.f * (y * z + r * x),
		1.f - 2.f * (x * x + y * y));
	glm::mat3 M = S * R;
	glm::mat3 Sigma = glm::transpose(M) * M;
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

__global__ void mixedPreprocessSurfacesCUDA(
	int surface_count,
	const float* orig_points,
	const glm::vec2* scales,
	float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* viewmatrix,
	const float* projmatrix,
	int W,
	int H,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* transMats,
	float3* normals,
	dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	const int idx = cg::this_grid().thread_rank();
	if (idx >= surface_count)
		return;
	radii[idx] = 0;
	tiles_touched[idx] = 0;
	float3 p_view;
	if (!in_frustum(
		idx,
		orig_points,
		viewmatrix,
		projmatrix,
		prefiltered,
		p_view))
		return;

	glm::mat3 T;
	float3 normal;
	compute_transmat(
		reinterpret_cast<const float3*>(orig_points)[idx],
		scales[idx],
		scale_modifier,
		rotations[idx],
		projmatrix,
		viewmatrix,
		W,
		H,
		T,
		normal);
#if DUAL_VISIABLE
	const float cos = -sumf3(p_view * normal);
	if (cos == 0)
		return;
	normal = (cos > 0 ? 1.0f : -1.0f) * normal;
#endif
#if TIGHTBBOX
	const float cutoff =
		sqrtf(max(9.f + 2.f * logf(opacities[idx]), 0.000001f));
#else
	const float cutoff = 3.0f;
#endif
	float2 point_image;
	float2 extent;
	if (!compute_aabb(T, cutoff, point_image, extent))
		return;
	const float radius = ceil(max(extent.x, extent.y));
	uint2 rect_min;
	uint2 rect_max;
	getRect(point_image, radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	float3* T_ptr = reinterpret_cast<float3*>(transMats);
	T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
	T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
	T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};
	normals[idx] = normal;
	depths[idx] = p_view.z;
	radii[idx] = static_cast<int>(radius);
	points_xy_image[idx] = point_image;
	tiles_touched[idx] =
		(rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

__global__ void mixedPreprocessVolumesCUDA(
	int surface_count,
	int volume_count,
	const float* orig_points,
	const glm::vec3* scales,
	float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* viewmatrix,
	const float* projmatrix,
	int W,
	int H,
	float tan_fovx,
	float tan_fovy,
	float focal_x,
	float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float4* conic_opacity,
	dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	const int local_idx = cg::this_grid().thread_rank();
	if (local_idx >= volume_count)
		return;
	const int idx = surface_count + local_idx;
	radii[idx] = 0;
	tiles_touched[idx] = 0;
	// Sequence/time ownership gates produce exact zero opacity for every
	// non-owner volume.  Such a primitive is mathematically absent from the
	// front-to-back integral; projecting it and emitting tile entries only
	// inflates sort memory/work (catastrophically for dense per-view leaves).
	// Keep the comparison strict so every positive contribution, however
	// small, retains bit-identical rendering semantics.
	if (!(opacities[local_idx] > 0.0f))
		return;
	float3 p_view;
	if (!in_frustum(
		local_idx,
		orig_points,
		viewmatrix,
		projmatrix,
		prefiltered,
		p_view))
		return;

	const float3 p_orig =
		reinterpret_cast<const float3*>(orig_points)[local_idx];
	const float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	const float p_w = 1.0f / (p_hom.w + 0.0000001f);
	const float3 p_proj = {
		p_hom.x * p_w,
		p_hom.y * p_w,
		p_hom.z * p_w
	};

	float* cov3D = cov3Ds + local_idx * 6;
	mixedComputeCov3D(
		scales[local_idx],
		scale_modifier,
		rotations[local_idx],
		cov3D);
	const float3 cov = mixedComputeCov2D(
		p_orig,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		cov3D,
		viewmatrix);
	const float det = cov.x * cov.z - cov.y * cov.y;
	if (det == 0.0f)
		return;
	const float det_inv = 1.0f / det;
	const float3 conic = {
		cov.z * det_inv,
		-cov.y * det_inv,
		cov.x * det_inv
	};
	const float mid = 0.5f * (cov.x + cov.z);
	const float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	const float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	const float radius = ceil(3.0f * sqrt(max(lambda1, lambda2)));
	const float2 point_image = {
		ndc2Pix(p_proj.x, W),
		ndc2Pix(p_proj.y, H)
	};
	uint2 rect_min;
	uint2 rect_max;
	getRect(point_image, radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	depths[idx] = p_view.z;
	radii[idx] = static_cast<int>(radius);
	points_xy_image[idx] = point_image;
	conic_opacity[local_idx] = {
		conic.x,
		conic.y,
		conic.z,
		opacities[local_idx]
	};
	tiles_touched[idx] =
		(rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
mixedRenderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int surface_count,
	int W,
	int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ opacities,
	const int* __restrict__ surface_gate_indices,
	const float* __restrict__ surface_gate_atlas,
	int gate_count,
	int gate_size,
	const float* __restrict__ surface_transMats,
	const float3* __restrict__ surface_normals,
	const float4* __restrict__ volume_conic,
	const float* __restrict__ depths,
	const float* __restrict__ audit_fields,
	int audit_field_count,
	float* __restrict__ primitive_responsibility,
	float* __restrict__ gate_responsibility,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = {
		block.group_index().x * BLOCK_X,
		block.group_index().y * BLOCK_Y
	};
	const uint2 pix = {
		pix_min.x + block.thread_index().x,
		pix_min.y + block.thread_index().y
	};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {static_cast<float>(pix.x), static_cast<float>(pix.y)};
	const bool inside = pix.x < W && pix.y < H;
	bool done = !inside;
	const uint2 range =
		ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds =
		(range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_shape[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	__shared__ float3 collected_normal[BLOCK_SIZE];

	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = {0};
	float N[3] = {0};
	float D = 0;
	float M1 = 0;
	float M2 = 0;
	float distortion = 0;
	float median_depth = 0;
	float median_contributor = -1;
	float surface_alpha = 0;
	float volume_alpha = 0;
	float surface_depth = 0;
	float volume_depth = 0;

	for (int round = 0; round < rounds; ++round, toDo -= BLOCK_SIZE)
	{
		if (__syncthreads_count(done) == BLOCK_SIZE)
			break;
		const int progress = round * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int id = point_list[range.x + progress];
			const int lane = block.thread_rank();
			collected_id[lane] = id;
			collected_xy[lane] = points_xy_image[id];
			if (id < surface_count)
			{
				collected_shape[lane] = {0, 0, 0, opacities[id]};
				collected_Tu[lane] = {
					surface_transMats[9 * id + 0],
					surface_transMats[9 * id + 1],
					surface_transMats[9 * id + 2]
				};
				collected_Tv[lane] = {
					surface_transMats[9 * id + 3],
					surface_transMats[9 * id + 4],
					surface_transMats[9 * id + 5]
				};
				collected_Tw[lane] = {
					surface_transMats[9 * id + 6],
					surface_transMats[9 * id + 7],
					surface_transMats[9 * id + 8]
				};
				collected_normal[lane] = surface_normals[id];
			}
			else
			{
				const int volume_id = id - surface_count;
				const float4 conic = volume_conic[volume_id];
				collected_shape[lane] = {
					conic.x,
					conic.y,
					conic.z,
					opacities[id]
				};
				collected_Tu[lane] = {0, 0, 0};
				collected_Tv[lane] = {0, 0, 0};
				collected_Tw[lane] = {0, 0, depths[id]};
				collected_normal[lane] = {0, 0, 0};
			}
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); ++j)
		{
			++contributor;
			const int id = collected_id[j];
			const float2 xy = collected_xy[j];
			const float4 shape = collected_shape[j];
			float power;
			float depth;
			float2 surface_uv = {0.0f, 0.0f};
			if (id < surface_count)
			{
				const float3 Tu = collected_Tu[j];
				const float3 Tv = collected_Tv[j];
				const float3 Tw = collected_Tw[j];
				const float3 k = pix.x * Tw - Tu;
				const float3 l = pix.y * Tw - Tv;
				const float3 p = cross(k, l);
				if (p.z == 0.0f)
					continue;
				surface_uv = {p.x / p.z, p.y / p.z};
				const float rho3d =
					surface_uv.x * surface_uv.x
					+ surface_uv.y * surface_uv.y;
				const float2 d = {xy.x - pixf.x, xy.y - pixf.y};
				const float rho2d =
					FilterInvSquare * (d.x * d.x + d.y * d.y);
				const float rho = min(rho3d, rho2d);
				depth = rho3d <= rho2d
					? surface_uv.x * Tw.x
						+ surface_uv.y * Tw.y + Tw.z
					: Tw.z;
				power = -0.5f * rho;
			}
			else
			{
				const float2 d = {xy.x - pixf.x, xy.y - pixf.y};
				power = -0.5f *
					(shape.x * d.x * d.x + shape.z * d.y * d.y)
					- shape.y * d.x * d.y;
				depth = depths[id];
			}
			if (depth < near_n || power > 0.0f)
				continue;
			int gate_index = -1;
			int gx0 = 0;
			int gy0 = 0;
			int gx1 = 0;
			int gy1 = 0;
			float gate_w00 = 1.0f;
			float gate_w10 = 0.0f;
			float gate_w01 = 0.0f;
			float gate_w11 = 0.0f;
			float gate_value = 1.0f;
			if (id < surface_count && gate_count > 0 && gate_size > 0)
			{
				gate_index = surface_gate_indices[id];
				if (gate_index >= 0 && gate_index < gate_count)
				{
					const float atlas_x = min(
						float(gate_size - 1),
						max(
							0.0f,
							(surface_uv.x / 3.0f + 1.0f)
								* 0.5f * float(gate_size - 1)));
					const float atlas_y = min(
						float(gate_size - 1),
						max(
							0.0f,
							(surface_uv.y / 3.0f + 1.0f)
								* 0.5f * float(gate_size - 1)));
					gx0 = int(floorf(atlas_x));
					gy0 = int(floorf(atlas_y));
					gx1 = min(gx0 + 1, gate_size - 1);
					gy1 = min(gy0 + 1, gate_size - 1);
					const float tx = atlas_x - float(gx0);
					const float ty = atlas_y - float(gy0);
					gate_w00 = (1.0f - tx) * (1.0f - ty);
					gate_w10 = tx * (1.0f - ty);
					gate_w01 = (1.0f - tx) * ty;
					gate_w11 = tx * ty;
					const int base = gate_index * gate_size * gate_size;
					gate_value =
						gate_w00 * surface_gate_atlas[
							base + gy0 * gate_size + gx0]
						+ gate_w10 * surface_gate_atlas[
							base + gy0 * gate_size + gx1]
						+ gate_w01 * surface_gate_atlas[
							base + gy1 * gate_size + gx0]
						+ gate_w11 * surface_gate_atlas[
							base + gy1 * gate_size + gx1];
					gate_value = min(1.0f, max(0.0f, gate_value));
				}
			}
			const float alpha = min(
				0.99f, shape.w * exp(power) * gate_value);
			if (alpha < 1.0f / 255.0f)
				continue;
			const float test_T = T * (1.0f - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
			const float w = alpha * T;
			if (id < surface_count)
			{
				surface_alpha += w;
				surface_depth += depth * w;
			}
			else
			{
				volume_alpha += w;
				volume_depth += depth * w;
			}
			if (audit_field_count > 0)
			{
				const int stride = audit_field_count + 1;
				atomicAdd(
					&primitive_responsibility[id * stride],
					w);
				for (int field = 0; field < audit_field_count; ++field)
				{
					atomicAdd(
						&primitive_responsibility[
							id * stride + field + 1],
						w * audit_fields[field * H * W + pix_id]);
				}
				if (gate_index >= 0)
				{
					const int gate_stride = audit_field_count + 1;
					const int texels = gate_size * gate_size;
					const int texel_indices[4] = {
						gy0 * gate_size + gx0,
						gy0 * gate_size + gx1,
						gy1 * gate_size + gx0,
						gy1 * gate_size + gx1
					};
					const float texel_weights[4] = {
						gate_w00, gate_w10, gate_w01, gate_w11
					};
					for (int corner = 0; corner < 4; ++corner)
					{
						const int gate_offset =
							(gate_index * texels + texel_indices[corner])
							* gate_stride;
						atomicAdd(
							&gate_responsibility[gate_offset],
							w * texel_weights[corner]);
						for (
							int field = 0;
							field < audit_field_count;
							++field)
						{
							atomicAdd(
								&gate_responsibility[
									gate_offset + field + 1],
								w * texel_weights[corner]
									* audit_fields[
										field * H * W + pix_id]);
						}
					}
				}
			}
			const float A = 1.0f - T;
			const float m =
				far_n / (far_n - near_n) * (1.0f - near_n / depth);
			distortion += (m * m * A + M2 - 2.0f * m * M1) * w;
			D += depth * w;
			M1 += m * w;
			M2 += m * m * w;
			if (T > 0.5f)
			{
				median_depth = depth;
				median_contributor = contributor;
			}
			const float3 normal = collected_normal[j];
			N[0] += normal.x * w;
			N[1] += normal.y * w;
			N[2] += normal.z * w;
			for (int ch = 0; ch < CHANNELS; ++ch)
				C[ch] += features[id * CHANNELS + ch] * w;
			T = test_T;
			last_contributor = contributor;
		}
	}

	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		n_contrib[pix_id + H * W] = median_contributor;
		final_T[pix_id + H * W] = M1;
		final_T[pix_id + 2 * H * W] = M2;
		for (int ch = 0; ch < CHANNELS; ++ch)
			out_color[ch * H * W + pix_id] =
				C[ch] + T * bg_color[ch];
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1.0f - T;
		for (int ch = 0; ch < 3; ++ch)
			out_others[pix_id + (NORMAL_OFFSET + ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
		out_others[pix_id + SURFACE_ALPHA_OFFSET * H * W] = surface_alpha;
		out_others[pix_id + VOLUME_ALPHA_OFFSET * H * W] = volume_alpha;
		out_others[pix_id + SURFACE_DEPTH_OFFSET * H * W] = surface_depth;
		out_others[pix_id + VOLUME_DEPTH_OFFSET * H * W] = volume_depth;
	}
}

void FORWARD::mixed_preprocess_surfaces(
	int surface_count,
	const float* means3D,
	const glm::vec2* scales,
	float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* viewmatrix,
	const float* projmatrix,
	int W,
	int H,
	int* radii,
	float2* means2D,
	float* depths,
	float* transMats,
	float3* normals,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	mixedPreprocessSurfacesCUDA<<<(surface_count + 255) / 256, 256>>>(
		surface_count,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		viewmatrix,
		projmatrix,
		W,
		H,
		radii,
		means2D,
		depths,
		transMats,
		normals,
		grid,
		tiles_touched,
		prefiltered);
}

void FORWARD::mixed_preprocess_volumes(
	int surface_count,
	int volume_count,
	const float* means3D,
	const glm::vec3* scales,
	float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* viewmatrix,
	const float* projmatrix,
	int W,
	int H,
	float focal_x,
	float focal_y,
	float tan_fovx,
	float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	mixedPreprocessVolumesCUDA<<<(volume_count + 255) / 256, 256>>>(
		surface_count,
		volume_count,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		viewmatrix,
		projmatrix,
		W,
		H,
		tan_fovx,
		tan_fovy,
		focal_x,
		focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered);
}

void FORWARD::mixed_render(
	const dim3 grid,
	dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int surface_count,
	int W,
	int H,
	const float2* means2D,
	const float* colors,
	const float* opacities,
	const int* surface_gate_indices,
	const float* surface_gate_atlas,
	int gate_count,
	int gate_size,
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
	float* out_others)
{
	mixedRenderCUDA<NUM_CHANNELS><<<grid, block>>>(
		ranges,
		point_list,
		surface_count,
		W,
		H,
		means2D,
		colors,
		opacities,
		surface_gate_indices,
		surface_gate_atlas,
		gate_count,
		gate_size,
		surface_transMats,
		surface_normals,
		volume_conic,
		depths,
		audit_fields,
		audit_field_count,
		primitive_responsibility,
		gate_responsibility,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_others);
}
