/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * Native mixed 2D-surface / 3D-EWA backward pass. The 3D covariance
 * derivatives follow the reference diff-gaussian-rasterization equations;
 * the compositing replay is shared by both representations.
 */

#include "mixed_backward.h"
#include "auxiliary.h"
#include "mixed_sort.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
mixedRenderBackwardCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const uint64_t* __restrict__ point_list_keys,
	int surface_count,
	int W,
	int H,
	const float* __restrict__ background,
	const float2* __restrict__ points_xy,
	const float* __restrict__ colors,
	const float* __restrict__ opacities,
	const int* __restrict__ surface_gate_indices,
	const float* __restrict__ surface_gate_atlas,
	int gate_count,
	int gate_size,
	const float* __restrict__ depth_query_bounds,
	bool has_depth_query,
	const float* __restrict__ surface_transMats,
	const float3* __restrict__ surface_normals,
	const float4* __restrict__ volume_conic,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dothers,
	const float* __restrict__ forward_others,
	float* __restrict__ dL_dsurface_transMat,
	float3* __restrict__ dL_dmean2D,
	float3* __restrict__ dL_dsurface_normal,
	float4* __restrict__ dL_dvolume_conic,
	float* __restrict__ dL_ddepth,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dsurface_gate_atlas,
	float* __restrict__ dL_dcolors)
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
	const float2 pixf = {
		static_cast<float>(pix.x),
		static_cast<float>(pix.y)
	};
	const bool inside = pix.x < W && pix.y < H;
	const uint2 range =
		ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	bool done = !inside;

	const float T_final = inside ? final_Ts[pix_id] : 0.0f;
	float T = T_final;
	const uint32_t last_contributor =
		inside ? n_contrib[pix_id] : UINT32_MAX;
	const uint32_t median_contributor =
		inside ? n_contrib[pix_id + H * W] : UINT32_MAX;
	float last_contributor_depth = -CUDART_INF_F;
	if (!done && last_contributor != UINT32_MAX)
	{
		const MixedPrimitiveData primitive = mixedLoadPrimitive(
			last_contributor,
			surface_count,
			points_xy,
			opacities,
			surface_transMats,
			surface_normals,
			volume_conic,
			depths);
		const MixedPixelCandidate candidate = mixedEvaluateCandidate(
			primitive,
			surface_count,
			pix,
			pixf,
			surface_gate_indices,
			surface_gate_atlas,
			gate_count,
			gate_size);
		if (!candidate.valid)
			done = true;
		else
			last_contributor_depth = candidate.depth;
	}
	else
	{
		done = true;
	}

	float dL_dpixel[C] = {0};
	float dL_dnormal2D[3] = {0};
	float dL_dreg = 0;
	float dL_drendered_depth = 0;
	float dL_daccum = 0;
	float dL_dmedian_depth = 0;
	float dL_dsurface_alpha = 0;
	float dL_dvolume_alpha = 0;
	float dL_dsurface_depth = 0;
	float dL_dvolume_depth = 0;
	float dL_dvolume_prehit_alpha = 0;
	float dL_dvolume_hit_interval_alpha = 0;
	float volume_prehit_T_final = 1.0f;
	float volume_hit_interval_T_final = 1.0f;
	float query_near = 0.0f;
	float query_far = 0.0f;
	bool query_valid = false;
	if (inside)
	{
		for (int ch = 0; ch < C; ++ch)
			dL_dpixel[ch] = dL_dpixels[ch * H * W + pix_id];
		dL_drendered_depth =
			dL_dothers[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum =
			dL_dothers[ALPHA_OFFSET * H * W + pix_id];
		dL_dreg =
			dL_dothers[DISTORTION_OFFSET * H * W + pix_id];
		dL_dmedian_depth =
			dL_dothers[MIDDEPTH_OFFSET * H * W + pix_id];
		dL_dsurface_alpha =
			dL_dothers[SURFACE_ALPHA_OFFSET * H * W + pix_id];
		dL_dvolume_alpha =
			dL_dothers[VOLUME_ALPHA_OFFSET * H * W + pix_id];
		dL_dsurface_depth =
			dL_dothers[SURFACE_DEPTH_OFFSET * H * W + pix_id];
		dL_dvolume_depth =
			dL_dothers[VOLUME_DEPTH_OFFSET * H * W + pix_id];
		dL_dvolume_prehit_alpha =
			dL_dothers[VOLUME_PREHIT_ALPHA_OFFSET * H * W + pix_id];
		dL_dvolume_hit_interval_alpha =
			dL_dothers[VOLUME_HIT_INTERVAL_ALPHA_OFFSET * H * W + pix_id];
		if (has_depth_query)
		{
			query_near = depth_query_bounds[pix_id];
			query_far = depth_query_bounds[H * W + pix_id];
			query_valid = isfinite(query_near) && isfinite(query_far)
				&& query_near > 0.0f && query_far >= query_near;
			if (query_valid)
			{
				volume_prehit_T_final = 1.0f - forward_others[
					VOLUME_PREHIT_ALPHA_OFFSET * H * W + pix_id];
				volume_hit_interval_T_final = 1.0f - forward_others[
					VOLUME_HIT_INTERVAL_ALPHA_OFFSET * H * W + pix_id];
			}
		}
		for (int ch = 0; ch < 3; ++ch)
			dL_dnormal2D[ch] =
				dL_dothers[(NORMAL_OFFSET + ch) * H * W + pix_id];
	}

	float accum_rec[C] = {0};
	float last_color[C] = {0};
	float last_alpha = 0;
	float last_depth = 0;
	float last_normal[3] = {0};
	float accum_depth_rec = 0;
	float accum_alpha_rec = 0;
	float accum_normal_rec[3] = {0};
	float accum_surface_alpha_rec = 0;
	float accum_volume_alpha_rec = 0;
	float accum_surface_depth_rec = 0;
	float accum_volume_depth_rec = 0;
	float last_surface_alpha = 0;
	float last_volume_alpha = 0;
	float last_surface_depth = 0;
	float last_volume_depth = 0;
	const float final_D =
		inside ? final_Ts[pix_id + H * W] : 0.0f;
	const float final_D2 =
		inside ? final_Ts[pix_id + 2 * H * W] : 0.0f;
	const float final_A = 1.0f - T_final;
	float last_dL_dT = 0;
	const float ddelx_dx = 0.5f * W;
	const float ddely_dy = 0.5f * H;

	// Replay the forward order in exact, bounded batches.  Capacity controls
	// scan granularity, not the number of supported overlapping surfaces.
	constexpr int MIXED_ACTIVE_CAPACITY = 256;
	MixedDepthHeapItem heap[MIXED_ACTIVE_CAPACITY];
	int heap_size = 0;
	uint32_t volume_cursor = range.y;
	bool surface_exhausted = false;
	bool replay_bound_inclusive = true;
	MixedDepthHeapItem replay_bound = {
		last_contributor_depth, last_contributor};
	while (!done)
	{
		if (heap_size == 0 && !surface_exhausted)
		{
			// Keep the k largest eligible surface tuples in a min heap and
			// convert it in-place to a max heap for reverse compositing.
			for (uint32_t offset = range.x; offset < range.y; ++offset)
			{
				const uint32_t id = point_list[offset];
				if (id >= static_cast<uint32_t>(surface_count))
					continue;
				const MixedPrimitiveData primitive = mixedLoadPrimitive(
					id, surface_count, points_xy, opacities,
					surface_transMats, surface_normals, volume_conic, depths);
				const MixedPixelCandidate candidate = mixedEvaluateCandidate(
					primitive, surface_count, pix, pixf,
					surface_gate_indices, surface_gate_atlas,
					gate_count, gate_size);
				if (!candidate.valid)
					continue;
				const MixedDepthHeapItem item = {candidate.depth, candidate.id};
				const bool before_bound = mixedHeapBefore(item, replay_bound);
				const bool equals_bound = item.depth == replay_bound.depth
					&& item.id == replay_bound.id;
				if (!before_bound
						&& !(replay_bound_inclusive && equals_bound))
					continue;
				if (heap_size < MIXED_ACTIVE_CAPACITY)
				{
					mixedHeapPush<MIXED_ACTIVE_CAPACITY, true>(
						heap, heap_size, item);
				}
				else if (mixedHeapBefore(heap[0], item))
				{
					(void)mixedHeapPop<MIXED_ACTIVE_CAPACITY, true>(
						heap, heap_size);
					mixedHeapPush<MIXED_ACTIVE_CAPACITY, true>(
						heap, heap_size, item);
				}
			}
			surface_exhausted = heap_size < MIXED_ACTIVE_CAPACITY;
			mixedMinHeapToMaxHeap<MIXED_ACTIVE_CAPACITY>(heap, heap_size);
		}

		MixedDepthHeapItem volume_item = {0.0f, 0};
		bool has_volume = false;
		while (volume_cursor > range.x)
		{
			const uint32_t offset = volume_cursor - 1;
			if (point_list[offset] < static_cast<uint32_t>(surface_count))
			{
				--volume_cursor;
				continue;
			}
			const MixedPrimitiveData primitive = mixedLoadPrimitive(
				point_list[offset], surface_count, points_xy, opacities,
				surface_transMats, surface_normals, volume_conic, depths);
			const MixedPixelCandidate candidate = mixedEvaluateCandidate(
				primitive, surface_count, pix, pixf,
				surface_gate_indices, surface_gate_atlas,
				gate_count, gate_size);
			if (!candidate.valid)
			{
				--volume_cursor;
				continue;
			}
			const MixedDepthHeapItem item = {candidate.depth, candidate.id};
			const bool before_bound = mixedHeapBefore(item, replay_bound);
			const bool equals_bound = item.depth == replay_bound.depth
				&& item.id == replay_bound.id;
			if (before_bound || (replay_bound_inclusive && equals_bound))
			{
				volume_item = item;
				has_volume = true;
				break;
			}
			--volume_cursor;
		}
		if (!has_volume && heap_size == 0)
			break;

		const bool choose_surface = heap_size > 0
			&& (!has_volume || mixedHeapBefore(volume_item, heap[0]));
		const MixedDepthHeapItem item = choose_surface
			? mixedHeapPop<MIXED_ACTIVE_CAPACITY, false>(heap, heap_size)
			: volume_item;
		if (!choose_surface)
			--volume_cursor;
		replay_bound = item;
		replay_bound_inclusive = false;
		const MixedPrimitiveData best_primitive = mixedLoadPrimitive(
			item.id, surface_count, points_xy, opacities,
			surface_transMats, surface_normals, volume_conic, depths);
		const MixedPixelCandidate best = mixedEvaluateCandidate(
			best_primitive, surface_count, pix, pixf,
			surface_gate_indices, surface_gate_atlas,
			gate_count, gate_size);
		if (!best.valid || best.id != item.id || best.depth != item.depth)
			asm("trap;");
			const int id = int(best.id);
			const bool is_surface = best.is_surface;
			const float4 shape = best_primitive.shape;
			const float2 d = best.d;
			const float3 Tu = best_primitive.Tu;
			const float3 Tv = best_primitive.Tv;
			const float3 Tw = best_primitive.Tw;
			const float3 k = best.k;
			const float3 l = best.l;
			const float3 p = best.p;
			const float2 s = best.surface_uv;
			const float rho3d = best.rho3d;
			const float rho2d = best.rho2d;
			const float depth = best.depth;
			const float G = best.G;
			const int gate_index = best.gate_index;
			const int gx0 = best.gx0;
			const int gy0 = best.gy0;
			const int gx1 = best.gx1;
			const int gy1 = best.gy1;
			const float gate_w00 = best.gate_w00;
			const float gate_w10 = best.gate_w10;
			const float gate_w01 = best.gate_w01;
			const float gate_w11 = best.gate_w11;
			const float gate_value = best.gate_value;
			const float alpha = best.alpha;

			T /= 1.0f - alpha;
			const float w = alpha * T;
			float dL_dalpha = 0;
			for (int ch = 0; ch < C; ++ch)
			{
				const float c = colors[id * C + ch];
				accum_rec[ch] =
					last_alpha * last_color[ch]
					+ (1.0f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;
				dL_dalpha +=
					(c - accum_rec[ch]) * dL_dpixel[ch];
				atomicAdd(
					&dL_dcolors[id * C + ch],
					w * dL_dpixel[ch]);
			}

			float dL_dz = 0;
			float dL_dweight = 0;
			const float m_d =
				far_n / (far_n - near_n) * (1.0f - near_n / depth);
			const float dmd_dd =
				(far_n * near_n)
				/ ((far_n - near_n) * depth * depth);
			if (best.id == median_contributor)
				dL_dz += dL_dmedian_depth;
#if !DETACH_WEIGHT
			dL_dweight +=
				(final_D2 + m_d * m_d * final_A - 2.0f * m_d * final_D)
				* dL_dreg;
#endif
			dL_dalpha += dL_dweight - last_dL_dT;
			last_dL_dT =
				dL_dweight * alpha + (1.0f - alpha) * last_dL_dT;
			const float dL_dmd =
				2.0f * w * (m_d * final_A - final_D) * dL_dreg;
			dL_dz += dL_dmd * dmd_dd;

			accum_depth_rec =
				last_alpha * last_depth
				+ (1.0f - last_alpha) * accum_depth_rec;
			last_depth = depth;
			dL_dalpha +=
				(depth - accum_depth_rec) * dL_drendered_depth;
			accum_alpha_rec =
				last_alpha + (1.0f - last_alpha) * accum_alpha_rec;
			dL_dalpha +=
				(1.0f - accum_alpha_rec) * dL_daccum;

			accum_surface_alpha_rec =
				last_alpha * last_surface_alpha
				+ (1.0f - last_alpha) * accum_surface_alpha_rec;
			accum_volume_alpha_rec =
				last_alpha * last_volume_alpha
				+ (1.0f - last_alpha) * accum_volume_alpha_rec;
			accum_surface_depth_rec =
				last_alpha * last_surface_depth
				+ (1.0f - last_alpha) * accum_surface_depth_rec;
			accum_volume_depth_rec =
				last_alpha * last_volume_depth
				+ (1.0f - last_alpha) * accum_volume_depth_rec;
			const float current_surface_alpha = is_surface ? 1.0f : 0.0f;
			const float current_volume_alpha = is_surface ? 0.0f : 1.0f;
			const float current_surface_depth = is_surface ? depth : 0.0f;
			const float current_volume_depth = is_surface ? 0.0f : depth;
			dL_dalpha +=
				(current_surface_alpha - accum_surface_alpha_rec)
				* dL_dsurface_alpha
				+ (current_volume_alpha - accum_volume_alpha_rec)
				* dL_dvolume_alpha
				+ (current_surface_depth - accum_surface_depth_rec)
				* dL_dsurface_depth
				+ (current_volume_depth - accum_volume_depth_rec)
				* dL_dvolume_depth;
			if (is_surface)
				dL_dz += w * dL_dsurface_depth;
			else
				dL_dz += w * dL_dvolume_depth;
			last_surface_alpha = current_surface_alpha;
			last_volume_alpha = current_volume_alpha;
			last_surface_depth = current_surface_depth;
			last_volume_depth = current_volume_depth;

			const float3 normal = best_primitive.normal;
			const float normal_values[3] = {
				normal.x,
				normal.y,
				normal.z
			};
			for (int ch = 0; ch < 3; ++ch)
			{
				accum_normal_rec[ch] =
					last_alpha * last_normal[ch]
					+ (1.0f - last_alpha) * accum_normal_rec[ch];
				last_normal[ch] = normal_values[ch];
				dL_dalpha +=
					(normal_values[ch] - accum_normal_rec[ch])
					* dL_dnormal2D[ch];
				if (is_surface)
				{
					atomicAdd(
						reinterpret_cast<float*>(
							dL_dsurface_normal + id) + ch,
						w * dL_dnormal2D[ch]);
				}
			}

			dL_dalpha *= T;
			// Conditional volume optical alpha in detached target-depth
			// intervals.  Membership is not differentiated, while opacity and
			// footprint receive the exact product-transmittance derivative.
			if (!is_surface && query_valid)
			{
				const float inv_one_minus_alpha =
					1.0f / max(1.0f - alpha, 1.0e-6f);
				if (depth < query_near)
				{
					dL_dalpha += volume_prehit_T_final
						* inv_one_minus_alpha
						* dL_dvolume_prehit_alpha;
				}
				else if (depth <= query_far)
				{
					dL_dalpha += volume_hit_interval_T_final
						* inv_one_minus_alpha
						* dL_dvolume_hit_interval_alpha;
				}
			}
			last_alpha = alpha;
			float bg_dot_dpixel = 0;
			for (int ch = 0; ch < C; ++ch)
				bg_dot_dpixel += background[ch] * dL_dpixel[ch];
			dL_dalpha +=
				(-T_final / (1.0f - alpha)) * bg_dot_dpixel;
			const float dL_dG = shape.w * gate_value * dL_dalpha;
			dL_dz += w * dL_drendered_depth;

			if (is_surface)
			{
				if (rho3d <= rho2d)
				{
					const float2 dL_ds = {
						-dL_dG * G * s.x + dL_dz * Tw.x,
						-dL_dG * G * s.y + dL_dz * Tw.y
					};
					const float3 dz_dTw = {s.x, s.y, 1.0f};
					const float dsx_pz = dL_ds.x / p.z;
					const float dsy_pz = dL_ds.y / p.z;
					const float3 dL_dp = {
						dsx_pz,
						dsy_pz,
						-(dsx_pz * s.x + dsy_pz * s.y)
					};
					const float3 dL_dk = cross(l, dL_dp);
					const float3 dL_dl = cross(dL_dp, k);
					const float3 dL_dTu = {
						-dL_dk.x,
						-dL_dk.y,
						-dL_dk.z
					};
					const float3 dL_dTv = {
						-dL_dl.x,
						-dL_dl.y,
						-dL_dl.z
					};
					const float3 dL_dTw = {
						pixf.x * dL_dk.x + pixf.y * dL_dl.x
							+ dL_dz * dz_dTw.x,
						pixf.x * dL_dk.y + pixf.y * dL_dl.y
							+ dL_dz * dz_dTw.y,
						pixf.x * dL_dk.z + pixf.y * dL_dl.z
							+ dL_dz * dz_dTw.z
					};
					const float grads[9] = {
						dL_dTu.x, dL_dTu.y, dL_dTu.z,
						dL_dTv.x, dL_dTv.y, dL_dTv.z,
						dL_dTw.x, dL_dTw.y, dL_dTw.z
					};
					for (int e = 0; e < 9; ++e)
						atomicAdd(
							&dL_dsurface_transMat[id * 9 + e],
							grads[e]);
				}
				else
				{
					atomicAdd(
						&dL_dmean2D[id].x,
						dL_dG * -G * FilterInvSquare * d.x);
					atomicAdd(
						&dL_dmean2D[id].y,
						dL_dG * -G * FilterInvSquare * d.y);
					atomicAdd(
						&dL_dsurface_transMat[id * 9 + 8],
						dL_dz);
				}
			}
			else
			{
				const int volume_id = id - surface_count;
				const float gdx = G * d.x;
				const float gdy = G * d.y;
				const float dG_ddelx =
					-gdx * shape.x - gdy * shape.y;
				const float dG_ddely =
					-gdy * shape.z - gdx * shape.y;
				atomicAdd(
					&dL_dmean2D[id].x,
					dL_dG * dG_ddelx * ddelx_dx);
				atomicAdd(
					&dL_dmean2D[id].y,
					dL_dG * dG_ddely * ddely_dy);
				atomicAdd(
					&dL_dvolume_conic[volume_id].x,
					-0.5f * gdx * d.x * dL_dG);
				atomicAdd(
					&dL_dvolume_conic[volume_id].y,
					-0.5f * gdx * d.y * dL_dG);
				atomicAdd(
					&dL_dvolume_conic[volume_id].w,
					-0.5f * gdy * d.y * dL_dG);
				atomicAdd(&dL_ddepth[id], dL_dz);
			}
			atomicAdd(
				&dL_dopacity[id], G * gate_value * dL_dalpha);
			if (gate_index >= 0)
			{
				const int base = gate_index * gate_size * gate_size;
				const float dL_dgate = shape.w * G * dL_dalpha;
				atomicAdd(
					&dL_dsurface_gate_atlas[
						base + gy0 * gate_size + gx0],
					gate_w00 * dL_dgate);
				atomicAdd(
					&dL_dsurface_gate_atlas[
						base + gy0 * gate_size + gx1],
					gate_w10 * dL_dgate);
				atomicAdd(
					&dL_dsurface_gate_atlas[
						base + gy1 * gate_size + gx0],
					gate_w01 * dL_dgate);
				atomicAdd(
					&dL_dsurface_gate_atlas[
						base + gy1 * gate_size + gx1],
					gate_w11 * dL_dgate);
			}
	}
}

void MIXED_BACKWARD::render(
	dim3 grid,
	dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const uint64_t* point_list_keys,
	int surface_count,
	int W,
	int H,
	const float* background,
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
	const float* final_T,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_dothers,
	const float* forward_others,
	float* dL_dsurface_transMat,
	float3* dL_dmean2D,
	float3* dL_dsurface_normal,
	float4* dL_dvolume_conic,
	float* dL_ddepth,
	float* dL_dopacity,
	float* dL_dsurface_gate_atlas,
	float* dL_dcolors)
{
	mixedRenderBackwardCUDA<NUM_CHANNELS><<<grid, block>>>(
		ranges,
		point_list,
		point_list_keys,
		surface_count,
		W,
		H,
		background,
		means2D,
		colors,
		opacities,
		surface_gate_indices,
		surface_gate_atlas,
		gate_count,
		gate_size,
		depth_query_bounds,
		has_depth_query,
		surface_transMats,
		surface_normals,
		volume_conic,
		depths,
		final_T,
		n_contrib,
		dL_dpixels,
		dL_dothers,
		forward_others,
		dL_dsurface_transMat,
		dL_dmean2D,
		dL_dsurface_normal,
		dL_dvolume_conic,
		dL_ddepth,
		dL_dopacity,
		dL_dsurface_gate_atlas,
		dL_dcolors);
}

__global__ void mixedComputeCov2DBackwardCUDA(
	int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	float focal_x,
	float focal_y,
	float tan_fovx,
	float tan_fovy,
	const float* view_matrix,
	const float4* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov)
{
	const int idx = cg::this_grid().thread_rank();
	if (idx >= P || radii[idx] <= 0)
		return;
	const float* cov3D = cov3Ds + 6 * idx;
	const float3 mean = means[idx];
	const float3 dL_dconic = {
		dL_dconics[idx].x,
		dL_dconics[idx].y,
		dL_dconics[idx].w
	};
	float3 t = transformPoint4x3(mean, view_matrix);
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	const float x_grad_mul =
		txtz < -limx || txtz > limx ? 0.0f : 1.0f;
	const float y_grad_mul =
		tytz < -limy || tytz > limy ? 0.0f : 1.0f;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0.0f, 0.0f, 0.0f);
	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);
	glm::mat3 T = W * J;
	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;
	const float a = cov2D[0][0] + 0.3f;
	const float b = cov2D[0][1];
	const float c = cov2D[1][1] + 0.3f;
	const float denom = a * c - b * b;
	const float denom2inv =
		1.0f / (denom * denom + 0.0000001f);
	float dL_da = 0;
	float dL_db = 0;
	float dL_dc = 0;
	if (denom2inv != 0.0f)
	{
		dL_da = denom2inv * (
			-c * c * dL_dconic.x
			+ 2.0f * b * c * dL_dconic.y
			+ (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (
			-a * a * dL_dconic.z
			+ 2.0f * a * b * dL_dconic.y
			+ (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2.0f * (
			b * c * dL_dconic.x
			- (denom + 2.0f * b * b) * dL_dconic.y
			+ a * b * dL_dconic.z);
		dL_dcov[6 * idx + 0] =
			T[0][0] * T[0][0] * dL_da
			+ T[0][0] * T[1][0] * dL_db
			+ T[1][0] * T[1][0] * dL_dc;
		dL_dcov[6 * idx + 3] =
			T[0][1] * T[0][1] * dL_da
			+ T[0][1] * T[1][1] * dL_db
			+ T[1][1] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 5] =
			T[0][2] * T[0][2] * dL_da
			+ T[0][2] * T[1][2] * dL_db
			+ T[1][2] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 1] =
			2.0f * T[0][0] * T[0][1] * dL_da
			+ (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db
			+ 2.0f * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] =
			2.0f * T[0][0] * T[0][2] * dL_da
			+ (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db
			+ 2.0f * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] =
			2.0f * T[0][2] * T[0][1] * dL_da
			+ (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db
			+ 2.0f * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int e = 0; e < 6; ++e)
			dL_dcov[6 * idx + e] = 0;
	}

	const float dL_dT00 =
		2.0f * (
			T[0][0] * Vrk[0][0]
			+ T[0][1] * Vrk[0][1]
			+ T[0][2] * Vrk[0][2]) * dL_da
		+ (
			T[1][0] * Vrk[0][0]
			+ T[1][1] * Vrk[0][1]
			+ T[1][2] * Vrk[0][2]) * dL_db;
	const float dL_dT01 =
		2.0f * (
			T[0][0] * Vrk[1][0]
			+ T[0][1] * Vrk[1][1]
			+ T[0][2] * Vrk[1][2]) * dL_da
		+ (
			T[1][0] * Vrk[1][0]
			+ T[1][1] * Vrk[1][1]
			+ T[1][2] * Vrk[1][2]) * dL_db;
	const float dL_dT02 =
		2.0f * (
			T[0][0] * Vrk[2][0]
			+ T[0][1] * Vrk[2][1]
			+ T[0][2] * Vrk[2][2]) * dL_da
		+ (
			T[1][0] * Vrk[2][0]
			+ T[1][1] * Vrk[2][1]
			+ T[1][2] * Vrk[2][2]) * dL_db;
	const float dL_dT10 =
		2.0f * (
			T[1][0] * Vrk[0][0]
			+ T[1][1] * Vrk[0][1]
			+ T[1][2] * Vrk[0][2]) * dL_dc
		+ (
			T[0][0] * Vrk[0][0]
			+ T[0][1] * Vrk[0][1]
			+ T[0][2] * Vrk[0][2]) * dL_db;
	const float dL_dT11 =
		2.0f * (
			T[1][0] * Vrk[1][0]
			+ T[1][1] * Vrk[1][1]
			+ T[1][2] * Vrk[1][2]) * dL_dc
		+ (
			T[0][0] * Vrk[1][0]
			+ T[0][1] * Vrk[1][1]
			+ T[0][2] * Vrk[1][2]) * dL_db;
	const float dL_dT12 =
		2.0f * (
			T[1][0] * Vrk[2][0]
			+ T[1][1] * Vrk[2][1]
			+ T[1][2] * Vrk[2][2]) * dL_dc
		+ (
			T[0][0] * Vrk[2][0]
			+ T[0][1] * Vrk[2][1]
			+ T[0][2] * Vrk[2][2]) * dL_db;
	const float dL_dJ00 =
		W[0][0] * dL_dT00
		+ W[0][1] * dL_dT01
		+ W[0][2] * dL_dT02;
	const float dL_dJ02 =
		W[2][0] * dL_dT00
		+ W[2][1] * dL_dT01
		+ W[2][2] * dL_dT02;
	const float dL_dJ11 =
		W[1][0] * dL_dT10
		+ W[1][1] * dL_dT11
		+ W[1][2] * dL_dT12;
	const float dL_dJ12 =
		W[2][0] * dL_dT10
		+ W[2][1] * dL_dT11
		+ W[2][2] * dL_dT12;
	const float tz = 1.0f / t.z;
	const float tz2 = tz * tz;
	const float tz3 = tz2 * tz;
	const float dL_dtx =
		x_grad_mul * -focal_x * tz2 * dL_dJ02;
	const float dL_dty =
		y_grad_mul * -focal_y * tz2 * dL_dJ12;
	const float dL_dtz =
		-focal_x * tz2 * dL_dJ00
		- focal_y * tz2 * dL_dJ11
		+ 2.0f * focal_x * t.x * tz3 * dL_dJ02
		+ 2.0f * focal_y * t.y * tz3 * dL_dJ12;
	dL_dmeans[idx] = transformVec4x3Transpose(
		{dL_dtx, dL_dty, dL_dtz},
		view_matrix);
}

__device__ void mixedComputeCov3DBackward(
	int idx,
	glm::vec3 scale,
	float mod,
	glm::vec4 rot,
	const float* dL_dcov3Ds,
	glm::vec3* dL_dscales,
	glm::vec4* dL_drots)
{
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
	const glm::vec3 s = mod * scale;
	glm::mat3 S(1.0f);
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;
	glm::mat3 M = S * R;
	const float* d = dL_dcov3Ds + 6 * idx;
	glm::mat3 dL_dSigma = glm::mat3(
		d[0], 0.5f * d[1], 0.5f * d[2],
		0.5f * d[1], d[3], 0.5f * d[4],
		0.5f * d[2], 0.5f * d[4], d[5]);
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;
	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);
	dL_dscales[idx] = mod * glm::vec3(
		glm::dot(Rt[0], dL_dMt[0]),
		glm::dot(Rt[1], dL_dMt[1]),
		glm::dot(Rt[2], dL_dMt[2]));
	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;
	glm::vec4 dq;
	dq.x =
		2 * z * (dL_dMt[0][1] - dL_dMt[1][0])
		+ 2 * y * (dL_dMt[2][0] - dL_dMt[0][2])
		+ 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dq.y =
		2 * y * (dL_dMt[1][0] + dL_dMt[0][1])
		+ 2 * z * (dL_dMt[2][0] + dL_dMt[0][2])
		+ 2 * r * (dL_dMt[1][2] - dL_dMt[2][1])
		- 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dq.z =
		2 * x * (dL_dMt[1][0] + dL_dMt[0][1])
		+ 2 * r * (dL_dMt[2][0] - dL_dMt[0][2])
		+ 2 * z * (dL_dMt[1][2] + dL_dMt[2][1])
		- 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dq.w =
		2 * r * (dL_dMt[0][1] - dL_dMt[1][0])
		+ 2 * x * (dL_dMt[2][0] + dL_dMt[0][2])
		+ 2 * y * (dL_dMt[1][2] + dL_dMt[2][1])
		- 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);
	dL_drots[idx] = dq;
}

__global__ void mixedVolumePreprocessBackwardCUDA(
	int P,
	const float3* means,
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	float scale_modifier,
	const float* view,
	const float* proj,
	const float3* dL_dmean2D,
	const float* dL_ddepth,
	glm::vec3* dL_dmeans,
	const float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	const int idx = cg::this_grid().thread_rank();
	if (idx >= P || radii[idx] <= 0)
		return;
	const float3 m = means[idx];
	const float4 m_hom = transformPoint4x4(m, proj);
	const float m_w = 1.0f / (m_hom.w + 0.0000001f);
	const float mul1 =
		(proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12])
		* m_w * m_w;
	const float mul2 =
		(proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13])
		* m_w * m_w;
	glm::vec3 dmean;
	dmean.x =
		(proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x
		+ (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dmean.y =
		(proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x
		+ (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dmean.z =
		(proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x
		+ (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;
	const float depth_grad = dL_ddepth[idx];
	dmean += glm::vec3(
		view[2] * depth_grad,
		view[6] * depth_grad,
		view[10] * depth_grad);
	dL_dmeans[idx] += dmean;
	mixedComputeCov3DBackward(
		idx,
		scales[idx],
		scale_modifier,
		rotations[idx],
		dL_dcov3D,
		dL_dscale,
		dL_drot);
}

void MIXED_BACKWARD::preprocessVolumes(
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
	glm::vec4* dL_drot)
{
	mixedComputeCov2DBackwardCUDA<<<(volume_count + 255) / 256, 256>>>(
		volume_count,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		reinterpret_cast<float3*>(dL_dmean3D),
		dL_dcov3D);
	mixedVolumePreprocessBackwardCUDA<<<
		(volume_count + 255) / 256,
		256>>>(
		volume_count,
		means3D,
		radii,
		scales,
		rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		dL_dmean2D,
		dL_ddepth,
		dL_dmean3D,
		dL_dcov3D,
		dL_dscale,
		dL_drot);
}
