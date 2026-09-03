/*
 * Shared per-pixel ordering helpers for the native mixed 2D/3D renderer.
 *
 * Tile binning limits the candidate set.  Forward and backward evaluate the
 * same ray-local surface tuple and merge exact top-k batches with the sorted
 * volume stream by (exact depth, primitive id).  Batches bound working memory
 * without bounding total overlap or changing image semantics.
 */

#pragma once

#include "auxiliary.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

struct MixedPrimitiveData
{
	uint32_t id;
	float2 xy;
	float4 shape;
	float3 Tu;
	float3 Tv;
	float3 Tw;
	float3 normal;
};

struct MixedPixelCandidate
{
	bool valid;
	bool is_surface;
	uint32_t id;
	float2 d;
	float3 k;
	float3 l;
	float3 p;
	float2 surface_uv;
	float rho3d;
	float rho2d;
	float depth;
	float power;
	float G;
	float alpha;
	int gate_index;
	int gx0;
	int gy0;
	int gx1;
	int gy1;
	float gate_w00;
	float gate_w10;
	float gate_w01;
	float gate_w11;
	float gate_value;
};

struct MixedDepthHeapItem
{
	float depth;
	uint32_t id;
};

__device__ __forceinline__ bool mixedHeapBefore(
	const MixedDepthHeapItem& left,
	const MixedDepthHeapItem& right)
{
	return left.depth < right.depth
		|| (left.depth == right.depth && left.id < right.id);
}

template <int CAPACITY, bool MIN_HEAP>
__device__ __forceinline__ void mixedHeapPush(
	MixedDepthHeapItem (&heap)[CAPACITY],
	int& size,
	MixedDepthHeapItem value)
{
	if (size >= CAPACITY)
	{
		// Callers maintain bounded selection heaps explicitly. Reaching this is
		// an internal invariant violation, never a supported density limit.
		printf(
			"mixed pixel-depth surface heap overflow: capacity=%d id=%u depth=%g\n",
			CAPACITY, value.id, value.depth);
		asm("trap;");
	}
	int child = size++;
	while (child > 0)
	{
		const int parent = (child - 1) >> 1;
		const bool value_precedes = mixedHeapBefore(value, heap[parent]);
		if (value_precedes != MIN_HEAP)
			break;
		heap[child] = heap[parent];
		child = parent;
	}
	heap[child] = value;
}

template <int CAPACITY, bool MIN_HEAP>
__device__ __forceinline__ MixedDepthHeapItem mixedHeapPop(
	MixedDepthHeapItem (&heap)[CAPACITY],
	int& size)
{
	const MixedDepthHeapItem result = heap[0];
	const MixedDepthHeapItem tail = heap[--size];
	if (size == 0)
		return result;
	int parent = 0;
	while (true)
	{
		int child = parent * 2 + 1;
		if (child >= size)
			break;
		if (child + 1 < size)
		{
			const bool right_precedes = mixedHeapBefore(
				heap[child + 1], heap[child]);
			if (right_precedes == MIN_HEAP)
				++child;
		}
		const bool child_precedes_tail = mixedHeapBefore(heap[child], tail);
		if (child_precedes_tail != MIN_HEAP)
			break;
		heap[parent] = heap[child];
		parent = child;
	}
	heap[parent] = tail;
	return result;
}

// Reuse one bounded array for exact top-k selection and ordered replay.  A
// max-heap popped into its vacated tail becomes an ascending array (and hence
// a valid min-heap); the converse holds for a min-heap.  This lets dense
// pixels continue in exact batches without a semantic capacity limit.
template <int CAPACITY>
__device__ __forceinline__ void mixedMaxHeapToMinHeap(
	MixedDepthHeapItem (&heap)[CAPACITY],
	int size)
{
	while (size > 0)
	{
		const MixedDepthHeapItem value =
			mixedHeapPop<CAPACITY, false>(heap, size);
		heap[size] = value;
	}
}

template <int CAPACITY>
__device__ __forceinline__ void mixedMinHeapToMaxHeap(
	MixedDepthHeapItem (&heap)[CAPACITY],
	int size)
{
	while (size > 0)
	{
		const MixedDepthHeapItem value =
			mixedHeapPop<CAPACITY, true>(heap, size);
		heap[size] = value;
	}
}

__device__ __forceinline__ MixedPrimitiveData mixedLoadPrimitive(
	uint32_t id,
	int surface_count,
	const float2* points_xy,
	const float* opacities,
	const float* surface_transMats,
	const float3* surface_normals,
	const float4* volume_conic,
	const float* depths)
{
	MixedPrimitiveData value;
	value.id = id;
	value.xy = points_xy[id];
	if (id < static_cast<uint32_t>(surface_count))
	{
		value.shape = {0, 0, 0, opacities[id]};
		value.Tu = {
			surface_transMats[9 * id + 0],
			surface_transMats[9 * id + 1],
			surface_transMats[9 * id + 2]};
		value.Tv = {
			surface_transMats[9 * id + 3],
			surface_transMats[9 * id + 4],
			surface_transMats[9 * id + 5]};
		value.Tw = {
			surface_transMats[9 * id + 6],
			surface_transMats[9 * id + 7],
			surface_transMats[9 * id + 8]};
		value.normal = surface_normals[id];
	}
	else
	{
		const int volume_id = int(id) - surface_count;
		const float4 conic = volume_conic[volume_id];
		value.shape = {conic.x, conic.y, conic.z, opacities[id]};
		value.Tu = {0, 0, 0};
		value.Tv = {0, 0, 0};
		value.Tw = {0, 0, depths[id]};
		value.normal = {0, 0, 0};
	}
	return value;
}

__device__ __forceinline__ MixedPixelCandidate mixedEvaluateCandidate(
	const MixedPrimitiveData& primitive,
	int surface_count,
	const uint2 pix,
	const float2 pixf,
	const int* surface_gate_indices,
	const float* surface_gate_atlas,
	int gate_count,
	int gate_size)
{
	MixedPixelCandidate value;
	value.valid = false;
	value.id = primitive.id;
	value.is_surface = primitive.id < static_cast<uint32_t>(surface_count);
	value.d = {primitive.xy.x - pixf.x, primitive.xy.y - pixf.y};
	value.k = {0, 0, 0};
	value.l = {0, 0, 0};
	value.p = {0, 0, 1};
	value.surface_uv = {0, 0};
	value.rho3d = 0;
	value.rho2d = 0;
	value.depth = 0;
	value.power = 0;
	value.G = 0;
	value.alpha = 0;
	value.gate_index = -1;
	value.gx0 = value.gy0 = value.gx1 = value.gy1 = 0;
	value.gate_w00 = 1.0f;
	value.gate_w10 = value.gate_w01 = value.gate_w11 = 0.0f;
	value.gate_value = 1.0f;

	if (value.is_surface)
	{
		value.k = pix.x * primitive.Tw - primitive.Tu;
		value.l = pix.y * primitive.Tw - primitive.Tv;
		value.p = cross(value.k, value.l);
		if (value.p.z == 0.0f)
			return value;
		value.surface_uv = {
			value.p.x / value.p.z,
			value.p.y / value.p.z};
		value.rho3d =
			value.surface_uv.x * value.surface_uv.x
			+ value.surface_uv.y * value.surface_uv.y;
		value.rho2d = FilterInvSquare
			* (value.d.x * value.d.x + value.d.y * value.d.y);
		const float rho = min(value.rho3d, value.rho2d);
		value.depth = value.rho3d <= value.rho2d
			? value.surface_uv.x * primitive.Tw.x
				+ value.surface_uv.y * primitive.Tw.y + primitive.Tw.z
			: primitive.Tw.z;
		value.power = -0.5f * rho;
	}
	else
	{
		value.power = -0.5f * (
			primitive.shape.x * value.d.x * value.d.x
			+ primitive.shape.z * value.d.y * value.d.y)
			- primitive.shape.y * value.d.x * value.d.y;
		value.depth = primitive.Tw.z;
	}
	if (!isfinite(value.depth) || value.depth < near_n || value.power > 0.0f)
		return value;

	if (value.is_surface && gate_count > 0 && gate_size > 0)
	{
		value.gate_index = surface_gate_indices[primitive.id];
		if (value.gate_index >= 0 && value.gate_index < gate_count)
		{
			const float atlas_x = min(
				float(gate_size - 1),
				max(0.0f, (value.surface_uv.x / 3.0f + 1.0f)
					* 0.5f * float(gate_size - 1)));
			const float atlas_y = min(
				float(gate_size - 1),
				max(0.0f, (value.surface_uv.y / 3.0f + 1.0f)
					* 0.5f * float(gate_size - 1)));
			value.gx0 = int(floorf(atlas_x));
			value.gy0 = int(floorf(atlas_y));
			value.gx1 = min(value.gx0 + 1, gate_size - 1);
			value.gy1 = min(value.gy0 + 1, gate_size - 1);
			const float tx = atlas_x - float(value.gx0);
			const float ty = atlas_y - float(value.gy0);
			value.gate_w00 = (1.0f - tx) * (1.0f - ty);
			value.gate_w10 = tx * (1.0f - ty);
			value.gate_w01 = (1.0f - tx) * ty;
			value.gate_w11 = tx * ty;
			const int base = value.gate_index * gate_size * gate_size;
			value.gate_value =
				value.gate_w00 * surface_gate_atlas[
					base + value.gy0 * gate_size + value.gx0]
				+ value.gate_w10 * surface_gate_atlas[
					base + value.gy0 * gate_size + value.gx1]
				+ value.gate_w01 * surface_gate_atlas[
					base + value.gy1 * gate_size + value.gx0]
				+ value.gate_w11 * surface_gate_atlas[
					base + value.gy1 * gate_size + value.gx1];
			value.gate_value = min(1.0f, max(0.0f, value.gate_value));
		}
	}
	value.G = exp(value.power);
	value.alpha = min(
		0.99f,
		primitive.shape.w * value.G * value.gate_value);
	value.valid = value.alpha >= MIN_RENDER_ALPHA;
	return value;
}

__device__ __forceinline__ bool mixedOrderAfter(
	const MixedPixelCandidate& candidate,
	float depth,
	uint32_t id)
{
	return candidate.depth > depth
		|| (candidate.depth == depth && candidate.id > id);
}

__device__ __forceinline__ bool mixedOrderBefore(
	const MixedPixelCandidate& candidate,
	float depth,
	uint32_t id)
{
	return candidate.depth < depth
		|| (candidate.depth == depth && candidate.id < id);
}
