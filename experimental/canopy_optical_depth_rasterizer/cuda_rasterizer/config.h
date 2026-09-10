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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16
// Seven legacy maps, four mixed surface/volume maps, and two conditional
// volume depth-query optical-alpha maps.
#define MIXED_AUX_CHANNELS 13
// Keep forward and backward in the same low-alpha precision contract.
// The former 8-bit cutoff (1/255) made every primitive below that peak alpha
// exactly invisible and therefore unable to receive an RGB opacity gradient.
// Static foliage logits are allowed down to sigmoid(-10) ~= 4.54e-5 and
// structural surfels down to sigmoid(-12) ~= 6.14e-6.  A 1e-6 cutoff is below
// both parameter floors, retaining a finite center-pixel recovery path while
// still dropping numerically irrelevant Gaussian tails.
#define MIN_RENDER_ALPHA 1.0e-6f

#endif
