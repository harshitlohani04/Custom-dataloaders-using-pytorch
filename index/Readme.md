# Index Pooling Implementation

This repository contains a PyTorch implementation of **Index Pooling** as described in the research paper *"MultiScale Probability Map guided Index Pooling with Attention-based learning for Road and Building Segmentation"*. Index Pooling is a downsampling technique designed to retain majority of the spatial information by preserving original feature map values at different indices within each pooling window. This method aims to provide more contextually aware representations for downstream tasks, especially in tasks requiring localization sensitivity.

## Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)

## Overview
Index Pooling is an alternative to conventional max pooling. Instead of discarding spatial details, Index Pooling retains the location of values in each pooling window. This is particularly useful in applications where retaining precise locations of high-activation features can improve model performance.

## How It Works
1. **Creating Feature Maps**: For a given `k x k` kernel size, `k²` unique kernels are created, each containing a single ‘one’ value at a distinct position within the kernel.
2. **Pooling**: These kernels are convolved over the feature map to produce `k²` downsampled feature maps, each preserving the value of the original map at different spatial position within the pooling window.
3. **Concatenation**: The `k²` feature maps are concatenated along the channel dimension.
4. **Reconstruction** (optional): The saved indices from Index Pooling allow precise upsampling, restoring spatial detail using `ConvTranspose2d` layers or custom upsampling.
