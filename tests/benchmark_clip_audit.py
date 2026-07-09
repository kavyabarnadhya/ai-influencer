import time
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
import sys
import os

# Add scripts to path
sys.path.insert(0, os.path.abspath("scripts"))
import clip_similarity_audit

def benchmark_similarity_logic():
    n = 100
    threshold = 0.92
    sim_matrix = np.random.rand(n, n).astype(np.float32)
    # Make it symmetric and 1.0 on diagonal
    sim_matrix = (sim_matrix + sim_matrix.T) / 2
    np.fill_diagonal(sim_matrix, 1.0)

    valid_paths = [Path(f"img_{i}.png") for i in range(n)]

    # Old logic simulation (O(N^2) loops)
    start_old = time.time()
    flagged_old = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim > threshold:
                flagged_old.append((sim, valid_paths[i], valid_paths[j]))
    flagged_old.sort(reverse=True)

    total_sim = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_sim += float(sim_matrix[i, j])
            count += 1
    avg_sim_old = total_sim / count if count > 0 else 0.0
    end_old = time.time()

    # New logic (Vectorized)
    start_new = time.time()
    triu_indices = np.triu_indices(n, k=1)
    upper_tri_sims = sim_matrix[triu_indices]
    avg_sim_new = float(np.mean(upper_tri_sims)) if n > 1 else 0.0

    flagged_mask = upper_tri_sims > threshold
    flagged_indices_i = triu_indices[0][flagged_mask]
    flagged_indices_j = triu_indices[1][flagged_mask]
    flagged_sims = upper_tri_sims[flagged_mask]

    flagged_new = [
        (float(sim), valid_paths[i], valid_paths[j])
        for sim, i, j in zip(flagged_sims, flagged_indices_i, flagged_indices_j)
    ]
    flagged_new.sort(key=lambda x: x[0], reverse=True)
    end_new = time.time()

    print(f"Similarity Analysis (N={n}):")
    print(f"  Old (Loops): {end_old - start_old:.6f}s")
    print(f"  New (Vector): {end_new - start_new:.6f}s")
    print(f"  Speedup: {(end_old - start_old) / (end_new - start_new):.2f}x")

    assert abs(avg_sim_old - avg_sim_new) < 1e-6
    assert len(flagged_old) == len(flagged_new)

def benchmark_encoding_batching():
    # Mock model and preprocess
    model = MagicMock()
    # model.encode_image returns a tensor of [batch_size, 512]
    model.encode_image.side_effect = lambda x: torch.randn(x.shape[0], 512)

    preprocess = MagicMock()
    preprocess.side_effect = lambda x: torch.randn(3, 224, 224)

    # Mock PIL Image
    import PIL.Image
    PIL.Image.open = MagicMock()

    image_paths = [Path(f"img_{i}.png") for i in range(32)]
    device = "cpu"

    # Batch size 1 (simulating old sequential behavior)
    start_seq = time.time()
    _ = clip_similarity_audit.encode_images(image_paths, model, preprocess, device, torch, batch_size=1)
    end_seq = time.time()

    # Batch size 16
    start_batch = time.time()
    _ = clip_similarity_audit.encode_images(image_paths, model, preprocess, device, torch, batch_size=16)
    end_batch = time.time()

    print(f"\nEncoding (N={len(image_paths)}):")
    print(f"  Sequential (BS=1): {end_seq - start_seq:.6f}s")
    print(f"  Batched (BS=16):    {end_batch - start_batch:.6f}s")
    print(f"  Speedup: {(end_seq - start_seq) / (end_batch - start_batch):.2f}x")

if __name__ == "__main__":
    benchmark_similarity_logic()
    benchmark_encoding_batching()
