"""Build semantic-neighbor positive pairs for HiGR PCRQ-VAE training."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def build_semantic_positive_item_map(
    embeddings: torch.Tensor,
    num_neighbors: int,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Return the cosine-nearest non-self item IDs for every item.

    HiGR permits positives from semantic neighbors or high-co-occurrence signals.
    This function implements the semantic-neighbor path using the bge-m3 item
    embeddings already produced by GRID.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (num_items, embedding_dim).")
    if embeddings.size(0) < 2:
        raise ValueError("At least two item embeddings are required.")
    if not 1 <= num_neighbors < embeddings.size(0):
        raise ValueError("num_neighbors must be in [1, num_items - 1].")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    normalized = F.normalize(embeddings.float(), dim=-1)
    positive_item_map = torch.empty(
        (normalized.size(0), num_neighbors), dtype=torch.long
    )

    for start in range(0, normalized.size(0), chunk_size):
        end = min(start + chunk_size, normalized.size(0))
        similarities = normalized[start:end] @ normalized.transpose(0, 1)
        row_ids = torch.arange(end - start)
        item_ids = torch.arange(start, end)
        similarities[row_ids, item_ids] = -torch.inf
        positive_item_map[start:end] = similarities.topk(
            k=num_neighbors, dim=1
        ).indices.cpu()

    return positive_item_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--num-neighbors", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    args = parser.parse_args()

    embeddings = torch.load(args.embedding_path, map_location="cpu")
    positive_item_map = build_semantic_positive_item_map(
        embeddings=embeddings,
        num_neighbors=args.num_neighbors,
        chunk_size=args.chunk_size,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(positive_item_map, args.output_path)


if __name__ == "__main__":
    main()
