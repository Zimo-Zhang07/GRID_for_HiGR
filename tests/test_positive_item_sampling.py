from types import SimpleNamespace

import torch

from src.data.loading.components.pre_processing import map_sparse_id_to_embedding
from src.data.preprocessing.build_positive_item_map import (
    build_semantic_positive_item_map,
)


def test_semantic_positive_map_excludes_the_item_itself():
    embeddings = torch.eye(4)
    positive_map = build_semantic_positive_item_map(
        embeddings, num_neighbors=2, chunk_size=2
    )

    assert positive_map.shape == (4, 2)
    for item_id in range(4):
        assert item_id not in positive_map[item_id].tolist()


def test_embedding_preprocessor_adds_a_configured_positive_embedding():
    embeddings = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    dataset_config = SimpleNamespace(
        embedding_map={"id": embeddings},
        positive_item_map=torch.tensor(
            [[1], [2], [3], [4], [0]], dtype=torch.long
        ),
    )

    row = map_sparse_id_to_embedding(
        {"id": torch.tensor([2])},
        dataset_config=dataset_config,
        sparse_id_field="id",
    )

    assert torch.equal(row["embedding"], embeddings[2])
    assert torch.equal(row["positive_embedding"], embeddings[3])
