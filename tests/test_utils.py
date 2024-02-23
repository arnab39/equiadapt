import torch

from equiadapt.common.utils import gram_schmidt


def test_gram_schmidt() -> None:
    vectors = torch.randn(1, 3, 3)  # batch of 1, 3 vectors of dimension 3
    orthogonal_vectors = gram_schmidt(vectors)

    # Check if the vectors are orthogonal
    for i in range(orthogonal_vectors.shape[1]):
        for j in range(i + 1, orthogonal_vectors.shape[1]):
            dot_product = torch.dot(orthogonal_vectors[0, i], orthogonal_vectors[0, j])
            assert abs(dot_product.item()) < 1e-6

    # Check if the vectors are normalized
    for i in range(orthogonal_vectors.shape[1]):
        norm = torch.norm(orthogonal_vectors[0, i]).item()
        assert abs(norm - 1) < 1e-6
