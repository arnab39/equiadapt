import torch

from equiadapt.common.utils import gram_schmidt


def test_gram_schmidt() -> None:
    torch.manual_seed(0)
    vectors = torch.randn(1, 3, 3)  # batch of 1, 3 vectors of dimension 3

    output = gram_schmidt(vectors)

    assert torch.allclose(output[0][0][0], torch.tensor(0.5740), atol=1e-4)
