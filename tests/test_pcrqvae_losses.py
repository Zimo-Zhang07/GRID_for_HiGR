import torch
import torch.nn.functional as F

from src.components.loss_functions import (
    GlobalQuantizationLoss,
    PrefixContrastiveLoss,
)


def test_global_quantization_loss_has_the_two_stop_gradient_paths():
    encoded = torch.tensor([[1.0, 2.0]], requires_grad=True)
    quantized = torch.tensor([[3.0, 4.0]], requires_grad=True)
    loss = GlobalQuantizationLoss(commitment_weight=0.1)(encoded, quantized)

    expected = F.mse_loss(quantized, encoded.detach()) + 0.1 * F.mse_loss(
        encoded, quantized.detach()
    )
    assert torch.allclose(loss, expected)

    loss.backward()
    assert torch.allclose(quantized.grad, torch.tensor([[2.0, 2.0]]))
    assert torch.allclose(encoded.grad, torch.tensor([[-0.2, -0.2]]))


def test_prefix_contrastive_loss_excludes_the_leaf_layer():
    anchor = torch.randn(4, 3, 8, requires_grad=True)
    positive = anchor.detach().clone().requires_grad_(True)
    loss_function = PrefixContrastiveLoss(
        temperature=0.1, layer_weights=[1.0, 0.1]
    )

    loss = loss_function(anchor, positive)
    changed_leaf = positive.detach().clone()
    changed_leaf[:, -1] = torch.randn_like(changed_leaf[:, -1])
    changed_loss = loss_function(anchor, changed_leaf)

    assert torch.allclose(loss, changed_loss)
    loss.backward()
    assert torch.count_nonzero(anchor.grad[:, -1]) == 0
    assert torch.count_nonzero(positive.grad[:, -1]) == 0


def test_prefix_contrastive_loss_prefers_the_paired_positive():
    codewords = torch.eye(4).unsqueeze(1).repeat(1, 3, 1)
    loss_function = PrefixContrastiveLoss(
        temperature=0.1, layer_weights=[1.0, 0.1]
    )

    paired_loss = loss_function(codewords, codewords)
    unpaired_loss = loss_function(codewords, codewords.roll(1, dims=0))

    assert paired_loss < unpaired_loss
