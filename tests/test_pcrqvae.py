import torch
from torch import nn

from src.components.loss_functions import (
    GlobalQuantizationLoss,
    PrefixContrastiveLoss,
)
from src.modules.clustering.prefix_contrastive_residual_quantization import (
    PrefixContrastiveResidualQuantization,
)


class DummyQuantizationLayer(nn.Module):
    def __init__(self, centroids: torch.Tensor):
        super().__init__()
        self.centroids = nn.Parameter(centroids)
        self.n_clusters = centroids.size(0)
        self.is_initialized = True
        self.is_initial_step = False
        self.init_buffer_size = 0

    def predict_step(self, batch):
        distances = torch.cdist(batch, self.centroids)
        ids = distances.argmin(dim=1)
        return ids, self.centroids.detach()[ids]

    def get_centroids(self):
        return self.centroids

    def on_train_start(self):
        pass


def test_quantizer_returns_real_codewords_and_their_global_sum():
    layers = nn.ModuleList(
        [
            DummyQuantizationLayer(torch.eye(4)),
            DummyQuantizationLayer(torch.eye(4) * 0.5),
            DummyQuantizationLayer(torch.eye(4) * 0.25),
        ]
    )
    model = PrefixContrastiveResidualQuantization(
        quantization_layer_list=layers,
        normalization_layer=nn.Identity(),
        encoder=nn.Identity(),
        decoder=nn.Identity(),
        reconstruction_loss_function=nn.MSELoss(),
        reconstruction_loss_weight=1.0,
        global_quantization_loss_function=GlobalQuantizationLoss(0.1),
        prefix_contrastive_loss_function=PrefixContrastiveLoss(
            temperature=0.1, layer_weights=[1.0, 0.1]
        ),
        optimizer=None,
        scheduler=None,
        training_loop_function=None,
        track_residuals=True,
        normalize_residuals=False,
    )

    output = model.quantize_with_details(torch.eye(4))

    assert output.cluster_ids.shape == (4, 3)
    assert output.selected_codewords.shape == (4, 3, 4)
    assert torch.allclose(
        output.codebook_quantized_embeddings,
        output.selected_codewords.sum(dim=1),
    )
