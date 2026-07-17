import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

class FullBatchCrossEntropyLoss(nn.Module):
    """
    Contrastive loss with negative samples being all candidates in the embedding table.
    """

    def __init__(
        self,
        normalize: bool = True,
        **kwargs,
    ):
        """
        Initialize the FullBatchContrastiveLoss.

        Parameters
        ----------
        contrastive_tau: float
            Temperature parameter for the contrastive loss.
        normalize: bool
            Whether to normalize the embeddings before computing the logits via dot product.
        """
        super().__init__()
        self.normalize = normalize
        self.cross_entroy_loss = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        label_locations: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the contrastive loss with negative samples from the full vocabulary.

        Parameters
        ----------
        query_embeddings: torch.Tensor (batch_size x sequence length x embedding_dim)
            The embeddings of the query items.
        key_embeddings: torch.Tensor (total number of items x embedding_dim)
            The embeddings of all items, i.e the full embedding table.
        label_locations: torch.Tensor (number of labels x 2)
            The locations of the labels in the input sequences.
        labels: torch.Tensor (number of labels)
            The labels for the input sequences.

        Returns
        -------
        torch.Tensor
            The contrastive loss.
        """
        # get representation of masked tokens
        # label_locations[:, 0] refers to the index of sequences
        # label_locations[:, 1] refers to the index of tokens in the sequences
        query_embeddings = query_embeddings[
            label_locations[:, 0], label_locations[:, 1]
        ]

        if self.normalize:
            query_embeddings = F.normalize(query_embeddings, dim=-1)
            key_embeddings = F.normalize(key_embeddings, dim=-1)

        logits = torch.mm(query_embeddings, key_embeddings.t())

        loss = self.cross_entroy_loss(logits, labels.long())

        return loss
    
class WeightedSquaredError(torch.nn.Module):
    def __init__(self):
        """Initialize the WeightedSquaredError loss function."""
        super().__init__()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the weighted squared error loss.

        Args:
            x: Predicted values of shape (n_points, n_features)
            y: Target values of shape (n_points, n_features)
            weights: Weights for each point of shape (n_points,)

        Returns:
            A tensor containing the weighted squared error loss of shape (1,)
        """
        error = x - y
        squared_error = torch.sum(error**2, dim=-1)
        # If weights are not provided, use uniform weights
        # This is equivalent to the standard squared error loss
        if weights is None:
            return torch.sum(squared_error)
        return torch.sum(weights * squared_error)
    
class BetaQuantizationLoss(torch.nn.Module):
    def __init__(self, beta: float = 0.25, reduction: str = "sum"):
        """Initialize the Beta Quantization Loss.

        Parameters
        ----------
        beta: float
            Weighting factor for the reconstruction loss.
        reduction: str
            Reduction method to apply to the loss. Options are 'none', 'mean', and 'sum'.
        """
        super().__init__()
        self.beta = beta
        self.criterion = torch.nn.MSELoss(reduction=reduction)

    def forward(self, x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
        """
        Compute the beta quantization loss.
        Args:
            x: Original tensor of shape (batch_size, n_features)
            x: Quantized tensor of shape (batch_size, n_features)
        Returns:
            A tensor containing the beta quantization loss of shape (1,)
        """
        x_no_grad = x.detach()
        xq_no_grad = xq.detach()
        loss = self.criterion(x_no_grad, xq) + self.beta * self.criterion(x, xq_no_grad)
        return loss

class GlobalQuantizationLoss(torch.nn.Module):
    def __init__(
        self,
        commitment_weight: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.commitment_weight = commitment_weight
        self.criterion = torch.nn.MSELoss(reduction=reduction)

    def forward(
        self,
        encoded_embeddings: torch.Tensor,
        quantized_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        codebook_loss = self.criterion(
            quantized_embeddings,
            encoded_embeddings.detach(),
        )
        commitment_loss = self.criterion(
            encoded_embeddings,
            quantized_embeddings.detach(),
        )

        return (
            codebook_loss
            + self.commitment_weight * commitment_loss
        )

class PrefixContrastiveLoss(torch.nn.Module):
    """HiGR prefix-level InfoNCE loss for residual-quantization codewords.

    For every layer except the final (leaf) layer, the codeword selected by an
    anchor is contrasted with the codeword selected by its paired positive.
    Other positives in the batch act as negatives. Excluding the final layer
    lets it retain fine-grained item identity instead of forcing semantically
    related items into the same complete semantic ID.

    Expected input shape is ``(batch_size, num_layers, embedding_dim)``. Pair
    ``anchor_codewords[i]`` with ``positive_codewords[i]`` before calling this
    loss.
    """

    def __init__(
        self,
        temperature: float,
        layer_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0.")
        if layer_weights is not None:
            if any(weight < 0 for weight in layer_weights):
                raise ValueError("layer_weights must be non-negative.")
            if not any(weight > 0 for weight in layer_weights):
                raise ValueError("At least one layer weight must be greater than 0.")

        self.temperature = float(temperature)
        self.layer_weights = (
            tuple(float(weight) for weight in layer_weights)
            if layer_weights is not None
            else None
        )

    def forward(
        self,
        anchor_codewords: torch.Tensor,
        positive_codewords: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted InfoNCE over the first ``D - 1`` codebooks.

        Args:
            anchor_codewords: Selected anchor codewords with shape ``(B, D, H)``.
            positive_codewords: Selected positive codewords with the same shape.

        Returns:
            A scalar equal to the weighted sum of per-layer InfoNCE losses,
            divided by ``D - 1`` as specified by HiGR.
        """
        if anchor_codewords.ndim != 3 or positive_codewords.ndim != 3:
            raise ValueError(
                "anchor_codewords and positive_codewords must have shape "
                "(batch_size, num_layers, embedding_dim)."
            )
        if anchor_codewords.shape != positive_codewords.shape:
            raise ValueError(
                "anchor_codewords and positive_codewords must have identical shapes; "
                f"got {tuple(anchor_codewords.shape)} and "
                f"{tuple(positive_codewords.shape)}."
            )

        batch_size, num_layers, _ = anchor_codewords.shape
        if batch_size < 2:
            raise ValueError(
                "PrefixContrastiveLoss requires at least two pairs so that each "
                "anchor has an in-batch negative."
            )
        if num_layers < 2:
            raise ValueError(
                "PrefixContrastiveLoss requires at least two quantization layers."
            )

        num_prefix_layers = num_layers - 1
        if (
            self.layer_weights is not None
            and len(self.layer_weights) != num_prefix_layers
        ):
            raise ValueError(
                "layer_weights must contain one weight for each prefix layer "
                f"(D - 1 = {num_prefix_layers}); got {len(self.layer_weights)}."
            )

        labels = torch.arange(batch_size, device=anchor_codewords.device)
        total_loss = anchor_codewords.new_zeros(())

        for layer_idx in range(num_prefix_layers):
            anchors = F.normalize(anchor_codewords[:, layer_idx, :], dim=-1)
            positives = F.normalize(positive_codewords[:, layer_idx, :], dim=-1)

            # logits[i, i] is the positive pair; all logits[i, j] where j != i
            # are in-batch negatives for anchor i.
            logits = anchors @ positives.transpose(0, 1)
            logits = logits / self.temperature

            layer_loss = F.cross_entropy(logits, labels)
            layer_weight = (
                self.layer_weights[layer_idx]
                if self.layer_weights is not None
                else 1.0
            )
            total_loss = total_loss + layer_weight * layer_loss

        return total_loss / num_prefix_layers
