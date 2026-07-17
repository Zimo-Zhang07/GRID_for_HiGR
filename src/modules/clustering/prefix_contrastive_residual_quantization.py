from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn
from torchmetrics import MeanMetric

from src.data.loading.components.interfaces import ItemData
from src.modules.clustering.residual_quantization import ResidualQuantization


@dataclass
class PCRQForwardOutput:
    """Intermediate values required by the PCRQ-VAE training objectives."""

    cluster_ids: torch.Tensor
    all_residuals: Optional[torch.Tensor]
    encoded_embeddings: torch.Tensor
    ste_quantized_embeddings: torch.Tensor
    codebook_quantized_embeddings: torch.Tensor
    selected_codewords: torch.Tensor
    initialization_loss: torch.Tensor


@dataclass
class PCRQLossOutput:
    """Losses and assignments produced for an anchor-positive batch."""

    cluster_ids: torch.Tensor
    all_residuals: Optional[torch.Tensor]
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    global_quantization_loss: torch.Tensor
    prefix_contrastive_loss: torch.Tensor


class PrefixContrastiveResidualQuantization(ResidualQuantization):
    """HiGR Prefix-Contrastive Residual Quantized VAE (PCRQ-VAE).

    The module retains the encoder, decoder, residual codebooks, initialization,
    checkpoint, and prediction behavior of :class:`ResidualQuantization`, but
    replaces its layer-wise quantization objective with two HiGR objectives:

    * global quantization aligns the sum of all selected codewords with the
      encoder latent;
    * prefix contrastive alignment applies InfoNCE to the selected codewords in
      the first ``D - 1`` layers.

    Reconstruction uses straight-through (STE) quantized embeddings so gradients
    reach the encoder. Global quantization and prefix contrastive alignment use
    codewords gathered directly from the codebooks so gradients reach the
    codebook parameters.
    """

    def __init__(
        self,
        global_quantization_loss_function: Optional[nn.Module] = None,
        prefix_contrastive_loss_function: Optional[nn.Module] = None,
        global_quantization_loss_weight: float = 0.1,
        prefix_contrastive_loss_weight: float = 0.01,
        **kwargs,
    ) -> None:
        if global_quantization_loss_weight < 0:
            raise ValueError("global_quantization_loss_weight must be non-negative.")
        if prefix_contrastive_loss_weight < 0:
            raise ValueError("prefix_contrastive_loss_weight must be non-negative.")

        # The parent layer-wise quantization loss is needed while codebooks are
        # initialized, but is deliberately excluded from the final PCRQ-VAE loss.
        kwargs["quantization_loss_weight"] = 0.0
        super().__init__(**kwargs)

        if self.n_layers < 2:
            raise ValueError("PCRQ-VAE requires at least two quantization layers.")
        if self.reconstruction_loss_function is None:
            raise ValueError("PCRQ-VAE requires a reconstruction_loss_function.")

        self.global_quantization_loss_function = (
            global_quantization_loss_function
        )
        self.prefix_contrastive_loss_function = (
            prefix_contrastive_loss_function
        )
        self.global_quantization_loss_weight = global_quantization_loss_weight
        self.prefix_contrastive_loss_weight = prefix_contrastive_loss_weight

        self.train_global_quantization_loss = MeanMetric()
        self.train_prefix_contrastive_loss = MeanMetric()

    def _is_fitting(self) -> bool:
        return (
            self._trainer is not None
            and self.trainer.state.fn == TrainerFn.FITTING
        )

    def _should_train_layer(self, layer_idx: int) -> bool:
        """Mirror the parent's staged codebook-initialization behavior."""
        if not self._is_fitting():
            return False
        if self.train_layer_wise:
            return layer_idx == self.current_layer

        layer = self.quantization_layer_list[layer_idx]
        if layer.is_initialized and not self.quantization_layer_list[-1].is_initialized:
            return False
        if layer_idx == 0:
            return True

        previous_layer = self.quantization_layer_list[layer_idx - 1]
        return previous_layer.is_initialized or previous_layer.is_initial_step

    def encode_and_quantize(
        self, input_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, PCRQForwardOutput]:
        """Normalize, encode, and residually quantize a batch of embeddings."""
        input_embeddings = input_embeddings.to(self.device)
        normalized_embeddings = self.normalization_layer(input_embeddings)
        encoded_embeddings = self.encoder(normalized_embeddings)
        quantization_output = self.quantize_with_details(encoded_embeddings)
        return normalized_embeddings, quantization_output

    def quantize_with_details(
        self, encoded_embeddings: torch.Tensor
    ) -> PCRQForwardOutput:
        """Quantize latents while preserving STE and real-codeword paths."""
        cluster_ids = []
        residuals = [] if self.track_residuals else None
        selected_codewords = []
        current_residuals = encoded_embeddings
        ste_quantized_embeddings = torch.zeros_like(encoded_embeddings)
        initialization_loss = encoded_embeddings.new_zeros(())

        for layer_idx, layer in enumerate(self.quantization_layer_list):
            if self.normalize_residuals:
                current_residuals = F.normalize(current_residuals, dim=-1)

            if self._should_train_layer(layer_idx):
                layer_ids, ste_layer_embeddings, layer_loss = layer.model_step(
                    current_residuals
                )
                if layer_loss is not None:
                    initialization_loss = initialization_loss + layer_loss
            else:
                layer_ids, ste_layer_embeddings = layer.predict_step(
                    current_residuals
                )

            # This lookup is intentionally separate from the STE value returned
            # above: it keeps a direct gradient path to the codebook parameter.
            layer_codewords = layer.get_centroids()[layer_ids]

            cluster_ids.append(layer_ids)
            selected_codewords.append(layer_codewords)
            ste_quantized_embeddings = (
                ste_quantized_embeddings + ste_layer_embeddings
            )
            current_residuals = current_residuals - ste_layer_embeddings

            if residuals is not None:
                residuals.append(current_residuals)

        stacked_cluster_ids = torch.stack(cluster_ids, dim=1)
        stacked_codewords = torch.stack(selected_codewords, dim=1)
        stacked_residuals = (
            torch.stack(residuals, dim=-1) if residuals is not None else None
        )

        return PCRQForwardOutput(
            cluster_ids=stacked_cluster_ids,
            all_residuals=stacked_residuals,
            encoded_embeddings=encoded_embeddings,
            ste_quantized_embeddings=ste_quantized_embeddings,
            codebook_quantized_embeddings=stacked_codewords.sum(dim=1),
            selected_codewords=stacked_codewords,
            initialization_loss=initialization_loss,
        )

    def pcrq_model_step(self, model_input: ItemData) -> PCRQLossOutput:
        """Compute the full PCRQ-VAE objective for anchor-positive pairs."""
        if self.global_quantization_loss_function is None:
            raise RuntimeError("PCRQ-VAE training requires a global quantization loss.")
        if self.prefix_contrastive_loss_function is None:
            raise RuntimeError("PCRQ-VAE training requires a prefix contrastive loss.")
        transformed_features = model_input.transformed_features
        if "input_embedding" not in transformed_features:
            raise KeyError("PCRQ-VAE batch is missing 'input_embedding'.")
        if "positive_embedding" not in transformed_features:
            raise KeyError("PCRQ-VAE training batch is missing 'positive_embedding'.")

        anchor_embeddings = transformed_features["input_embedding"].to(self.device)
        positive_embeddings = transformed_features["positive_embedding"].to(
            self.device
        )
        if anchor_embeddings.shape != positive_embeddings.shape:
            raise ValueError(
                "Anchor and positive embeddings must have identical shapes; got "
                f"{tuple(anchor_embeddings.shape)} and "
                f"{tuple(positive_embeddings.shape)}."
            )

        batch_size = anchor_embeddings.shape[0]

        # Codebook initialization must follow the original item distribution.
        # Positive neighbors are used only by the prefix-contrastive objective;
        # including them in the initialization buffer would over-sample popular
        # semantic neighbors and bias K-means initialization.
        all_codebooks_initialized = all(
            layer.is_initialized for layer in self.quantization_layer_list
        )

        if not all_codebooks_initialized:
            _, output = self.encode_and_quantize(anchor_embeddings)
            zero = anchor_embeddings.new_zeros(())

            return PCRQLossOutput(
                cluster_ids=output.cluster_ids,
                all_residuals=output.all_residuals,
                loss=output.initialization_loss,
                reconstruction_loss=zero,
                global_quantization_loss=zero,
                prefix_contrastive_loss=zero,
            )

        # Positive samples participate only after all residual codebooks have
        # completed initialization.
        combined_embeddings = torch.cat(
            (anchor_embeddings, positive_embeddings), dim=0
        )
        normalized_embeddings, output = self.encode_and_quantize(
            combined_embeddings
        )

        reconstructed_embeddings = self.decoder(
            output.ste_quantized_embeddings
        )
        reconstruction_loss = self.reconstruction_loss_function(
            reconstructed_embeddings, normalized_embeddings
        )
        global_quantization_loss = self.global_quantization_loss_function(
            encoded_embeddings=output.encoded_embeddings,
            quantized_embeddings=output.codebook_quantized_embeddings,
        )

        anchor_codewords = output.selected_codewords[:batch_size]
        positive_codewords = output.selected_codewords[batch_size:]
        prefix_contrastive_loss = self.prefix_contrastive_loss_function(
            anchor_codewords=anchor_codewords,
            positive_codewords=positive_codewords,
        )

        loss = (
            self.reconstruction_loss_weight * reconstruction_loss
            + self.global_quantization_loss_weight * global_quantization_loss
            + self.prefix_contrastive_loss_weight * prefix_contrastive_loss
        )
        return PCRQLossOutput(
            cluster_ids=output.cluster_ids[:batch_size],
            all_residuals=(
                output.all_residuals[:batch_size]
                if output.all_residuals is not None
                else None
            ),
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            global_quantization_loss=global_quantization_loss,
            prefix_contrastive_loss=prefix_contrastive_loss,
        )

    def model_step(
        self, model_input: ItemData
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Anchor-only step used by SID prediction and legacy evaluation code."""
        input_embeddings = model_input.transformed_features["input_embedding"]
        normalized_embeddings, output = self.encode_and_quantize(input_embeddings)

        reconstruction_loss = output.encoded_embeddings.new_zeros(())
        if self._trainer is not None and self.trainer.state.fn != TrainerFn.PREDICTING:
            reconstructed_embeddings = self.decoder(
                output.ste_quantized_embeddings
            )
            reconstruction_loss = self.reconstruction_loss_function(
                reconstructed_embeddings, normalized_embeddings
            )

        return (
            output.cluster_ids,
            output.all_residuals,
            output.initialization_loss,
            reconstruction_loss,
        )

    def training_step(self, batch, batch_idx: int = 0) -> torch.Tensor:
        """Optimize reconstruction, global quantization, and prefix InfoNCE."""
        model_input = batch[0] if isinstance(batch, (tuple, list)) else batch
        result = self.pcrq_model_step(model_input)

        self.train_loss(result.loss)
        self.train_reconstruction_loss(result.reconstruction_loss)
        self.train_global_quantization_loss(result.global_quantization_loss)
        self.train_prefix_contrastive_loss(result.prefix_contrastive_loss)
        self.log_dict(
            {
                "train/loss": self.train_loss,
                "train/reconstruction_loss": self.train_reconstruction_loss,
                "train/global_quantization_loss": self.train_global_quantization_loss,
                "train/prefix_contrastive_loss": self.train_prefix_contrastive_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if self.training_loop_function is not None:
            self.training_loop_function(
                self,
                loss=result.loss,
                world_size=self.trainer.world_size,
                is_initialized=all(
                    layer.is_initialized
                    for layer in self.quantization_layer_list
                ),
            )

        return result.loss

    def _shared_evaluation_step(
        self, batch: ItemData, stage: str
    ) -> torch.Tensor:
        """Evaluate the full objective when pairs exist, otherwise reconstruction."""
        if "positive_embedding" in batch.transformed_features:
            result = self.pcrq_model_step(batch)
            loss = result.loss
            metrics = {
                f"{stage}/loss": loss,
                f"{stage}/reconstruction_loss": result.reconstruction_loss,
                f"{stage}/global_quantization_loss": (
                    result.global_quantization_loss
                ),
                f"{stage}/prefix_contrastive_loss": (
                    result.prefix_contrastive_loss
                ),
            }
        else:
            if self.global_quantization_loss_function is None:
                raise RuntimeError(
                    "PCRQ-VAE evaluation requires a global quantization loss."
                )
            input_embeddings = batch.transformed_features["input_embedding"]
            normalized_embeddings, output = self.encode_and_quantize(
                input_embeddings
            )
            reconstructed_embeddings = self.decoder(
                output.ste_quantized_embeddings
            )
            reconstruction_loss = self.reconstruction_loss_function(
                reconstructed_embeddings, normalized_embeddings
            )
            global_quantization_loss = self.global_quantization_loss_function(
                encoded_embeddings=output.encoded_embeddings,
                quantized_embeddings=output.codebook_quantized_embeddings,
            )
            loss = (
                self.reconstruction_loss_weight * reconstruction_loss
                + self.global_quantization_loss_weight
                * global_quantization_loss
            )
            metrics = {
                f"{stage}/loss": loss,
                f"{stage}/reconstruction_loss": reconstruction_loss,
                f"{stage}/global_quantization_loss": global_quantization_loss,
            }

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: ItemData, batch_idx: int = 0) -> torch.Tensor:
        return self._shared_evaluation_step(batch, "val")

    def test_step(self, batch: ItemData, batch_idx: int = 0) -> torch.Tensor:
        return self._shared_evaluation_step(batch, "test")

    def on_train_start(self) -> None:
        super().on_train_start()
        self.train_global_quantization_loss.reset()
        self.train_prefix_contrastive_loss.reset()
