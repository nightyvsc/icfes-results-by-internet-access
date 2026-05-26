"""
Autoencoder PyTorch para detección de anomalías territoriales.

Arquitectura:
  Input (14) → Encoder → Bottleneck (4) → Decoder → Output (14)
"""

import torch
import torch.nn as nn
from typing import Tuple


class AutoencoderMunicipal(nn.Module):
    """
    Autoencoder para detectar municipios anómalos.

    Features de entrada: 14 (conectividad + socioeconómicas)
    Latent space: 4 dimensiones (compresión ~71%)
    """

    def __init__(self, input_dim: int = 14, latent_dim: int = 4):
        """
        Args:
            input_dim: Dimensión de entrada (# features)
            latent_dim: Dimensión del espacio latente (bottleneck)
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: comprime de input_dim a latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),  # Bottleneck
        )

        # Decoder: expande de latent_dim a input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 12),
            nn.ReLU(),
            nn.Linear(12, input_dim),  # Sin activación (features normalizadas)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Codifica entrada al espacio latente.

        Args:
            x: Tensor (batch_size, input_dim)

        Returns:
            Tensor (batch_size, latent_dim)
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodifica desde espacio latente.

        Args:
            z: Tensor (batch_size, latent_dim)

        Returns:
            Tensor (batch_size, input_dim)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: input → encoder → decoder → output (reconstrucción).

        Args:
            x: Tensor (batch_size, input_dim)

        Returns:
            Tensor (batch_size, input_dim) - reconstrucción
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Obtiene representación latente (para análisis posterior).

        Args:
            x: Tensor (batch_size, input_dim)

        Returns:
            Tensor (batch_size, latent_dim)
        """
        return self.encode(x)


def reconstruction_error(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Calcula error de reconstrucción (MSE).

    Args:
        x: Entrada original (batch_size, features)
        x_recon: Reconstrucción (batch_size, features)
        reduction: "mean" para MSE medio, "none" para error por sample

    Returns:
        Tensor con error(es) de reconstrucción
    """
    mse = nn.MSELoss(reduction=reduction)
    return mse(x_recon, x)


def reconstruction_error_per_feature(
    x: torch.Tensor,
    x_recon: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calcula error de reconstrucción desglosado por feature.

    Útil para interpretabilidad: saber qué features contribuyen más al error.

    Args:
        x: Entrada original (batch_size, features)
        x_recon: Reconstrucción (batch_size, features)

    Returns:
        Tuple:
        - error_per_feature: (features,) - error medio por feature
        - error_per_sample: (batch_size,) - error MSE por muestra
    """
    # Error por feature: MSE medio across all samples
    error_per_feature = ((x - x_recon) ** 2).mean(dim=0)

    # Error por sample: MSE medio across all features
    error_per_sample = ((x - x_recon) ** 2).mean(dim=1)

    return error_per_feature, error_per_sample


if __name__ == "__main__":
    # Test simple de arquitectura
    print("=== Autoencoder Municipal ===\n")

    model = AutoencoderMunicipal(input_dim=14, latent_dim=4)
    print(f"Modelo: {model}\n")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 14)
    x_recon = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")

    # Test error de reconstrucción
    error = reconstruction_error(x, x_recon)
    print(f"MSE error (mean): {error.item():.4f}")

    # Test error por feature
    error_feat, error_sample = reconstruction_error_per_feature(x, x_recon)
    print(f"Error per feature shape: {error_feat.shape}")
    print(f"Error per sample shape: {error_sample.shape}")
    print(f"Error per feature (sample): {error_feat[:5]}")

    print("\n✓ Test passed")
