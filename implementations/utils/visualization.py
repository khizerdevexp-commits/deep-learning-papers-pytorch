"""
Visualization utilities for embeddings and results.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional, List


def plot_embeddings(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    method: str = "pca",
    figsize: tuple = (10, 8),
    title: str = "Embedding Visualization",
) -> None:
    """
    Visualize high-dimensional embeddings in 2D.
    
    Args:
        embeddings: Embedding matrix [vocab_size, embedding_dim]
        labels: Optional word labels
        method: Dimensionality reduction method ('pca' or 'tsne')
        figsize: Figure size
        title: Plot title
    """
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Reduce dimensions
    reduced = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5)
    
    if labels:
        for i, label in enumerate(labels):
            plt.annotate(label, (reduced[i, 0], reduced[i, 1]), fontsize=8)
    
    plt.title(title)
    plt.xlabel(f"{method.upper()} 1")
    plt.ylabel(f"{method.upper()} 2")
    plt.tight_layout()
    plt.show()


def plot_training_curve(
    train_losses: list,
    val_losses: Optional[list] = None,
    figsize: tuple = (10, 5),
    title: str = "Training Curves",
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: Optional list of validation losses per epoch
        figsize: Figure size
        title: Plot title
    """
    plt.figure(figsize=figsize)
    plt.plot(train_losses, label="Training Loss", marker="o")
    
    if val_losses:
        plt.plot(val_losses, label="Validation Loss", marker="s")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
