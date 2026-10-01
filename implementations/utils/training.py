"""
Training utilities and loops.
"""

import torch
import torch.nn as nn
from typing import Callable, Dict, Tuple
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: torch.device,
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to train on
    
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move batch to device
        if isinstance(batch, (list, tuple)):
            batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
        else:
            batch = batch.to(device) if isinstance(batch, torch.Tensor) else batch
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(*batch) if isinstance(batch, (list, tuple)) else model(batch)
        loss = loss_fn(outputs)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader,
    loss_fn: Callable,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation data loader
        loss_fn: Loss function
        device: Device to evaluate on
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            else:
                batch = batch.to(device) if isinstance(batch, torch.Tensor) else batch
            
            # Forward pass
            outputs = model(*batch) if isinstance(batch, (list, tuple)) else model(batch)
            loss = loss_fn(outputs)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return {"loss": avg_loss}
