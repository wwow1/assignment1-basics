import torch

def cross_entropy_loss(logits, labels):
    """
    Compute the cross entropy loss between logits and labels.

    Args:
        logits (torch.Tensor): Logits tensor of shape (..., num_classes).
        labels (torch.Tensor): Labels tensor of shape (...).

    Returns:
        torch.Tensor: Cross entropy loss tensor of shape (...).
    """
    # Flatten dimensions to handle arbitrary batch shapes (e.g., [batch, seq_len, vocab])
    vocab_size = logits.shape[-1]
    
    # 1. Flatten logits to (N, vocab_size) and labels to (N,)
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)

    # 2. Subtract max for numerical stability (Max Trick)
    # Find max along vocab dimension (dim=-1)
    max_logits, _ = logits_flat.max(dim=-1, keepdim=True)
    shifted_logits = logits_flat - max_logits
    
    # 3. Compute log(sum(exp(shifted_logits)))
    # This gives us the normalization constant in log space
    log_sum_exp = torch.logsumexp(shifted_logits, dim=-1)
    
    # 4. Get the logits corresponding to the correct classes
    # Use advanced indexing: arange generates [0, 1, ..., N-1], labels_flat provides the column indices
    target_logits = shifted_logits[torch.arange(logits_flat.shape[0]), labels_flat]
    
    # 5. Calculate loss per element
    # Loss = -log(softmax(target)) = -target_logit + log_sum_exp
    loss_flat = log_sum_exp - target_logits
    
    # 6. Reshape back to original batch dimensions
    # The output should match the shape of the labels tensor
    return loss_flat.view(labels.shape).mean()