import math

def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    # Linear warmup
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    # Cosine learning rate decay
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    progress = min(progress, 1.0)
    return min_learning_rate + (max_learning_rate - min_learning_rate) * 0.5 * (1 + math.cos(math.pi * progress))