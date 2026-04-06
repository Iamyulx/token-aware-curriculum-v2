import math

class TokenLRScheduler:
    def __init__(self, optimizer, warmup_tokens, total_tokens, base_lr):
        self.optimizer = optimizer
        self.warmup_tokens = warmup_tokens
        self.total_tokens = total_tokens
        self.base_lr = base_lr
        self.tokens_processed = 0

    def step(self, tokens_in_batch):
        self.tokens_processed += tokens_in_batch

        if self.tokens_processed < self.warmup_tokens:
            lr = self.base_lr * self.tokens_processed / self.warmup_tokens
        else:
            progress = (self.tokens_processed - self.warmup_tokens) / (
                self.total_tokens - self.warmup_tokens
            )
            lr = 0.5 * self.base_lr * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr
