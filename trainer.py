def train_step(batch, model, optimizer, scheduler, rl_selector=None):
    input_ids = batch["input_ids"]

    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss

    loss.backward()

    tokens = input_ids.numel()
    scheduler.step(tokens)

    optimizer.step()
    optimizer.zero_grad()

    return loss.item()