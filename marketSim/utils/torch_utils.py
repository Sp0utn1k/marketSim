import torch

def pre_process(out):
    body = out[:, [0, 3]]
    real_body = 10 * torch.log(body[:, 1] / body[:, 0])
    sorted_body = torch.sort(body, dim=1, descending=True).values
    shadow = out[:, 1:3]
    shadow = 10 * torch.log(shadow / sorted_body)
    shadow_up = shadow[:, 0]
    shadow_low = -shadow[:, 1]
    processed = torch.stack((real_body, shadow_up, shadow_low, out[:, 3], out[:, 4]), dim=1)
    return processed

def make_metabatch(out, metadata, batch_size, sample_size, pred_len):
    device = out.device
    max_idx = out.shape[0] - (sample_size + pred_len)
    ids = tuple(torch.arange(idx, idx + sample_size + pred_len, device=device) for idx in torch.randint(max_idx, (batch_size,)))
    ids = torch.stack(ids, dim=0)
    batch = out[ids, :].transpose(1, 2)
    batch, target = batch[:, :, :sample_size], batch[:, 3, sample_size:]
    metabatch = metadata[ids[:, 0], :]
    return batch, metabatch, target

def make_batch(out, batch_size, sample_size, pred_len):
    device = out.device
    max_idx = out.shape[0] - (sample_size + pred_len)
    ids = tuple(torch.arange(idx, idx + sample_size + pred_len, device=device) for idx in torch.randint(max_idx, (batch_size,)))
    ids = torch.stack(ids, dim=0)
    batch = out[ids, :].transpose(1, 2)
    batch, target = batch[:, :, :sample_size], batch[:, 3, sample_size:]
    return batch, target.flatten()
