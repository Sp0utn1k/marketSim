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


def create_batches(out, metadata=None, batch_size=32, sample_size=50, pred_len=10, include_meta=False):
    device = out.device
    max_idx = out.shape[0] - (sample_size + pred_len)
    ids = torch.randint(max_idx, (batch_size,))
    ids = torch.stack([torch.arange(idx, idx + sample_size + pred_len, device=device) for idx in ids])
    batch = out[ids, :].transpose(1, 2)
    batch, target = batch[:, :, :sample_size], batch[:, 3, sample_size:]

    if include_meta and metadata is not None:
        metabatch = metadata[ids[:, 0], :]
        return batch, metabatch, target
    else:
        return batch, target.flatten()


def make_metabatch(out, metadata, batch_size=32, sample_size=50, pred_len=10):
    return create_batches(out, metadata, batch_size, sample_size, pred_len, include_meta=True)


def make_batch(out, batch_size=32, sample_size=50, pred_len=10):
    return create_batches(out, batch_size=batch_size, sample_size=sample_size, pred_len=pred_len)
