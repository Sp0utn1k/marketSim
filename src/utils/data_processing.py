from datetime import datetime
import pandas as pd
import torch


def get_ohlcvt(datafolder, instrument_name, interval=1, **kwargs):
    if interval not in [1, 5, 15, 30, 60, 240, 720, 1440]:
        raise ValueError("Unknown interval")

    datafile = f"{datafolder}/{instrument_name}_{interval}.csv"
    names = ["time", "open", "high", "low", "close", "volume", "trades"]
    return pd.read_csv(datafile, header=None, names=names, skiprows=1, **kwargs)


def expand_timestamp(stamp):
    time = datetime.fromtimestamp(stamp)
    return [
        time.year - 2000,
        time.month,
        time.day,
        time.weekday(),
        time.hour,
        time.minute,
    ]


def load_data(datafolder, instrument_name, device=None, to_torch=True, correction_factor=1, **kwargs):
    out = get_ohlcvt(datafolder, instrument_name, **kwargs).to_numpy()[:, :-1]
    if to_torch:
        metadata = torch.tensor([expand_timestamp(out[i, 0]) for i in range(out.shape[0])], dtype=torch.float,
                                device=device)
        out = torch.tensor(out, dtype=torch.float, device=device)
        out *= correction_factor
        return out, metadata
    else:
        return out
