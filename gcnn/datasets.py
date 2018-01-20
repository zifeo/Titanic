import pandas as pd
import numpy as np
from torchvision import datasets


def data_icebergs(set: str):
    icebergs = pd.read_json('../data/{}.json'.format(set)).set_index('id')
    icebergs = icebergs.assign(
        inc_angle=pd.to_numeric(icebergs.inc_angle, 'coerce'),
        band_1=icebergs.band_1.apply(np.array),
        band_2=icebergs.band_2.apply(np.array),
    )
    return icebergs


def mnist(train: bool):
    x, y = zip(*datasets.MNIST('../data', train=train, download=True))
    return pd.DataFrame(dict(x=x, y=y))
