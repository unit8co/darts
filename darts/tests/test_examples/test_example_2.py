import os.path

import pytest

from darts.datasets import AirPassengersDataset
from darts.models import TFTModel


def test_add():
    assert 1 + 2 == 3


def test_save(tmpdir_module, tfm_kwargs):
    s = AirPassengersDataset().load()
    m = TFTModel(
        input_chunk_length=12,
        output_chunk_length=6,
        add_relative_index=True,
        **tfm_kwargs
    )
    m.fit(s, epochs=1, verbose=False)
    m.save(os.path.join(tmpdir_module, "tft1.pt"))


@pytest.mark.parametrize(
    "model_config", [(0, {"full_attention": False}), (1, {"full_attention": True})]
)
def test_save2(tmpdir_module, tfm_kwargs, model_config):
    idx, idx_model_kwargs = model_config
    assert idx, "blabla error message"

    s = AirPassengersDataset().load()
    m = TFTModel(
        input_chunk_length=12,
        output_chunk_length=6,
        add_relative_index=True,
        **tfm_kwargs
    )
    m.fit(s, epochs=1, verbose=False)
    m.save(os.path.join(tmpdir_module, "tft2.pt"))
