import os.path

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
    m.fit(s, epochs=1)
    m.save(os.path.join(tmpdir_module, "tft1.pt"))


def test_save2(tmpdir_module, tfm_kwargs):
    s = AirPassengersDataset().load()
    m = TFTModel(
        input_chunk_length=12,
        output_chunk_length=6,
        add_relative_index=True,
        **tfm_kwargs
    )
    m.fit(s, epochs=1)
    m.save(os.path.join(tmpdir_module, "tft2.pt"))
