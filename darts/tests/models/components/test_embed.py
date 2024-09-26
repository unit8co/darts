import pytest
import torch

from darts.models.components.embed import (
    DataEmbedding,
    DataEmbedding_inverted,
    DataEmbedding_wo_pos,
    FixedEmbedding,
    PatchEmbedding,
    PositionalEmbedding,
    TemporalEmbedding,
    TimeFeatureEmbedding,
    TokenEmbedding,
)


class TestEmbedding:
    def test_PositionalEmbedding(self):
        d_model = 64
        max_len = 500
        embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)
        x = torch.randn(10, 100, d_model)  # batch_size=10, seq_len=100, d_model
        pe = embedding(x)
        assert pe.shape == (
            1,
            100,
            d_model,
        ), "PositionalEmbedding output shape mismatch."
        # Test that pe does not require gradient
        assert (
            not pe.requires_grad
        ), "PositionalEmbedding output should not require grad."

    def test_TokenEmbedding(self):
        c_in = 10
        d_model = 64
        batch_size = 32
        seq_len = 100
        embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        x = torch.randn(batch_size, seq_len, c_in)  # [B, L, C_in]
        output = embedding(x)
        assert output.shape == (
            batch_size,
            seq_len,
            d_model,
        ), "TokenEmbedding output shape mismatch."

    def test_FixedEmbedding(self):
        c_in = 32
        d_model = 64
        embedding = FixedEmbedding(c_in, d_model)
        x = torch.arange(0, c_in).unsqueeze(0)  # [1, c_in]
        output = embedding(x)
        assert output.shape == (
            1,
            c_in,
            d_model,
        ), "FixedEmbedding output shape mismatch."

    def test_TemporalEmbedding(self):
        d_model = 64
        embed_type = "fixed"
        freq = "h"
        embedding = TemporalEmbedding(d_model, embed_type, freq)
        batch_size = 32
        seq_len = 100

        month = torch.randint(0, 13, (batch_size, seq_len, 1))  # 0-12 for months
        day = torch.randint(1, 32, (batch_size, seq_len, 1))  # 1-31 for days
        weekday = torch.randint(0, 7, (batch_size, seq_len, 1))  # 0-6 for weekdays
        hour = torch.randint(0, 24, (batch_size, seq_len, 1))  # 0-23 for hours
        minute = torch.randint(
            0, 4, (batch_size, seq_len, 1)
        )  # 0-3 for minutes (assuming 15-minute intervals)

        x = torch.cat([month, day, weekday, hour, minute], dim=2)

        output = embedding(x)
        assert output.shape == (
            batch_size,
            seq_len,
            d_model,
        ), "TemporalEmbedding output shape mismatch."

    def test_DataEmbedding_no_x_mark(self):
        c_in = 10
        d_model = 64
        embed_type = "fixed"
        freq = "h"
        dropout = 0.1
        embedding = DataEmbedding(c_in, d_model, embed_type, freq, dropout)
        batch_size = 32
        seq_len = 100
        x = torch.randn(batch_size, seq_len, c_in)
        output = embedding(x, None)
        assert output.shape == (
            batch_size,
            seq_len,
            d_model,
        ), "DataEmbedding output shape mismatch when x_mark is None."

    def test_DataEmbedding_wo_pos_no_x_mark(self):
        c_in = 10
        d_model = 64
        embed_type = "fixed"
        freq = "h"
        dropout = 0.1
        embedding = DataEmbedding_wo_pos(c_in, d_model, embed_type, freq, dropout)
        batch_size = 32
        seq_len = 100
        x = torch.randn(batch_size, seq_len, c_in)
        output = embedding(x, None)
        assert output.shape == (
            batch_size,
            seq_len,
            d_model,
        ), "DataEmbedding_wo_pos output shape mismatch when x_mark is None."

    def test_DataEmbedding_with_x_mark(self):
        c_in = 10
        d_model = 64
        embed_type = "fixed"
        freq = "h"
        dropout = 0.1
        embedding = DataEmbedding(c_in, d_model, embed_type, freq, dropout)
        batch_size = 32
        seq_len = 100
        x = torch.randn(batch_size, seq_len, c_in)

        # Create x_mark with appropriate integer indices
        month = torch.randint(0, 12, (batch_size, seq_len, 1))
        day = torch.randint(1, 32, (batch_size, seq_len, 1))
        weekday = torch.randint(0, 7, (batch_size, seq_len, 1))
        hour = torch.randint(0, 24, (batch_size, seq_len, 1))
        minute = torch.randint(0, 60, (batch_size, seq_len, 1))
        x_mark = torch.cat([month, day, weekday, hour, minute], dim=2)

        output = embedding(x, x_mark)

        assert output.shape == (
            batch_size,
            seq_len,
            d_model,
        ), "DataEmbedding output shape mismatch."

    def test_DataEmbedding_inverted(self):
        c_in = 10
        d_model = 64
        embed_type = "fixed"
        freq = "h"
        dropout = 0.1
        embedding = DataEmbedding_inverted(c_in, d_model, embed_type, freq, dropout)
        batch_size = 32
        seq_len = 100

        # Change the input shape to (batch_size, c_in, seq_len)
        x = torch.randn(batch_size, c_in, seq_len)

        # Change x_mark shape to (batch_size, seq_len, 5) if x_mark is used
        # or set it to None if it's not used in this embedding
        x_mark = None  # or torch.randn(batch_size, seq_len, 5) if it's used

        output = embedding(x, x_mark)

        # The expected output shape should be (batch_size, d_model, seq_len)
        assert output.shape == (
            batch_size,
            seq_len,
            d_model,
        ), "DataEmbedding_inverted output shape mismatch."

    def test_PatchEmbedding(self):
        d_model = 64
        patch_len = 16
        stride = 8
        padding = 8
        dropout = 0.1
        embedding = PatchEmbedding(d_model, patch_len, stride, padding, dropout)
        batch_size = 32
        n_vars = 10
        seq_len = 100
        x = torch.randn(batch_size, n_vars, seq_len)
        output, n_vars_output = embedding(x)
        num_patches = ((seq_len + padding) - patch_len) // stride + 1
        expected_shape = (batch_size * n_vars, num_patches, d_model)
        assert output.shape == expected_shape, "PatchEmbedding output shape mismatch."
        assert n_vars_output == n_vars, "PatchEmbedding n_vars output mismatch."

    def test_TimeFeatureEmbedding_invalid_input(self):
        d_model = 64
        embed_type = "timeF"
        freq = "h"
        embedding = TimeFeatureEmbedding(d_model, embed_type, freq)
        batch_size = 32
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 10)  # Incorrect feature size
        with pytest.raises(RuntimeError):
            embedding(x)
