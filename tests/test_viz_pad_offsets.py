from nbh.viz_utils import compute_pad_offsets


def test_compute_pad_offsets_16_divisor():
    pad_h1, pad_h2, pad_w1, pad_w2 = compute_pad_offsets(1325, 1428, divisor=16)
    assert (pad_h1, pad_h2) == (1, 2)
    assert (pad_w1, pad_w2) == (6, 6)
