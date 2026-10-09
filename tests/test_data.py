from pathlib import Path

import numpy as np
import pytest

from revllm.data import TokenTape


def test_exact_partial_batch_and_next_token_targets(tmp_path: Path):
    np.arange(14, dtype=np.uint16).tofile(tmp_path / "train.bin")
    tape = TokenTape(tmp_path / "train.bin", target_count=13, block_size=4)
    x, y, valid = tape.batch(8, physical_batch=2)
    assert valid == 5
    assert x.tolist() == [[8, 9, 10, 11], [12, 0, 0, 0]]
    assert y.tolist() == [[9, 10, 11, 12], [13, -100, -100, -100]]
    assert tape.batch(0, physical_batch=3, remaining=7)[2] == 7
    with pytest.raises(ValueError):
        tape.batch(13, physical_batch=1)
