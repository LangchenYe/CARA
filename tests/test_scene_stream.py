import numpy as np
import pytest

from cara.scene_stream import SCENE_LAWS, SceneStream, sample_stream
from cara.synthetic import LAW, make_environment


def test_every_law_is_prefix_consistent_across_irregular_batch_shapes():
    environment = make_environment(145003)
    for law in SCENE_LAWS:
        whole = SceneStream(environment, law, 15).next(97)
        stream = SceneStream(environment, law, 15)
        split = np.concatenate([stream.next(n) for n in (1, 31, 2, 32, 17, 14)])
        assert np.array_equal(whole, split)
        assert np.array_equal(whole, sample_stream(environment, law, 97, 15, 32))
        assert np.array_equal(whole[:32], SceneStream(environment, law, 15).next(32))
        assert stream.rows == 97 and np.all((whole >= 0) & (whole <= 1))


def test_role_streams_are_distinct_and_reproducible():
    environment = make_environment(145004)
    first = SceneStream(environment, "bursty_common", 15).next(64)
    repeated = SceneStream(environment, "bursty_common", 15).next(64)
    other = SceneStream(environment, "bursty_common", 16).next(64)
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, other)
    assert first.shape == (64, LAW["vms"], LAW["horizon"])


def test_stream_rejects_invalid_requests():
    environment = make_environment(145005)
    with pytest.raises(ValueError):
        SceneStream(environment, "unknown", 15)
    with pytest.raises(ValueError):
        SceneStream(environment, "complementary", -1)
    with pytest.raises(ValueError):
        SceneStream(environment, "complementary", 15).next(0)
