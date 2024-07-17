import numpy as np
from cosas.transforms import get_image_stats


def test_get_image_stats():

    images = np.random.randint(0, 256, size=(10, 224, 224, 3), dtype=np.uint8)

    # get image stats
    means, stds = get_image_stats(images)

    assert means.shape == (3,)
    assert stds.shape == (3,)


def test_get_image_stats_list():

    images = [
        np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
        for i in range(0, 10)
    ]

    # get image stats
    means, stds = get_image_stats(images)

    assert means.shape == (3,)
    assert stds.shape == (3,)
