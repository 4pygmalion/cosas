import numpy as np
from cosas.transforms import (
    get_image_stats,
    get_lab_distribution,
    find_representative_lab_image,
)


def test_get_image_stats():

    images = np.random.randint(0, 256, size=(10, 224, 224, 3), dtype=np.uint8)

    # get image stats
    means, stds = get_image_stats(images)

    assert len(means) == 3
    assert len(stds) == 3


def test_get_image_stats_list():

    images = [
        np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
        for i in range(0, 10)
    ]

    # get image stats
    means, stds = get_image_stats(images)

    assert len(means) == 3
    assert len(stds) == 3


def test_get_lab_distribution():

    images = np.random.randint(0, 256, size=(10, 224, 224, 3), dtype=np.uint8)

    # get image stats
    means, stds = get_lab_distribution(images)

    assert len(means) == 3
    assert len(stds) == 3


def test_find_mode_image():
    images = np.random.randint(0, 256, size=(10, 224, 224, 3), dtype=np.uint8)

    image = find_representative_lab_image(
        images, means=np.array([100.2, 100.1, 100.0], dtype=np.float32)
    )

    assert image.shape == (224, 224, 3)
