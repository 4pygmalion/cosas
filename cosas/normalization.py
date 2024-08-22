import cv2
import numpy as np
import spams
from PIL import Image

from seestaina.misc import od_to_rgb, hash_image, rgb_to_od, normalize_rows


def find_median_lab_image(images: list[np.ndarray]) -> np.ndarray:
    # 각 이미지의 LAB 채널에서의 중앙값을 계산
    lab_medians = []

    for image in images:
        # 이미지를 LAB 색상 공간으로 변환
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # LAB 각 채널의 중앙값을 계산
        lab_median = np.median(lab_image.reshape(-1, 3), axis=0)
        lab_medians.append(lab_median)

    # 모든 이미지의 LAB 중앙값에 대해 중앙값을 구함
    lab_medians = np.array(lab_medians)
    overall_median = np.median(lab_medians, axis=0)

    # 중앙값과 가장 가까운 이미지를 찾음
    min_distance = float("inf")
    median_image = None

    for i, lab_median in enumerate(lab_medians):
        distance = np.linalg.norm(lab_median - overall_median)
        if distance < min_distance:
            min_distance = distance
            median_image = images[i]

    return median_image


class SPCNNormalizer:

    def __init__(self, numThreads: int = 16, od_threshold: float = 0.05):
        self.numThreads = numThreads
        self.hash2stain_matrix = dict()
        self.hash2stain_density = dict()
        self.od_threshold = od_threshold

    def get_tissue_mask(
        self, image: np.ndarray, luminosity_threshold=0.8
    ) -> np.ndarray:

        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_values = image_lab[:, :, 0] / 255.0

        return l_values < luminosity_threshold

    def standardize_brightness(self, image: np.ndarray):
        """

        Params
            image (np.ndarray): image array (np.uint8)
        """
        p = np.percentile(image, 90)
        return np.clip(image * 255.0 / p, 0, 255).astype(np.uint8)

    def get_stain_matrix(
        self,
        image: np.ndarray,
        resize: tuple = (224, 224),
    ) -> np.ndarray:
        """Image array

        Args:
            image (np.ndarray): rgb_image_array

        Returns:
            np.ndarray: OD space (channel OD, stains)

        References:
            https://github.com/wanghao14/Stain_Normalization/blob/master/stainNorm_Vahadane.py
        """

        resized_image = cv2.resize(image, dsize=resize, interpolation=cv2.INTER_NEAREST)

        digest = hash_image(Image.fromarray(resized_image))
        if digest in self.hash2stain_matrix:
            return self.hash2stain_matrix[digest]

        mask_indices = self.get_tissue_mask(resized_image)
        od_image = rgb_to_od(resized_image)
        od_image = od_image[mask_indices, :].reshape(-1, 3)
        if len(od_image) < resized_image.shape[0] * resized_image.shape[1] * 0.05:
            error_msg = f"too small tissue: len(od_image)={len(od_image)}"
            error_msg += " saved error_image.png"
            raise np.linalg.LinAlgError(error_msg)

        dictionary = spams.trainDL(
            od_image.T,
            K=2,  # size of dictionary (기저)
            mode=2,
            modeD=0,
            lambda1=0.1,
            numThreads=self.numThreads,
            posAlpha=True,
            posD=True,
            verbose=False,
        ).T

        if dictionary[0, 0] < dictionary[1, 0]:
            dictionary = dictionary[[1, 0], :]

        stain_matrix = normalize_rows(dictionary)
        self.hash2stain_matrix[digest] = stain_matrix

        return stain_matrix

    def get_stain_density(
        self, image: np.ndarray, stain_matrix: np.ndarray
    ) -> np.ndarray:
        """Retrun stain density matrix

        Return
            stain_density (np.ndarray): (pixels, N stains)
        """
        size = tuple(image.shape[:2])
        digest = hash_image(Image.fromarray(image), size=size)
        if digest in self.hash2stain_density:
            return self.hash2stain_density[digest]

        od_image = rgb_to_od(image).reshape((-1, 3))
        stain_density = (
            spams.lasso(
                od_image.T,
                D=stain_matrix.T,
                mode=2,
                lambda1=0.01,
                pos=True,
                numThreads=self.numThreads,
            )
            .toarray()
            .T
        )

        self.hash2stain_density[digest] = stain_density
        return stain_density

    def fit(self, image: np.ndarray) -> None:
        """image(RGB array)

        Args:
            image (np.ndarray, uint8)

        """
        target_image = self.standardize_brightness(image)
        self.stain_matrix = self.get_stain_matrix(target_image)
        return

    def transform(self, image: np.ndarray):
        """image(RGB array)

        Args:
            image (np.ndarray, uint8)

        """
        src_image = self.standardize_brightness(image)
        src_stain_matrix = self.get_stain_matrix(src_image)
        src_stain_density = self.get_stain_density(src_image, src_stain_matrix)

        reconstruction = np.dot(src_stain_density, self.stain_matrix).reshape(
            image.shape
        )

        out = od_to_rgb(reconstruction)
        return self.standardize_brightness(out)
