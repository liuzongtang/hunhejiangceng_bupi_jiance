"""
Image preprocessing for fabric defect detection inference.

Pipeline:
  1. Decode (bytes/array → numpy array)
  2. Resize to model input size
  3. Normalize (mean/std)
  4. Channel conversion (BGR→RGB if needed)
  5. Batch collation
"""

from typing import List, Tuple

import numpy as np

# ImageNet normalization (commonly used by pre-trained models)
DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImagePreprocessor:
    """
    Preprocesses raw images for model inference.

    Handles resize, normalization, channel conversion, and batching.
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (640, 640),
        mean: np.ndarray = DEFAULT_MEAN,
        std: np.ndarray = DEFAULT_STD,
        normalize: bool = True,
    ):
        self.input_size = input_size
        self.mean = mean.reshape(1, 3, 1, 1)
        self.std = std.reshape(1, 3, 1, 1)
        self.normalize = normalize

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image.

        Args:
            image: (H, W, 3) uint8 BGR or RGB image.

        Returns:
            (1, 3, H', W') float32 normalized tensor.
        """
        # Resize
        h, w = image.shape[:2]
        scale = min(self.input_size[0] / w, self.input_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        import cv2

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to input size
        canvas = np.zeros((*self.input_size, 3), dtype=np.float32)
        canvas[:new_h, :new_w] = resized.astype(np.float32)

        # HWC → CHW
        canvas = canvas.transpose(2, 0, 1)

        # Normalize
        if self.normalize:
            canvas = canvas / 255.0
            canvas = (canvas - self.mean.reshape(3, 1, 1)) / self.std.reshape(3, 1, 1)

        return canvas[np.newaxis, ...].astype(np.float32)

    def preprocess_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess a batch of images.

        Args:
            images: List of (H, W, 3) uint8 images.

        Returns:
            (B, 3, H', W') float32 normalized tensor.
        """
        batch = [self.preprocess(img) for img in images]
        return np.concatenate(batch, axis=0)

    def decode_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Decode raw image bytes to numpy array."""
        import cv2

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image bytes")
        return image

    def decode_base64(self, b64_string: str) -> np.ndarray:
        """Decode base64-encoded image string to numpy array."""
        import base64

        image_bytes = base64.b64decode(b64_string)
        return self.decode_bytes(image_bytes)
