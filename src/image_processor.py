"""
Модуль для обработки и валидации изображений.
"""

import logging
from typing import Tuple
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Класс для обработки изображений перед передачей в ML модель."""

    # Поддерживаемые форматы
    SUPPORTED_FORMATS = ["JPEG", "JPG", "PNG", "WebP"]
    MAX_FILE_SIZE_MB = 10
    MIN_RESOLUTION = (224, 224)

    @staticmethod
    def validate_image(file) -> bool:
        """
        Проверить формат и размер загруженного изображения.

        Args:
            file: Файл, загруженный через st.file_uploader

        Returns:
            bool: True если изображение валидно, False иначе
        """
        if file is None:
            return False

        # Проверка размера файла
        file_size_mb = len(file.getvalue()) / (1024 * 1024)
        if file_size_mb > ImageProcessor.MAX_FILE_SIZE_MB:
            logger.warning(f"Файл слишком большой: {file_size_mb:.2f} МБ")
            return False

        try:
            # Проверка формата
            image = Image.open(io.BytesIO(file.getvalue()))
            format_name = image.format.upper() if image.format else ""

            if format_name not in ImageProcessor.SUPPORTED_FORMATS:
                logger.warning(f"Неподдерживаемый формат: {format_name}")
                return False

            # Проверка разрешения
            width, height = image.size
            if (
                width < ImageProcessor.MIN_RESOLUTION[0]
                or height < ImageProcessor.MIN_RESOLUTION[1]
            ):
                logger.warning(f"Изображение слишком маленькое: {width}x{height}")
                return False

            # Проверка целостности
            image.verify()

            return True

        except Exception as e:
            logger.error(f"Ошибка при валидации изображения: {e}")
            return False

    @staticmethod
    def preprocess_image(
        image: Image.Image, target_size: Tuple[int, int] = (224, 224)
    ) -> Image.Image:
        """
        Предварительная обработка изображения для ML модели.

        Args:
            image: PIL Image объект
            target_size: Целевой размер (ширина, высота)

        Returns:
            PIL.Image: Обработанное изображение
        """
        try:
            import numpy as np

            img_array = np.array(image)

            if len(img_array.shape) == 2:
                # Grayscale -> RGB
                img_array = np.stack([img_array] * 3, axis=-1)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                # RGBA -> RGB
                img_array = img_array[:, :, :3]
            elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
                # Single channel -> RGB
                img_array = np.repeat(img_array, 3, axis=2)

            image = Image.fromarray(img_array.astype(np.uint8), "RGB")

            image.thumbnail(target_size, Image.Resampling.LANCZOS)

            new_image = Image.new("RGB", target_size, (255, 255, 255))
            paste_x = (target_size[0] - image.size[0]) // 2
            paste_y = (target_size[1] - image.size[1]) // 2
            new_image.paste(image, (paste_x, paste_y))

            return new_image

        except Exception as e:
            logger.error(f"Ошибка при предобработке изображения: {e}")
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image.resize(target_size, Image.Resampling.LANCZOS)

    @staticmethod
    def convert_to_array(image: Image.Image) -> np.ndarray:
        """
        Конвертировать PIL Image в numpy array.

        Args:
            image: PIL Image объект

        Returns:
            np.ndarray: Массив изображения в формате (H, W, C)
        """
        return np.array(image)

    @staticmethod
    def load_from_bytes(file_bytes: bytes) -> Image.Image:
        """
        Загрузить изображение из bytes.

        Args:
            file_bytes: Байты изображения

        Returns:
            PIL.Image: Загруженное изображение
        """
        return Image.open(io.BytesIO(file_bytes))
