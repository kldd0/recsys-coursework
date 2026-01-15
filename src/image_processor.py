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
    SUPPORTED_FORMATS = ['JPEG', 'JPG', 'PNG', 'WebP']
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
            if width < ImageProcessor.MIN_RESOLUTION[0] or height < ImageProcessor.MIN_RESOLUTION[1]:
                logger.warning(f"Изображение слишком маленькое: {width}x{height}")
                return False
            
            # Проверка целостности
            image.verify()
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при валидации изображения: {e}")
            return False
    
    @staticmethod
    def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
        """
        Предварительная обработка изображения для ML модели.
        
        Args:
            image: PIL Image объект
            target_size: Целевой размер (ширина, высота)
            
        Returns:
            PIL.Image: Обработанное изображение
        """
        # Конвертация в RGB если необходимо
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Изменение размера с сохранением пропорций
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Создание нового изображения с целевым размером (центрирование)
        new_image = Image.new('RGB', target_size, (255, 255, 255))
        paste_x = (target_size[0] - image.size[0]) // 2
        paste_y = (target_size[1] - image.size[1]) // 2
        new_image.paste(image, (paste_x, paste_y))
        
        return new_image
    
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
