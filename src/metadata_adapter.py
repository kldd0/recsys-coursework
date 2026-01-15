"""
Адаптер для работы с метаданными из pickle файла (ноутбук).
Конвертирует данные из формата ноутбука в формат приложения.

Структура pickle файла (из ноутбука):
    metadata = {'df_filtered': df_filtered, 'images': images}
    - df_filtered: pandas DataFrame с колонкой 'caption' и другими полями
    - images: список PIL.Image (10000 изображений)
"""
import pickle
import os
import logging
from typing import Dict, Optional
import pandas as pd
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class PickleMetadataAdapter:
    """
    Адаптер для загрузки метаданных из pickle файла ноутбука.
    """
    
    def __init__(self, pickle_path: str):
        """
        Инициализация адаптера.
        
        Args:
            pickle_path: Путь к pickle файлу с метаданными
        """
        self.pickle_path = os.path.abspath(pickle_path)
        self._df_filtered: Optional[pd.DataFrame] = None
        self._images: Optional[list] = None
        self._load_metadata()
    
    def _load_metadata(self):
        """Загрузка метаданных из pickle файла (точно как в ноутбуке)."""
        if not os.path.exists(self.pickle_path):
            logger.warning(f"Файл метаданных не найден: {self.pickle_path}")
            return
        
        try:
            logger.info(f"Загрузка метаданных из: {self.pickle_path}")
            
            # Загружаем метаданные (точно как в ноутбуке)
            # Структура: metadata = {'df_filtered': df_filtered, 'images': images}
            with open(self.pickle_path, "rb") as f:
                metadata = pickle.load(f)
            
            # Извлекаем данные (точно как в ноутбуке)
            self._df_filtered = metadata['df_filtered']
            self._images = metadata['images']
            
            logger.info(f"✅ Метаданные загружены из {os.path.basename(self.pickle_path)}")
            logger.info(f"  - Датафрейм: {len(self._df_filtered)} записей")
            logger.info(f"  - Изображения: {len(self._images)} штук")
                
        except (ValueError, AttributeError) as e:
            error_str = str(e)
            if "not enough values to unpack" in error_str or "JpegImagePlugin" in error_str or "__setstate__" in error_str:
                logger.error("=" * 60)
                logger.error("ПРОБЛЕМА СОВМЕСТИМОСТИ PILLOW")
                logger.error("=" * 60)
                logger.error("Ошибка возникает из-за несовместимости версий Pillow:")
                logger.error("  - Pickle файл создан с одной версией Pillow")
                logger.error("  - Текущая версия Pillow отличается")
                logger.error("")
                logger.error("РЕШЕНИЕ: Пересоздайте metadata.pkl в ноутбуке")
                logger.error("")
                logger.error("В ноутбуке выполните:")
                logger.error("  1. Проверьте версию Pillow: import PIL; print(PIL.__version__)")
                logger.error("  2. Убедитесь, что версия совпадает с проектом")
                logger.error("  3. Пересоздайте pickle файл:")
                logger.error("     metadata = {'df_filtered': df_filtered, 'images': images}")
                logger.error("     with open('metadata.pkl', 'wb') as f:")
                logger.error("         pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)")
                logger.error("")
                logger.error(f"Оригинальная ошибка: {e}")
                logger.error("=" * 60)
                raise ValueError(
                    "Несовместимость версий Pillow. "
                    "Пересоздайте metadata.pkl в ноутбуке с текущей версией Pillow. "
                    f"Оригинальная ошибка: {e}"
                ) from e
            raise
        except (ModuleNotFoundError, AttributeError, ImportError) as e:
            error_str = str(e)
            if 'numpy._core' in error_str or 'numpy.core' in error_str:
                logger.error("Проблема совместимости numpy")
                logger.error(f"Ошибка: {e}")
                logger.error("Решение: Пересоздайте pickle файл с numpy < 2.0")
                raise ImportError(
                    f"Несовместимость версий numpy. "
                    f"Пересоздайте pickle файл с numpy 1.x. "
                    f"Оригинальная ошибка: {e}"
                ) from e
            raise
        except KeyError as e:
            logger.error(f"Ключ не найден в метаданных: {e}")
            logger.error(f"Доступные ключи: {list(metadata.keys()) if 'metadata' in locals() else 'N/A'}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при загрузке метаданных: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def get_product_count(self) -> int:
        """Получить количество товаров."""
        if self._df_filtered is not None:
            return len(self._df_filtered)
        return 0
    
    def get_product_by_index(self, index: int) -> Optional[Dict]:
        """
        Получить данные товара по индексу (соответствует индексу в FAISS).
        
        Args:
            index: Индекс в FAISS (0-based, соответствует iloc в DataFrame)
            
        Returns:
            Dict: Метаданные товара или None
        """
        if self._df_filtered is None:
            return None
        
        if index < 0 or index >= len(self._df_filtered):
            logger.debug(f"Индекс {index} вне диапазона [0, {len(self._df_filtered)})")
            return None
        
        try:
            row = self._df_filtered.iloc[index]
            
            # Базовые поля
            product_data = {
                'product_id': str(index),
                'index': index,
            }
            
            # Извлекаем caption (обязательное поле из ноутбука)
            if 'caption' in row:
                product_data['name'] = str(row['caption'])
                product_data['description'] = str(row['caption'])
            elif 'name' in row:
                product_data['name'] = str(row['name'])
                product_data['description'] = str(row.get('description', row['name']))
            
            # Дополнительные поля из DataFrame
            for field in ['category', 'brand', 'color', 'size', 'price']:
                if field in row:
                    product_data[field] = str(row[field])
            
            return product_data
            
        except Exception as e:
            logger.warning(f"Ошибка при получении товара по индексу {index}: {e}")
            return None
    
    def get_product_image_by_index(self, index: int) -> Optional[Image.Image]:
        """
        Получить изображение товара по индексу.
        
        Args:
            index: Индекс в FAISS (0-based, соответствует индексу в списке images)
            
        Returns:
            PIL.Image: Изображение товара или None
        """
        if self._images is None:
            return None
        
        if index < 0 or index >= len(self._images):
            logger.debug(f"Индекс {index} вне диапазона [0, {len(self._images)})")
            return None
        
        try:
            image = self._images[index]
            
            # Если это уже PIL Image, возвращаем
            if isinstance(image, Image.Image):
                return image
            
            # Если это numpy array, конвертируем
            if isinstance(image, np.ndarray):
                # Обработка разных форматов numpy arrays
                if image.dtype != np.uint8:
                    # Нормализуем, если нужно
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                
                # Обрабатываем shape
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # RGB
                    return Image.fromarray(image, 'RGB')
                elif len(image.shape) == 3 and image.shape[2] == 4:
                    # RGBA
                    return Image.fromarray(image, 'RGBA')
                elif len(image.shape) == 2:
                    # Grayscale
                    return Image.fromarray(image, 'L')
                else:
                    return Image.fromarray(image)
            
            # Если это байты, загружаем
            if isinstance(image, bytes):
                import io
                return Image.open(io.BytesIO(image))
            
            logger.warning(f"Неизвестный формат изображения для индекса {index}: {type(image)}")
            return None
            
        except Exception as e:
            logger.warning(f"Ошибка при получении изображения по индексу {index}: {e}")
            return None
    
    def get_product_caption_by_index(self, index: int) -> Optional[str]:
        """
        Получить описание товара по индексу.
        
        Args:
            index: Индекс в FAISS (0-based)
            
        Returns:
            str: Описание товара (caption) или None
        """
        if self._df_filtered is None:
            return None
        
        if index < 0 or index >= len(self._df_filtered):
            return None
        
        try:
            row = self._df_filtered.iloc[index]
            if 'caption' in row:
                return str(row['caption'])
            return None
        except Exception as e:
            logger.debug(f"Ошибка при получении описания для индекса {index}: {e}")
            return None
