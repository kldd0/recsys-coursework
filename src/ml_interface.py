"""
Интерфейс для подключения ML модели эмбеддингов.
"""

import logging
from typing import Tuple, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class DummyEmbeddingModel:
    """
    Dummy модель для тестирования UI.
    Генерирует случайные эмбеддинги.
    """

    def __init__(self, embedding_dim: int = 512):
        """
        Инициализация dummy модели.

        Args:
            embedding_dim: Размерность эмбеддинга
        """
        self.embedding_dim = embedding_dim

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Генерация случайного визуального эмбеддинга.

        Args:
            image: PIL Image объект

        Returns:
            np.ndarray: Вектор эмбеддинга
        """
        # Нормализация для стабильности
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def encode_text(self, text: str) -> np.ndarray:
        """
        Генерация случайного текстового эмбеддинга.

        Args:
            text: Текстовая строка

        Returns:
            np.ndarray: Вектор эмбеддинга
        """
        # Нормализация для стабильности
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding


class MLInterface:
    """
    Интерфейс для работы с ML моделью эмбеддингов.
    Позволяет легко заменить dummy модель на реальную.
    """

    def __init__(self, model_path: Optional[str] = None, embedding_dim: int = 512):
        """
        Инициализация интерфейса ML модели.

        Args:
            model_path: Путь к модели (опционально)
            embedding_dim: Размерность эмбеддинга
        """
        self.model = None
        self.embedding_dim = embedding_dim
        self.model_path = model_path

        # Если модель не предоставлена, используем dummy модель
        if model_path is None:
            logger.info("Используется dummy модель для тестирования")
            self.model = DummyEmbeddingModel(embedding_dim=embedding_dim)

    def set_model(self, model):
        """
        Установить модель в runtime.

        Args:
            model: Объект модели с методами encode_image и encode_text
        """
        if not hasattr(model, "encode_image") or not hasattr(model, "encode_text"):
            raise ValueError("Модель должна иметь методы encode_image и encode_text")
        self.model = model
        logger.info("ML модель успешно установлена")

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Получить визуальный эмбеддинг изображения.

        Args:
            image: PIL Image объект

        Returns:
            np.ndarray: Вектор эмбеддинга размера [embedding_dim]
        """
        if self.model is None:
            raise RuntimeError("Модель не инициализирована. Используйте set_model()")

        try:
            embedding = self.model.encode_image(image)
            if embedding.shape[0] != self.embedding_dim:
                raise ValueError(
                    f"Размерность эмбеддинга не совпадает: "
                    f"ожидается {self.embedding_dim}, получено {embedding.shape[0]}"
                )
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Ошибка при кодировании изображения: {e}")
            raise

    def encode_text(self, text: str) -> np.ndarray:
        """
        Получить текстовый эмбеддинг.

        Args:
            text: Текстовая строка

        Returns:
            np.ndarray: Вектор эмбеддинга размера [embedding_dim]
        """
        if self.model is None:
            raise RuntimeError("Модель не инициализирована. Используйте set_model()")

        try:
            embedding = self.model.encode_text(text)
            if embedding.shape[0] != self.embedding_dim:
                raise ValueError(
                    f"Размерность эмбеддинга не совпадает: "
                    f"ожидается {self.embedding_dim}, получено {embedding.shape[0]}"
                )
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Ошибка при кодировании текста: {e}")
            raise

    def encode_both(
        self, image: Image.Image, text: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Параллельное кодирование обоих модальностей.

        Args:
            image: PIL Image объект
            text: Текстовая строка

        Returns:
            Tuple[np.ndarray, np.ndarray]: (визуальный эмбеддинг, текстовый эмбеддинг)
        """
        visual_embedding = self.encode_image(image)
        text_embedding = self.encode_text(text)
        return visual_embedding, text_embedding
