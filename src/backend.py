"""
Бизнес-логика рекомендательной системы.
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from src.vector_store import VectorStore
from src.data_loader import DataLoader
from src.ml_interface import MLInterface
from src.image_processor import ImageProcessor
import config

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Ядро рекомендательной системы."""

    def __init__(
        self,
        vector_store: VectorStore,
        data_loader: DataLoader,
        ml_interface: MLInterface,
    ):
        """
        Инициализация ядра рекомендаций.

        Args:
            vector_store: Векторное хранилище (FAISS/Qdrant)
            data_loader: Загрузчик данных
            ml_interface: Интерфейс ML модели
        """
        self.vector_store = vector_store
        self.data_loader = data_loader
        self.ml_interface = ml_interface
        self.image_processor = ImageProcessor()
        logger.info("RecommendationEngine инициализирован")

    def get_visual_recommendations(
        self, image: Image.Image, top_k: int = 5
    ) -> List[Dict]:
        """
        Получить рекомендации на основе изображения.

        Args:
            image: PIL Image объект
            top_k: Количество результатов

        Returns:
            List[Dict]: Список словарей с информацией о товарах
        """
        try:
            # Предобработка изображения
            processed_image = self.image_processor.preprocess_image(
                image, target_size=(config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE)
            )

            # Получение эмбеддинга от ML модели
            query_embedding = self.ml_interface.encode_image(processed_image)

            # Убеждаемся, что эмбеддинг в правильном формате (float32, contiguous)
            import numpy as np

            if query_embedding.dtype != np.float32:
                query_embedding = query_embedding.astype(np.float32)
            if not query_embedding.flags["C_CONTIGUOUS"]:
                query_embedding = np.ascontiguousarray(query_embedding)

            # Поиск в векторном хранилище
            search_results = self.vector_store.search(query_embedding, top_k=top_k)

            # Получение метаданных товаров
            recommendations = []
            for product_id, similarity in search_results:
                product_metadata = self.data_loader.get_product_metadata(product_id)
                if product_metadata is None:
                    logger.warning(f"Метаданные для товара {product_id} не найдены")
                    continue

                product_image = self.data_loader.get_product_image(product_id)

                recommendation = {
                    "product_id": product_id,
                    "name": product_metadata.get("name", "Неизвестный товар"),
                    "category": product_metadata.get("category", ""),
                    "brand": product_metadata.get("brand", ""),
                    "description": product_metadata.get("description", ""),
                    "color": product_metadata.get("color", ""),
                    "size": product_metadata.get("size", ""),
                    "price": product_metadata.get("price", ""),
                    "similarity": similarity * 100,  # Преобразование в проценты
                    "image": product_image,
                }
                recommendations.append(recommendation)

            logger.info(f"Найдено {len(recommendations)} визуальных рекомендаций")
            return recommendations

        except Exception as e:
            logger.error(f"Ошибка при получении визуальных рекомендаций: {e}")
            raise

    def get_text_recommendations(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Получить рекомендации на основе текстового запроса.

        Args:
            query: Текстовая строка запроса
            top_k: Количество результатов

        Returns:
            List[Dict]: Список словарей с информацией о товарах
        """
        try:
            # Валидация запроса
            query = query.strip()
            if len(query) < 3:
                logger.warning("Текстовый запрос слишком короткий")
                return []

            # Получение эмбеддинга от ML модели
            query_embedding = self.ml_interface.encode_text(query)

            # Убеждаемся, что эмбеддинг в правильном формате (float32, contiguous)
            import numpy as np

            if query_embedding.dtype != np.float32:
                query_embedding = query_embedding.astype(np.float32)
            if not query_embedding.flags["C_CONTIGUOUS"]:
                query_embedding = np.ascontiguousarray(query_embedding)

            # Поиск в векторном хранилище
            search_results = self.vector_store.search(query_embedding, top_k=top_k)

            # Получение метаданных товаров
            recommendations = []
            for product_id, similarity in search_results:
                product_metadata = self.data_loader.get_product_metadata(product_id)
                if product_metadata is None:
                    logger.warning(f"Метаданные для товара {product_id} не найдены")
                    continue

                product_image = self.data_loader.get_product_image(product_id)

                recommendation = {
                    "product_id": product_id,
                    "name": product_metadata.get("name", "Неизвестный товар"),
                    "category": product_metadata.get("category", ""),
                    "brand": product_metadata.get("brand", ""),
                    "description": product_metadata.get("description", ""),
                    "color": product_metadata.get("color", ""),
                    "size": product_metadata.get("size", ""),
                    "price": product_metadata.get("price", ""),
                    "similarity": similarity * 100,  # Преобразование в проценты
                    "image": product_image,
                }
                recommendations.append(recommendation)

            logger.info(f"Найдено {len(recommendations)} текстовых рекомендаций")
            return recommendations

        except Exception as e:
            logger.error(f"Ошибка при получении текстовых рекомендаций: {e}")
            raise

    def get_hybrid_recommendations(
        self,
        image: Image.Image,
        text: str,
        top_k: int = 5,
        visual_weight: float = 0.5,
        text_weight: float = 0.5,
    ) -> List[Dict]:
        """
        Комбинированный поиск по изображению и тексту.

        Args:
            image: PIL Image объект
            text: Текстовая строка
            top_k: Количество результатов
            visual_weight: Вес визуального поиска (0-1)
            text_weight: Вес текстового поиска (0-1)

        Returns:
            List[Dict]: Список словарей с информацией о товарах
        """
        try:
            # Нормализация весов
            total_weight = visual_weight + text_weight
            if total_weight > 0:
                visual_weight /= total_weight
                text_weight /= total_weight

            # Параллельное выполнение обоих поисков
            with ThreadPoolExecutor(max_workers=2) as executor:
                visual_future = executor.submit(
                    self.get_visual_recommendations, image, top_k * 2
                )
                text_future = executor.submit(
                    self.get_text_recommendations, text, top_k * 2
                )

                try:
                    visual_results = visual_future.result(
                        timeout=config.SEARCH_TIMEOUT_SEC
                    )
                    text_results = text_future.result(timeout=config.SEARCH_TIMEOUT_SEC)
                except FutureTimeoutError:
                    logger.error("Таймаут при гибридном поиске")
                    return []

            # Объединение результатов
            product_scores = {}

            # Добавление визуальных результатов
            for product in visual_results:
                product_id = product["product_id"]
                if product_id not in product_scores:
                    product_scores[product_id] = product.copy()
                    product_scores[product_id]["similarity"] = 0.0
                product_scores[product_id]["similarity"] += (
                    product["similarity"] * visual_weight
                )

            # Добавление текстовых результатов
            for product in text_results:
                product_id = product["product_id"]
                if product_id not in product_scores:
                    product_scores[product_id] = product.copy()
                    product_scores[product_id]["similarity"] = 0.0
                product_scores[product_id]["similarity"] += (
                    product["similarity"] * text_weight
                )

            # Сортировка по итоговой схожести
            recommendations = sorted(
                product_scores.values(), key=lambda x: x["similarity"], reverse=True
            )[:top_k]

            logger.info(f"Найдено {len(recommendations)} гибридных рекомендаций")
            return recommendations

        except Exception as e:
            logger.error(f"Ошибка при получении гибридных рекомендаций: {e}")
            raise

    def rank_results(
        self, results: List[Dict], weights: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Ранжирование результатов с учетом разных метрик.

        Args:
            results: Список результатов
            weights: Словарь весов для разных метрик (опционально)

        Returns:
            List[Dict]: Отсортированные результаты
        """
        if weights is None:
            # Простая сортировка по similarity
            return sorted(results, key=lambda x: x.get("similarity", 0), reverse=True)

        # Взвешенное ранжирование (можно расширить в будущем)
        for result in results:
            weighted_score = 0.0
            total_weight = 0.0

            if "similarity" in weights:
                weighted_score += result.get("similarity", 0) * weights["similarity"]
                total_weight += weights["similarity"]

            if total_weight > 0:
                result["weighted_score"] = weighted_score / total_weight

        return sorted(results, key=lambda x: x.get("weighted_score", 0), reverse=True)
