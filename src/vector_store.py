"""
Модуль для управления векторным хранилищем (FAISS/Qdrant).
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


def _ensure_contiguous_float32(array: np.ndarray) -> np.ndarray:
    """
    Убедиться, что массив в правильном формате для FAISS.
    FAISS требует contiguous float32 массивы.

    Args:
        array: Входной numpy массив

    Returns:
        np.ndarray: Массив в формате float32, C-contiguous
    """
    # Конвертируем в float32 и делаем contiguous
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)
    return array


class VectorStore(ABC):
    """Абстрактный класс для векторного хранилища."""

    @abstractmethod
    def add_vectors(
        self, vectors: np.ndarray, ids: List[str], metadata: Optional[List[Dict]] = None
    ):
        """
        Добавить векторы с метаданными.

        Args:
            vectors: Матрица векторов (N, D)
            ids: Список ID векторов
            metadata: Опциональные метаданные
        """
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """
        Поиск top-k похожих векторов.

        Args:
            query_vector: Вектор запроса (D,)
            top_k: Количество результатов

        Returns:
            List[Tuple[str, float]]: Список (ID, similarity_score)
        """
        pass


class FAISSVectorStore(VectorStore):
    """Реализация векторного хранилища на основе FAISS."""

    def __init__(self, dimension: int, metric: str = "cosine"):
        """
        Инициализация FAISS индекса.

        Args:
            dimension: Размерность векторов
            metric: Метрика расстояния ('cosine', 'L2', 'inner_product')
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS не установлен. Установите: pip install faiss-cpu или faiss-gpu"
            )

        self.dimension = dimension
        self.metric = metric
        self.index = None
        self.id_to_index = {}  # Маппинг ID товара -> индекс в FAISS
        self.index_to_id = {}  # Обратный маппинг

        # Проверка совместимости numpy версии
        try:
            numpy_version = np.__version__
            logger.info(f"NumPy версия: {numpy_version}")
            # FAISS 1.10.0 не поддерживает numpy 2.x
            if numpy_version.startswith("2."):
                logger.warning(
                    f"⚠️ NumPy {numpy_version} может быть несовместим с faiss-cpu 1.10.0. "
                    f"Рекомендуется использовать numpy < 2.0.0"
                )
        except Exception:
            pass

        # Инициализация индекса
        try:
            if metric == "cosine":
                # Для cosine similarity используем L2 нормализацию
                self.index = faiss.IndexFlatIP(
                    dimension
                )  # Inner Product для нормализованных векторов
            elif metric == "L2":
                self.index = faiss.IndexFlatL2(dimension)
            else:
                raise ValueError(f"Неподдерживаемая метрика: {metric}")

            logger.info(
                f"FAISS индекс инициализирован: dimension={dimension}, metric={metric}"
            )
        except Exception as e:
            logger.error(f"Ошибка при инициализации FAISS индекса: {e}")
            raise

    def add_vectors(
        self, vectors: np.ndarray, ids: List[str], metadata: Optional[List[Dict]] = None
    ):
        """
        Добавить векторы в индекс.

        Args:
            vectors: Матрица векторов (N, D)
            ids: Список ID векторов
            metadata: Опциональные метаданные (не используется в FAISS)
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Размерность векторов не совпадает: "
                f"ожидается {self.dimension}, получено {vectors.shape[1]}"
            )

        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"Количество ID не совпадает с количеством векторов: "
                f"{len(ids)} != {vectors.shape[0]}"
            )

        # Подготовка векторов - убеждаемся, что они в правильном формате
        vectors = _ensure_contiguous_float32(vectors)

        # Нормализация для cosine similarity
        if self.metric == "cosine":
            import faiss

            try:
                faiss.normalize_L2(vectors)
            except Exception as e:
                logger.error(f"Ошибка при нормализации векторов: {e}")
                raise

        # Добавление векторов в индекс
        try:
            start_idx = self.index.ntotal
            self.index.add(vectors)
        except Exception as e:
            logger.error(f"Ошибка при добавлении векторов в FAISS: {e}")
            logger.error(
                f"Vectors shape: {vectors.shape}, dtype: {vectors.dtype}, contiguous: {vectors.flags['C_CONTIGUOUS']}"
            )
            raise

        # Обновление маппингов
        for i, vector_id in enumerate(ids):
            idx = start_idx + i
            self.id_to_index[vector_id] = idx
            self.index_to_id[idx] = vector_id

        logger.info(f"Добавлено {len(ids)} векторов в индекс")

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """
        Поиск ближайших соседей.

        Args:
            query_vector: Вектор запроса (D,)
            top_k: Количество результатов

        Returns:
            List[Tuple[str, float]]: Список (ID, similarity_score)
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Индекс пуст")
            return []

        if query_vector.shape[0] != self.dimension:
            raise ValueError(
                f"Размерность вектора запроса не совпадает: "
                f"ожидается {self.dimension}, получено {query_vector.shape[0]}"
            )

        # Подготовка запроса - убеждаемся, что он в правильном формате
        query = query_vector.reshape(1, -1)
        query = _ensure_contiguous_float32(query)

        # Нормализация для cosine similarity
        if self.metric == "cosine":
            import faiss

            try:
                faiss.normalize_L2(query)
            except Exception as e:
                logger.error(f"Ошибка при нормализации запроса: {e}")
                raise

        # Поиск
        try:
            distances, indices = self.index.search(query, min(top_k, self.index.ntotal))
        except Exception as e:
            logger.error(f"Ошибка при поиске в FAISS: {e}")
            logger.error(
                f"Query shape: {query.shape}, dtype: {query.dtype}, contiguous: {query.flags['C_CONTIGUOUS']}"
            )
            raise

        # Преобразование результатов
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS возвращает -1 для пустых результатов
                continue

            # Если есть маппинг, используем его, иначе используем индекс напрямую
            if self.index_to_id:
                vector_id = self.index_to_id.get(idx)
                if vector_id is None:
                    continue
            else:
                # Если маппинга нет, используем индекс как ID
                # Форматируем индекс как строку без ведущих нулей (для больших индексов)
                vector_id = str(idx)

            # Для cosine similarity используем inner product как similarity
            # Для L2 используем отрицательное расстояние (чем меньше расстояние, тем больше similarity)
            if self.metric == "cosine":
                similarity = float(dist)
            else:  # L2
                similarity = float(
                    1.0 / (1.0 + dist)
                )  # Преобразование расстояния в similarity

            results.append((vector_id, similarity))

        return results

    def load_from_file(self, path: str):
        """
        Загрузить индекс из файла.

        Args:
            path: Путь к файлу индекса
        """
        try:
            import faiss
            import pickle

            # Загрузка индекса
            self.index = faiss.read_index(path)

            # Загрузка маппингов (опционально)
            mappings_path = path + ".mappings.pkl"
            if os.path.exists(mappings_path):
                with open(mappings_path, "rb") as f:
                    self.id_to_index, self.index_to_id = pickle.load(f)
                logger.info(f"Маппинги загружены из {mappings_path}")
            else:
                logger.warning(
                    f"Файл маппингов не найден: {mappings_path}. Будет использован прямой доступ по индексу."
                )
                self.id_to_index = {}
                self.index_to_id = {}

            logger.info(
                f"Индекс загружен из {path}: {self.index.ntotal} векторов, размерность {self.index.d}"
            )

        except Exception as e:
            logger.error(f"Ошибка при загрузке индекса: {e}")
            raise

    def save_to_file(self, path: str):
        """
        Сохранить индекс в файл.

        Args:
            path: Путь для сохранения
        """
        try:
            import faiss
            import pickle
            import os

            # Сохранение индекса
            faiss.write_index(self.index, path)

            # Сохранение маппингов
            mappings_path = path + ".mappings.pkl"
            with open(mappings_path, "wb") as f:
                pickle.dump((self.id_to_index, self.index_to_id), f)

            logger.info(f"Индекс сохранен в {path}")

        except Exception as e:
            logger.error(f"Ошибка при сохранении индекса: {e}")
            raise


class QdrantVectorStore(VectorStore):
    """Реализация векторного хранилища на основе Qdrant (для production)."""

    def __init__(
        self,
        collection_name: str,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        dimension: int = 512,
    ):
        """
        Инициализация Qdrant клиента.

        Args:
            collection_name: Имя коллекции
            url: URL Qdrant сервера
            api_key: API ключ (опционально)
            dimension: Размерность векторов
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "Qdrant client не установлен. Установите: pip install qdrant-client"
            )

        self.collection_name = collection_name
        self.dimension = dimension
        self.client = QdrantClient(url=url, api_key=api_key)

        # Создание коллекции если не существует
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=dimension, distance=Distance.COSINE
                    ),
                )
                logger.info(f"Создана коллекция Qdrant: {collection_name}")
            else:
                logger.info(f"Используется существующая коллекция: {collection_name}")

        except Exception as e:
            logger.error(f"Ошибка при инициализации Qdrant: {e}")
            raise

    def add_vectors(
        self, vectors: np.ndarray, ids: List[str], metadata: Optional[List[Dict]] = None
    ):
        """Добавить векторы в Qdrant."""
        from qdrant_client.models import PointStruct

        points = []
        for i, (vector, vector_id) in enumerate(zip(vectors, ids)):
            payload = metadata[i] if metadata else {}
            points.append(
                PointStruct(
                    id=i,
                    vector=vector.tolist(),
                    payload={"product_id": vector_id, **payload},
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

        logger.info(f"Добавлено {len(ids)} векторов в Qdrant")

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Поиск через Qdrant API."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
        )

        return [
            (hit.payload.get("product_id", str(hit.id)), hit.score) for hit in results
        ]
