"""
Модуль для кэширования ресурсов Streamlit.
"""

import logging
import streamlit as st
import numpy as np

from src.vector_store import FAISSVectorStore, QdrantVectorStore, VectorStore
from src.data_loader import DataLoader
from src.ml_interface import MLInterface
from src.backend import RecommendationEngine
import config

logger = logging.getLogger(__name__)


@st.cache_resource
def load_vector_store() -> VectorStore:
    """
    Кэширование загрузки векторного хранилища.

    Returns:
        VectorStore: Инициализированное векторное хранилище
    """
    logger.info("Загрузка векторного хранилища...")

    if config.VECTOR_STORE_TYPE == "faiss":
        vector_store = FAISSVectorStore(
            dimension=config.VECTOR_DIMENSION, metric=config.VECTOR_METRIC
        )

        # Попытка загрузить существующий индекс
        index_path = f"{config.VECTOR_STORE_PATH}/faiss_index.bin"
        try:
            import os

            if os.path.exists(index_path):
                vector_store.load_from_file(index_path)
                logger.info("FAISS индекс загружен из файла")
        except Exception as e:
            logger.warning(f"Не удалось загрузить индекс из файла: {e}")

        return vector_store

    elif config.VECTOR_STORE_TYPE == "qdrant":
        vector_store = QdrantVectorStore(
            collection_name=config.QDRANT_COLLECTION,
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            dimension=config.VECTOR_DIMENSION,
        )
        return vector_store

    else:
        raise ValueError(f"Неподдерживаемый тип хранилища: {config.VECTOR_STORE_TYPE}")


@st.cache_resource
def load_data_loader() -> DataLoader:
    """
    Кэширование загрузки данных.

    Returns:
        DataLoader: Инициализированный загрузчик данных
    """
    logger.info("Загрузка загрузчика данных...")

    data_loader = DataLoader(
        data_dir=config.DATA_DIR,
        metadata_file=config.METADATA_FILE,
        images_dir=config.IMAGES_DIR,
        embeddings_dir=config.EMBEDDINGS_DIR,
        pickle_metadata_path=getattr(config, "PICKLE_METADATA_FILE", None),
    )

    return data_loader


@st.cache_resource
def load_ml_interface() -> MLInterface:
    """
    Кэширование инициализации ML интерфейса.

    Returns:
        MLInterface: Инициализированный интерфейс ML модели
    """
    logger.info("Загрузка ML интерфейса...")

    # Попытка загрузить реальную CLIP модель
    try:
        from models.clothing_model import CLIPEmbeddingModel

        # Инициализация CLIP модели
        model_name = getattr(config, "MODEL_NAME", "openai/clip-vit-base-patch32")
        clip_model = CLIPEmbeddingModel(
            model_name=model_name, device=None  # Автоматический выбор устройства
        )

        # Создание интерфейса с реальной моделью
        ml_interface = MLInterface(embedding_dim=clip_model.embedding_dim)
        ml_interface.set_model(clip_model)

        logger.info("CLIP модель успешно загружена и интегрирована")
        return ml_interface

    except ImportError:
        logger.warning("CLIP модель не найдена, используется dummy модель")
        # Fallback на dummy модель
        ml_interface = MLInterface(
            model_path=config.MODEL_PATH, embedding_dim=config.EMBEDDING_DIM
        )
        return ml_interface
    except Exception as e:
        logger.error(f"Ошибка при загрузке CLIP модели: {e}")
        logger.warning("Используется dummy модель")
        # Fallback на dummy модель
        ml_interface = MLInterface(
            model_path=config.MODEL_PATH, embedding_dim=config.EMBEDDING_DIM
        )
        return ml_interface


@st.cache_resource
def load_recommendation_engine() -> RecommendationEngine:
    """
    Кэширование инициализации ядра рекомендаций.

    Returns:
        RecommendationEngine: Инициализированное ядро рекомендаций
    """
    logger.info("Инициализация RecommendationEngine...")

    vector_store = load_vector_store()
    data_loader = load_data_loader()
    ml_interface = load_ml_interface()

    engine = RecommendationEngine(
        vector_store=vector_store, data_loader=data_loader, ml_interface=ml_interface
    )

    return engine


@st.cache_data
def load_metadata() -> dict:
    """
    Кэширование метаданных товаров.

    Returns:
        dict: Метаданные товаров
    """
    data_loader = load_data_loader()
    return data_loader.load_metadata()
