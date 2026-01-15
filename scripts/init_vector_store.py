"""
Скрипт для инициализации векторного хранилища с тестовыми данными.
Создает dummy эмбеддинги и заполняет FAISS индекс.
"""

import sys
import os
import numpy as np
import pickle
import json

# Добавление корневой директории в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store import FAISSVectorStore
from src.data_loader import DataLoader
import config


def create_dummy_embeddings(num_products: int, embedding_dim: int = 512):
    """
    Создать dummy эмбеддинги для тестирования.

    Args:
        num_products: Количество товаров
        embedding_dim: Размерность эмбеддинга

    Returns:
        tuple: (visual_embeddings, text_embeddings, embedding_ids)
    """
    print(f"Создание dummy эмбеддингов для {num_products} товаров...")

    visual_embeddings = {}
    text_embeddings = {}
    embedding_ids = []

    for i in range(1, num_products + 1):
        product_id = f"{i:03d}"
        visual_id = f"visual_{product_id}"
        text_id = f"text_{product_id}"

        # Генерация случайных нормализованных эмбеддингов
        visual_emb = np.random.randn(embedding_dim).astype(np.float32)
        visual_emb = visual_emb / np.linalg.norm(visual_emb)

        text_emb = np.random.randn(embedding_dim).astype(np.float32)
        text_emb = text_emb / np.linalg.norm(text_emb)

        visual_embeddings[visual_id] = visual_emb
        text_embeddings[text_id] = text_emb
        embedding_ids.append((product_id, visual_id, text_id))

    return visual_embeddings, text_embeddings, embedding_ids


def save_embeddings(visual_embeddings: dict, text_embeddings: dict):
    """Сохранить эмбеддинги в файлы."""
    os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)

    visual_path = os.path.join(config.EMBEDDINGS_DIR, "visual_embeddings.pkl")
    text_path = os.path.join(config.EMBEDDINGS_DIR, "text_embeddings.pkl")

    with open(visual_path, "wb") as f:
        pickle.dump(visual_embeddings, f)
    print(f"Визуальные эмбеддинги сохранены: {visual_path}")

    with open(text_path, "wb") as f:
        pickle.dump(text_embeddings, f)
    print(f"Текстовые эмбеддинги сохранены: {text_path}")


def init_vector_store():
    """Инициализировать векторное хранилище."""
    print("Инициализация векторного хранилища...")

    # Загрузка метаданных
    data_loader = DataLoader(
        data_dir=config.DATA_DIR,
        metadata_file=config.METADATA_FILE,
        images_dir=config.IMAGES_DIR,
        embeddings_dir=config.EMBEDDINGS_DIR,
    )

    metadata = data_loader.load_metadata()
    products = metadata.get("products", [])
    num_products = len(products)

    if num_products == 0:
        print("⚠️  Нет товаров в метаданных. Создайте файл products_metadata.json")
        return

    print(f"Найдено {num_products} товаров в метаданных")

    # Создание dummy эмбеддингов
    visual_embeddings, text_embeddings, embedding_ids = create_dummy_embeddings(
        num_products, config.VECTOR_DIMENSION
    )

    # Сохранение эмбеддингов
    save_embeddings(visual_embeddings, text_embeddings)

    # Инициализация FAISS индекса для визуальных эмбеддингов
    print("\nСоздание FAISS индекса для визуальных эмбеддингов...")
    visual_store = FAISSVectorStore(
        dimension=config.VECTOR_DIMENSION, metric=config.VECTOR_METRIC
    )

    # Подготовка векторов и ID
    vectors = []
    ids = []

    for product_id, visual_id, text_id in embedding_ids:
        if visual_id in visual_embeddings:
            vectors.append(visual_embeddings[visual_id])
            ids.append(product_id)

    if vectors:
        vectors_array = np.array(vectors)
        visual_store.add_vectors(vectors_array, ids)

        # Сохранение индекса
        os.makedirs(config.VECTOR_STORE_PATH, exist_ok=True)
        index_path = os.path.join(config.VECTOR_STORE_PATH, "faiss_index.bin")
        visual_store.save_to_file(index_path)
        print(f"FAISS индекс сохранен: {index_path}")

    print("\n✅ Инициализация завершена!")
    print(f"   - Эмбеддинги: {config.EMBEDDINGS_DIR}")
    print(f"   - Индекс: {index_path}")
    print(f"   - Товаров: {num_products}")


if __name__ == "__main__":
    init_vector_store()
