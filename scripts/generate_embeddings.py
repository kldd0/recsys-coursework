"""
Скрипт для генерации эмбеддингов всех товаров с использованием реальной ML модели.
Запустите этот скрипт после интеграции реальной модели для создания векторного индекса.
"""

import sys
import os
import numpy as np
import pickle
from tqdm import tqdm

# Добавление корневой директории в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.vector_store import FAISSVectorStore
import config

# Импорт модели (адаптируйте под вашу модель)
try:
    from models.clothing_model import ClothingEmbeddingModel
except ImportError:
    print("⚠️  Модель не найдена. Создайте models/clothing_model.py")
    print("См. инструкцию в docs/MODEL_INTEGRATION.md")
    sys.exit(1)


def generate_embeddings_for_products():
    """
    Генерация эмбеддингов для всех товаров из метаданных.
    """
    print("Инициализация модели...")

    # Загрузка модели
    model = ClothingEmbeddingModel(
        model_path=(
            config.MODEL_PATH
            if hasattr(config, "MODEL_PATH") and config.MODEL_PATH
            else "models/clothing_embedding_model.pth"
        ),
        device="cpu",  # или "cuda" если есть GPU
    )

    print(f"Модель загружена. Размерность эмбеддинга: {model.embedding_dim}")

    # Загрузка данных
    data_loader = DataLoader(
        data_dir=config.DATA_DIR,
        metadata_file=config.METADATA_FILE,
        images_dir=config.IMAGES_DIR,
        embeddings_dir=config.EMBEDDINGS_DIR,
    )

    metadata = data_loader.load_metadata()
    products = metadata.get("products", [])

    if len(products) == 0:
        print("⚠️  Нет товаров в метаданных")
        return

    print(f"Найдено {len(products)} товаров")

    # Генерация эмбеддингов
    visual_embeddings = {}
    text_embeddings = {}
    vectors_for_index = []
    product_ids = []

    print("\nГенерация эмбеддингов...")
    for product in tqdm(products):
        product_id = product.get("product_id")
        if not product_id:
            continue

        # Визуальный эмбеддинг
        try:
            product_image = data_loader.get_product_image(product_id)
            if product_image:
                visual_emb = model.encode_image(product_image)
                visual_id = product.get("image_embedding_id", f"visual_{product_id}")
                visual_embeddings[visual_id] = visual_emb
                vectors_for_index.append(visual_emb)
                product_ids.append(product_id)
        except Exception as e:
            print(f"⚠️  Ошибка при обработке изображения товара {product_id}: {e}")
            continue

        # Текстовый эмбеддинг
        try:
            # Используем описание товара для текстового эмбеддинга
            text_description = f"{product.get('name', '')} {product.get('description', '')} {product.get('category', '')}"
            text_emb = model.encode_text(text_description)
            text_id = product.get("text_embedding_id", f"text_{product_id}")
            text_embeddings[text_id] = text_emb
        except Exception as e:
            print(f"⚠️  Ошибка при обработке текста товара {product_id}: {e}")
            continue

    # Сохранение эмбеддингов
    print("\nСохранение эмбеддингов...")
    os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)

    visual_path = os.path.join(config.EMBEDDINGS_DIR, "visual_embeddings.pkl")
    text_path = os.path.join(config.EMBEDDINGS_DIR, "text_embeddings.pkl")

    with open(visual_path, "wb") as f:
        pickle.dump(visual_embeddings, f)
    print(f"Визуальные эмбеддинги сохранены: {visual_path}")

    with open(text_path, "wb") as f:
        pickle.dump(text_embeddings, f)
    print(f"Текстовые эмбеддинги сохранены: {text_path}")

    # Создание FAISS индекса
    if vectors_for_index:
        print("\nСоздание FAISS индекса...")
        vectors_array = np.array(vectors_for_index)

        # Обновление размерности в конфиге если нужно
        if vectors_array.shape[1] != config.VECTOR_DIMENSION:
            print(
                f"⚠️  Размерность эмбеддинга ({vectors_array.shape[1]}) не совпадает с конфигом ({config.VECTOR_DIMENSION})"
            )
            print(f"Обновите VECTOR_DIMENSION в config.py или .env")

        visual_store = FAISSVectorStore(
            dimension=vectors_array.shape[1], metric=config.VECTOR_METRIC
        )

        visual_store.add_vectors(vectors_array, product_ids)

        # Сохранение индекса
        os.makedirs(config.VECTOR_STORE_PATH, exist_ok=True)
        index_path = os.path.join(config.VECTOR_STORE_PATH, "faiss_index.bin")
        visual_store.save_to_file(index_path)
        print(f"FAISS индекс сохранен: {index_path}")

    print("\n✅ Генерация эмбеддингов завершена!")
    print(f"   - Обработано товаров: {len(product_ids)}")
    print(f"   - Визуальных эмбеддингов: {len(visual_embeddings)}")
    print(f"   - Текстовых эмбеддингов: {len(text_embeddings)}")


if __name__ == "__main__":
    generate_embeddings_for_products()
