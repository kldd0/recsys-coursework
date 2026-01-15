"""
Модуль для загрузки метаданных товаров и эмбеддингов.
"""

import json
import logging
import os
from typing import Dict, List, Optional
import numpy as np
import pickle
from PIL import Image

logger = logging.getLogger(__name__)


class DataLoader:
    """Класс для загрузки данных о товарах."""

    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        images_dir: str,
        embeddings_dir: str,
        pickle_metadata_path: Optional[str] = None,
    ):
        """
        Инициализация загрузчика данных.

        Args:
            data_dir: Корневая директория с данными
            metadata_file: Путь к файлу метаданных (JSON)
            images_dir: Директория с изображениями
            embeddings_dir: Директория с эмбеддингами
            pickle_metadata_path: Путь к pickle файлу метаданных из ноутбука (опционально)
        """
        self.data_dir = data_dir
        self.metadata_file = metadata_file
        self.images_dir = images_dir
        self.embeddings_dir = embeddings_dir
        # Определение пути к pickle файлу метаданных
        # Проверяем несколько возможных путей
        possible_paths = []
        if pickle_metadata_path:
            possible_paths.append(pickle_metadata_path)
        # Добавляем возможные пути по умолчанию
        possible_paths.extend(
            [
                os.path.join(embeddings_dir, "metadata.pkl"),
                os.path.join(data_dir, "metadata.pkl"),
                os.path.join(data_dir, "embeddings", "metadata.pkl"),
                os.path.join(os.path.dirname(data_dir), "models", "metadata.pkl"),
            ]
        )

        # Ищем существующий файл
        self.pickle_metadata_path = None
        for path in possible_paths:
            if path and os.path.exists(path):
                self.pickle_metadata_path = os.path.abspath(path)
                break

        self._metadata = None
        self._visual_embeddings = None
        self._text_embeddings = None
        self._embedding_id_to_product_id = {}
        self._pickle_adapter = None
        self._use_pickle = False

        # Проверяем наличие pickle файла и пытаемся загрузить
        if self.pickle_metadata_path:
            try:
                from src.metadata_adapter import PickleMetadataAdapter

                logger.info(
                    f"Попытка загрузить pickle метаданные из: {self.pickle_metadata_path}"
                )
                self._pickle_adapter = PickleMetadataAdapter(self.pickle_metadata_path)
                # Проверяем, что адаптер успешно загрузил данные
                # У PickleMetadataAdapter нет поля _metadata, только _df_filtered и _images
                if (
                    self._pickle_adapter._df_filtered is not None
                    and len(self._pickle_adapter._df_filtered) > 0
                ):
                    self._use_pickle = True
                    product_count = len(self._pickle_adapter._df_filtered)
                    images_count = (
                        len(self._pickle_adapter._images)
                        if self._pickle_adapter._images
                        else 0
                    )
                    logger.info(
                        f"✅ Используется pickle метаданные из {self.pickle_metadata_path}"
                    )
                    logger.info(f"   Товаров в DataFrame: {product_count}")
                    logger.info(f"   Изображений: {images_count}")
                    if images_count > 0 and product_count != images_count:
                        logger.warning(
                            f"⚠️ Количество товаров ({product_count}) не совпадает с количеством изображений ({images_count})"
                        )
                else:
                    logger.warning(
                        "⚠️ Pickle метаданные не загружены (DataFrame пуст), используется JSON"
                    )
                    logger.warning(
                        "⚠️ ВНИМАНИЕ: JSON содержит только 10 товаров, а FAISS индекс имеет 10000!"
                    )
                    self._pickle_adapter = None
                    self._use_pickle = False
            except Exception as e:
                logger.error(f"❌ Не удалось загрузить pickle метаданные: {e}")
                logger.error(f"   Тип ошибки: {type(e).__name__}")
                logger.error(f"   Подробности: {str(e)}")
                import traceback

                logger.debug(f"   Traceback: {traceback.format_exc()}")
                logger.warning("⚠️ Будет использован JSON файл метаданных")
                logger.warning(
                    "⚠️ ВНИМАНИЕ: JSON содержит только 10 товаров, а FAISS индекс имеет 10000!"
                )
                logger.warning(
                    "⚠️ Результаты поиска могут быть пустыми для большинства запросов!"
                )
                self._pickle_adapter = None
                self._use_pickle = False
        else:
            logger.warning(
                "⚠️ Pickle файл метаданных не найден ни в одном из проверенных мест"
            )
            logger.warning("   Проверялись пути:")
            for path in possible_paths:
                logger.warning(f"   - {path}")
            logger.warning(
                "⚠️ Будет использован JSON файл метаданных (только 10 товаров)"
            )

    def load_metadata(self) -> Dict:
        """
        Загрузить метаданные товаров.
        Если используется pickle, возвращает метаданные из pickle адаптера.
        Иначе загружает из JSON файла.

        Returns:
            Dict: Словарь с метаданными товаров
        """
        # Если используем pickle метаданные, формируем словарь из них
        if (
            self._use_pickle
            and self._pickle_adapter
            and self._pickle_adapter._df_filtered is not None
        ):
            if self._metadata is not None:
                return self._metadata

            # Формируем словарь с метаданными из pickle
            products = []
            df_len = len(self._pickle_adapter._df_filtered)

            # Для обратной совместимости создаем структуру как в JSON
            # Но не загружаем все товары в память сразу (их 10000!)
            # Вместо этого создаем минимальную структуру с корректным количеством
            self._metadata = {
                "products": products,  # Пустой список, но структура правильная
                "_pickle_count": df_len,  # Добавляем информацию о реальном количестве
                "_use_pickle": True,
            }

            logger.info(f"Используются pickle метаданные: {df_len} товаров")
            return self._metadata

        # Стандартная логика для JSON метаданных
        if self._metadata is not None:
            return self._metadata

        try:
            if not os.path.exists(self.metadata_file):
                logger.warning(f"Файл метаданных не найден: {self.metadata_file}")
                return {"products": []}

            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

            logger.info(
                f"Загружено {len(self._metadata.get('products', []))} товаров из JSON"
            )
            return self._metadata

        except Exception as e:
            logger.error(f"Ошибка при загрузке метаданных: {e}")
            return {"products": []}

    def load_visual_embeddings(self) -> Optional[np.ndarray]:
        """
        Загрузить визуальные эмбеддинги из файла.

        Returns:
            np.ndarray: Матрица визуальных эмбеддингов или None
        """
        if self._visual_embeddings is not None:
            return self._visual_embeddings

        embeddings_path = os.path.join(self.embeddings_dir, "visual_embeddings.pkl")

        try:
            if not os.path.exists(embeddings_path):
                logger.warning(
                    f"Файл визуальных эмбеддингов не найден: {embeddings_path}"
                )
                return None

            with open(embeddings_path, "rb") as f:
                data = pickle.load(f)

            # Поддержка разных форматов данных
            if isinstance(data, dict):
                # Если это словарь с ключами embedding_id
                self._visual_embeddings = data
            elif isinstance(data, np.ndarray):
                # Если это просто массив
                self._visual_embeddings = data
            else:
                logger.error(f"Неожиданный формат данных: {type(data)}")
                return None

            logger.info("Визуальные эмбеддинги загружены")
            return self._visual_embeddings

        except Exception as e:
            logger.error(f"Ошибка при загрузке визуальных эмбеддингов: {e}")
            return None

    def load_text_embeddings(self) -> Optional[np.ndarray]:
        """
        Загрузить текстовые эмбеддинги из файла.

        Returns:
            np.ndarray: Матрица текстовых эмбеддингов или None
        """
        if self._text_embeddings is not None:
            return self._text_embeddings

        embeddings_path = os.path.join(self.embeddings_dir, "text_embeddings.pkl")

        try:
            if not os.path.exists(embeddings_path):
                logger.warning(
                    f"Файл текстовых эмбеддингов не найден: {embeddings_path}"
                )
                return None

            with open(embeddings_path, "rb") as f:
                data = pickle.load(f)

            # Поддержка разных форматов данных
            if isinstance(data, dict):
                self._text_embeddings = data
            elif isinstance(data, np.ndarray):
                self._text_embeddings = data
            else:
                logger.error(f"Неожиданный формат данных: {type(data)}")
                return None

            logger.info("Текстовые эмбеддинги загружены")
            return self._text_embeddings

        except Exception as e:
            logger.error(f"Ошибка при загрузке текстовых эмбеддингов: {e}")
            return None

    def get_product_image(self, product_id: str) -> Optional[Image.Image]:
        """
        Получить изображение товара по ID.

        Args:
            product_id: ID товара (может быть индексом "0", "1", "8109", ...)

        Returns:
            PIL.Image: Изображение товара или None
        """
        # Если используем pickle метаданные, работаем по индексу
        if self._use_pickle and self._pickle_adapter:
            try:
                # Преобразуем ID в индекс (product_id - это строка с индексом, например "8109")
                index = int(product_id)

                # Проверяем диапазон индекса
                max_index = (
                    len(self._pickle_adapter._images) - 1
                    if self._pickle_adapter._images
                    else -1
                )
                if index < 0 or (max_index >= 0 and index > max_index):
                    logger.debug(
                        f"Индекс {index} вне диапазона [0, {max_index}] для product_id '{product_id}' "
                        f"(изображения: {max_index + 1} штук)"
                    )
                    return None

                image = self._pickle_adapter.get_product_image_by_index(index)
                if image is None:
                    logger.debug(
                        f"Не удалось получить изображение для индекса {index} из pickle адаптера (product_id: '{product_id}')"
                    )
                return image
            except (ValueError, TypeError) as e:
                logger.debug(
                    f"Ошибка преобразования product_id '{product_id}' в индекс для изображения: {e}"
                )
                # Если не удалось преобразовать, пробуем другие способы
                pass
            except Exception as e:
                logger.warning(
                    f"Ошибка при получении изображения для product_id '{product_id}': {e}"
                )
                return None

        # Стандартная логика для JSON метаданных
        metadata = self.load_metadata()
        product = self.get_product_metadata(product_id)

        if product and "image_path" in product:
            image_path = product["image_path"]
            if os.path.exists(image_path):
                try:
                    return Image.open(image_path)
                except Exception as e:
                    logger.warning(
                        f"Не удалось загрузить изображение по пути {image_path}: {e}"
                    )

        # Попробуем стандартные имена файлов
        possible_names = [
            f"product_{product_id}.jpg",
            f"product_{product_id}.png",
            f"product_{product_id.zfill(3)}.jpg",
            f"product_{product_id.zfill(3)}.png",
        ]

        for filename in possible_names:
            image_path = os.path.join(self.images_dir, filename)
            if os.path.exists(image_path):
                try:
                    return Image.open(image_path)
                except Exception as e:
                    logger.warning(
                        f"Не удалось загрузить изображение {image_path}: {e}"
                    )

        logger.warning(f"Изображение для товара {product_id} не найдено")
        return None

    def get_product_metadata(self, product_id: str) -> Optional[Dict]:
        """
        Получить метаданные товара по ID.

        Args:
            product_id: ID товара (может быть индексом "0", "1", "8109", ...)

        Returns:
            Dict: Метаданные товара или None
        """
        # Если используем pickle метаданные, работаем по индексу
        if self._use_pickle and self._pickle_adapter:
            try:
                # Преобразуем ID в индекс (product_id - это строка с индексом, например "8109")
                index = int(product_id)

                # Проверяем диапазон индекса
                max_index = (
                    len(self._pickle_adapter._df_filtered) - 1
                    if self._pickle_adapter._df_filtered is not None
                    else -1
                )
                if index < 0 or (max_index >= 0 and index > max_index):
                    logger.warning(
                        f"Индекс {index} вне диапазона [0, {max_index}] для product_id '{product_id}' "
                        f"(используется pickle метаданные с {max_index + 1} товарами)"
                    )
                    return None

                product_data = self._pickle_adapter.get_product_by_index(index)

                if product_data:
                    # Добавляем дополнительные поля для совместимости
                    product_data["product_id"] = product_id
                    if "description" not in product_data and "name" in product_data:
                        product_data["description"] = product_data["name"]
                    return product_data
                else:
                    logger.warning(
                        f"Не удалось получить метаданные для индекса {index} из pickle адаптера (product_id: '{product_id}')"
                    )
                    return None
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Ошибка преобразования product_id '{product_id}' в индекс: {e}"
                )
                # Если не удалось преобразовать, пробуем другие способы
                pass
            except Exception as e:
                logger.error(
                    f"Неожиданная ошибка при получении метаданных для product_id '{product_id}': {e}"
                )
                import traceback

                logger.debug(f"Traceback: {traceback.format_exc()}")
                return None

        # Стандартная логика для JSON метаданных
        metadata = self.load_metadata()
        products = metadata.get("products", [])

        for product in products:
            if product.get("product_id") == product_id:
                return product

        # Если не нашли и не используем pickle, выводим предупреждение
        if not self._use_pickle:
            logger.debug(
                f"Метаданные для product_id '{product_id}' не найдены в JSON метаданных "
                f"(всего {len(products)} товаров). "
                f"Возможно, используется FAISS индекс с {product_id} векторами, "
                f"но pickle метаданные не загружены."
            )

        return None

    def get_all_product_ids(self) -> List[str]:
        """
        Получить список всех ID товаров.

        Returns:
            List[str]: Список ID товаров
        """
        # Если используем pickle метаданные, генерируем ID из индексов
        if (
            self._use_pickle
            and self._pickle_adapter
            and self._pickle_adapter._df_filtered is not None
        ):
            df_len = len(self._pickle_adapter._df_filtered)
            # Возвращаем список ID как строки индексов: ["0", "1", "2", ..., "9999"]
            return [str(i) for i in range(df_len)]

        # Стандартная логика для JSON метаданных
        metadata = self.load_metadata()
        products = metadata.get("products", [])
        return [p.get("product_id") for p in products if "product_id" in p]
