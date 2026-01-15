# Мультимодальная Рекомендательная Система Гардероба

Streamlit приложение для поиска похожих предметов одежды на основе визуальных и текстовых эмбеддингов.

## Описание

Это приложение позволяет находить похожие товары в каталоге одежды используя:
- **Текстовый поиск**: описание товара текстом (например, "красная футболка найк")
- **Визуальный поиск**: загрузка фотографии товара
- **Гибридный поиск**: комбинация текста и изображения

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd recsys-cw
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Инициализируйте векторное хранилище с тестовыми данными:
```bash
python scripts/init_vector_store.py
```

Этот скрипт создаст dummy эмбеддинги для товаров из `data/products_metadata.json` и инициализирует FAISS индекс.

4. Запустите приложение:
```bash
streamlit run app.py
```

Приложение будет доступно по адресу: `http://localhost:8501`

## Структура проекта

```
recsys-cw/
├── app.py                    # Главный Streamlit файл
├── config.py                 # Конфигурация и переменные окружения
├── requirements.txt          # Зависимости Python
├── .streamlit/
│   └── config.toml           # Конфигурация Streamlit
├── src/
│   ├── __init__.py
│   ├── backend.py            # Бизнес-логика (поиск, интеграция с ML)
│   ├── vector_store.py       # Управление векторным хранилищем (FAISS/Qdrant)
│   ├── data_loader.py        # Загрузка метаданных и эмбеддингов
│   ├── image_processor.py    # Обработка загруженных изображений
│   ├── cache.py              # Кэширование и оптимизация
│   └── ml_interface.py       # Интерфейс для подключения ML модели
├── data/
│   ├── products_metadata.json    # Метаданные товаров
│   ├── embeddings/
│   │   ├── visual_embeddings.pkl
│   │   └── text_embeddings.pkl
│   └── images/
│       ├── product_001.jpg
│       └── ...
├── logs/
│   └── app.log               # Логирование
└── README.md                 # Документация
```

## Использование

1. **Текстовый поиск**: Введите описание товара в текстовое поле (минимум 3 символа)
2. **Визуальный поиск**: Загрузите изображение товара (JPG, PNG, WebP, макс. 10 МБ)
3. **Гибридный поиск**: Используйте оба метода одновременно
4. Нажмите "Начать поиск"
5. Получите топ-5 похожих товаров с информацией о схожести

## Конфигурация

Настройки приложения находятся в файле `config.py` и могут быть переопределены через переменные окружения (файл `.env`):

- `DATA_DIR`: Директория с данными (по умолчанию: `./data`)
- `VECTOR_STORE_TYPE`: Тип хранилища (`faiss` или `qdrant`)
- `VECTOR_DIMENSION`: Размерность эмбеддингов (по умолчанию: 512)
- `TOP_K_RESULTS`: Количество результатов (по умолчанию: 5)
- `MAX_FILE_SIZE_MB`: Максимальный размер загружаемого файла (по умолчанию: 10)

## Интеграция с ML моделью

Приложение использует интерфейс `MLInterface` для работы с ML моделью. По умолчанию используется dummy модель для тестирования UI.

### Что нужно извлечь из ноутбука

1. **Веса модели** (`.pth`, `.pt`, `.ckpt` и т.д.) → сохраните в `models/`
2. **Код инициализации модели** → создайте класс-обертку в `models/clothing_model.py`
3. **Препроцессинг** (трансформации изображений, токенизация текста)
4. **Параметры** (размерность эмбеддинга, размер входного изображения)

### Подключение реальной модели

**Подробная инструкция**: см. [`docs/MODEL_INTEGRATION.md`](docs/MODEL_INTEGRATION.md)

**Кратко:**

1. Создайте класс-обертку в `models/clothing_model.py`:
```python
class ClothingEmbeddingModel:
    def __init__(self, model_path: str):
        # Загрузка вашей модели
        pass
    
    def encode_image(self, image: PIL.Image) -> np.ndarray:
        # Ваша логика кодирования изображения
        return embedding  # 1D numpy array
    
    def encode_text(self, text: str) -> np.ndarray:
        # Ваша логика кодирования текста
        return embedding  # 1D numpy array
```

2. Обновите `src/cache.py` для загрузки вашей модели

3. Пересоздайте векторный индекс:
```bash
python scripts/generate_embeddings.py
```

### Требования к модели

Модель должна иметь следующие методы:

```python
class EmbeddingModel:
    def encode_image(self, image: PIL.Image) -> np.ndarray:
        """Возвращает визуальный эмбеддинг (1D вектор)"""
        pass
    
    def encode_text(self, text: str) -> np.ndarray:
        """Возвращает текстовый эмбеддинг (1D вектор)"""
        pass
```

**Важно:**
- Эмбеддинги должны быть нормализованы (L2 норма)
- Размерность должна совпадать с `VECTOR_DIMENSION` в `config.py`
- Изображения должны обрабатываться так же, как при обучении

## Векторное хранилище

### FAISS (по умолчанию)

Используется для MVP и локальной разработки. Индекс сохраняется в `./vector_store/faiss_index.bin`.

### Qdrant (для production)

Для использования Qdrant установите переменные окружения:

```bash
VECTOR_STORE_TYPE=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=clothing_embeddings
```

## Подготовка данных

1. **Метаданные товаров**: Создайте файл `data/products_metadata.json` со структурой:
```json
{
  "products": [
    {
      "product_id": "001",
      "name": "Название товара",
      "category": "категория",
      "brand": "бренд",
      "description": "описание",
      "color": "цвет",
      "size": "размер",
      "image_path": "data/images/product_001.jpg",
      "image_embedding_id": "visual_001",
      "text_embedding_id": "text_001",
      "price": "цена"
    }
  ]
}
```

2. **Эмбеддинги**: Создайте файлы:
   - `data/embeddings/visual_embeddings.pkl` - визуальные эмбеддинги
   - `data/embeddings/text_embeddings.pkl` - текстовые эмбеддинги

3. **Изображения**: Разместите изображения товаров в `data/images/`

## Развертывание

### Streamlit Cloud

1. Загрузите проект на GitHub
2. Перейдите на https://share.streamlit.io
3. Нажмите "New app"
4. Выберите репозиторий и укажите `app.py`
5. Нажмите Deploy

### Docker

```bash
docker build -t clothing-recommender .
docker run -p 8501:8501 clothing-recommender
```

## Логирование

Логи приложения сохраняются в `logs/app.log` и выводятся в консоль.

## Тестирование

Запуск тестов:
```bash
pytest tests/ -v
```

## Лицензия

MIT

## Автор

Разработано для курсовой работы по рекомендательным системам.
