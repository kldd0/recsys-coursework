"""
Конфигурация приложения для мультимодальной рекомендательной системы.
"""
import os
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# Paths
DATA_DIR = os.getenv("DATA_DIR", "./data")
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", os.path.join(DATA_DIR, "embeddings"))
IMAGES_DIR = os.getenv("IMAGES_DIR", os.path.join(DATA_DIR, "images"))
METADATA_FILE = os.getenv("METADATA_FILE", os.path.join(DATA_DIR, "products_metadata.json"))
# Путь к pickle файлу метаданных
PICKLE_METADATA_FILE = os.getenv("PICKLE_METADATA_FILE", os.path.join(EMBEDDINGS_DIR, "metadata.pkl"))

# Vector Store
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "512"))
VECTOR_METRIC = os.getenv("VECTOR_METRIC", "cosine")

# Qdrant (если используется)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "clothing_embeddings")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# ML Model
MODEL_PATH = os.getenv("MODEL_PATH", None)
MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")  # CLIP модель
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))

# Приложение
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
TARGET_IMAGE_SIZE = int(os.getenv("TARGET_IMAGE_SIZE", "224"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
SEARCH_TIMEOUT_SEC = int(os.getenv("SEARCH_TIMEOUT_SEC", "30"))

# Логирование
LOG_DIR = os.getenv("LOG_DIR", "./logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Создание необходимых директорий
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
