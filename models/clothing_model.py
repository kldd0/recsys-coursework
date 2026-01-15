"""
Класс-обертка для CLIP модели эмбеддингов одежды.
Интегрировано из notebook/main.ipynb
"""
import torch
import numpy as np
from PIL import Image
from typing import Optional
from transformers import CLIPProcessor, CLIPModel
import logging

logger = logging.getLogger(__name__)


class CLIPEmbeddingModel:
    """
    Обертка для CLIP модели от OpenAI.
    Использует openai/clip-vit-base-patch32 для генерации эмбеддингов.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        """
        Инициализация CLIP модели.
        
        Args:
            model_name: Имя модели из HuggingFace (по умолчанию: openai/clip-vit-base-patch32)
            device: Устройство для вычислений ('cuda' или 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        logger.info(f"Загрузка CLIP модели: {model_name}")
        logger.info(f"Устройство: {self.device}")
        
        # Загрузка модели и процессора
        try:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            
            # Перемещение модели на устройство
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Размерность эмбеддинга для CLIP ViT-B/32
            self.embedding_dim = 512
            
            logger.info(f"CLIP модель успешно загружена. Размерность эмбеддинга: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке CLIP модели: {e}")
            raise
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Получить визуальный эмбеддинг изображения.
        
        Args:
            image: PIL Image объект
            
        Returns:
            np.ndarray: Вектор эмбеддинга размера [512]
        """
        try:
            # Конвертация в RGB если необходимо
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Обработка изображения через CLIP processor
            inputs = self.processor(images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Получение эмбеддинга изображения
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                embedding = image_features.cpu().numpy()[0]
            
            # Нормализация эмбеддинга (L2 норма)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
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
            np.ndarray: Вектор эмбеддинга размера [512]
        """
        try:
            # Обработка текста через CLIP processor
            inputs = self.processor(
                text=[text], 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Получение эмбеддинга текста
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                embedding = text_features.cpu().numpy()[0]
            
            # Нормализация эмбеддинга (L2 норма)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Ошибка при кодировании текста: {e}")
            raise
