"""
Главное Streamlit приложение для мультимодальной рекомендательной системы гардероба.
"""
import logging
import os
import sys
import time
from typing import Optional, Tuple

import streamlit as st
from PIL import Image
import numpy as np

# Добавление корневой директории в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cache import load_recommendation_engine, load_metadata
from src.image_processor import ImageProcessor
from src.data_loader import DataLoader
import config

# Настройка логирования
os.makedirs(config.LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Конфигурация страницы
st.set_page_config(
    page_title="Рекомендательная Система Гардероба",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Инициализация session_state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

if 'last_search_type' not in st.session_state:
    st.session_state.last_search_type = None  # 'text', 'image', 'hybrid'

if 'search_in_progress' not in st.session_state:
    st.session_state.search_in_progress = False


def initialize_system():
    """Инициализация системы рекомендаций."""
    try:
        engine = load_recommendation_engine()
        return engine, None
    except Exception as e:
        logger.error(f"Ошибка при инициализации системы: {e}")
        return None, str(e)


def validate_text_input(text: str) -> Tuple[bool, Optional[str]]:
    """
    Валидация текстового ввода.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    text = text.strip()
    
    if not text:
        return False, "Введите описание товара"
    
    if len(text) < 3:
        return False, "Запрос должен содержать минимум 3 символа"
    
    return True, None


def display_product_card(product: dict, col):
    """
    Отображение карточки товара.
    
    Args:
        product: Словарь с данными товара
        col: Streamlit колонка
    """
    with col:
        # Изображение товара
        if product.get('image'):
            st.image(product['image'], use_column_width=True)
        else:
            st.image("https://via.placeholder.com/300x300?text=No+Image", use_column_width=True)
        
        # Название
        st.subheader(product.get('name', 'Неизвестный товар'))
        
        # Метаданные
        if product.get('category'):
            st.write(f"📦 **Категория:** {product['category']}")
        
        if product.get('brand'):
            st.write(f"🏷️ **Бренд:** {product['brand']}")
        
        if product.get('description'):
            st.caption(product['description'])
        
        # Дополнительная информация
        info_parts = []
        if product.get('color'):
            info_parts.append(f"Цвет: {product['color']}")
        if product.get('size'):
            info_parts.append(f"Размер: {product['size']}")
        
        if info_parts:
            st.write(" | ".join(info_parts))
        
        # Схожесть
        similarity = product.get('similarity', 0)
        st.metric("Схожесть", f"{similarity:.1f}%")
        
        # Цена
        if product.get('price'):
            st.caption(f"💰 Цена: {product['price']} руб.")


def main():
    """Главная функция приложения."""
    # Заголовок
    st.header("👕 Мультимодальная Рекомендательная Система Гардероба")
    st.markdown("---")
    
    # Инициализация системы
    engine, init_error = initialize_system()
    
    if engine is None:
        st.error(f"❌ Ошибка инициализации системы: {init_error}")
        st.info("Проверьте логи в файле logs/app.log")
        return
    
    # Основной интерфейс
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Поиск товаров")
        
        # Текстовое поле
        text_query = st.text_input(
            "Введите описание товара",
            placeholder="Например: футболка найк красная",
            help="Опишите товар, который вы ищете"
        )
        
        # Загрузка изображения
        uploaded_file = st.file_uploader(
            "Прикрепить фотографию",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Поддерживаемые форматы: JPG, PNG, WebP (макс. 10 МБ)"
        )
        
        # Отображение загруженного изображения
        if uploaded_file is not None:
            image_processor = ImageProcessor()
            if image_processor.validate_image(uploaded_file):
                image = Image.open(uploaded_file)
                st.image(image, caption="Загруженное изображение", width=300)
            else:
                st.warning("⚠️ Изображение не прошло валидацию. Проверьте формат и размер.")
                uploaded_file = None
        
        # Кнопка поиска
        search_button = st.button(
            "🔍 Начать поиск",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        st.subheader("ℹ️ Информация")
        st.info("""
        **Как использовать:**
        1. Введите текстовое описание товара ИЛИ
        2. Загрузите фотографию товара
        3. Нажмите "Начать поиск"
        4. Получите топ-5 похожих товаров
        """)
        
        # Статистика
        try:
            metadata = load_metadata()
            # Если используются pickle метаданные, берем количество оттуда
            if metadata.get('_use_pickle', False) and '_pickle_count' in metadata:
                product_count = metadata['_pickle_count']
            else:
                product_count = len(metadata.get('products', []))
            st.metric("Товаров в базе", product_count)
        except:
            pass
    
    # Обработка поиска
    if search_button:
        # Валидация входных данных
        has_text = text_query and text_query.strip()
        has_image = uploaded_file is not None
        
        if not has_text and not has_image:
            st.warning("⚠️ Пожалуйста, введите описание товара или загрузите изображение")
            return
        
        # Валидация текста
        if has_text:
            is_valid, error_msg = validate_text_input(text_query)
            if not is_valid:
                st.warning(f"⚠️ {error_msg}")
                return
        
        # Валидация изображения
        if has_image:
            image_processor = ImageProcessor()
            if not image_processor.validate_image(uploaded_file):
                st.warning("⚠️ Изображение не прошло валидацию")
                return
        
        # Выполнение поиска
        st.session_state.search_in_progress = True
        
        with st.spinner("🔍 Поиск похожих товаров..."):
            try:
                start_time = time.time()
                
                if has_image and has_text:
                    # Гибридный поиск
                    image = Image.open(uploaded_file)
                    results = engine.get_hybrid_recommendations(
                        image=image,
                        text=text_query.strip(),
                        top_k=config.TOP_K_RESULTS
                    )
                    search_type = "hybrid"
                    
                elif has_image:
                    # Визуальный поиск
                    image = Image.open(uploaded_file)
                    results = engine.get_visual_recommendations(
                        image=image,
                        top_k=config.TOP_K_RESULTS
                    )
                    search_type = "image"
                    
                else:
                    # Текстовый поиск
                    results = engine.get_text_recommendations(
                        query=text_query.strip(),
                        top_k=config.TOP_K_RESULTS
                    )
                    search_type = "text"
                
                elapsed_time = time.time() - start_time
                
                st.session_state.search_results = results
                st.session_state.last_search_type = search_type
                st.session_state.search_in_progress = False
                
                logger.info(
                    f"Поиск завершен. Тип: {search_type}, "
                    f"Результатов: {len(results)}, Время: {elapsed_time:.2f}с"
                )
                
            except Exception as e:
                st.error(f"❌ Ошибка при выполнении поиска: {str(e)}")
                logger.error(f"Ошибка поиска: {e}", exc_info=True)
                st.session_state.search_in_progress = False
    
    # Отображение результатов
    if st.session_state.search_results is not None:
        st.markdown("---")
        st.subheader("📦 Результаты поиска")
        
        results = st.session_state.search_results
        
        if len(results) == 0:
            st.info("😔 Товары не найдены. Попробуйте другой поиск.")
        else:
            # Отображение в виде карточек
            num_cols = min(len(results), 5)
            cols = st.columns(num_cols)
            
            for idx, product in enumerate(results[:num_cols]):
                display_product_card(product, cols[idx])
            
            # Информация о типе поиска
            search_type_labels = {
                "text": "📝 Текстовый поиск",
                "image": "🖼️ Визуальный поиск",
                "hybrid": "🔀 Гибридный поиск"
            }
            search_type_label = search_type_labels.get(
                st.session_state.last_search_type,
                "Поиск"
            )
            st.caption(f"Тип поиска: {search_type_label}")


if __name__ == "__main__":
    main()
