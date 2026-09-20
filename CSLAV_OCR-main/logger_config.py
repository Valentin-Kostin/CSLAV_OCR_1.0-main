import logging
import os
import sys
from datetime import datetime

def setup_logger(name="CSLAV_OCR", log_file="csalv_ocr.log", level=logging.INFO):
    """
    Настраивает логгер для проекта.
    
    Args:
        name: Имя логгера.
        log_file: Путь к файлу лога.
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    
    Returns:
        Настроенный объект logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Очистка предыдущих хендлеров, чтобы не дублировать записи при перезапуске в той же сессии
    if logger.handlers:
        logger.handlers.clear()

    # Формат сообщения: Дата Время | Уровень | Модуль | Сообщение
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. Файловый хендлер (запись всех сообщений от DEBUG и выше)
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Critical Error: Could not create log file {log_file}: {e}")

    # 2. Консольный хендлер (вывод только WARNING и выше, чтобы не спамить в GUI)
    # В режиме GUI консоль может быть скрыта, но при запуске из терминала это полезно
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logger initialized. Log file: {os.path.abspath(log_file)}")
    return logger

# Создаем экземпляр логгера по умолчанию
logger = setup_logger()
