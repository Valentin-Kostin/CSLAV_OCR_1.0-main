# -*- coding: utf-8 -*-
"""
Графический интерфейс для CSLAV_OCR
Приложение для распознавания церковнославянских текстов
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
from tensorflow.keras import models
import cv2
import numpy as np
import pymupdf  # PyMuPDF (используем новый API)
import tempfile
import shutil

# Импорт утилит и логгера
from ocr_utils import (
    decode_predictions, prepare_img, get_boxes_from_prepared,
    get_symbols_from_prepared, get_edges, symbols_to_rows, get_raw_str,
    process_prepared_image_to_text, process_image_to_text
)
from logger_config import logger


class CSLAVOCRApp:
    """Основной класс приложения с графическим интерфейсом."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("CSLAV OCR - Распознавание церковнославянских текстов")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Переменные
        self.model = None
        self.predictions_list = None
        self.selected_file = None
        self.file_type = None  # 'image' или 'pdf'
        self.is_processing = False
        self.current_image = None  # Для отображения изображения
        self.photo_image = None  # Ссылка на PhotoImage
        self.rotated_temp_dir = None  # Временная папка для повёрнутых изображений
        self.rotation_angle = tk.DoubleVar(value=0.0)  # Угол поворота в градусах
        self.original_file_path = None  # Путь к оригинальному файлу для сброса поворота
        self.image_scale = tk.DoubleVar(value=1.0)  # Масштаб изображения
        
        # Пути по умолчанию
        self.default_model_path = os.path.join(os.path.dirname(__file__), 'machine.h5')
        self.default_predictions_path = os.path.join(os.path.dirname(__file__), 'predictions.txt')
        
        # Настройки распознавания
        self.min_h_symbols = tk.IntVar(value=15)
        self.min_h_boxes = tk.IntVar(value=50)
        self.edge_threshn = tk.IntVar(value=70)
        self.space_size = tk.IntVar(value=50)
        self.save_interim = tk.BooleanVar(value=False)
        self.zoom_pdf = tk.DoubleVar(value=4.166)
        
        logger.info("Инициализация GUI приложения CSLAV OCR")
        self.pdf_page = tk.IntVar(value=1)
        
        # Создание интерфейса
        self._create_menu()
        self._create_top_panel()
        self._create_main_area()
        self._create_bottom_panel()
        
        # Привязка события изменения размера для обновления Canvas
        self.root.bind("<Configure>", self._on_resize)
        
        # Загрузка модели при запуске
        self._load_model_async()
    
    def _create_menu(self):
        """Создание меню приложения."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Меню Файл
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Открыть изображение...", command=self._select_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Открыть PDF...", command=self._select_pdf, accelerator="Ctrl+P")
        file_menu.add_separator()
        file_menu.add_command(label="Сохранить результат...", command=self._save_result, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit, accelerator="Alt+F4")
        
        # Меню Настройки
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Настройки", menu=settings_menu)
        settings_menu.add_command(label="Выбрать модель...", command=self._select_model)
        settings_menu.add_command(label="Выбрать файл предсказаний...", command=self._select_predictions)
        settings_menu.add_command(label="Сбросить настройки", command=self._reset_settings)
        
        # Меню Справка
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Справка", menu=help_menu)
        help_menu.add_command(label="О программе", command=self._show_about)
    
    def _create_top_panel(self):
        """Создание верхней панели с кнопками и настройками."""
        top_frame = ttk.Frame(self.root, padding="5")
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Верхний ряд - кнопки файлов и параметры
        top_row = ttk.Frame(top_frame)
        top_row.pack(side=tk.TOP, fill=tk.X)
        
        # Левая часть - кнопки выбора файла
        file_frame = ttk.LabelFrame(top_row, text="Файл", padding="5")
        file_frame.pack(side=tk.LEFT, fill=tk.X, padx=5)
        
        ttk.Button(file_frame, text="📁 Изображение", command=self._select_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="📄 PDF", command=self._select_pdf).pack(side=tk.LEFT, padx=2)
        
        self.file_label = ttk.Label(file_frame, text="Не выбран", foreground="gray")
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        # Центральная часть - параметры распознавания
        params_frame = ttk.LabelFrame(top_row, text="Параметры распознавания", padding="5")
        params_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Параметры в две строки для компактности
        params_inner1 = ttk.Frame(params_frame)
        params_inner1.pack(fill=tk.X)
        
        ttk.Label(params_inner1, text="Мин. высота символа:").pack(side=tk.LEFT, padx=2)
        ttk.Spinbox(params_inner1, from_=5, to=100, textvariable=self.min_h_symbols, width=5).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(params_inner1, text="Мин. высота строки:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Spinbox(params_inner1, from_=20, to=200, textvariable=self.min_h_boxes, width=5).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(params_inner1, text="Порог строк:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Spinbox(params_inner1, from_=20, to=150, textvariable=self.edge_threshn, width=5).pack(side=tk.LEFT, padx=2)
        
        params_inner2 = ttk.Frame(params_frame)
        params_inner2.pack(fill=tk.X, pady=(2, 0))
        
        ttk.Label(params_inner2, text="Размер пробела:").pack(side=tk.LEFT, padx=2)
        ttk.Spinbox(params_inner2, from_=10, to=150, textvariable=self.space_size, width=5).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(params_inner2, text="Масштаб PDF:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Spinbox(params_inner2, from_=2.0, to=10.0, increment=0.5, textvariable=self.zoom_pdf, width=5).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(params_inner2, text="Страница:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Spinbox(params_inner2, from_=1, to=9999, textvariable=self.pdf_page, width=5).pack(side=tk.LEFT, padx=2)
        
        self.save_interim_check = ttk.Checkbutton(params_inner2, text="Сохранять промежуточные", variable=self.save_interim)
        self.save_interim_check.pack(side=tk.LEFT, padx=10)
        
        # Нижний ряд - поворот и управление
        bottom_row = ttk.Frame(top_frame)
        bottom_row.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        
        # Панель поворота изображения (перемещена вниз)
        rotate_frame = ttk.LabelFrame(bottom_row, text="Поворот изображения", padding="5")
        rotate_frame.pack(side=tk.LEFT, fill=tk.X, padx=5)
        
        ttk.Label(rotate_frame, text="Угол (°):").pack(side=tk.LEFT, padx=2)
        ttk.Spinbox(rotate_frame, from_=-360, to=360, increment=0.1, textvariable=self.rotation_angle, width=6).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(rotate_frame, text="↻ Применить", command=self._apply_rotation).pack(side=tk.LEFT, padx=2)
        ttk.Button(rotate_frame, text="⟲ 90°", command=lambda: self._rotate_by_90(90)).pack(side=tk.LEFT, padx=2)
        ttk.Button(rotate_frame, text="⟳ -90°", command=lambda: self._rotate_by_90(-90)).pack(side=tk.LEFT, padx=2)
        ttk.Button(rotate_frame, text="✕ Сброс", command=self._reset_rotation).pack(side=tk.LEFT, padx=2)
        
        # Панель масштабирования (добавлена рядом с поворотом)
        scale_frame = ttk.LabelFrame(bottom_row, text="Масштабирование", padding="5")
        scale_frame.pack(side=tk.LEFT, fill=tk.X, padx=5)
        
        ttk.Label(scale_frame, text="Масштаб:").pack(side=tk.LEFT, padx=2)
        ttk.Spinbox(scale_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.image_scale, width=5).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(scale_frame, text="+", command=self._increase_scale, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(scale_frame, text="-", command=self._decrease_scale, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(scale_frame, text="1:1", command=self._reset_scale, width=3).pack(side=tk.LEFT, padx=2)
        
        # Правая часть - кнопки управления
        control_frame = ttk.LabelFrame(bottom_row, text="Управление", padding="5")
        control_frame.pack(side=tk.RIGHT, padx=5)
        
        self.process_button = ttk.Button(control_frame, text="▶ Распознать", command=self._start_processing, width=12)
        self.process_button.pack(side=tk.LEFT, padx=2)
        
        self.cancel_button = ttk.Button(control_frame, text="⏹ Отмена", command=self._cancel_processing, state=tk.DISABLED, width=8)
        self.cancel_button.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(control_frame, text="🗑 Очистить", command=self._clear_all, width=8).pack(side=tk.LEFT, padx=2)
    
    def _create_main_area(self):
        """Создание основной области с изображением слева и текстом справа."""
        main_area = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Левая панель - изображение
        left_frame = ttk.LabelFrame(main_area, text="Изображение", padding="5")
        main_area.add(left_frame, weight=1)
        
        # Canvas для прокрутки изображения
        self.image_canvas = tk.Canvas(left_frame, bg='white', highlightthickness=0)
        self.image_scrollbar_y = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        self.image_scrollbar_x = ttk.Scrollbar(left_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        self.image_scrollable_frame = ttk.Frame(self.image_canvas)
        
        self.image_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
        )
        
        self.image_canvas.create_window((0, 0), window=self.image_scrollable_frame, anchor="nw")
        self.image_canvas.configure(yscrollcommand=self.image_scrollbar_y.set, xscrollcommand=self.image_scrollbar_x.set)
        
        self.image_label = ttk.Label(self.image_scrollable_frame, text="Нет изображения", foreground="gray")
        self.image_label.pack(padx=5, pady=5)
        
        self.image_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Правая панель - распознанный текст
        right_frame = ttk.LabelFrame(main_area, text="Распознанный текст", padding="5")
        main_area.add(right_frame, weight=1)
        
        self.result_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, font=("Consolas", 11))
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Кнопки действий с результатом
        result_btn_frame = ttk.Frame(right_frame)
        result_btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(result_btn_frame, text="Копировать", command=self._copy_result).pack(side=tk.LEFT, padx=2)
        ttk.Button(result_btn_frame, text="Очистить", command=self._clear_result).pack(side=tk.LEFT, padx=2)
        ttk.Button(result_btn_frame, text="Сохранить в файл...", command=self._save_result).pack(side=tk.LEFT, padx=2)
    
    def _create_bottom_panel(self):
        """Создание нижней панели с прогресс-баром и сообщениями."""
        bottom_frame = ttk.Frame(self.root, padding="5")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Прогресс-бар
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(bottom_frame, variable=self.progress_var, maximum=100, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        # Строка состояния с сообщениями
        self.status_var = tk.StringVar(value="Готов к работе")
        status_bar = ttk.Label(bottom_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.LEFT, expand=True)
        
        self.model_status_var = tk.StringVar(value="Модель: не загружена")
        model_status_bar = ttk.Label(bottom_frame, textvariable=self.model_status_var, relief=tk.SUNKEN, anchor=tk.W)
        model_status_bar.pack(fill=tk.X, side=tk.RIGHT, padx=(5, 0))
    
    def _load_model_async(self):
        """Асинхронная загрузка модели."""
        def load():
            try:
                self.status_var.set("Загрузка модели...")
                self.root.update_idletasks()
                
                logger.info(f"Попытка загрузки модели из: {self.default_model_path}")
                
                if os.path.exists(self.default_model_path):
                    logger.debug("Загрузка файла модели...")
                    self.model = models.load_model(self.default_model_path)
                    logger.info("Модель успешно загружена")
                    self.model_status_var.set(f"Модель: machine.h5 (загружена)")
                    
                    if os.path.exists(self.default_predictions_path):
                        logger.info(f"Загрузка предсказаний из: {self.default_predictions_path}")
                        self.predictions_list = decode_predictions(self.default_predictions_path)
                        self.status_var.set("Готов к работе")
                        logger.info("Приложение готово к работе")
                    else:
                        logger.error(f"Файл предсказаний не найден: {self.default_predictions_path}")
                        self.status_var.set("Файл predictions.txt не найден")
                else:
                    logger.error(f"Файл модели не найден: {self.default_model_path}")
                    self.status_var.set("Файл machine.h5 не найден. Выберите модель вручную.")
            except Exception as e:
                error_msg = f"Ошибка загрузки модели: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.status_var.set(error_msg)
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def _on_resize(self, event):
        """Обработка изменения размера окна для обновления Canvas."""
        # Обновляем отображение изображения при изменении размера окна
        if hasattr(self, 'photo_image') and self.photo_image is not None:
            # Небольшая задержка чтобы окно успело перерисоваться
            self.root.after(100, self._refresh_image_display)
    
    def _refresh_image_display(self):
        """Перерисовка текущего изображения с учётом новых размеров."""
        if self.selected_file and self.file_type == 'image':
            self._display_image(self.selected_file)
    
    def _select_image(self):
        """Выбор изображения для распознавания."""
        # Очистка предыдущих временных файлов и сброс пути к оригиналу
        self._cleanup_temp_dir()
        self.original_file_path = None
        
        filename = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png *.tiff *.tif *.bmp"),
                ("Все файлы", "*.*")
            ]
        )
        if filename:
            self.selected_file = filename
            self.file_type = 'image'
            self.rotation_angle.set(0.0)  # Сброс угла поворота
            self.image_scale.set(1.0)  # Сброс масштаба
            self.file_label.config(text=os.path.basename(filename), foreground="black")
            self.status_var.set(f"Выбран файл: {os.path.basename(filename)}")
            # Обновляем отображение после завершения основного цикла событий
            self.root.after(100, lambda: self._display_image(filename))
    
    def _display_image(self, filepath):
        """Отображение изображения в левой панели."""
        try:
            # Чтение изображения с поддержкой кириллических путей
            img_array = np.fromfile(filepath, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Не удалось декодировать изображение")
            
            # Конвертация BGR -> RGB для PIL
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Получение размеров области отображения
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            # Если canvas ещё не инициализирован, используем значения по умолчанию
            if canvas_width < 2:
                canvas_width = 500
            if canvas_height < 2:
                canvas_height = 600
            
            # Масштабирование изображения если оно слишком большое
            img_pil = Image.fromarray(img_rgb)
            orig_width, orig_height = img_pil.size
            
            scale = min(canvas_width / orig_width, canvas_height / orig_height, 1.0)
            if scale < 1.0:
                new_width = int(orig_width * scale)
                new_height = int(orig_height * scale)
                img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Конвертация в PhotoImage
            self.photo_image = ImageTk.PhotoImage(img_pil)
            
            # Очистка предыдущего содержимого
            for widget in self.image_scrollable_frame.winfo_children():
                widget.destroy()
            
            # Отображение изображения
            image_label = ttk.Label(self.image_scrollable_frame, image=self.photo_image)
            image_label.pack(padx=5, pady=5)
            
            logger.debug(f"Изображение отображено: {orig_width}x{orig_height} -> {img_pil.size}")
            
        except Exception as e:
            logger.error(f"Ошибка при отображении изображения: {e}")
            messagebox.showwarning("Предупреждение", f"Не удалось отобразить изображение:\n{e}")
    
    def _select_pdf(self):
        """Выбор PDF файла для распознавания."""
        # Очистка предыдущих временных файлов и сброс пути к оригиналу
        self._cleanup_temp_dir()
        if hasattr(self, 'original_file_path'):
            self.original_file_path = None
        
        filename = filedialog.askopenfilename(
            title="Выберите PDF файл",
            filetypes=[
                ("PDF файлы", "*.pdf"),
                ("Все файлы", "*.*")
            ]
        )
        if filename:
            self.selected_file = filename
            self.file_type = 'pdf'
            self.rotation_angle.set(0.0)  # Сброс угла поворота (для PDF не применяется)
            self.file_label.config(text=os.path.basename(filename), foreground="black")
            self.status_var.set(f"Выбран PDF: {os.path.basename(filename)}")
    
    def _select_model(self):
        """Выбор файла модели."""
        filename = filedialog.askopenfilename(
            title="Выберите файл модели",
            filetypes=[
                ("H5 файлы", "*.h5"),
                ("Все файлы", "*.*")
            ]
        )
        if filename:
            try:
                self.status_var.set("Загрузка модели...")
                self.root.update_idletasks()
                self.model = models.load_model(filename)
                self.model_status_var.set(f"Модель: {os.path.basename(filename)} (загружена)")
                self.status_var.set("Модель успешно загружена")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить модель:\n{str(e)}")
                self.status_var.set("Ошибка загрузки модели")
    
    def _select_predictions(self):
        """Выбор файла предсказаний."""
        filename = filedialog.askopenfilename(
            title="Выберите файл предсказаний",
            filetypes=[
                ("Текстовые файлы", "*.txt"),
                ("Все файлы", "*.*")
            ]
        )
        if filename:
            try:
                self.predictions_list = decode_predictions(filename)
                self.status_var.set(f"Предсказания загружены: {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл предсказаний:\n{str(e)}")
                self.status_var.set("Ошибка загрузки предсказаний")
    
    def _reset_settings(self):
        """Сброс настроек к значениям по умолчанию."""
        self.min_h_symbols.set(15)
        self.min_h_boxes.set(50)
        self.edge_threshn.set(70)
        self.space_size.set(50)
        self.save_interim.set(False)
        self.zoom_pdf.set(4.166)
        self.pdf_page.set(1)
        self.status_var.set("Настройки сброшены")
    
    def _start_processing(self):
        """Запуск процесса распознавания."""
        if not self.selected_file:
            messagebox.showwarning("Предупреждение", "Выберите файл для распознавания!")
            return
        
        if self.model is None:
            messagebox.showwarning("Предупреждение", "Модель не загружена! Выберите файл модели.")
            return
        
        if self.predictions_list is None:
            messagebox.showwarning("Предупреждение", "Файл предсказаний не загружен!")
            return
        
        self.is_processing = True
        self.process_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.progress_bar.start()
        self.status_var.set("Обработка...")
        
        # Запуск в отдельном потоке
        thread = threading.Thread(target=self._process_file, daemon=True)
        thread.start()
    
    def _process_file(self):
        """Обработка файла в отдельном потоке."""
        try:
            logger.info(f"Начало обработки файла: {self.selected_file}")
            logger.debug(f"Параметры: min_h_symbols={self.min_h_symbols.get()}, min_h_boxes={self.min_h_boxes.get()}, "
                        f"edge_threshn={self.edge_threshn.get()}, space_size={self.space_size.get()}")
            
            if self.file_type == 'image':
                logger.info("Обработка изображения...")
                text = process_image_to_text(
                    self.selected_file,
                    self.default_predictions_path if self.predictions_list else None,
                    self.default_model_path if self.model else None,
                    min_h_symbols=self.min_h_symbols.get(),
                    min_h_boxes=self.min_h_boxes.get(),
                    edge_threshn=self.edge_threshn.get(),
                    space_size=self.space_size.get(),
                    save_interim_results=self.save_interim.get()
                )
            elif self.file_type == 'pdf':
                # Извлечение страницы из PDF
                page_number = self.pdf_page.get()
                zoom = self.zoom_pdf.get()
                
                logger.info(f"Извлечение страницы {page_number} из PDF (zoom={zoom})...")
                self.status_var.set(f"Извлечение страницы {page_number} из PDF...")
                self.root.update_idletasks()
                
                doc = pymupdf.open(self.selected_file)
                if page_number > len(doc):
                    raise ValueError(f"В документе всего {len(doc)} страниц")
                
                page = doc.load_page(page_number - 1)
                mat = pymupdf.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # Сохранение временного изображения
                temp_dir = os.path.join(os.path.dirname(self.selected_file), 'temp_ocr')
                os.makedirs(temp_dir, exist_ok=True)
                temp_image = os.path.join(temp_dir, 'page.png')
                logger.debug(f"Сохранение временного изображения: {temp_image}")
                pix.save(temp_image)
                doc.close()
                
                self.status_var.set("Распознавание текста...")
                self.root.update_idletasks()
                
                logger.info("Распознавание текста из страницы PDF...")
                text = process_prepared_image_to_text(
                    temp_image,
                    prepare_img(temp_image, self.save_interim.get()),
                    self.model,
                    self.predictions_list,
                    min_h_symbols=self.min_h_symbols.get(),
                    min_h_boxes=self.min_h_boxes.get(),
                    edge_threshn=self.edge_threshn.get(),
                    space_size=self.space_size.get(),
                    save_interim_results=self.save_interim.get()
                )
                
                # Очистка временных файлов
                if not self.save_interim.get():
                    try:
                        os.remove(temp_image)
                        os.rmdir(temp_dir)
                        logger.debug("Временные файлы удалены")
                    except Exception as e:
                        logger.warning(f"Не удалось удалить временные файлы: {e}")
            
            logger.info(f"Распознавание завершено успешно. Получено символов: {len(text)}")
            
            # Обновление результата в главном потоке
            self.root.after(0, self._update_result, text)
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Ошибка при обработке файла: {error_msg}", exc_info=True)
            self.root.after(0, self._processing_error, error_msg)
    
    def _update_result(self, text):
        """Обновление поля результата."""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.is_processing = False
        self.process_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
        self.progress_bar.stop()
        self.status_var.set("Распознавание завершено")
    
    def _processing_error(self, error_msg):
        """Обработка ошибки распознавания."""
        messagebox.showerror("Ошибка", f"Ошибка при распознавании:\n{error_msg}")
        self.is_processing = False
        self.process_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
        self.progress_bar.stop()
        self.status_var.set("Ошибка распознавания")
    
    def _cancel_processing(self):
        """Отмена обработки (пока не реализовано полное прерывание)."""
        if messagebox.askyesno("Подтверждение", "Отменить обработку?"):
            self.is_processing = False
            self.process_button.config(state=tk.NORMAL)
            self.cancel_button.config(state=tk.DISABLED)
            self.progress_bar.stop()
            self.status_var.set("Обработка отменена")
    
    def _copy_result(self):
        """Копирование результата в буфер обмена."""
        text = self.result_text.get(1.0, tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set("Текст скопирован в буфер обмена")
    
    def _clear_result(self):
        """Очистка поля результата."""
        self.result_text.delete(1.0, tk.END)
        self.status_var.set("Результат очищен")
    
    def _clear_all(self):
        """Очистка всего: изображения и результата."""
        self._clear_result()
        # Очистка изображения
        for widget in self.image_scrollable_frame.winfo_children():
            widget.destroy()
        self.image_label = ttk.Label(self.image_scrollable_frame, text="Нет изображения", foreground="gray")
        self.image_label.pack(padx=5, pady=5)
        self.photo_image = None
        self.selected_file = None
        self.file_type = None
        self.file_label.config(text="Не выбран", foreground="gray")
        self.rotation_angle.set(0.0)
        self.image_scale.set(1.0)
        # Очистка временной папки
        self._cleanup_temp_dir()
        self.status_var.set("Все очищено")
    
    def _cleanup_temp_dir(self):
        """Очистка временной папки с повёрнутыми изображениями."""
        if self.rotated_temp_dir and os.path.exists(self.rotated_temp_dir):
            try:
                shutil.rmtree(self.rotated_temp_dir)
                self.rotated_temp_dir = None
                logger.debug("Временная папка очищена")
            except Exception as e:
                logger.error(f"Ошибка при очистке временной папки: {e}")
    
    def _rotate_by_90(self, degrees):
        """Поворот на 90 градусов (быстрая кнопка)."""
        current = self.rotation_angle.get()
        new_angle = current + degrees
        # Нормализация угла
        while new_angle >= 360:
            new_angle -= 360
        while new_angle < 0:
            new_angle += 360
        self.rotation_angle.set(new_angle)
        self._apply_rotation()
    
    def _reset_rotation(self):
        """Сброс поворота и масштаба к исходному изображению."""
        self.rotation_angle.set(0.0)
        self.image_scale.set(1.0)
        # Восстановление оригинального файла если он был (ДО очистки временной папки!)
        if self.original_file_path:
            self.selected_file = self.original_file_path
        # Очистка временной папки после восстановления пути
        if self.rotated_temp_dir:
            self._cleanup_temp_dir()
        # Отображение изображения
        if self.original_file_path and os.path.exists(self.original_file_path):
            self._display_image(self.selected_file)
            logger.info("Поворот и масштаб сброшены, восстановлено оригинальное изображение")
        else:
            logger.warning("Оригинальный файл не найден для сброса")
    
    def _increase_scale(self):
        """Увеличение масштаба изображения."""
        current = self.image_scale.get()
        new_scale = min(current + 0.1, 5.0)
        self.image_scale.set(round(new_scale, 1))
        self._apply_rotation()
    
    def _decrease_scale(self):
        """Уменьшение масштаба изображения."""
        current = self.image_scale.get()
        new_scale = max(current - 0.1, 0.1)
        self.image_scale.set(round(new_scale, 1))
        self._apply_rotation()
    
    def _reset_scale(self):
        """Сброс масштаба к 1:1."""
        self.image_scale.set(1.0)
        self._apply_rotation()
    
    def _apply_rotation(self):
        """Применение поворота к изображению с одновременным масштабированием."""
        if not self.selected_file or not self.file_type == 'image':
            messagebox.showwarning("Предупреждение", "Сначала выберите изображение для поворота")
            return
        
        angle = self.rotation_angle.get()
        scale = self.image_scale.get()
        
        # Если угол 0 и масштаб 1 - сбрасываем к оригиналу
        if abs(angle) < 0.01 and abs(scale - 1.0) < 0.01:
            self._reset_rotation()
            return
        
        try:
            # Чтение оригинального изображения (всегда из оригинала!)
            source_file = self.original_file_path if self.original_file_path else self.selected_file
            
            # Проверка существования оригинального файла
            if not os.path.exists(source_file):
                raise FileNotFoundError(f"Оригинальный файл не найден: {source_file}")
            
            img_array = np.fromfile(source_file, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Не удалось декодировать изображение")
            
            # Сначала применяем поворот если есть
            if abs(angle) > 0.01:
                h, w = img.shape[:2]
                center = (w / 2, h / 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, rotation_matrix, (w, h), 
                                     borderMode=cv2.BORDER_REPLICATE, 
                                     flags=cv2.INTER_CUBIC)
            
            # Применяем масштабирование
            if abs(scale - 1.0) > 0.01:
                h, w = img.shape[:2]
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                # Убираем артефакты с помощью легкой медианной фильтрации
                img = cv2.medianBlur(img, 3)
            
            # Создание временной папки если ещё не создана
            if self.rotated_temp_dir is None:
                self.rotated_temp_dir = tempfile.mkdtemp(prefix="cslav_rotated_")
                logger.debug(f"Создана временная папка: {self.rotated_temp_dir}")
            
            # Сохранение обработанного изображения во временный файл
            temp_filename = os.path.join(self.rotated_temp_dir, "processed_image.png")
            
            # Используем imencode и сохраняем через numpy для поддержки кириллических путей
            ret, buffer = cv2.imencode('.png', img)
            if not ret:
                raise ValueError("Не удалось закодировать обработанное изображение")
            
            # Запись через numpy для поддержки кириллических путей
            with open(temp_filename, 'wb') as f:
                f.write(buffer.tobytes())
            
            # Проверка что файл успешно сохранён
            if not os.path.exists(temp_filename):
                raise FileNotFoundError(f"Не удалось сохранить временный файл: {temp_filename}")
            
            # Сохранение пути к оригиналу для возможности сброса
            if not self.original_file_path:
                self.original_file_path = self.selected_file
            
            # Обновление текущего файла (только если файл существует!)
            self.selected_file = temp_filename
            logger.info(f"Изображение обработано: поворот {angle}°, масштаб {scale}x")
            
            # Отображение обработанного изображения
            self._display_image(temp_filename)
            self.status_var.set(f"Поворот: {angle}°, Масштаб: {scale}x")
            
        except FileNotFoundError as e:
            logger.error(f"Файл не найден: {e}")
            messagebox.showerror("Ошибка", f"Файл не найден:\n{e}\n\nПопробуйте выбрать изображение заново.")
            self._reset_rotation()
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения: {e}", exc_info=True)
            messagebox.showerror("Ошибка", f"Не удалось обработать изображение:\n{e}")
    
    def _save_result(self):
        """Сохранение результата в файл."""
        text = self.result_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Предупреждение", "Нет текста для сохранения!")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Сохранить результат",
            defaultextension=".txt",
            filetypes=[
                ("Текстовые файлы", "*.txt"),
                ("Все файлы", "*.*")
            ],
            initialfile="ocr_result.txt"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.status_var.set(f"Результат сохранен в {os.path.basename(filename)}")
                messagebox.showinfo("Успех", f"Текст успешно сохранен в файл:\n{filename}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{str(e)}")
    
    def _show_about(self):
        """Показать информацию о программе."""
        about_text = """CSLAV OCR v2.0
        
Программа для оптического распознавания
церковнославянских текстов.

Особенности:
• Распознавание из изображений и PDF
• Поддержка киновари и диакритических знаков
• 49 классов символов
• Экспорт в UTF-8

Разработано для проекта CSLAV_OCR"""
        
        messagebox.showinfo("О программе", about_text)


def main():
    """Точка входа приложения."""
    root = tk.Tk()
    
    # Установка иконки (если есть)
    try:
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.ico')
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except:
        pass
    
    app = CSLAVOCRApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
