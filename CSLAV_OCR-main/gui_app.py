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
from tensorflow.keras import models
import cv2
import pymupdf  # PyMuPDF (используем новый API)

# Импорт утилит из ocr_utils
from ocr_utils import (
    decode_predictions, prepare_img, get_boxes_from_prepared,
    get_symbols_from_prepared, get_edges, symbols_to_rows, get_raw_str,
    process_prepared_image_to_text, process_image_to_text
)


class CSLAVOCRApp:
    """Основной класс приложения с графическим интерфейсом."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("CSLAV OCR - Распознавание церковнославянских текстов")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Переменные
        self.model = None
        self.predictions_list = None
        self.selected_file = None
        self.file_type = None  # 'image' или 'pdf'
        self.is_processing = False
        
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
        self.pdf_page = tk.IntVar(value=1)
        
        # Создание интерфейса
        self._create_menu()
        self._create_main_frame()
        self._create_status_bar()
        
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
    
    def _create_main_frame(self):
        """Создание основной панели интерфейса."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Настройка растягивания
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Панель выбора файла
        file_frame = ttk.LabelFrame(main_frame, text="Выбор файла", padding="5")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Label(file_frame, text="Файл:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.file_label = ttk.Label(file_frame, text="Не выбран", foreground="gray")
        self.file_label.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        btn_frame = ttk.Frame(file_frame)
        btn_frame.grid(row=0, column=2, padx=5)
        
        ttk.Button(btn_frame, text="Изображение", command=self._select_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="PDF", command=self._select_pdf).pack(side=tk.LEFT, padx=2)
        
        # Панель настроек
        settings_frame = ttk.LabelFrame(main_frame, text="Параметры распознавания", padding="5")
        settings_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        settings_frame.columnconfigure(1, weight=1)
        
        # Параметры в две колонки
        ttk.Label(settings_frame, text="Мин. высота символа:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(settings_frame, from_=5, to=100, textvariable=self.min_h_symbols, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(settings_frame, text="Мин. высота строки:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(settings_frame, from_=20, to=200, textvariable=self.min_h_boxes, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(settings_frame, text="Порог строк:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(settings_frame, from_=20, to=150, textvariable=self.edge_threshn, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(settings_frame, text="Размер пробела:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(settings_frame, from_=10, to=150, textvariable=self.space_size, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(settings_frame, text="Масштаб PDF:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(settings_frame, from_=2.0, to=10.0, increment=0.5, textvariable=self.zoom_pdf, width=10).grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(settings_frame, text="Страница PDF:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(settings_frame, from_=1, to=9999, textvariable=self.pdf_page, width=10).grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)
        
        self.save_interim_check = ttk.Checkbutton(settings_frame, text="Сохранять промежуточные результаты", variable=self.save_interim)
        self.save_interim_check.grid(row=6, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Кнопка запуска
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, pady=(0, 10))
        
        self.process_button = ttk.Button(button_frame, text="Распознать текст", command=self._start_processing)
        self.process_button.pack(side=tk.LEFT, padx=5)
        
        self.cancel_button = ttk.Button(button_frame, text="Отмена", command=self._cancel_processing, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        
        # Прогресс-бар
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100, mode='indeterminate')
        self.progress_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Поле результата
        result_frame = ttk.LabelFrame(main_frame, text="Результат распознавания", padding="5")
        result_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, height=20, font=("Consolas", 11))
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Кнопки действий с результатом
        result_btn_frame = ttk.Frame(result_frame)
        result_btn_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Button(result_btn_frame, text="Копировать", command=self._copy_result).pack(side=tk.LEFT, padx=2)
        ttk.Button(result_btn_frame, text="Очистить", command=self._clear_result).pack(side=tk.LEFT, padx=2)
        ttk.Button(result_btn_frame, text="Сохранить в файл...", command=self._save_result).pack(side=tk.LEFT, padx=2)
    
    def _create_status_bar(self):
        """Создание строки состояния."""
        self.status_var = tk.StringVar(value="Готов к работе")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        self.model_status_var = tk.StringVar(value="Модель: не загружена")
        model_status_bar = ttk.Label(self.root, textvariable=self.model_status_var, relief=tk.SUNKEN, anchor=tk.W)
        model_status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=5, pady=2)
    
    def _load_model_async(self):
        """Асинхронная загрузка модели."""
        def load():
            try:
                self.status_var.set("Загрузка модели...")
                self.root.update_idletasks()
                
                if os.path.exists(self.default_model_path):
                    self.model = models.load_model(self.default_model_path)
                    self.model_status_var.set(f"Модель: machine.h5 (загружена)")
                    
                    if os.path.exists(self.default_predictions_path):
                        self.predictions_list = decode_predictions(self.default_predictions_path)
                        self.status_var.set("Готов к работе")
                    else:
                        self.status_var.set("Файл predictions.txt не найден")
                else:
                    self.status_var.set("Файл machine.h5 не найден. Выберите модель вручную.")
            except Exception as e:
                self.status_var.set(f"Ошибка загрузки модели: {str(e)}")
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def _select_image(self):
        """Выбор изображения для распознавания."""
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
            self.file_label.config(text=os.path.basename(filename), foreground="black")
            self.status_var.set(f"Выбран файл: {os.path.basename(filename)}")
    
    def _select_pdf(self):
        """Выбор PDF файла для распознавания."""
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
            if self.file_type == 'image':
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
                pix.save(temp_image)
                doc.close()
                
                self.status_var.set("Распознавание текста...")
                self.root.update_idletasks()
                
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
                    except:
                        pass
            
            # Обновление результата в главном потоке
            self.root.after(0, self._update_result, text)
            
        except Exception as e:
            self.root.after(0, self._processing_error, str(e))
    
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
