#крч есть +- документация, вот
#всякие импорты

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import fitz
from tkinter import *
from tkinter import filedialog
from ocr_utils import (
    Symbol, decode_predictions, prepare_img, get_boxes_from_prepared,
    get_symbols_from_prepared, get_edges, symbols_to_rows, get_raw_str,
    process_prepared_image_to_text
)
from tensorflow.keras import models


def clicked():
    # функция выбора пути к файлу
    trek = filedialog.askopenfilename(filetypes=(
        ("PDF files", "*.pdf"), ("all files", "*.*")))
    print(trek)
    return trek


def load_page_from_pdf(pdffile, page_number, zoom=4.166):
    """Загрузка страницы из PDF в файл PNG.
    
    Args:
        pdffile: имя файла
        page_number: номер страницы (начиная с 1)
        zoom: масштаб
        
    Returns:
        путь к сохраненному изображению
    """
    doc = fitz.open(pdffile)
    page = doc.load_page(page_number - 1)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    output_fname = pdffile[0:-4] + '\\\\page' + str(page_number) + '\\\\' + 'page.png'
    pix.save(output_fname)
    print('Страница сохранена в файл page.png')
    return 'page.png'


def get_text(filename, model, predictions_list, save_interim_results=False):
    """Получение текста из изображения с использованием общих утилит."""
    img_prepared = prepare_img(filename, save_interim_results)
    text = process_prepared_image_to_text(
        filename, img_prepared, model, predictions_list,
        min_h_symbols=15, min_h_boxes=50, edge_threshn=70,
        space_size=50, save_interim_results=save_interim_results
    )
    return text


def main(model_name, predictions_file):
    """Основная функция для обработки PDF файлов."""
    pdffile = clicked()
    page_number = int(input('Введите номер страницы: '))
    result_dir = pdffile[0:-4] + '/page' + str(page_number)
    os.makedirs(result_dir, exist_ok=True)
    print('Результаты будут сохранены в папке ' + result_dir)
    print('0')
    fname = load_page_from_pdf(pdffile, page_number)
    print('1')
    model = models.load_model(model_name)
    print('2', model)
    predictions_list = decode_predictions(predictions_file)
    os.chdir(result_dir)
    save_interim_results = int(input('Сохранить промежуточные результаты? 1 - да, 0 - нет  '))
    with open('text.txt', 'w', encoding='utf-8') as f:
        f.write(get_text(fname, model, predictions_list, save_interim_results=save_interim_results))
    print('Распознанный текст в файле text.txt')


"""window = Tk()
window.title("Распознование текста")
window.geometry('700x400')
lbl = Label(window, text="Привет!")
lbl.grid(column=2, row=0)
btn = Button(window, text="загрузить файл", command=main('machine.h5', 'predictions.txt'))
btn.grid(column=2, row=1)

window.mainloop()"""

#main('C:/Coding/CSLAV/CSLAV_OCR_1.0-main/CSLAV_OCR-main/machine.h5', 'C:/Coding/CSLAV/CSLAV_OCR_1.0-main/CSLAV_OCR-main/predictions.txt')
