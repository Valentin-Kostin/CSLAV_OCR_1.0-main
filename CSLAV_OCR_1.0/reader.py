import os ; os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import collections
import numpy
import cv2
import fitz
import keras
from tkinter import *
from tkinter import filedialog


class Symbol:
    
    def __init__(self, matrix, rectangle, model, predictions_list):
        cv2.imwrite('symbol.png', matrix)
        image = keras.preprocessing.image.load_img('symbol.png', target_size=(56, 56, 3))
        input_arr = keras.preprocessing.image.img_to_array(image)
        num_arr = numpy.array([input_arr])
        result = model.predict([num_arr])
        for prediction in predictions_list:
            if len(prediction) == 2 and prediction[1] == str(result):
                self.text = prediction[0]
                break
        else:
            self.text = ''
        os.remove('symbol.png')
        self.coordinates = rectangle[0], rectangle[1], rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]


def clicked():    
    trek = filedialog.askopenfilename(filetypes=(
        ("PDF files", "*.pdf"), ("all files", "*.*")))
    #file = fitz.open(trek, 'r', encoding='utf-16')
    #trek = trek[0:-4]
    print(trek)
    return trek


def load_page_from_pdf(pdffile, page_number, zoom=4.166):
    print('10')
    doc = fitz.open(pdffile)
    print(doc)
    page = doc.load_page(page_number - 1)
    print('12')
    mat = fitz.Matrix(zoom, zoom)
    print('13')
    pix = page.get_pixmap(matrix=mat)
    print('14', pix)
    output_fname = pdffile[0:-4]+'\\page'+str(page_number)+'\\'+'page.png'
    print('15')
    """cv2.imwrite(output_fname, pix)
    print('16')"""
    pix.save(output_fname)
    print('Страница сохранена в файл page.png')
    return 'page.png'


def decode_predictions(prediction_file):
    with open(prediction_file, 'r', encoding='utf-8') as f_predictions:
        predictions = f_predictions.read()
    predictions_list = predictions.split('\n\n')
    for i in range(len(predictions_list)):
        predictions_list[i] = predictions_list[i].split('\t')
    return predictions_list


def get_text(filename, model, predictions_list, save_interim_results=False):

    def kinovar2black(img):
        h = img.shape[0]
        w = img.shape[1]
        img = img.reshape(h * w, 3)
        img = img.T
        a = numpy.logical_and(img[2] / (img[0] + 1) > 1.4, img[2] / (img[1] + 1) > 1.4)
        for idx in numpy.arange(len(a)):
            x = a.item(idx)
            if x:
                img.itemset((0, idx), 30)
                img.itemset((1, idx), 30)
                img.itemset((2, idx), 30)
        img = img.T
        img = img.reshape(h, w, 3)
        return img

    def prepare_img(filename, save_interim_results):
        img = cv2.imread(filename)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_ex = cv2.morphologyEx(img, cv2.MORPH_DILATE, se)
        img_no_kinovar = kinovar2black(img_ex)
        if save_interim_results:
            cv2.imwrite('img_no_kinovar.png', img_no_kinovar)
            print('Изображение после перекрашивания киновари в файле img_no_kinovar.png')
        img_gray = cv2.cvtColor(img_no_kinovar, cv2.COLOR_BGR2GRAY)
        img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        img_erode = cv2.erode(img_binary, numpy.ones((3, 3), numpy.uint8), iterations=1)
        if save_interim_results:
            cv2.imwrite('img_binary.png', img_erode)
            print('Изображение после предобработки в файле img_binary.png')
        return img_erode

    def get_boxes(img_prepared, min_h, save_interim_results=False, max_h=700, max_w=400):
        boxes = []
        contours, hierarchy = cv2.findContours(img_prepared, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        hierarchy_counter = collections.Counter()
        for el in hierarchy[0]:
            hierarchy_counter[el[3]] += 1
        h_mark = hierarchy_counter.most_common(1)[0][0]
        for idx, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            if hierarchy[0][idx][3] == h_mark and min_h < h < max_h and w < max_w:
                boxes.append([x, y, w, h])
        return boxes

    def get_symbols(img_prepared, model, predictions_list, min_h, max_h=700, max_w=400, save_interim_results=False):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if save_interim_results:
            img_contours = cv2.imread(filename)
        boxes = get_boxes(img_prepared, min_h, save_interim_results, max_h, max_w)
        symbols = []
        for box in boxes:
            (x, y, w, h) = box
            if save_interim_results:
                cv2.rectangle(img_contours, (x, y), (x + w, y + h), (30, 30, 30), 2)
            symbol_pic = img[y:y + h, x:x + w]
            size_max = max(w, h)
            out_pic = 255 * numpy.ones(shape=[size_max, size_max], dtype=numpy.uint8)
            if w > h:
                y_pos = size_max // 2 - h // 2
                out_pic[y_pos:y_pos + h, 0:w] = symbol_pic
            elif h > w:
                x_pos = size_max // 2 - w // 2
                out_pic[0:h, x_pos:x_pos + w] = symbol_pic
            else:
                out_pic = symbol_pic
            binary_out_pic = cv2.threshold(out_pic, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            binary_out_pic = cv2.resize(binary_out_pic, (56, 56))
            new_symbol = Symbol(binary_out_pic, (x, y, w, h), model, predictions_list)
            symbols.append(new_symbol)
        if save_interim_results:
            cv2.imwrite('img_contours.png', img_contours)
            print('Границы контуров в файле img_contours.png')
        return symbols

    def get_edges(boxes, threshn):
        boxes.sort(key=lambda box: box[1])
        new_list = []
        for el in boxes:
            new_list.append([])
            new_list[-1].append(1)
            new_list[-1].append(el)
        rows = []
        for idx in range(len(new_list)):
            if new_list[idx][0] == 1:
                rows.append([])
                rows[-1].append(new_list[idx][1])
                new_list[idx][0] = 0
                for idx1 in range(idx + 1, len(new_list)):
                    if new_list[idx1][0] == 1 and abs(new_list[idx1][1][1] - new_list[idx][1][1]) < threshn:
                        rows[-1].append(new_list[idx1][1])
                        new_list[idx1][0] = 0
        img_for_rows = cv2.imread(filename)
        edges = []
        for row in rows:
            x = img_for_rows.shape[0]
            for box in row:
                if box[1] + box[3] < x:
                    x = box[1] + box[3]
            edges.append(x)
        return edges

    def symbols_to_rows(symbols_list, edges, save_interim_results=False):
        symbols_list.sort(key=lambda symbol: symbol.coordinates[1], reverse=True)
        rows = []
        for i in range(len(edges)):
            rows.append([])
        i = 0
        for symbol in symbols_list:
            if i < len(edges) - 1:
                if symbol.coordinates[1] > edges[-i - 2]:
                    rows[i].append(symbol)
                else:
                    i += 1
            else:
                rows[i].append(symbol)
        rows.reverse()
        diacs = '҆,́̾҇҃ⷮ.ѣⷣ'
        final_rows = []
        m = True
        for row in rows:
            if row == []:
                continue
            if m:
                final_rows.append([])
            for symbol in row:
                final_rows[-1].append(symbol)
            for symbol in row:
                if symbol.text not in diacs:
                    m = True
                    break
            else:
                m = False
        for idx in range(len(final_rows)):
            final_rows[idx].sort(key=lambda symbol: symbol.coordinates[0])
        if save_interim_results:
            img_for_rows = cv2.imread(filename)
            for row in final_rows:
                x1, y1, x2, y2 = img_for_rows.shape[1], img_for_rows.shape[0], 0, 0
                for symbol in row:
                    if symbol.coordinates[0] < x1:
                        x1 = symbol.coordinates[0]
                    if symbol.coordinates[1] < y1:
                        y1 = symbol.coordinates[1]
                    if symbol.coordinates[2] > x2:
                        x2 = symbol.coordinates[2]
                    if symbol.coordinates[3] > y2:
                        y2 = symbol.coordinates[3]
                cv2.rectangle(img_for_rows, (x1, y1), (x2, y2), (30, 30, 30), 5)
            cv2.imwrite('img_rows.png', img_for_rows)
            print('Границы строк в файле img_rows.png')
        return final_rows

    def get_raw_str(row, space):
        raw_str = ''
        for idx in range(len(row)):
            if idx > 0 and row[idx].coordinates[0] - row[idx - 1].coordinates[2] > space:
                raw_str += ' '
            raw_str += row[idx].text
        raw_str = raw_str.replace('iк', 'к')
        raw_str = raw_str.replace(' ᾿ ', ' , ')
        return raw_str

    img_prepared = prepare_img(filename, save_interim_results)
    symbols = get_symbols(img_prepared, model, predictions_list, min_h=15, save_interim_results=save_interim_results)
    boxes = get_boxes(img_prepared, min_h=50)
    edges = get_edges(boxes, 70)
    rows = symbols_to_rows(symbols, edges, save_interim_results=save_interim_results)
    text = ''
    for row in rows:
        text += get_raw_str(row, 50) + '\n'
    return text


def main(model_name, predictions_file):
    pdffile = clicked() #input('Введите имя пдф-файла: ')
    page_number = int(input('Введите номер страницы: '))
    result_dir = pdffile[0:-4]+'/page'+str(page_number)
    os.makedirs(result_dir, exist_ok=True)
    print('Результаты будут сохранены в папке '+result_dir)
    print('0')
    fname = load_page_from_pdf(pdffile, page_number)
    print('1')
    model = keras.models.load_model(model_name)
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
main('c:/Coding/CSLAV/CSLAV_OCR_1.0-main/CSLAV_OCR_1.0/machine.h5', 'c:/Coding/CSLAV/CSLAV_OCR_1.0-main/CSLAV_OCR_1.0/predictions.txt')
