#крч есть +- документация и есть опциональные визуализации тоже за решетками

import os
from ocr_utils import process_image_to_text


def main():
    fname = input('Введите имя файла: ')
    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write(process_image_to_text(fname, 'predictions.txt', 'machine.h5'))


if __name__ == '__main__':
    main()

