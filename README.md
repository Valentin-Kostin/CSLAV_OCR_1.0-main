# CSLAV_OCR_1.0

Оптическое распознавание печатных церковнославянских текстов XXVII в.
Сейчас код заточен на единственный документ (<a href = "https://drive.google.com/file/d/1Kopdod_E4U6r7Ft9FukMAtBI655PzUQc/view?usp=sharing">Евангелие 1606 г.</a>), однако планируется расширение за счет других документов.

<a href="https://docs.google.com/presentation/d/1R4N4ky2yfZKIDexc6jP9J3RDOxKdCvDcEwbooAx9UWw/edit?usp=sharing">Презентация проекта.</a>

Для работы программы необходимы следующие файлы:
- файл с итоговым кодом (<a href="https://github.com/PavelAstafyev/CSLAV_OCR_2.0/blob/main/reader.py">reader.py</a>). Выгружает страницу из pdf-файла, обрабатывает её, находит символы, сортирует их по строкам и внутри строк и получает текст, используя обученную нами нейросеть. Распознанный текст записывается в файл формата txt в кодировке utf-8. Также добавлено опциональное сохранение промежуточных результатов (обработанное изображение, границы символов, границы строк), иллюстрирующих работу программы (а также позволяющих отслеживать, на каких этапах происходят ошибки).
- файл с обученной нейросетью для распознавания отдельных символов (<a href="https://github.com/PavelAstafyev/CSLAV_OCR_2.0/blob/main/machine.h5">machine.h5</a>)
- файл с соответствиями предсказаний и символов (<a href="https://github.com/PavelAstafyev/CSLAV_OCR_2.0/blob/main/predictions.txt">predictions.txt</a>)
- собственно файл pdf для распознавания

Требуется установка библиотек:
- **PyMuPDF** для работы с pdf-файлами
- **opencv-python** для работы с изображениями
- **numpy** для работы с многомерными массивами
- **tensorflow** для работы с нейросетями

В файле <a href="https://github.com/PavelAstafyev/CSLAV_OCR_2.0/blob/main/model.py">model.py</a> содержится код, который мы использовали для создания, обучения и оценки нейросети. Для его запуска требуется установка библиотеки **matplotlib**, которую мы использовали для визуализации процесса обучения. <a href = "https://drive.google.com/drive/folders/1P8BmnVK_i-LL06Xi3sTd1LJ4WUQ1hb8h?usp=sharing">Наш датасет</a>. Модель можно переобучать на других датасетах (при соответствующей корректировке кода). Например, на втором нашем <a href = "https://drive.google.com/drive/folders/1652vWGuTw_keJiAn1P3hsQaxxpWM8RkR?usp=sharing">датасете</a> на основе <a href = "https://disk.yandex.ru/d/3PhffLECcvDdq/УставОхристианскомЖитии">Устава о домашней молитве</a>.
