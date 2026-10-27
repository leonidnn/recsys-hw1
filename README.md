# Домашнее задание 1 по рекомендательным системам

Студент: Некрылов Леонид

Датасет: MTS Library (МТС Библиотека) — https://www.kaggle.com/datasets/sharthz23/mts-library

Реализованы модели (папка `models/`):
- `knn.py` — item-based kNN
- `als.py` — iALS
- `ease.py` — EASE
- `fm.py` — факторизационная машина с метапризнаками (BPR)
- `timesvdpp.py` — timeSVD++
- `utils.py` — подготовка данных, разбиение, метрики

Все эксперименты в `experiments.ipynb`.

Запуск:
```
pip install -r requirements.txt
jupyter notebook experiments.ipynb
```
Данные скачиваются в папку `data/` первыми ячейками ноутбука.
