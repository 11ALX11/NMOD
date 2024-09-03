import math
from random import random


# Вариант №3


#   Функция для тестирования:
#   y = a * sin( sin( b * x ) ) + d

#   |-----------------------------------|
#   |   a   b    d   Кол-во входов ИНС  |
#   |   3	7	0.3	        5           |
#   |-----------------------------------|

#   т.е. имеем:
#   y = 3 * sin( sin( 7 * x ) ) + 0.3
#   Период функции = (2*Pi)/7 = 0.8975979010256552
INPUT_DATA_MAX_STEP = 0.07
INPUT_DATA_MIN_STEP = 0.01

#   Графическая интерпретация НС на 5 входов
#   x1 --> () w1 --\
#   x2 --> () w2 -\ \
#   x3 --> () w3 ----> () --> y
#   x4 --> () w4 -/ /
#   x5 --> () w5 --/


ALPHA = 0.3         # шаг обучения 0 < a < 1
E_OPTIMAL = 1e-4    # минимальная среднеквадратичная ошибка НС
THETA = 0           # "theta is some constant" - цитата из вики;
                    # используется при вычислении выходного значения НС

NN_WIDTH = 5        # количество входных образов (Кол-во входов ИНС)

#   Количество значений функции; для обучения и тестирования
LEARN_DATA_AMOUNT    = 30   # для обучения
TEST_DATA_AMOUNT     = 15   # для тестирования
DATA_AMOUNT          = LEARN_DATA_AMOUNT + TEST_DATA_AMOUNT     # кол-во значений

WEIGHTS_RANGE_LOW    = -0.5  # нижняя  граница для весов
WEIGHTS_RANGE_TOP    =  0.5  # верхняя граница для весов
WEIGHTS_RANGE_LENGTH = WEIGHTS_RANGE_TOP - WEIGHTS_RANGE_LOW    # длина диапазона


input_data  = []    # x's (иксы)
data_values = []    # y's (игрики)

weights     = []    # веса НС -0.5 < w < 0.5

# подготовка, инициализация данных
def init_data():
    x = 0
    i = 0

    while i < DATA_AMOUNT:

        y = 3 * math.sin( math.sin( 7 * x ) ) + 0.3

        input_data.append(x)
        data_values.append(y)

        x += INPUT_DATA_MAX_STEP * random() + INPUT_DATA_MIN_STEP
        i += 1

    print_stage12()

def print_stage12():
    print("Stage 1&2: data preparing and split:\n")

    print("train_data:")
    i = 0
    while i < LEARN_DATA_AMOUNT:
        print(f"x{i+1} = {input_data[i]}; y{i+1} = {data_values[i]}")
        i += 1

    print("\ntest_data:")
    while i < DATA_AMOUNT:
        print(f"x{i + 1} = {input_data[i]}; y{i + 1} = {data_values[i]}")
        i += 1


# подготовить начальные значения для весов
def init_weights():
    i = 0
    while i < NN_WIDTH:
        weight = random() * WEIGHTS_RANGE_LENGTH + WEIGHTS_RANGE_LOW    # удерживаем в диапазоне
        weights.append(weight)
        i += 1


def main():
    init_data()
    init_weights()
    # TODO: stage 3-5


if __name__ == '__main__':
    main()
