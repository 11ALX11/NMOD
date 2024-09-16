import math
from random import random
import matplotlib.pyplot as plt


# Вариант №3


#   Функция для тестирования:
#   y = a * sin( sin( b * x ) ) + d

#   |-----------------------------------|
#   |   a   b    d   Кол-во входов ИНС  |
#   |   3	7	0.3	        5           |
#   |-----------------------------------|

#   т.е. имеем:
#   y = 3 * sin( sin( 7 * x ) ) + 0.3
#   Период функции = (2*Pi)/7 = 0,8975979010256552
INPUT_DATA_MAX_STEP = 0.007
INPUT_DATA_MIN_STEP = 0.015

#   Графическая интерпретация НС на 5 входов
#   x1 --> () w1 --\
#   x2 --> () w2 -\ \
#   x3 --> () w3 ----> () --> y
#   x4 --> () w4 -/ /
#   x5 --> () w5 --/


ALPHA = 0.003       # шаг обучения 0 < a < 1
E_OPTIMAL = 1e-4    # минимальная среднеквадратичная ошибка НС

NN_WIDTH = 5        # количество входных образов (Кол-во входов ИНС)

#   Количество значений функции; для обучения и тестирования
LEARN_DATA_AMOUNT    = 50   # для обучения
TEST_DATA_AMOUNT     = 25   # для тестирования
DATA_AMOUNT          = LEARN_DATA_AMOUNT + TEST_DATA_AMOUNT     # = 45 - кол-во значений

WEIGHTS_RANGE_LOW    = -1 # нижняя  граница для весов
WEIGHTS_RANGE_TOP    =  1 # верхняя граница для весов
WEIGHTS_RANGE_LENGTH = WEIGHTS_RANGE_TOP - WEIGHTS_RANGE_LOW    # длина диапазона


input_values    = []    # x's (иксы)
data_values     = []    # y's (игрики)
NN_inout_values = []    # значения для входа в НС

weights         = []                # веса НС -0.5 < w < 0.5
error_current   = E_OPTIMAL + 1     # текущая ошибка НС
theta           = 0                 # порог НС;
                                    # используется при вычислении выходного значения НС


def clamp(n, smallest, largest): return max(smallest, min(n, largest))


# подготовка, инициализация данных
def init_data():
    x = 0
    i = 0

    while i < DATA_AMOUNT:

        y = 3 * math.sin( math.sin( 7 * x ) ) + 0.3

        input_values.append(x)
        data_values.append(y)

        x += INPUT_DATA_MAX_STEP * random() + INPUT_DATA_MIN_STEP
        i += 1

    print_stage12()

def print_stage12():
    print("\nStage 1&2: data preparing and split:\n")

    print("train_data:")
    i = 0
    while i < LEARN_DATA_AMOUNT:
        print(f"x{i+1} = {input_values[i]}; y{i + 1} = {data_values[i]}")
        i += 1

    print("\ntest_data:")
    while i < DATA_AMOUNT:
        print(f"x{i + 1} = {input_values[i]}; y{i + 1} = {data_values[i]}")
        i += 1


# создать список входных значений на входные нейроны
def prepare_data():
    # для обучения
    i = 0
    while i < LEARN_DATA_AMOUNT - NN_WIDTH:
        in_value = []

        j = i
        while j < NN_WIDTH + i:
            in_value.append(data_values[j])
            j += 1

        NN_inout_values.append([in_value, data_values[NN_WIDTH + i]])

        i += 1

    # и 1 начальный для тестирования
    in_value = []
    j = LEARN_DATA_AMOUNT - NN_WIDTH
    while j < LEARN_DATA_AMOUNT:
        in_value.append(data_values[j])
        j += 1
    NN_inout_values.append([in_value, 0]) # 0 - при тестировании выходное значение заранее не известно

    print_stage3()

def print_stage3():
    print("\nStage 3: prepare train/test data for NN:\n")

    print("train_data:")
    i = 0
    for inout_value in NN_inout_values:
        in_value = inout_value[0]
        out_value = inout_value[1]

        inputs = ", ".join(f"y{j+1}({in_value[j-i]})" for j in range(i, i + NN_WIDTH))
        output = f"y{i + NN_WIDTH + 1}({out_value})"

        if i + NN_WIDTH < LEARN_DATA_AMOUNT:
            print(f"{inputs} -> {output}")
        else:
            print("\ntest_data (y -> original value, y’ -> model output value):")
            print(f"{inputs} -> y'{i + NN_WIDTH + 1}")

        i += 1


# возвращает y`, найденный с помощью НС
def get_y_NN(in_value: list) -> float:
    w_sum = 0.
    #print() # TODO: debug cleanup

    i = 0
    while i < NN_WIDTH:
        w_sum += weights[i] * in_value[i]

        #print(f"w_sum = {w_sum}, w{i+1} = {weights[i]}, x = {in_value[i]}")# TODO: debug cleanup
        i += 1

    #print(f"theta = {theta}")# TODO: debug cleanup
    return w_sum - theta

# возвращает ошибку
# @param y - значение НС
# @param e - эталонное значение
def get_error(y, e) -> float: return 0.5 * (abs(y - e) ** 2)

# изменяет веса и порог НС
# @param y - значение НС
# @param e - эталонное значение
def mutate_weights(y, e, in_values: list):
    i = 0
    while i < NN_WIDTH:
        weights[i] = clamp(weights[i] - ALPHA * (y - e) * in_values[i],
                           WEIGHTS_RANGE_LOW,
                           WEIGHTS_RANGE_TOP)
        i += 1

    global theta
    theta = clamp(theta + ALPHA * (y - e),
                           WEIGHTS_RANGE_LOW,
                           WEIGHTS_RANGE_TOP)

# подготовить начальные значения для весов
def init_weights():
    i = 0
    while i < NN_WIDTH:
        weight = random() * WEIGHTS_RANGE_LENGTH + WEIGHTS_RANGE_LOW    # удерживаем в диапазоне
        weights.append(weight)
        i += 1

    global theta
    theta = random() * WEIGHTS_RANGE_LENGTH + WEIGHTS_RANGE_LOW

# запустить тренировку (и тестирование) НС
def train():
    print("\nStage 4: train & test model")

    init_weights()
    global error_current

    last_train_loss = 1000.
    last_test_loss = 1000.
    generation_counter = 1

    while error_current > E_OPTIMAL:
        if generation_counter > 500: break # TODO: debug cleanup
        print(f"\nGeneration №{generation_counter}")
        old_weights_debug = weights.copy() # TODO: debug cleanup
        error_current = 0.
        train_loss = 0
        inout_values = NN_inout_values.copy()

        i = 0
        while i < DATA_AMOUNT - NN_WIDTH:
            inout_value = inout_values[i]
            in_value = inout_value[0]
            # print(f"i - {i}; in_value - {in_value}") # TODO: debug cleanup
            out_value = inout_value[1]

            y = get_y_NN(in_value)
            #print(f"y`{i+NN_WIDTH+1} = {y}, y{i+NN_WIDTH+1} = {out_value}") # TODO: debug cleanup

            if i + NN_WIDTH < LEARN_DATA_AMOUNT:
                # тренировка
                #print(f"{i} - {get_error(y, out_value)}")  # TODO: debug cleanup
                mutate_weights(y, out_value, in_value)
                #print(f"{i} - {get_error(get_y_NN(in_value), out_value)}")  # TODO: debug cleanup
                y = get_y_NN(in_value)
            else:
                # тестирование
                j = 1
                new_in_value = []
                while j < NN_WIDTH:
                    new_in_value.append(in_value[j])
                    if j+1 == NN_WIDTH:
                        new_in_value.append(y)
                    j += 1

                inout_values.append([new_in_value, 0])

                out_value = data_values[i + NN_WIDTH] # переопределяем для проверки

            error = get_error(y, out_value)
            error_current += error

            if i + NN_WIDTH + 1 == LEARN_DATA_AMOUNT:
                train_loss = error_current

            i += 1

        print(f"train_loss: {train_loss}\ttest_loss: {error_current}")
        if error_current > E_OPTIMAL:
            if (last_train_loss < train_loss) and (last_test_loss < error_current) and (generation_counter > 5):
                print(f"train_loss > last_train_loss -> stop training")
                break
            else:
                print(f"test_loss > {E_OPTIMAL} -> continue training")
        else:
            print(f"test_loss < {E_OPTIMAL} -> stop training")

        #print(f"old_weights {old_weights_debug} -> \nnew_weights {weights}") # TODO: debug cleanup

        last_train_loss = train_loss
        last_test_loss = error_current
        generation_counter += 1


# вывести предсказания модели для лучшей эпохи
def print_stage5():
    print("\nStage 5: print full model outputs for best epoch\n")

    inout_values = NN_inout_values.copy()

    i = LEARN_DATA_AMOUNT - NN_WIDTH
    while i < DATA_AMOUNT - NN_WIDTH:
        inout_value = inout_values[i]
        in_value = inout_value[0]

        y = get_y_NN(in_value)

        inputs = ", ".join(f"y{j + 1}({in_value[j - i]})" for j in range(i, i + NN_WIDTH))
        output = f"y{i + NN_WIDTH + 1}({y})"
        print(f"{inputs} -> y'{i + NN_WIDTH + 1}({y})")

        j = 1
        new_in_value = []
        while j < NN_WIDTH:
            new_in_value.append(in_value[j])
            if j + 1 == NN_WIDTH:
                new_in_value.append(y)
            j += 1

        inout_values.append([new_in_value, 0])

        i += 1

    #print(f"y45 = {data_values[44]}")# TODO: debug cleanup


# выводит графики данных для сравнения
def plot_func():
    plt.figure()
    plt.subplot(211)
    plt.plot(input_values, data_values)

    NN_data_predictions = []
    i = 0
    while i < NN_WIDTH:
        NN_data_predictions.append(data_values[i])
        i += 1

    while i < DATA_AMOUNT:
        in_value = NN_data_predictions[-NN_WIDTH:]  # последние NN_WIDTH значений
        y = get_y_NN(in_value)
        NN_data_predictions.append(y)
        i += 1

    plt.subplot(212)
    plt.plot(input_values, NN_data_predictions)

    plt.show()


def main():
    init_data()

    prepare_data()
    # TODO: stage 4
    train()

    print_stage5()

    plot_func()


if __name__ == '__main__':
    main()
