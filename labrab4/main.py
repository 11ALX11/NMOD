import math
from random import random

import matplotlib.pyplot as plt

# Вариант №2


#   Функция для тестирования:
#   f(x) = cos(x) + sin(4x) / 2 - cos(3x)
#   Период функции =  6.28319
INPUT_DATA_MIN_STEP = 0.10


ALPHA = 0.05        # шаг обучения 0 < a < 1
E_OPTIMAL = 1e-4    # минимальная среднеквадратичная ошибка НС

NN_WIDTH = 10        # количество входных образов (Кол-во входов ИНС)
# n = NN_WIDTH and = m; so  n x m x 1

#   Количество значений функции; для обучения и тестирования
LEARN_DATA_AMOUNT    = 60   # для обучения
TEST_DATA_AMOUNT     = 30   # для тестирования
DATA_AMOUNT          = LEARN_DATA_AMOUNT + TEST_DATA_AMOUNT     # = 90 - кол-во значений

input_values    = []    # x's (иксы)
data_values     = []    # y's (игрики)

#weights1        = [[random()/2 - 0.25] * NN_WIDTH for i in range(NN_WIDTH)]
weights1 = [[0.11160823972369743, 0.11160823972369743, 0.11160823972369743, 0.11160823972369743, 0.11160823972369743, 0.11160823972369743, 0.11160823972369743, 0.11160823972369743, 0.11160823972369743, 0.11160823972369743], [-0.24705139643554985, -0.24705139643554985, -0.24705139643554985, -0.24705139643554985, -0.24705139643554985, -0.24705139643554985, -0.24705139643554985, -0.24705139643554985, -0.24705139643554985, -0.24705139643554985], [-0.1539637603281181, -0.1539637603281181, -0.1539637603281181, -0.1539637603281181, -0.1539637603281181, -0.1539637603281181, -0.1539637603281181, -0.1539637603281181, -0.1539637603281181, -0.1539637603281181], [-0.2305872611214148, -0.2305872611214148, -0.2305872611214148, -0.2305872611214148, -0.2305872611214148, -0.2305872611214148, -0.2305872611214148, -0.2305872611214148, -0.2305872611214148, -0.2305872611214148], [-0.24889380831464342, -0.24889380831464342, -0.24889380831464342, -0.24889380831464342, -0.24889380831464342, -0.24889380831464342, -0.24889380831464342, -0.24889380831464342, -0.24889380831464342, -0.24889380831464342], [0.06313087150584368, 0.06313087150584368, 0.06313087150584368, 0.06313087150584368, 0.06313087150584368, 0.06313087150584368, 0.06313087150584368, 0.06313087150584368, 0.06313087150584368, 0.06313087150584368], [-0.18762330619124018, -0.18762330619124018, -0.18762330619124018, -0.18762330619124018, -0.18762330619124018, -0.18762330619124018, -0.18762330619124018, -0.18762330619124018, -0.18762330619124018, -0.18762330619124018], [0.10212046501687089, 0.10212046501687089, 0.10212046501687089, 0.10212046501687089, 0.10212046501687089, 0.10212046501687089, 0.10212046501687089, 0.10212046501687089, 0.10212046501687089, 0.10212046501687089], [-0.1744751744155199, -0.1744751744155199, -0.1744751744155199, -0.1744751744155199, -0.1744751744155199, -0.1744751744155199, -0.1744751744155199, -0.1744751744155199, -0.1744751744155199, -0.1744751744155199], [-0.12448939593913061, -0.12448939593913061, -0.12448939593913061, -0.12448939593913061, -0.12448939593913061, -0.12448939593913061, -0.12448939593913061, -0.12448939593913061, -0.12448939593913061, -0.12448939593913061]]
                        # w входной -> промежуточный
#theta1          = []    # порог
theta1 = [-0.09763942924945213, 0.26979981763492755, -0.24375289439714143, -0.07117584629553697, -0.18752803762879644, 0.41598525947228326, -0.11416763722886503, 0.10626460044476926, 0.012928600149857084, 0.36760267518412937]
#weights2        = []    # v промежуточный -> выходной
weights2 = [-0.13163808405766841, 0.4707180620440672, 0.3513396736912622, -0.3840186689753289, 0.4928004546881847, -0.05166867269455577, 0.08354264676491363, -0.06512108427839014, -0.3734508829646682, -0.06309994176690004]
#theta2          = random() - 0.5     # порог
theta2 = -0.4659100335197712
#weights3        = []    # контекстный -> промежуточный
weights3 = [0.03565954023376128, -0.11139317073758115, 0.46186012041872637, -0.29523543027021315, 0.2997178998587985, -0.10849077197612211, 0.027604010690461922, -0.3716995924023483, 0.09814665538604461, 0.15512207439081493]
last_y          = 0     # y(t-1)
p               = []
last_p          = []    # p[i](t-1)

error_current   = E_OPTIMAL + 1     # текущая ошибка НС

NN_data_predictions = []


def clamp(n, smallest, largest): return max(smallest, min(n, largest))


# подготовка, инициализация данных
def init_data():
    x = 0
    i = 0

    while i < DATA_AMOUNT:

        y = math.cos(x) + math.sin(4*x) / 2 - math.cos(3*x)

        input_values.append(x)
        data_values.append(y)

        x += INPUT_DATA_MIN_STEP
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
# (удалено в пользу простого обращения через список y (data_values))
def prepare_data():
    print_stage3()

def print_stage3():
    print("\nStage 3: prepare train/test data for NN:\n")

    print("train_data:")

    i = 0
    while i < LEARN_DATA_AMOUNT - NN_WIDTH + 1:
        in_value = data_values[i:i+NN_WIDTH]
        out_value = data_values[i+NN_WIDTH]

        inputs = ", ".join(f"y{i+j+1}({in_value[j]})" for j in range(0, NN_WIDTH))
        output = f"y{i+NN_WIDTH+1}({out_value})"

        if i + NN_WIDTH < LEARN_DATA_AMOUNT:
            print(f"{inputs} -> {output}")
        else:
            print("\ntest_data (y -> original value, y’ -> model output value):")
            print(f"{inputs} -> y'{i+NN_WIDTH+1}")

        i += 1


# возвращает y`, найденный с помощью НС
def get_y_NN(in_value: list) -> float:
    global last_y
    wp_sum = 0

    i = 0
    while i < NN_WIDTH:
        wx_sum = 0
        k = 0
        while k < NN_WIDTH:
            wx_sum += weights1[k][i] * in_value[k]
            k += 1

        p[i] = math.tanh(wx_sum + weights3[i] * last_y - theta1[i])
        wp_sum += weights2[i] * p[i]

        i += 1

    last_y = wp_sum - theta2
    return last_y

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

        fs = 1 - p[i] ** 2

        gamma = (y - e) * weights2[i]
        weights2[i]         = weights2[i]    - ALPHA * (y - e) * p[i]

        weights3[i]         = weights3[i]    - ALPHA * gamma * fs * last_p[i]
        theta1[i]           = theta1[i]      + ALPHA * gamma * fs

        k = 0
        while k < NN_WIDTH:
            weights1[k][i]  = weights1[k][i] - ALPHA * gamma * fs * in_values[k]
            k += 1

        last_p[i] = p[i]

        i += 1

    global theta2
    theta2                  = theta2         + ALPHA * (y - e)


# подготовить начальные значения для весов
def init_weights():
    i = 0
    while i < NN_WIDTH:
        #theta1.append(random() - 0.5)
        #weights2.append(random() - 0.5)
        #weights3.append(random() - 0.5)

        p.append(0.)
        last_p.append(0.)

        i += 1

    #print(weights1)
    #print(theta1)
    #print(weights2)
    #print(theta2)
    #print(weights3)
    #print(last_y)
    #print(p)
    #print(last_p)


# запустить тренировку (и тестирование) НС
def train():
    print("\nStage 4: train & test model")

    init_weights()
    global error_current

    last_train_loss = 1000.
    last_test_loss = 1000.
    generation_counter = 1

    while error_current > E_OPTIMAL:
        print(f"\nGeneration №{generation_counter}")
        error_current = 0.
        train_loss = 0

        global last_y
        last_y = 0
        i = 0
        while i < NN_WIDTH:
            p[i] = 0.
            last_p[i] = 0.
            i += 1

        global NN_data_predictions
        NN_data_predictions = []
        # i = LEARN_DATA_AMOUNT - NN_WIDTH
        # while i < LEARN_DATA_AMOUNT:
        #     NN_data_predictions.append(data_values[i])
        #     i += 1
        i = 0
        while i < NN_WIDTH:
            #append first NN_WIDTH
            NN_data_predictions.append((data_values[i]))
            i += 1

        i = 0
        while i < DATA_AMOUNT - NN_WIDTH:
            in_value = data_values[i:i + NN_WIDTH]
            expected_value = data_values[i + NN_WIDTH]

            y = get_y_NN(in_value)

            if i + NN_WIDTH < LEARN_DATA_AMOUNT:
                # тренировка
                mutate_weights(y, expected_value, in_value)
                y = get_y_NN(in_value)
            else:
                # тестирование
                #in_value = NN_data_predictions[-NN_WIDTH:]  # последние NN_WIDTH значений
                mutate_weights(y, expected_value, in_value)
                y = get_y_NN(in_value)
            # remember y's for plot
            NN_data_predictions.append(y)

            error = get_error(y, expected_value)
            error_current += error

            if i + NN_WIDTH + 1 == LEARN_DATA_AMOUNT:
                train_loss = error_current
                error_current = 0 # train and test separate

            i += 1

        print(f"train_loss: {train_loss}\ttest_loss: {error_current}")
        if error_current > E_OPTIMAL:
            print(f"test_loss > {E_OPTIMAL} -> continue training")
        else:
            print(f"test_loss < {E_OPTIMAL} -> stop training")
            break

        last_train_loss = train_loss
        last_test_loss = error_current
        generation_counter += 1

        if generation_counter > 1720:
            break


# вывести предсказания модели для лучшей эпохи
def print_stage5():
    print("\nStage 5: print full model outputs for best epoch\n")

    NN_data_predictions = []

    global last_y
    last_y = 0
    i = 0
    while i < NN_WIDTH:
        p[i] = 0.
        last_p[i] = 0.
        i += 1

    i = LEARN_DATA_AMOUNT - NN_WIDTH
    while i < LEARN_DATA_AMOUNT:
        NN_data_predictions.append(data_values[i])
        i += 1

    while i < DATA_AMOUNT:
        in_value = NN_data_predictions[-NN_WIDTH:]  # последние NN_WIDTH значений
        y = get_y_NN(in_value)

        NN_data_predictions.append(y)

        inputs = (", ".join(f"y{"'" if i + j - NN_WIDTH + 1 > LEARN_DATA_AMOUNT else ""}"
                            f"{i + j - NN_WIDTH + 1}"
                            f"({in_value[j]})"
                            for j in range(0, NN_WIDTH)))
        print(f"{inputs} -> y'{i + 1}({y})")

        i += 1


# выводит графики данных для сравнения
def plot_func():
    plt.figure()
    plt.subplot(211)
    plt.plot(input_values, data_values)

    # global last_y
    # last_y = 0
    # i = 0
    # while i < NN_WIDTH:
    #     p[i] = 0.
    #     last_p[i] = 0.
    #     i += 1
    #
    # NN_data_predictions = []
    # i = 0
    # while i < NN_WIDTH:
    #     NN_data_predictions.append(data_values[i])
    #     i += 1
    #
    # while i < DATA_AMOUNT:
    #     in_value = NN_data_predictions[-NN_WIDTH:]  # последние NN_WIDTH значений
    #     y = get_y_NN(in_value)
    #     NN_data_predictions.append(y)
    #     i += 1

    plt.subplot(212)
    plt.plot(input_values, NN_data_predictions)

    plt.show()


def main():
    init_data()
    prepare_data()

    train()
    print_stage5()

    plot_func()


if __name__ == '__main__':
    main()
