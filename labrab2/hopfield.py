import numpy as np
import random

# Вариант №2
# | n   |	m	| № векторов |
# | --- | ----- | ---------- |
# | 11  |	4	| 1,8,2,11   |

# | №  | Данные вектора                                                                |
# | 1  | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 |
# | 2  | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
# | 8  | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |
# | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |


Y1 = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0])
Y2 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
Y3 = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1])
Y4 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

Y = np.array([Y1, Y2, Y3 ,Y4])

max_nn_algorithm_iteration = 10


def sign(x):
    if isinstance(x, (list, np.ndarray)):
        return np.array([1 if el > 0 else 0 for el in x])
    elif isinstance(x, (int, float)):
        return 1 if x > 0 else 0


def copy_noisy(array, bits_count: int) -> list:
    noisy_positions = random.sample(range(len(array)), bits_count)
    y_noisy = array.copy()
    for i in noisy_positions:
        y_noisy[i] = y_noisy[i] ^ 1
    return y_noisy


def async_method(y_noisy, W, y_original):
    y_noisy = y_noisy.copy()
    print(f"y_original = {y_original}")
    print(f"y_noisy    = {y_noisy}")
    for i in range(max_nn_algorithm_iteration):
        y_in = y_noisy.copy()
        print(f"\nStage {i+1}:")
        for j in range(len(y_noisy)):
            s_j = np.dot(y_noisy, W[:, j])
            y_noisy[j] = sign(s_j)
            print(f"y_model({j+1:02}) = [{" ".join(f" {x} " if i != j else f"({x})" for i, x in enumerate(y_noisy))}]")
        if np.array_equal(y_in, y_noisy):
            if np.array_equal(y_in, y_original):
                print(f"y_stage_{i+1} == y_original, relaxation with correct value")
                return True
            else:
                print(f"y_stage_{i+1} == y_stage_{i} != y_original, relaxation with wrong value")
                return False
        else:
            print(f"y_stage{i+1} != y_stage{i}, continue calculation")
    print(f"model can’t find relaxation, max iteration = {max_nn_algorithm_iteration}")
    return False

def sync_method(y_noisy, W, y_original):
    y_noisy = y_noisy.copy()
    print(f"y_original = {y_original}")
    print(f"y_noisy    = {y_noisy}")
    y_out = y_noisy.copy()
    for i in range(max_nn_algorithm_iteration):
        print(f"\nStage {i+1}:")
        y_in = y_out
        s = np.dot(y_in, W)
        y_out = sign(s)
        print(f"y_model({i+1}) = {y_out}")
        if np.array_equal(y_out, y_in):
            if np.array_equal(y_out, y_original):
                print(f"y_stage_{i + 1} == y_original, relaxation with correct value")
                return True
            else:
                print(f"y_stage_{i + 1} == y_stage_{i} != y_original, relaxation with wrong value")
                return False
        else:
            print(f"y_stage{i + 1} != y_stage_{i}, continue calculation")
            print(f"model can’t find relaxation, max iteration = {max_nn_algorithm_iteration}")
    return False


def main():
    print("Сеть Хопфилда:")

    print("\nSource vectors:")
    for y_idx, y_original in enumerate(Y):
        print(f"y{y_idx + 1} = {y_original}")

    W = np.dot(np.matrix.transpose(Y * 2 - 1), (Y * 2 - 1)) - np.identity(Y1.size)

    y1_noisy = copy_noisy(Y1, 1)

    print(f"\n\tAsync example for y1:")
    async_method(y1_noisy, W, Y1)

    print(f"\n\tSync example for y1:")
    sync_method(y1_noisy, W, Y1)

    # for y_idx, y_original in enumerate(Y):
    #     y_noisy = copy_noisy(y_original, 1)
    #
    #     print(f"\n\tAsync y{y_idx + 1}:")
    #     async_method(y_noisy, W, y_original)
    #
    #     print(f"\n\tSync y{y_idx + 1}:")
    #     sync_method(y_noisy, W, y_original)

if __name__ == '__main__':
    main()
