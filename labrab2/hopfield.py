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


def main():
    print("Сеть Хопфилда:")

    W = np.dot(np.matrix.transpose(Y * 2 - 1), (Y * 2 - 1)) - np.identity(Y1.size)

    Y1Mod = copy_noisy(Y1, 1)

    y1 = sign(np.dot(Y1Mod, W[:][0]))
    y2 = sign(np.dot(Y1Mod, W[:][1]))
    y3 = sign(np.dot(Y1Mod, W[:][2]))
    y4 = sign(np.dot(Y1Mod, W[:][3]))


if __name__ == '__main__':
    main()
