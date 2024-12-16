# Вариант №2
# | n   |	m	| № векторов |
# | --- | ----- | ---------- |
# | 11  |	4	| 1,8,2,11   |

# | №  | Данные вектора                                                                |
# | 1  | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 |
# | 2  | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
# | 8  | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |
# | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

import copy
import random

#Hopfield Utils
asyncMethod = 1
syncMethod = 2

#Matrix Utils
add = 1
substract = 2
multiply = 3
divide = 4

#Printing Utils
counter = 0
isPrintingAvailable = True

def matrixAndNumberOperation(matrixA, number, operation):
    if(operation == add):
        matrix = [0] * len(matrixA)
        for i in range(len(matrixA)):
            matrix[i] = [0] * len(matrixA[i])
            for j in range(len(matrixA[i])):
                matrix[i][j] = matrixA[i][j] + number
        return matrix
    elif(operation == substract):
        matrix = [0] * len(matrixA)
        for i in range(len(matrixA)):
            matrix[i] = [0] * len(matrixA[i])
            for j in range(len(matrixA[i])):
                matrix[i][j] = matrixA[i][j] - number
        return matrix
    elif(operation == multiply):
        matrix = [0] * len(matrixA)
        for i in range(len(matrixA)):
            matrix[i] = [0] * len(matrixA[i])
            for j in range(len(matrixA[i])):
                matrix[i][j] =  matrixA[i][j] * number
        return matrix
    elif(operation == divide):
        matrix = [0] * len(matrixA)
        for i in range(len(matrixA)):
            matrix[i] = [0] * len(matrixA[i])
            for j in range(len(matrixA[i])):
                matrix[i][j] = matrixA[i][j] / number
        return matrix
    return None

def matrixAndMatrixOperation(matrixA, matrixB, operation):
    if(operation == add):
        if(len(matrixA) != len(matrixB) or len(matrixA[0]) != len(matrixB[0])):
            return None
        matrix = [0] * len(matrixA)
        for i in range(len(matrixA)):
            matrix[i] = [0] * len(matrixA[i])
            for j in range(len(matrixA[i])):
                matrix[i][j] = matrixA[i][j] + matrixB[i][j]
        return matrix
    elif(operation == substract):
        if(len(matrixA) != len(matrixB) or len(matrixA[0]) != len(matrixB[0])):
            return None
        matrix = [0] * len(matrixA)
        for i in range(len(matrixA)):
            matrix[i] = [0] * len(matrixA[i])
            for j in range(len(matrixA[i])):
                matrix[i][j] -= matrixB[i][j]
        return matrix
    elif(operation == multiply):
        if(len(matrixA[0]) != len(matrixB)):
            return None
        matrix = [0] * len(matrixA)
        for i in range(len(matrixA)):
            matrix[i] = [0] * len(matrixB[0])
            for j in range(len(matrixB[0])):
                for k in range(len(matrixA[0])):
                    matrix[i][j] += matrixA[i][k] * matrixB[k][j]
        return matrix
    return None

def transposeMatrix(matrixA):
    matrix = [0] * len(matrixA[0])
    for i in range(len(matrix)):
        matrix[i] = [0] * len(matrixA)
        for j in range(len(matrix[i])):
            matrix[i][j] = matrixA[j][i]
    return matrix

def addNoise(noisedVariant, bit):
    length = len(noisedVariant)
    # Случайно выбираем уникальные индексы для зашумления
    indices_to_flip = random.sample(range(length), min(bit, length))

    for i in indices_to_flip:
        noisedVariant[i] = 1 if noisedVariant[i] == 0 else 0

    return noisedVariant


#Сеть Хопфилда:
def initWeightsHopfield(INPUT_VALUES):
    # Инициализация весовых коэффициентов (2Y - 1)^T * (2Y - 1) - I
    weights = copy.deepcopy(INPUT_VALUES)
    weights = matrixAndNumberOperation(weights, 2, multiply)
    weights = matrixAndNumberOperation(weights, 1, substract)
    weights = matrixAndMatrixOperation(transposeMatrix(weights), weights, multiply)

    for i in range(len(weights)):
        weights[i][i] -= 1

    return weights

def clamp(number):
    return 1 if number > 0 else 0

def asyncMethod(variant, noisedVariant, variantWeight):
    global counter, isPrintingAvailable
    numberOfTries = 0

    if(isPrintingAvailable):
        print(f"\ty{counter + 1}_original = {variant}\n\ty{counter + 1}_noised   = {noisedVariant}\n")

    while(numberOfTries < 10):
        previousVariant = noisedVariant[:]

        if(isPrintingAvailable):
            print(f"\tStage {numberOfTries + 1}:")

        for i in range(len(variantWeight)):
            sum = 0
            for j in range(len(variantWeight[i])):
                sum += noisedVariant[j] * variantWeight[j][i]

            noisedVariant[i] = clamp(sum)
            if(isPrintingAvailable):
                str = f"["
                isBracketSet = False
                for k in range(len(noisedVariant) - 1):
                    if(k == i):
                        str += f"({noisedVariant[k]}), "
                        isBracketSet = True
                    else:
                        str += f"{noisedVariant[k]}, "

                if(not isBracketSet):
                    str += f"({noisedVariant[k]})]"
                else:
                    str += f"{noisedVariant[k]}]"

                print(f"\ty{counter + 1}_model ({i + 1}) = {str}")

        if(noisedVariant == previousVariant):
            if(isPrintingAvailable):
                print(f"\ty{counter + 1}_stage_{numberOfTries + 1} == y_previous -> relaxation, incorrect\n")
            return False

        if(noisedVariant != variant):
            numberOfTries += 1
            continue

        if(isPrintingAvailable):
            print(f"\ty{counter + 1}_stage_{numberOfTries + 1} == y_original -> relaxation, correct\n")
        return True

    if(isPrintingAvailable):
        print(f"\ty{counter + 1}_stage_{numberOfTries + 1} != y_original -> incorrect\n")
    return False

def syncMethod(variant, noisedVariant, variantWeight):
    global counter, isPrintingAvailable
    numberOfTries = 0

    if(isPrintingAvailable):
        print(f"\ty{counter + 1}_original = {variant}\n\ty{counter + 1}_noised   = {noisedVariant}\n")

    while (numberOfTries < 10):
        previousVariant = noisedVariant[:]
        noisedVariant = matrixAndMatrixOperation([noisedVariant], variantWeight, multiply)[0]

        for i in range(len(variantWeight)):
            noisedVariant[i] = clamp(noisedVariant[i])

        if(isPrintingAvailable):
            print(f"\tStage {numberOfTries + 1}:\n\ty{counter + 1}_model (1) = {noisedVariant}")

        if(noisedVariant == previousVariant):
            if(isPrintingAvailable):
                print(f"\ty{counter + 1}_stage_{numberOfTries + 1} == y_previous -> relaxation, incorrect\n")
            return False

        if(noisedVariant != variant):
            numberOfTries += 1
            continue

        if(isPrintingAvailable):
            print(f"\ty{counter + 1}_stage_{numberOfTries + 1} == y_original -> relaxation, correct\n")

        return True

    if(isPrintingAvailable):
        print(f"\ty{counter + 1}_stage_{numberOfTries + 1} != y_original -> incorrect\n")

    return False

def hopfieldNetwork(variant, noisedVariant, variantWeight, method):
    if(method == asyncMethod):
        return asyncMethod(variant, noisedVariant, variantWeight)
    elif(method == syncMethod):
        return syncMethod(variant, noisedVariant, variantWeight)

    return False

def hopfieldResults(VECTORS):
    global counter, isPrintingAvailable
    INPUT_VALUES_HOPFIELD = [sublist[:] for sublist in VECTORS]

    print(f"\nСеть Хопфилда:\n")
    print(f"1. Source vectors:\n")

    for i in range(len(INPUT_VALUES_HOPFIELD)):
        print(f"\ty{i + 1} = {INPUT_VALUES_HOPFIELD[i]}")

    WEIGHTS_HOPFIELD = initWeightsHopfield(INPUT_VALUES_HOPFIELD)
    maxAsync = [0] * len(INPUT_VALUES_HOPFIELD)
    maxSync = [0] * len(INPUT_VALUES_HOPFIELD)

    print(f"\n2. Async method:\n")

    counter = 0
    isPrintingAvailable = True

    for i in range(len(INPUT_VALUES_HOPFIELD)):
        isPrintingAvailable = True
        for j in range(len(INPUT_VALUES_HOPFIELD[i])):
            noisedVariant = addNoise(INPUT_VALUES_HOPFIELD[i][:], j + 1)
            if(hopfieldNetwork(INPUT_VALUES_HOPFIELD[i], noisedVariant, WEIGHTS_HOPFIELD, asyncMethod)):
                if(maxAsync[i] < j + 1):
                    maxAsync[i] = j + 1
            isPrintingAvailable = False
        counter += 1

    print(f"3. Sync method:\n")

    counter = 0
    for i in range(len(INPUT_VALUES_HOPFIELD)):
        isPrintingAvailable = True
        for j in range(len(INPUT_VALUES_HOPFIELD[i])):
            noisedVariant = addNoise(INPUT_VALUES_HOPFIELD[i][:], j + 1)
            if(hopfieldNetwork(INPUT_VALUES_HOPFIELD[i], noisedVariant, WEIGHTS_HOPFIELD, syncMethod)):
                if(maxSync[i] < j + 1):
                    maxSync[i] = j + 1
            isPrintingAvailable = False
        counter += 1

    print(f"4. Maximum number of recognised noisy bits:")
    print(f"\tAsync:")

    for i in range(len(maxAsync)):
        print(f"\ty_{i + 1} = {maxAsync[i]}")

    print(f"\tSync:")
    for i in range(len(maxSync)):
        print(f"\ty_{i + 1} = {maxSync[i]}")


def main():
    global counter, isPrintingAvailable

    # VECTORS_HOPFIELD = [
    #     [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0], # 1
    #     [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], # 2
    #     [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1], # 8
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # 11
    # ]

    VECTORS_HOPFIELD = [
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    ]

    hopfieldResults(VECTORS_HOPFIELD)

    return

if __name__ =="__main__":
    main()