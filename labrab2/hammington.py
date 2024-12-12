# Вариант №2
# | n   |	m	| № векторов |
# | --- | ----- | ---------- |
# | 11  |	4	| 1,8,2,11   |

# | №  | Данные вектора                                                                |
# | 1  | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 |
# | 2  | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
# | 8  | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |
# | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

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
    for i in range(bit):
        noisedVariant[i] = 1 if noisedVariant[i] == 0 else 0
    return noisedVariant


#Сеть Хэмминга
def hammingNetwork(noisedVariant, originalVariants, neededOriginal, E):
    if(isPrintingAvailable):
        print(f"\ty{counter + 1}_original = {neededOriginal}\n\ty{counter + 1}_noised   = {noisedVariant}")

    midValues = [0] * len(originalVariants)
    for i in range(len(midValues)):
        for j in range(len(originalVariants[i])):
            midValues[i] += originalVariants[i][j] * noisedVariant[j]

        midValues[i] += len(originalVariants[i])
        midValues[i] /= 2

    numberOfTries = 0
    newOutput = midValues[:]

    while(numberOfTries < 20):
        originalOutput = newOutput[:]
        for i in range(len(newOutput)):
            for j in range(len(newOutput)):
                if(i != j):
                    newOutput[i] -= E * originalOutput[j]
            newOutput[i] = newOutput[i] if newOutput[i] > 0 else 0

        if(isPrintingAvailable):
            print(f"\twinner ({numberOfTries + 1}) = {newOutput}")

        numberOfWinners = 0
        for i in range(len(newOutput)):
            if(newOutput[i] > 0):
                numberOfWinners += 1

        if(numberOfWinners == 1):
            break

        numberOfTries += 1

    for i in range(len(newOutput)):
        if(newOutput[i] <= 0):
            continue

        if(isPrintingAvailable):
            print(f"\ty{counter + 1}_model ({numberOfTries + 1})\t= {originalVariants[i]}")
            print(f"\ty{counter + 1}_original \t= {neededOriginal}")

        if(originalVariants[i] == neededOriginal):
            if(isPrintingAvailable):
                print(f"\ty{counter + 1}_model ({numberOfTries + 1}) == y_original -> correct\n")
            return True

        if(isPrintingAvailable):
            print(f"\ty{counter + 1}_model ({numberOfTries + 1}) != y_original -> incorrect\n")

        return False

def hammingNetworkResults(VECTORS):
    global counter, isPrintingAvailable

    INPUT_VALUES_X = [sublist[:] for sublist in VECTORS]
    E = 1 / len(INPUT_VALUES_X) / 2
    maxY = [0] * len(INPUT_VALUES_X)

    print(f"\nСеть Хэмминга\n")
    print(f"1. Source vectors:\n")

    for i in range(len(INPUT_VALUES_X)):
        print(f"y{i + 1} = {INPUT_VALUES_X[i]}")

    print(f"\n2. Y-input:\n")

    counter = 0
    for i in range(len(INPUT_VALUES_X)):
        isPrintingAvailable = True
        for j in range(len(INPUT_VALUES_X[i])):
            noisedVariant = addNoise(INPUT_VALUES_X[i][:], j + 1)
            if(hammingNetwork(noisedVariant, INPUT_VALUES_X, INPUT_VALUES_X[i], E)):
                if(maxY[i] < j + 1):
                    maxY[i] = j + 1
            isPrintingAvailable = False
        counter += 1

    print(f"3. Maximum number of recognised noisy bits:\n")
    for i in range(len(maxY)):
        print(f"\ty_{i + 1} = {maxY[i]}")


def main():
    global counter, isPrintingAvailable

    VECTORS_HAMMINGTON = [
        [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0], # 1
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], # 2
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1], # 8
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # 11
    ]

    hammingNetworkResults(VECTORS_HAMMINGTON)

    return

if __name__ =="__main__":
    main()