# Вариант №2
# | n   |	m	| № векторов |
# | --- | ----- | ---------- |
# | 11  |	4	| 1,8,2,11   |

# | №  | Данные вектора                                                                |
# | 1  | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 |
# | 2  | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
# | 8  | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |
# | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

#Bidirectional Associative Memory Utils
startWithY = 1
startWithX = 2

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
        noisedVariant[i] = 1 if noisedVariant[i] == -1 else -1
    return noisedVariant


#Двунаправленная ассоциативная память:
def notLinearTransformation(previous, current):
    for i in range(len(current)):
        current[i] = previous[i] if current[i] == 0 else (1 if current[i] > 1 else -1)
    return current

def associativeMemoryY(originalY, noisedVariant, variantWeight):
    numberOfTries = 0
    previousY = noisedVariant[:]

    if(isPrintingAvailable):
        print(f"\ty{counter + 1}_original = {originalY}\n\ty{counter + 1}_noised   = {noisedVariant}")

    while(numberOfTries < 10):
        noisedVariant = matrixAndMatrixOperation([previousY], transposeMatrix(variantWeight), multiply)[0]#y->x
        notLinearTransformation(previousY, noisedVariant)

        if(isPrintingAvailable):
            print(f"\tStage {numberOfTries + 1}:\n\tx{counter + 1}_model (1) = {noisedVariant}")

        previousX = noisedVariant[:]
        noisedVariant = matrixAndMatrixOperation([noisedVariant], variantWeight, multiply)[0]#x->y
        notLinearTransformation(previousX, noisedVariant)

        if(isPrintingAvailable):
            print(f"\ty{counter + 1}_model (1) = {noisedVariant}")
        if(noisedVariant == originalY):
            if(isPrintingAvailable):
                print(f"\ty{counter + 1}_model (1) == y_original -> relaxation, correct\n")
            return True
        if(noisedVariant == previousY):
            if(isPrintingAvailable):
                print(f"\ty{counter + 1}_stage_{numberOfTries + 1} == y_previous -> relaxation, incorrect\n")
            return False

        previousY = noisedVariant[:]
        numberOfTries += 1

    if(isPrintingAvailable):
        print(f"\ty{counter + 1}_stage_{numberOfTries + 1} != y_original -> incorrect\n")

    return False

def associativeMemoryX(originalX, noisedVariant, variantWeight):
    numberOfTries = 0
    previousX = noisedVariant[:]

    if(isPrintingAvailable):
        print(f"\tx{counter + 1}_original = {originalX}\n\ty{counter + 1}_noised   = {noisedVariant}")

    while(numberOfTries < 10):
        noisedVariant = matrixAndMatrixOperation([noisedVariant], variantWeight, multiply)[0]#x->y
        notLinearTransformation(previousX, noisedVariant)

        if(isPrintingAvailable):
            print(f"\tStage {numberOfTries + 1}:\n\ty{counter + 1}_model (1) = {noisedVariant}")

        previousY = noisedVariant[:]
        noisedVariant = matrixAndMatrixOperation([previousY], transposeMatrix(variantWeight), multiply)[0]#y->x
        notLinearTransformation(previousY, noisedVariant)

        if(isPrintingAvailable):
            print(f"\tx{counter + 1}_model (1) = {noisedVariant}")

        if(noisedVariant == originalX):
            if(isPrintingAvailable):
                print(f"\tx{counter + 1}_model (1) == x_original -> relaxation, correct\n")
            return True
        if(noisedVariant == previousX):
            if(isPrintingAvailable):
                print(f"\tx{counter + 1}_stage_{numberOfTries + 1} == x_previous -> relaxation, incorrect\n")
            return False

        previousX = noisedVariant[:]
        numberOfTries += 1

    if(isPrintingAvailable):
        print(f"\tx{counter + 1}_stage_{numberOfTries + 1} != x_original -> incorrect\n")
    return False

def bidirectionalAssociativeMemory(variant, noisedVariant, variantWeight, method):
    if(method == startWithY):
        return associativeMemoryY(variant, noisedVariant, variantWeight)
    elif(method == startWithX):
        return associativeMemoryX(variant, noisedVariant, variantWeight)

def bidirectionalAssociativeMemoryResults(VECTORS, N, M):
    global counter, isPrintingAvailable

    INPUT_VALUES_X = [sublist[:N] for sublist in VECTORS]
    INPUT_VALUES_Y = [sublist[(len(VECTORS[0]) - M):] for sublist in VECTORS]

    WEIGHTS_BIDIRECTIONAL = matrixAndMatrixOperation(transposeMatrix(INPUT_VALUES_X), INPUT_VALUES_Y, multiply)

    maxY = [0] * len(INPUT_VALUES_Y)
    maxX = [0] * len(INPUT_VALUES_X)

    print(f"\nДвунаправленная ассоциативная память: \n")
    print(f"1. Source vectors:\n")

    for i in range(len(INPUT_VALUES_X)):
        print(f"x{i + 1} = {INPUT_VALUES_X[i]}; y{i + 1} = {INPUT_VALUES_Y[i]}")

    print(f"\n2. Y-input:\n")

    counter = 0
    for i in range(len(INPUT_VALUES_Y)):
        isPrintingAvailable = True
        noisedVariant = INPUT_VALUES_Y[i][:]
        for j in range(len(INPUT_VALUES_Y[i])):
            noisedVariant = addNoise(INPUT_VALUES_Y[i][:], j + 1)
            if(bidirectionalAssociativeMemory(INPUT_VALUES_Y[i], noisedVariant, WEIGHTS_BIDIRECTIONAL, startWithY)):
                if(maxY[i] < j + 1):
                    maxY[i] = j + 1
            isPrintingAvailable = False
        counter += 1

    print(f"3. X-input:\n")

    counter = 0
    for i in range(len(INPUT_VALUES_X)):
        isPrintingAvailable = True
        for j in range(len(INPUT_VALUES_X[i])):
            noisedVariant = addNoise(INPUT_VALUES_X[i][:], j + 1)
            if(bidirectionalAssociativeMemory(INPUT_VALUES_X[i], noisedVariant, WEIGHTS_BIDIRECTIONAL, startWithX)):
                if(maxX[i] < j + 1):
                    maxX[i] = j + 1
            isPrintingAvailable = False
        counter += 1

    print(f"4. Maximum number of recognised noisy bits:")
    print(f"\tY-input:")

    for i in range(len(maxY)):
        print(f"\ty_{i + 1} = {maxY[i]}")

    print(f"\n\tX-input:")

    for i in range(len(maxX)):
        print(f"\tx_{i + 1} = {maxX[i]}")


def main():
    global counter, isPrintingAvailable

    N = 13
    M = 13

    # VECTORS_BIDIRECTIONAL = [
    #     [-1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1, 1, -1,  1, -1, -1, -1], # 1
    #     [-1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1, 1,  1, -1, -1, -1, -1], # 2
    #     [ 1,  1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, 1, -1, -1, -1,  1,  1], # 8
    #     [-1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1]  # 11
    # ]
    VECTORS_BIDIRECTIONAL = [
        [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1],
        [-1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1]
    ]

    bidirectionalAssociativeMemoryResults(VECTORS_BIDIRECTIONAL, N, M)

if __name__ =="__main__":
    main()