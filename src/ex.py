
import numpy as np

def read_matrix():
    with  open("city/matrix.txt") as matrixFile:
        resultList = []
        for line in matrixFile:
            line = line.rstrip('\n')
            sVals = line.split(" ")
            iVals = list(map(np.int, sVals))
            resultList.append(iVals)
        matrixFile.close()
    return np.asmatrix(resultList)

matrix = read_matrix()
matrixBool = np.full(matrix.shape, 0)
print(matrixBool)