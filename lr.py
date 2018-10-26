

from numpy import *

def sigmoid(inX):
    return 1.0/(1+exp(-inX))


def predictfunc(inX, weights):
    prob = sigmoid(sum(inX*weights)) #利用sigmoid函数生成预测值，若大与0.5则判断为1
    if prob > 0.5: return 1.0
    else: return 0.0


# def stocGradAscent1(traindata, classLabels, Iter=150): #利用随机梯度上升求解weight
#
#     m,n = shape(traindata) #获得训练集的维数m*n
#     weights = ones(n)   #初始化weight为全1
#     for j in range(Iter):
#
#         dataIndex = range(m)
#         for i in range(m):
#             alpha = 4/(1.0+j+i)+0.0001    #alpha会随着每次迭代从而减小，这样可以减小随机梯度上升的回归系数波动问题，同时也
#                                           #同时比普通梯度上升收敛更快，加入常数项避免alpha变成0
#             randnum = int(random.uniform(0,len(traindata)))#随机选一个值更新回归系数
#             h = sigmoid(sum(traindata[randnum]*weights))
#             error = classLabels[randnum] - h#计算预测误差
#             weight = weights + alpha * error * traindata[randnum] #更新weights值
#             del (list(dataIndex)[randnum])
#     return weight

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #将输入数据变成矩阵形式
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)             #获得训练集的维数m*n
    alpha = 0.001                       #学习率设置
    Cycle =5000                #迭代次数
    weights = ones((n,1))               #设定weight值的初始值
    for k in range(Cycle):              #开始迭代
        h = sigmoid(dataMatrix*weights)     #计算sigmiod函数值
        error = (labelMat - h)              #计算偏差，用于更新weight值
        weights = weights + alpha * dataMatrix.transpose()* error #更新weight值
    return weights


def colicTest():
    frTrain = open('horseColicTraining.txt') #导入训练集
    frTest = open('horseColicTest.txt')#导入测试集
    trainingSet = []
    trainingLabels = [] #初始化训练集
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')#以制表符分割列，并形成数组形式
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21])) #建立训练集x和y
    trainWeights = gradAscent(array(trainingSet), trainingLabels) #使用梯度上升算法求解weight
    errorCount = 0; numTestVec = 0.0 #初始化错误率和测试集个数
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i])) #建立测试集数据
        if int(predictfunc(array(lineArr), trainWeights))!= int(currLine[21]): #利用predictfunc预测测试集结果,这里必须强制变成int 不然结果全是1
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %f" % errorRate)  #和测试集相比较，输出测试结果

if __name__ == '__main__':
    colicTest()
