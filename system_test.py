from env import *
from time import time

@njit
def numbaRandomBot1(state, tempData, perData):
    validActions = getValidActions(state)
    validActions = np.where(validActions==1)[0]
    idx = np.random.randint(0, len(validActions))
    a = getReward(state)
    if a == 1:
        perData[0][0][1] += 1
    elif a == -1:
        perData[0][0][0] += 1

    return validActions[idx], tempData, perData

perData = List()
perData.append(np.array([[0., 0.]]))
t_ = time()
a, b = numbaMain(numbaRandomBot1,numbaRandomBot1,numbaRandomBot1,numbaRandomBot1,100,perData,True,10)
print(time() - t_)
print(b)

perData = List()
perData.append(np.array([[0., 0.]]))
t_ = time()
a, b = numbaMain(numbaRandomBot1,numbaRandomBot1,numbaRandomBot1,numbaRandomBot1,1000,perData,True,100)
print(time() - t_)
print(b)

perData = List()
perData.append(np.array([[0., 0.]]))
t_ = time()
a, b = numbaMain(numbaRandomBot1,numbaRandomBot1,numbaRandomBot1,numbaRandomBot1,10000,perData,True,1000)
print(time() - t_)
print(b)

perData = List()
perData.append(np.array([[0., 0.]]))
t_ = time()
a, b = numbaMain(numbaRandomBot1,numbaRandomBot1,numbaRandomBot1,numbaRandomBot1,100000,perData,True,1000)
print(time() - t_)
print(b)