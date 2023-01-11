from sub_func import *
from numba.typed import List
from numba import jit

__STATE_SIZE__ = 319
__ACTION_SIZE__ = 45
__AGENT_SIZE__ = 4


@njit
def initEnv():
    lv1 = np.arange(41) # Khởi tạo chồng thẻ ẩn cấp 1
    lv2 = np.arange(40, 71) # Khởi tạo chồng thẻ ẩn cấp 2
    lv3 = np.arange(70, 91) # Khởi tạo chồng thẻ ẩn cấp 3

    np.random.shuffle(lv1[:-1]) # Xáo trộn chồng thẻ ẩn cấp 1
    np.random.shuffle(lv2[:-1]) # Xáo trộn chồng thẻ ẩn cấp 2
    np.random.shuffle(lv3[:-1]) # Xáo trộn chồng thẻ ẩn cấp 3

    lv1[-1] = 4 # Số thẻ cấp 1 chia ra bàn chơi: 4
    lv2[-1] = 4 # Số thẻ cấp 2 chia ra bàn chơi: 4
    lv3[-1] = 4 # Số thẻ cấp 3 chia ra bàn chơi: 4

    env = np.full(103, 0)

    env[0:6] = np.array([7,7,7,7,7,5]) # Nguyên liệu trên bàn chơi

    for i in range(4):
        env[6+i] = np.random.randint(0, 2) + 2*i

    env[10:14] = lv1[:4] # Lấy 4 thẻ lv1 và xếp trên bàn chơi
    env[14:18] = lv2[:4] # Lấy 4 thẻ lv2 và xếp trên bàn chơi
    env[18:22] = lv3[:4] # Lấy 4 thẻ lv3 và xếp trên bàn chơi

    for pIdx in range(4): # Thông tin của các người chơi [22, 39, 56, 73, 90]
        tempVal = 17*pIdx
        # env[22+tempVal:36+tempVal] = 0 # Nguyên liệu (6), nguyên liệu vv (6), điểm (1) và điểm A (1)
        env[36+tempVal:39+tempVal] = -1

    env[90] = 0 # Turn
    env[91] = 0 # Phase
    env[92:97] = 0 # Nguyên liệu người chơi đã lấy trong turn đó
    env[97:101] = 0 # Số thẻ đã mua của các người chơi
    env[101] = 0 # Game đã kết thúc hay chưa (1 là kết thúc rồi)
    env[102] = -1 # Người chơi đang nắm giữ Avenger

    return env, lv1, lv2, lv3



def visualizeEnv(env_, lv1, lv2, lv3):
    env = env_.copy()
    dict_ = {}
    dict_['BoardGems'] = env[0:6]
    dict_['Noble'] = env[6:10]

    dict_['Cards'] = env[10:22]

    for i in range(4):
        dict_[f'Player_{i}'] = {}
        pInfor = env[22+17*i:39+17*i]
        dict_[f'Player_{i}']['Gems'] = pInfor[0:6]
        dict_[f'Player_{i}']['PerGems'] = pInfor[6:12]
        dict_[f'Player_{i}']['Score'] = pInfor[12]
        dict_[f'Player_{i}']['AvengerScore'] = pInfor[13]
        dict_[f'Player_{i}']['HidingCards'] = pInfor[14:17]

    dict_['Turn'] = env[90]
    dict_['Phase'] = env[91]
    dict_['TakenStocks'] = env[92:97]
    dict_['NumBoughtCards'] = env[97:101]
    dict_['EndGame'] = env[101]
    dict_['OwnAvenger'] = env[102]
    dict_['HideCardsLv1Order'] = lv1[:-1], lv1[-1]
    dict_['HideCardsLv2Order'] = lv2[:-1], lv2[-1]
    dict_['HideCardsLv3Order'] = lv3[:-1], lv3[-1]

    return dict_


@njit
def getStateSize():
    return __STATE_SIZE__


@njit
def getAgentState(env, lv1, lv2, lv3):
    state = np.zeros(__STATE_SIZE__)

    state[0:6] = env[0:6] # Nguyên liệu trên bàn chơi

    for i in range(4): # Thông tin các thẻ noble
        cardId = env[6+i]
        if cardId != -1:
            tempVal = 6*i
            state[6+tempVal:12+tempVal] = nobleCardInfor[cardId]

    for i in range(12): # Thông tin các thẻ thường
        cardId = env[10+i]
        if cardId != -1:
            tempVal = 12*i
            cardInfor = normalCardInfor[cardId]
            state[30+tempVal:32+tempVal] = cardInfor[0:2]
            state[37+tempVal:42+tempVal] = cardInfor[3:8]
            state[32+tempVal+cardInfor[2]] = 1

    if lv1[-1] < 40: # Còn thẻ ẩn cấp 1
        state[174] = 1
    if lv2[-1] < 30: # Còn thẻ ẩn cấp 2
        state[175] = 1
    if lv3[-1] < 20: # Còn thẻ ẩn cấp 3
        state[176] = 1

    pIdx = env[90] % 4 # Index của người chơi nhận state
    for i in range(4): # Sắp xếp lại thông tin env theo góc nhìn người chơi
        pEnvIdx = (pIdx + i) % 4
        tempVal = 14*i
        tempValEnv = 17*pEnvIdx
        state[177+tempVal:191+tempVal] = env[22+tempValEnv:36+tempValEnv] # gems, per_gems, score, A_score
        if i == 0: # Truyền thông tin chi tiết thẻ úp người chơi vào
            for pos in range(3):
                cardId = env[36+tempValEnv+pos]
                if cardId != -1:
                    cardInfor = normalCardInfor[cardId]
                    tempValCard = 12*pos
                    state[233+tempValCard:235+tempValCard] = cardInfor[0:2]
                    state[240+tempValCard:245+tempValCard] = cardInfor[3:8]
                    state[235+tempValCard+cardInfor[2]] = 1

        else: # Truyền cấp của thẻ úp
            tempValP = 9*(i - 1)
            for pos in range(3):
                cardId = env[36+tempValEnv+pos]
                if cardId != -1:
                    tempValCardLv = 3*pos
                    if cardId < 40:
                        state[269+tempValP+tempValCardLv] = 1
                    elif cardId < 70:
                        state[270+tempValP+tempValCardLv] = 1
                    elif cardId < 90:
                        state[271+tempValP+tempValCardLv] = 1

        state[296+i] = env[97+pEnvIdx] # Số thẻ đã mua

    state[300+pIdx] = 1 # Vị trí của người chơi

    state[304+env[91]] = 1 # Phase

    state[309:314] = env[92:97] # Nguyên liệu đã lấy trong turn

    state[314] = env[101] # Check xem game đã thắng hay chưa

    if env[102] != -1:
        state[315+(env[102]-pIdx)%4] = 1

    return state


@njit
def getActionSize():
    return __ACTION_SIZE__


@njit
def getValidActions(state):
    phase = state[304:309]
    validActions = np.full(__ACTION_SIZE__, 0)

    if phase[0] == 1: # Lựa chọn kiểu hành động
        boardStocks = state[0:6]
        if (boardStocks[0:5] > 0).any(): # Bàn chơi còn nguyên liệu thường
            validActions[1] = 1 # Lấy nguyên liệu
        else: # Bàn chơi hết nguyên liệu thường
            validActions[0] = 1 # Bỏ lượt

        for i in range(3): # Check xem có thể úp thêm thẻ không
            tempVal = 12*i
            if (state[240+tempVal:245+tempVal] == 0).all():
                check = False
                for j in range(12):
                    tempVal1 = 12*j
                    cardPrice = state[37+tempVal1:42+tempVal1]
                    if (cardPrice > 0).any():
                        validActions[2] = 1
                        check = True
                        break

                if check:
                    break

        checkActionBuy = False
        for i in range(12): # Check thẻ trên bàn
            tempVal = 12*i
            cardPrice = state[37+tempVal:42+tempVal]
            if (cardPrice > 0).any() and checkBuyCard(state[177:183], state[183:188], cardPrice):
                checkActionBuy = True
                break

        if checkActionBuy:
            validActions[3] = 1
        else:
            for i in range(3): # Check thẻ úp
                tempVal = 12*i
                cardPrice = state[240+tempVal:245+tempVal]
                if (cardPrice > 0).any() and checkBuyCard(state[177:183], state[183:188], cardPrice):
                    validActions[3] = 1
                    break

    elif phase[1] == 1: # Lấy nguyên liệu
        takenStocks = state[309:314]
        boardStocks = state[0:6]
        temp = np.where(boardStocks[0:5] > 0)[0] + 4

        s_ = np.sum(takenStocks)
        if s_ == 0:
            validActions[temp] = 1

        elif s_ == 1:
            validActions[temp] = 1
            t_ = np.where(takenStocks==1)[0][0]
            if boardStocks[t_] < 3: # Không thể lấy double
                validActions[t_+4] = 0

        elif s_ == 2:
            temp_ = np.where(takenStocks == 1)[0] + 4
            validActions[temp] = 1
            validActions[temp_] = 0

        if np.sum(state[177:183]) + state[188] > 9:
            validActions[9] = 0

    elif phase[2] == 1: # Úp thẻ
        for lv in range(1, 4):
            tempVal = 5*lv
            if state[173+lv] == 1: # Còn thẻ ẩn
                validActions[5+tempVal:10+tempVal] = 1
            else:
                for i in range(4):
                    tempValCard = 48*(lv-1) + 12*i
                    cardPrice = state[37+tempValCard:42+tempValCard]
                    if (cardPrice > 0).any():
                        validActions[6+tempVal+i] = 1

    elif phase[3] == 1: # Mua thẻ
        for i in range(12): # Kiểm tra 12 thẻ trên bàn
            tempVal = 12*i
            cardPrice = state[37+tempVal:42+tempVal]
            if (cardPrice > 0).any() and checkBuyCard(state[177:183], state[183:188], cardPrice):
                validActions[25+i] = 1

        for i in range(3): # Check thẻ úp
            tempVal = 12*i
            cardPrice = state[240+tempVal:245+tempVal]
            if (cardPrice > 0).any() and checkBuyCard(state[177:183], state[183:188], cardPrice):
                validActions[37+i] = 1

    elif phase[4] == 1: # Trả nguyên liệu
        temp = np.where(state[177:182] > 0)[0] + 40
        validActions[temp] = 1

    return validActions


@njit
def stepEnv(action, env, lv1, lv2, lv3):
    phase = env[91]

    if phase == 0: # Lựa chọn hành động
        if action == 0:
            env[90] += 1
        else:
            env[91] = action

    elif phase == 1: # Lấy nguyên liệu
        checkP1 = False
        if action == 9:
            checkP1 = True
        else:
            gem = action - 4

            takenStocks = env[92:97]
            takenStocks[gem] += 1 # Thêm vào nguyên liệu đã lấy trong turn

            pIdx = env[90] % 4
            tempVal = 17*pIdx
            pStocks = env[22+tempVal:28+tempVal]
            pStocks[gem] += 1 # Thêm nguyên liệu cho người chơi hiện tại

            bStocks = env[0:6]
            bStocks[gem] -= 1 # Trừ nguyên liệu ở bàn chơi

            s_ = np.sum(takenStocks)
            if s_ == 1:
                # Còn đúng một loại nguyên liệu và nguyên liệu đó có số lượng < 3
                if bStocks[gem] < 3 and (np.sum(bStocks[0:5]) - bStocks[gem]) == 0:
                    checkP1 = True
            elif s_ == 2:
                # Lấy double hoặc không còn nguyên liệu nào khác 2 cái vừa lấy
                if np.max(takenStocks) == 2 or (np.sum(bStocks[0:5]) - np.sum(bStocks[np.where(takenStocks==1)[0]])) == 0:
                    checkP1 = True
            elif s_ == 3: # Đã lấy 3 nguyên liệu
                checkP1 = True

        if checkP1:
            env[92:97] = 0
            if np.sum(pStocks) + env[33+tempVal] > 10:
                env[91] = 4 # Trả nguyên liệu
            else:
                env[91] = 0
                env[90] += 1 # Sang turn mới

    elif phase == 2: # Úp thẻ
        bStocks = env[0:6]
        pIdx = env[90] % 4
        tempVal = 17*pIdx
        pStocks = env[22+tempVal:28+tempVal]
        tempValHiCa = 36+tempVal
        posP = np.where(env[tempValHiCa:tempValHiCa+3] == -1)[0][0] + tempValHiCa
        if bStocks[5] > 0: # Trên bàn còn nguyên liệu gold
            pStocks[5] += 1
            bStocks[5] -= 1

        if action == 10: # Úp thẻ ẩn cấp 1
            env[posP] = lv1[lv1[-1]]
            lv1[-1] += 1
        elif action == 15: # Úp thẻ ẩn cấp 2
            env[posP] = lv2[lv2[-1]]
            lv2[-1] += 1
        elif action == 20: # Úp thẻ ẩn cấp 3
            env[posP] = lv3[lv3[-1]]
            lv3[-1] += 1
        else: # Úp thẻ trên bàn chơi
            posE = action - (action-11)//5 - 1
            cardId = env[posE]
            env[posP] = cardId
            if cardId < 40:
                if lv1[-1] < 40:
                    env[posE] = lv1[lv1[-1]]
                    lv1[-1] += 1
                else:
                    env[posE] = -1
            elif cardId < 70:
                if lv2[-1] < 30:
                    env[posE] = lv2[lv2[-1]]
                    lv2[-1] += 1
                else:
                    env[posE] = -1
            elif cardId < 90:
                if lv3[-1] < 20:
                    env[posE] = lv3[lv3[-1]]
                    lv3[-1] += 1
                else:
                    env[posE] = -1

        if np.sum(pStocks) + env[33+tempVal] > 10:
            env[91] = 4 # Trả nguyên liệu
        else:
            env[91] = 0
            env[90] += 1 # Sang turn mới

    elif phase == 3: # Mua thẻ
        bStocks = env[0:6]
        pIdx = env[90] % 4
        tempVal = 17*pIdx
        pStocks = env[22+tempVal:28+tempVal]
        pPerStocks = env[28+tempVal:34+tempVal]
        if action < 37:
            posE = action - 15
        else:
            posE = -1 + action + tempVal

        cardId = env[posE]
        if cardId > 69:
            pPerStocks[5] = 1

        cardInfor = normalCardInfor[cardId]
        price = cardInfor[3:8]

        nlMat = (price > pPerStocks[0:5]) * (price - pPerStocks[0:5])
        nlBt = np.minimum(nlMat, pStocks[0:5])
        nlG = np.sum(nlMat - nlBt)

        pStocks[0:5] -= nlBt # Trả nguyên liệu
        pStocks[5] -= nlG
        bStocks[0:5] += nlBt
        bStocks[5] += nlG

        # Nhận các phần thưởng từ thẻ
        env[97+pIdx] += 1 # Tăng số thẻ đã mua
        if action < 37: # Mua thẻ trên bàn
            if cardId < 40:
                if lv1[-1] < 40:
                    env[posE] = lv1[lv1[-1]]
                    lv1[-1] += 1
                else:
                    env[posE] = -1
            elif cardId < 70:
                if lv2[-1] < 30:
                    env[posE] = lv2[lv2[-1]]
                    lv2[-1] += 1
                else:
                    env[posE] = -1
            elif cardId < 90:
                if lv3[-1] < 20:
                    env[posE] = lv3[lv3[-1]]
                    lv3[-1] += 1
                else:
                    env[posE] = -1
        else: # Mua thẻ giữ trong tay
            env[posE] = -1

        env[34+tempVal:36+tempVal] += cardInfor[0:2] # Cộng điểm
        pPerStocks[cardInfor[2]] += 1 # Tăng nguyên liệu vĩnh viễn

        # Check noble
        if pPerStocks[cardInfor[2]] >= 3 and pPerStocks[cardInfor[2]] <= 4:
            for i in range(4):
                nobleId = env[6+i]
                if nobleId != -1:
                    nobleInfor = nobleCardInfor[nobleId]
                    price = nobleInfor[1:6]
                    if (price <= pPerStocks[0:5]).all():
                        env[6+i] = -1 # Loại thẻ noble khỏi bàn chơi
                        env[34+tempVal] += nobleInfor[0]

        lstA = env[np.array([35, 52, 69, 86])]
        max_ = np.max(lstA)
        # Check avenger
        if cardInfor[1] > 0 and env[35+tempVal] >= 3 and env[35+tempVal] == max_ and env[102] != pIdx:
            lstCheck = np.where(lstA==max_)[0]
            if len(lstCheck) == 1:
                if env[102] == -1:
                    env[34+tempVal] += 3
                    env[102] = pIdx
                else:
                    env[34+17*env[102]] -= 3
                    env[34+tempVal] += 3
                    env[102] = pIdx

        if np.sum(pStocks) + env[33+tempVal] > 10:
            env[91] = 4 # Trả nguyên liệu
        else:
            env[91] = 0
            env[90] += 1 # Sang turn mới

    elif phase == 4: # Trả nguyên liệu
        gem = action - 40
        bStocks = env[0:6]
        pIdx = env[90] % 4
        tempVal = 17*pIdx
        pStocks = env[22+tempVal:28+tempVal]

        pStocks[gem] -= 1
        bStocks[gem] += 1

        if np.sum(pStocks) + env[33+tempVal] < 11:
            env[91] = 0
            env[90] += 1 # Sang turn mới


@njit
def getAgentSize():
    return __AGENT_SIZE__


@njit
def checkEnded(env):
    scoreArr = env[np.array([34, 51, 68, 85])]
    maxScore = np.max(scoreArr)
    if maxScore < 16 or env[91] != 0 or env[90] % 4 != 0:
        return -1

    lstEnPer = np.full(4, 0)
    for i in range(4):
        tempVal = 17*i
        if (env[28+tempVal:34+tempVal] > 0).all():
            lstEnPer[i] = 1

    lst_ = np.where(lstEnPer==1)[0]
    if len(lst_) == 0:
        return -1

    maxScore_ = np.max(scoreArr[lst_])
    if maxScore_ < 16:
        return -1

    lstP = np.where((lstEnPer==1)&(scoreArr==maxScore_))[0]

    lenLstP = len(lstP)
    if lenLstP == 0:
        return -1

    elif lenLstP == 1:
        env[101] = 1
        return lstP[0]

    else:
        if env[102] in lstP:
            env[101] = 1
            return env[102]
        else:
            playerBoughtCards = env[lstP+97]
            min_ = np.min(playerBoughtCards)
            # Trường hợp có người cùng điểm, cùng số thẻ và không ai có Avenger card
            winnerIdx = np.where(playerBoughtCards==min_)[0][-1]
            env[101] = 1
            return lstP[winnerIdx]


@njit
def getReward(state):
    if state[314] == 0:
        return -1
    else:
        if (state[183:189] == 0).any() or state[189] < 16: # Thiếu nguyên liệu hoặc điểm
            return 0
        else: # Đủ nguyên liệu và đủ điểm
            lstEnPer = np.array([1, 0, 0, 0])
            for i in range(1, 4):
                tempVal = 14*i
                if (state[183+tempVal:189+tempVal]>0).all():
                    lstEnPer[i] = 1

            lst_ = np.where(lstEnPer==1)[0]
            if len(lst_) == 1: # Mỗi bản thân đủ nguyên liệu và điểm
                return 1

            scoreArr = state[np.array([189, 203, 217, 231])]
            maxScore_ = np.max(scoreArr[lst_])

            lstP = np.where((lstEnPer==1)&(scoreArr==maxScore_))[0]
            if 0 not in lstP: # Không trong danh sách đủ nguyên liệu và điểm cao nhất (trong những người đủ nguyên liệu)
                return 0

            if len(lstP) == 1: # Duy nhất trong danh sách
                return 1

            # Danh sách có 2 người trở lên
            pAven = np.where(state[315:319]==1)[0]
            if len(pAven) > 0 and pAven[0] in lstP:
                if pAven[0] == 0:
                    return 1
                else:
                    return 0

            playerBoughtCards = state[lstP+296]
            min_ = np.min(playerBoughtCards)
            if playerBoughtCards[0] > min_:
                return 0
            # Trường hợp có người cùng điểm, cùng số thẻ và không ai có Avenger card
            else: # Số lượng thẻ ít nhất
                lstChk = lstP[np.where(playerBoughtCards==min_)[0]]
                if len(lstChk) == 1:
                    return 1
                else: # Xét vị trị của bản thân
                    selfId = np.where(state[300:304] == 1)[0][0]
                    if selfId + lstChk[1] >= 4:
                        return 1
                    else:
                        return 0



def one_game(listAgent, perData):
    env, lv1, lv2, lv3 = initEnv()
    tempData = []
    for _ in range(__AGENT_SIZE__):
        dataOnePlayer = List()
        dataOnePlayer.append(np.array([[0.]]))
        tempData.append(dataOnePlayer)

    winner = -1
    while env[90] < 400:
        pIdx = env[90] % 4
        action, tempData[pIdx], perData = listAgent[pIdx](getAgentState(env, lv1, lv2, lv3), tempData[pIdx], perData)
        stepEnv(action, env, lv1, lv2, lv3)
        winner = checkEnded(env)
        if winner != -1:
            break

    env[101] = 1

    for pIdx in range(4):
        env[90] = pIdx
        action, tempData[pIdx], perData = listAgent[pIdx](getAgentState(env, lv1, lv2, lv3), tempData[pIdx], perData)

    return winner, perData


@njit
def numba_one_game(p0, p1, p2, p3, perData, pIdOrder):
    env, lv1, lv2, lv3 = initEnv()
    tempData = []
    for _ in range(__AGENT_SIZE__):
        dataOnePlayer = List()
        dataOnePlayer.append(np.array([[0.]]))
        tempData.append(dataOnePlayer)

    winner = -1
    while env[90] < 400:
        pIdx = env[90] % 4
        if pIdOrder[pIdx] == 0:
            action, tempData[pIdx], perData = p0(getAgentState(env, lv1, lv2, lv3), tempData[pIdx], perData)
        elif pIdOrder[pIdx] == 1:
            action, tempData[pIdx], perData = p1(getAgentState(env, lv1, lv2, lv3), tempData[pIdx], perData)
        elif pIdOrder[pIdx] == 2:
            action, tempData[pIdx], perData = p2(getAgentState(env, lv1, lv2, lv3), tempData[pIdx], perData)
        elif pIdOrder[pIdx] == 3:
            action, tempData[pIdx], perData = p3(getAgentState(env, lv1, lv2, lv3), tempData[pIdx], perData)

        stepEnv(action, env, lv1, lv2, lv3)
        winner = checkEnded(env)
        if winner != -1:
            break

    # temp = np.full(4, 0)

    env[101] = 1

    for pIdx in range(4):
        env[90] = pIdx
        # temp[pIdx] = getReward(getAgentState(env, lv1, lv2, lv3))
        if pIdOrder[pIdx] == 0:
            action, tempData[pIdx], perData = p0(getAgentState(env, lv1, lv2, lv3), tempData[pIdx], perData)
        elif pIdOrder[pIdx] == 1:
            action, tempData[pIdx], perData = p1(getAgentState(env, lv1, lv2, lv3), tempData[pIdx], perData)
        elif pIdOrder[pIdx] == 2:
            action, tempData[pIdx], perData = p2(getAgentState(env, lv1, lv2, lv3), tempData[pIdx], perData)
        elif pIdOrder[pIdx] == 3:
            action, tempData[pIdx], perData = p3(getAgentState(env, lv1, lv2, lv3), tempData[pIdx], perData)

    return winner, perData



def normal_main(listAgent, times, perData, printMode=False, k=100):
    if len(listAgent) != __AGENT_SIZE__:
        raise Exception('Hệ thống chỉ cho phép có đúng 4 người chơi!!!')

    numWin = np.full(5, 0)
    pIdOrder = np.arange(__AGENT_SIZE__)
    for _ in range(times):
        if printMode and _ != 0 and _ % k == 0:
            print(_, numWin)

        np.random.shuffle(pIdOrder)
        shuffledListAgent = [listAgent[i] for i in pIdOrder]
        winner, perData = one_game(shuffledListAgent, perData)

        if winner == -1:
            numWin[4] += 1
        else:
            numWin[pIdOrder[winner]] += 1

    if printMode:
        print(_+1, numWin)

    return numWin, perData


@njit
def numba_main(p0, p1, p2, p3, times, perData, printMode=False, k=100):
    numWin = np.full(5, 0)
    pIdOrder = np.arange(__AGENT_SIZE__)
    for _ in range(times):
        if printMode and _ != 0 and _ % k == 0:
            print(_, numWin)

        np.random.shuffle(pIdOrder)
        winner, perData = numba_one_game(p0, p1, p2, p3, perData, pIdOrder)

        if winner == -1:
            numWin[4] += 1
        else:
            numWin[pIdOrder[winner]] += 1

    if printMode:
        print(_+1, numWin)

    return numWin, perData



def randomBot(state, tempData, perData):
    validActions = getValidActions(state)
    validActions = np.where(validActions==1)[0]
    idx = np.random.randint(0, len(validActions))
    return validActions[idx], tempData, perData


@njit
def numbaRandomBot(state, tempData, perData):
    validActions = getValidActions(state)
    validActions = np.where(validActions==1)[0]
    idx = np.random.randint(0, len(validActions))
    return validActions[idx], tempData, perData


@njit
def one_game_numba(p0, list_other, per_player, per1, per2, per3, p1, p2, p3):
    env, lv1, lv2, lv3 = initEnv()

    winner = -1
    while env[90] < 400:
        pIdx = env[90] % 4
        p_state = getAgentState(env, lv1, lv2, lv3)
        list_action = getValidActions(p_state)

        if list_other[pIdx] == -1:
            action, per_player = p0(p_state, per_player)
        elif list_other[pIdx] == 1:
            action, per1 = p1(p_state, per1)
        elif list_other[pIdx] == 2:
            action, per2 = p2(p_state, per2)
        elif list_other[pIdx] == 3:
            action, per3 = p3(p_state, per3)

        if list_action[action] != 1:
            raise Exception("Action không hợp lệ")

        stepEnv(action, env, lv1, lv2, lv3)
        winner = checkEnded(env)
        if winner != -1:
            break

    env[101] = 1

    for pIdx in range(4):
        env[90] = pIdx
        if list_other[pIdx] == -1:
            p_state = getAgentState(env, lv1, lv2, lv3)
            action, per_player = p0(p_state, per_player)

    check = False
    if winner != -1 and list_other[winner] == -1:
        check = True

    return check, per_player


@njit
def n_game_numba(p0, num_game, per_player, list_other, per1, per2, per3, p1, p2, p3):
    win = 0
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_numba(p0, list_other, per_player, per1, per2, per3, p1, p2, p3)
        win += winner

    return win, per_player


import importlib.util, json, sys
from setup import SHOT_PATH


def load_module_player(player):
    return importlib.util.spec_from_file_location('Agent_player',
    f"{SHOT_PATH}Agent/{player}/Agent_player.py").loader.load_module()


@jit
def numba_main_2(p0, n_game, per_player, level, *args):
    list_other = np.array([1, 2, 3, -1])
    if level == 0:
        per_agent_env = np.array([0])
        return n_game_numba(p0, n_game, per_player, list_other, per_agent_env, per_agent_env, per_agent_env, numbaRandomBot, numbaRandomBot, numbaRandomBot)
    else:
        env_name = sys.argv[1]
        if len(args) > 0:
            dict_level = json.load(open(f'{SHOT_PATH}Log/check_system_about_level.json'))
        else:
            dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))

        if str(level) not in dict_level[env_name]:
            raise Exception('Hiện tại không có level này') 
            
        lst_agent_level = dict_level[env_name][str(level)][2]

        p1 = load_module_player(lst_agent_level[0]).Test
        p2 = load_module_player(lst_agent_level[1]).Test
        p3 = load_module_player(lst_agent_level[2]).Test
        per_level = []
        for id in range(getAgentSize()-1):
            data_agent_env = list(np.load(f'{SHOT_PATH}Agent/{lst_agent_level[id]}/Data/{env_name}_{level}/Train.npy',allow_pickle=True))
            per_level.append(data_agent_env)
        
        return n_game_numba(p0, n_game, per_player, list_other, per_level[0], per_level[1], per_level[2], p1, p2, p3)
