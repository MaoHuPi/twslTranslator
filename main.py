'''
2023 © MaoHuPi
twslTranslator/main.py
'''

import cv2
import mediapipe as mp
import time
import pygame
import numpy as np
import json
from train import predict

path = '.'
camW , camH = 640, 480
handType = '1.1.5'
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(f'{path}/video/a_few_days.mp4')
# cap = cv2.VideoCapture(f'{path}/video/kawaikutegomenn.mp4')
cap.set(3, camW)
cap.set(4, camH)

holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

timeNow = 0
timeLast = 0

pygame.init()
win = pygame.display.set_mode((camW, camH))
font = pygame.font.SysFont(None, 24)

def landmarks2pointList(landmarks):
    pointList = [[int(m.x*imgW), int(m.y*imgH), int(-m.z) + 1] for i, m in enumerate(landmarks.landmark)]
    return(pointList)

def distance(p1, p2):
    d = False
    if type(p1) == type(p2):
        if type(p1) in [int, float]:
            d = abs(p1 - p2)
        elif type(p1) in [list, np.ndarray]:
            d = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
    return(d)

def center(*ps):
    d = max(*[len(p) for p in ps])
    c = [0 for _ in range(d)]
    for i in range(d):
        l = [p[i] if len(p) > i else False for p in ps]
        l = list(filter(lambda x: x != False, l))
        c[i] = sum(l)/len(l) if len(l) > 0 else 0
    return(c)

def normalizationHandData(hand):
    if len(hand) < 1:
        print('[Error] No hand!')
        return
    handX = [point[0] for point in hand]
    handY = [point[1] for point in hand]
    xMax = max(*handX)
    xMin = min(*handX)
    yMax = max(*handY)
    yMin = min(*handY)
    for i in range(len(hand)):
        hand[i][0] = (hand[i][0]-xMin) / (xMax-xMin)
        hand[i][1] = (hand[i][1]-yMin) / (yMax-yMin)
    return(hand)

def saveHandData(hand, handType):
    hand = normalizationHandData(hand)
    file = open(f'{path}/data/{handType}/{time.time()}.json', 'w+', encoding = 'utf-8')
    file.write(json.dumps(hand))
    file.close()

frame = 0
datas = []
while True:
    ret, img = cap.read()
    if ret:
        # img = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)

        imgW = img.shape[1]
        imgH = img.shape[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)
        
        # cv2.imshow('img', img)

        win.fill((0, 0, 0))

        partPointList = {
            'face': [], 
            'pose': [], 
            'leftHand': [], 
            'rightHand': []
        }
        partColorList = {
            'face': (255, 0, 0), 
            'pose': (0, 255, 0), 
            'leftHand': (255, 0, 255), 
            'rightHand': (0, 255, 255)
        }
        holisticResult = holistic.process(img)
        handsResult = hands.process(img)
        if holisticResult.face_landmarks:
            partPointList['face'] = landmarks2pointList(holisticResult.face_landmarks)
            # print(min(*[point[2] for point in partPointList['face']]))
        if holisticResult.pose_landmarks:
            partPointList['pose'] = landmarks2pointList(holisticResult.pose_landmarks)
        handsPointList = []
        if handsResult.multi_hand_landmarks:
            for hand_landmarks in handsResult.multi_hand_landmarks:
                handpointList = landmarks2pointList(hand_landmarks)
                handRoot = center(handpointList[0], handpointList[1])
                handsPointList.append([handpointList, handRoot])
        
        if holisticResult.pose_landmarks and handsResult.multi_hand_landmarks:
            leftHandRoot = center(*[partPointList['pose'][i] for i in [16, 18, 20, 22]])
            rightHandRoot = center(*[partPointList['pose'][i] for i in [15, 17, 19, 21]])
            secondIsRight = True
            if len(handsPointList) > 0:
                rootDistanceList = [distance(handpointList[1], leftHandRoot) for handpointList in handsPointList]
                handIndex = rootDistanceList.index(min(rootDistanceList))
                firstIsLeft = distance(leftHandRoot, handsPointList[handIndex][1]) < distance(rightHandRoot, handsPointList[handIndex][1])
                partPointList['leftHand' if firstIsLeft else 'rightHand'] = handsPointList[handIndex][0]
                handsPointList.remove(handsPointList[handIndex])
            if len(handsPointList) > 0:
                rootDistanceList = [distance(handpointList[1], rightHandRoot if secondIsRight else leftHandRoot) for handpointList in handsPointList]
                handIndex = rootDistanceList.index(min(rootDistanceList))
                partPointList['rightHand' if secondIsRight else 'leftHand'] = handsPointList[handIndex][0]
                handsPointList.remove(handsPointList[handIndex])

        for hand in [partPointList['leftHand'], partPointList['rightHand']]:
            [distance(point, hand[0]) for point in hand]
            pass

        for part in partPointList:
            for i, point in enumerate(partPointList[part]):
                if part == 'face' and i not in [*[10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338], *[12, 16]]:
                    continue
                pygame.draw.circle(win, (255, 255, 255), (point[0], point[1]), radius=point[2], width=0)
                text = font.render(str(i), True, partColorList[part])
                win.blit(text, (point[0], point[1]))

        timeNow = time.time()
        fps = 1/(timeNow-timeLast)
        timeLast = timeNow
        # print(f'FPS: {int(fps)}')

        pygame.display.update()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
            break
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                print('S')
                saveHandData(partPointList['leftHand' if len(partPointList['leftHand']) > 0 else 'rightHand'], handType)
    
    frame += 1
    try:
        data = normalizationHandData(partPointList['leftHand' if len(partPointList['leftHand']) > 0 else 'rightHand'])
        datas.append(data)
    except Exception as e:
        print(e)
        pass
    if frame > 10:
        frame = 0
        try:
            print('x-', predict(datas))
        except Exception as e:
            print(e)
            pass
        datas = []