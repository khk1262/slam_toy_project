#!/usr/bin/env python3

import cv2
from extract import FeaturExtractor
import numpy as np

W = 1920 // 2
H = 1080 // 2
F = 270
# intrinsic parameters
# u0, v0는 화면의 중앙을 뜻함

K = np.array(([F, 0, W//2], [0,F,H//2], [0, 0, 1]))
# print(K)

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
orb = cv2.ORB_create()

fe = FeaturExtractor(K)

def process_frame(img):
    img = cv2.resize(img, (W, H))
    matches, pose = fe.extract(img)
    if pose is None:
        return

    # denormalize for display
    # 정규 좌표계로 이동한 포인트들을 다시 화면에 맞춤

    print(f'{len(matches)} matches')
    for pt1, pt2 in matches:
        u1, v1 = fe.denormalize(pt1)
        u2, v2 = fe.denormalize(pt2)

        cv2.circle(img, (u1, v1), color=(0,255,0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2),color=(255,255,0))

    cv2.imshow("image", img)
    cv2.waitKey(1)

if __name__ == "__main__":
    cap  = cv2.VideoCapture("videos/test.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
