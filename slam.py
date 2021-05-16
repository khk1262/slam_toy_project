#!/usr/bin/env python3

import cv2
from frame import Frame, denormalize, match
import numpy as np
import g2o

# intrinsic parameters
W = 1920 // 2
H = 1080 // 2
F = 270

K = np.array(([F, 0, W//2], [0,F,H//2], [0, 0, 1]))

cv2.namedWindow("image", cv2.WINDOW_NORMAL)

frames = []

def process_frame(img):
    img = cv2.resize(img, (W, H))
    frame = Frame(img, K)
    frames.append(frame)
    if len(frames) <= 1:
        return 

    ret, Rt = match(frames[-1], frames[-2])
    # denormalize for display
    # 정규화한 포인트들을 다시 화면에 맞춤
    for pt1, pt2 in ret:
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

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
