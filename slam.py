#!/usr/bin/env python3

import cv2
from extract import FeaturExtractor

W = 1920 // 2
H = 1080 // 2


cv2.namedWindow("image", cv2.WINDOW_NORMAL)
orb = cv2.ORB_create()

fe = FeaturExtractor()

def process_frame(img):
    img = cv2.resize(img, (W, H))
    matches = fe.extract(img)

    print(f'{len(matches)} matches')

    for pt1, pt2 in matches:
        u1, v1 = map(lambda x : int(round(x)), pt1)
        u2, v2 = map(lambda x : int(round(x)), pt2)

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
