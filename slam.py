#!/usr/bin/env python3

import cv2
from frame import Frame, denormalize, match_frames, IRt
import numpy as np
import g2o

# intrinsic parameters
W = 1920 // 2
H = 1080 // 2
# F = 270

# K = np.array(([F, 0, W//2], [0,F,H//2], [0, 0, 1]))

# intrinsic parameter
K = np.array(([602, 0, 309], [0,605,250], [0, 0, 1]))

cv2.namedWindow("image", cv2.WINDOW_NORMAL)

class Point(object):
    # A point is a 3-D point in ther world
    # Each Point is observed in multiple frames
    def __init__(self, loc):
        self.frames = []
        self.location = loc
        self.idxs= []

    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)

def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T


frames = []
def process_frame(img):
    img = cv2.resize(img, (W, H))
    frame = Frame(img, K)
    frames.append(frame)
    if len(frames) <= 1:
        return 

    idx1, idx2, Rt = match_frames(frames[-1], frames[-2])

    # 삼각측량
    # triangulatePoints(첫번째 카메라의 3X4 투영 행렬, 두번째 카메라의 3X4 투영 행렬, 첫번째 이미지의 특징점, 두번째 이미지의 특징점)
    # 반환되는 내용은 homogeneous 좌표에서 재구성된 4XN 배열
    frames[-1].pose = np.dot(Rt, frames[-2].pose)

    pts4d = triangulate(frames[-1].pose, frames[-2].pose, frames[-1].pts[idx1], frames[-2].pts[idx2])
    print(pts4d)
    # homogenous 좌표계의 w를 나누는 것?
    # homogenous 3-D coords
    pts4d /= pts4d[:, 3:]

    # reject pts without enough "parallax"
    # reject pts behind camera

    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)
    pts4d = pts4d[good_pts4d]

    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        pt = Point(p)
        pt.add_observation(frames[-1], idx1[i])
        pt.add_observation(frames[-2], idx2[i])
        
    # denormalize for display
    # 정규화한 포인트들을 다시 화면에 맞춤
    for pt1, pt2 in zip(frames[-1].pts[idx1], frames[-2].pts[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        cv2.circle(img, (u1, v1), color=(0,255,0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2),color=(255,0,0))

    cv2.imshow("image", img)
    cv2.waitKey(1)

if __name__ == "__main__":
    cap  = cv2.VideoCapture("videos/test.mp4")
    # cap  = cv2.VideoCapture(4)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
