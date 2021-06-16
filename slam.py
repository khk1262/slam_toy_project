#!/usr/bin/env python3

import cv2
from numpy.core import einsumfunc
from frame import Frame, denormalize, match_frames, IRt
from pointmap import Map, Point
import numpy as np
import g2o

# intrinsic parameters
W = 1920 // 2
H = 1080 // 2
F = 270

K = np.array(([F, 0, W//2], [0,F,H//2], [0, 0, 1]))
Kinv = np.linalg.inv(K)
# intrinsic parameter
# W = 640
# H = 480

# K = np.array(([632.950, 0, 326.063], [0,632.211,224.883], [0, 0, 1]))

# cv2.namedWindow("image", cv2.WINDOW_NORMAL)

mapp = Map()


# orb_slam triangulation code, Linear Triangulation method
def triangulate(pose1, pose2, pts1, pts2):
    ret = np.zeros((pts1.shape[0], 4))

    for i, p in enumerate(zip(pts1, pts2)):
        A = np.zeros((4,4))
        A[0] = p[0][0] * pose1[2] -  pose1[0]
        A[1] = p[0][1] * pose1[2] -  pose1[1]
        A[2] = p[1][0] * pose2[2] -  pose2[0]
        A[3] = p[1][1] * pose2[2] -  pose2[1]
        print("p00")
        print(pose2, pose1)
        _, _, vt = np.linalg.svd(A)

        ret[i] = vt[3]
    return ret

def process_frame(img):
    img = cv2.resize(img, (W, H))

    frame = Frame(mapp, img, K)
    if frame.id == 0:
        return

    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)

    # 삼각측량
    # triangulatePoints(첫번째 카메라의 3X4 투영 행렬, 두번째 카메라의 3X4 투영 행렬, 첫번째 이미지의 특징점, 두번째 이미지의 특징점)
    # 반환되는 내용은 homogeneous 좌표에서 재구성된 4XN 배열
    f1.pose = np.dot(Rt, f2.pose)
    # homogenous 3-D coords


    # 사용되는 파라미터, 이전 프레임에서의 카메라 위치, 현재 프레임 카메라 위치, 매칭되는 이전 프레임 특징점, 매칭 현재 프레임 특징점
    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])

    # 삼각측량으로 나타날 수 있는 3차원 포인트는 
    pts4d /= pts4d[:, 3:]

    # reject pts without enough "parallax"
    # reject pts behind camera

    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)
    print(sum(good_pts4d), len(good_pts4d))
    # pts4d = pts4d[good_pts4d]

    # print(pts4d)
    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue

        pt = Point(mapp, p)
        pt.add_observation(f1, idx1)
        pt.add_observation(f2, idx2)
        
    # denormalize for display
    # 정규화한 포인트들을 다시 화면에 맞춤
    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        cv2.circle(img, (u1, v1), color=(0,255,0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2),color=(255,255,0))

    cv2.imshow("image", img)
    mapp.display()

    cv2.waitKey(1)

if __name__ == "__main__":
    cap  = cv2.VideoCapture("videos/test.mp4")
    # cap  = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:
            process_frame(frame)
        else:
            break
