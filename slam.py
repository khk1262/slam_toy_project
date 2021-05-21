#!/usr/bin/env python3

import cv2
from frame import Frame, denormalize, match_frames, IRt
import numpy as np
import g2o
import OpenGL.GL as gl
import pangolin

from multiprocessing import Process, Queue

# intrinsic parameters
W = 1920 // 2
H = 1080 // 2
F = 270

K = np.array(([F, 0, W//2], [0,F,H//2], [0, 0, 1]))

# intrinsic parameter
# W = 640
# H = 480

# K = np.array(([618.035, 0, 301.457], [0,619.002,234.492], [0, 0, 1]))

cv2.namedWindow("image", cv2.WINDOW_NORMAL)


#global Map
class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.state = None
        self.q = Queue()

        p = Process(target=self.viewer_thread, args=(self.q, ))
        p.daemon = True
        p.start()


    def viewer_thread(self, q):
        self.viewer_init()
        while 1:
            self.viewer_refresh(q)

    def viewer_init(self):
        pangolin.CreateWindowAndBind('Main', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
            pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
        handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(handler)

    def viewer_refresh(self, q):
        if self.state is None or not q.empty():
            self.state = q.get()
        ppts = np.array([d[:3, 3] for d in self.state[0]])
        spts = np.array(self.state[1])

        # print(ppts.shape)
        # print(spts.shape)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)


        gl.glPointSize(10)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawPoints(ppts)

        gl.glPointSize(2)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawPoints(spts)

        pangolin.FinishFrame()


    def display(self):
        poses, pts = [], []
        for f in self.frames:
            poses.append(f.pose)
        for p in self.points:
            pts.append(p.pt)
        self.q.put((poses, pts))

mapp = Map()


class Point(object):
    # A point is a 3-D point in ther world
    # Each Point is observed in multiple frames
    def __init__(self, mapp, loc):
        self.frames = []
        self.pt = loc
        self.idxs= []
        self.id = len(mapp.frames)
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)

def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T


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

    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
    pts4d /= pts4d[:, 3:]

    # reject pts without enough "parallax"
    # reject pts behind camera

    good_pts4d = (np.abs(pts4d[:, 3]) > 0.001) & (pts4d[:, 2] > 0)
    pts4d = pts4d[good_pts4d]

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
    # cap  = cv2.VideoCapture(4)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:
            process_frame(frame)
        else:
            break
