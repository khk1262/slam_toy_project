import OpenGL.GL as gl
import pangolin
import numpy as np

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

from multiprocessing import Process, Queue

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
        self.viewer_init(1024, 768)
        while 1:
            self.viewer_refresh(q)

    def viewer_init(self, w, h):
        pangolin.CreateWindowAndBind('Main', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, w//2, h//2, 0.2, 100),
            pangolin.ModelViewLookAt(0, -10, -8, 
                                     0, 0, 0,
                                     0, -1, 0))
        handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -w/h)
        self.dcam.SetHandler(handler)

    def viewer_refresh(self, q):
        if self.state is None or not q.empty():
            self.state = q.get()


        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)


        # gl.glColor3f(0.0, 1.0, 0.0)
        # for pose in self.state[0]:
        #     pangolin.glDrawFrustrum(Kinv, 2, 2, pose, 1)

        # draw poses
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawCameras(self.state[0])


        # draw keypoints
        gl.glPointSize(2)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawPoints(self.state[1])

        pangolin.FinishFrame()


    def display(self):
        poses, pts = [], []
        for f in self.frames:
            poses.append(f.pose)
            #print(f.pose)

        for p in self.points:
            pts.append(p.pt)
        self.q.put((np.array(poses), np.array(pts)))
