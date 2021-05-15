import cv2
import numpy as np
np.set_printoptions(suppress=True)

from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform
from skimage.transform import FundamentalMatrixTransform


# make homogeneous matrix
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def extractRt(E):
    W=np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
    U, d, Vt  = np.linalg.svd(E)

    assert np.linalg.det(U) > 0
    # if np.linalg.det(u) < 0:
    #     u *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0 

    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    
    t = U[:, 2]
    Rt = np.concatenate([R, t.reshape(3,1)], axis=1)
    print(Rt)
    return Rt


class FeaturExtractor(object):

    def __init__(self, K):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher (cv2.NORM_HAMMING)
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

    def normalize(self, pts):
        return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]



    def denormalize(self, pt):
        ret = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))
        return int(round(ret[0])), int(round(ret[1]))


    #특징 검출
    def extract(self, img):

        #detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feats = cv2.goodFeaturesToTrack(gray, 3000, 0.01, 3) # Shi-Tomasi Corner Detection을 통해 corner를 검출
        
        #extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats] # 검출한 코너들을 키포인트로써 저장
        kps, des = self.orb.compute(img, kps) # 위에서 만든 키포인트들에 대한 기술자를 계산한다.


        # match 결과로 나오는 것은 Dmatch 오브젝트들의 리스트가 나온다.

        #matching
        ret = []
        if self.last is not None:
            # bf.match(queryImage_des, trainImage_des), train에서 query를 찾아낸다.즉 querImage의 des와 가장 유사도가 높은 trainImage의 기술자를 찾음
            matches = self.bf.knnMatch(des, self.last['des'], k=2)

            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))
        
        #filter
        Rt = None

        if len(ret)>0:
            ret = np.array(ret)
            # normalize coords : dot product with Kinv
            # 정규화를 하기 위해서 화면의 중점으로 좌표를 이동
            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])

            #estimate the epipolar geometry between the prev and cur image.
            # FundamentalMatrix는 보정되지 않은(캘리브레이션 x) 이미지의 페어 사이의 점 대응에 관한 것
            model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                    EssentialMatrixTransform,
                                    # FundamentalMatrixTransform,
                                    min_samples=8,
                                    # residual_threshold=1,
                                    residual_threshold=0.005, 
                                    max_trials=200)
            

            # 노이즈를 제거한다. 여기서 노이즈는 이전 프레임과 현재 프레임 사이의 연관이 확실히 있지 않는 것들
            ret = ret[inliers]

            # question1, why do svd compose? and why choose sqrt(2)
            Rt = extractRt(model.params)

        #return
        self.last = {'kps' : kps, 'des': des}
        return ret, Rt