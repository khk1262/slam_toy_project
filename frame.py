import cv2
import numpy as np
np.set_printoptions(suppress=True)

from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform
from skimage.transform import FundamentalMatrixTransform


IRt = np.eye(4)


# make homogeneous matrix
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

#pose
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
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret

#특징 검출
def extract(img):
    orb = cv2.ORB_create()

    #detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(gray, 3000, 0.01, 3) # Shi-Tomasi Corner Detection을 통해 corner를 검출
    
    #extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts] # 검출한 코너들을 키포인트로써 저장
    kps, des = orb.compute(img, kps) # 위에서 만든 키포인트들에 대한 기술자를 계산한다.

    #return pts and des
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des
        # match 결과로 나오는 것은 Dmatch 오브젝트들의 리스트가 나온다.

def normalize(Kinv, pts):
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

def denormalize(K, pt):
    ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))


def match_frames(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # bf.match(queryImage_des, trainImage_des), train에서 query를 찾아낸다.즉 querImage의 des와 가장 유사도가 높은 trainImage의 기술자를 찾음
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    ret = []
    idx1, idx2 = [], []

    # Lowe's ratio test
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            idx1.append(m.queryIdx)
            idx2.append(m.trainIdx)

            p1 = f1.pts[m.queryIdx]
            p2 = f2.pts[m.trainIdx]
            ret.append((p1, p2))
    
    # 8인 이유는 8 points algorithmn 때문에
    assert len(ret) >= 8
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    #filter
    #estimate the epipolar geometry between the prev and cur image.
    # FundamentalMatrix는 보정되지 않은(캘리브레이션 x) 이미지의 페어 사이의 점 대응에 관한 것
    # fit matrix
    model, inliers = ransac((ret[:, 0], ret[:, 1]),
                            EssentialMatrixTransform,
                            # FundamentalMatrixTransform,
                            min_samples=8,
                            # residual_threshold=1,
                            residual_threshold=0.005, 
                            max_trials=200)
    

    # 노이즈를 제거한다. 여기서 노이즈는 이전 프레임과 현재 프레임 사이의 연관이 확실히 있지 않는 것들
    # 아웃라이어 제거
    # ret = ret[inliers]

    # question1, why do svd compose? and why choose sqrt(2)
    Rt = extractRt(model.params)

    #return
    # return ret, Rt
    return idx1[inliers], idx2[inliers], Rt
 

class Frame(object):

    def __init__(self, img, K):
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
        self.pts, self.des = extract(img)
        self.pts = normalize(self.Kinv, self.pts)
        self.pose = IRt