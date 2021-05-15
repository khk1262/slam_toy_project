import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


class FeaturExtractor(object):

    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher (cv2.NORM_HAMMING)
        self.last = None

    # 격자로 나눠서 특징 검출
    def extract(self, img):

        #detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feats = cv2.goodFeaturesToTrack(gray, 3000, 0.01, 3) # Shi-Tomasi Corner Detection을 통해 corner를 검출
        
        #extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats] # 검출한 코너들을 키포인트로써 저장
        kps, des = self.orb.compute(img, kps) # 위에서 만든 키포인트들에 대한 기술자를 계산한다.


        # match 결과로 나오는 것은 Dmatch 오브젝트들의 리스트가 나온다.
        '''
        DMatch.distance - Distance between descriptors. The lower, the better it is.
        DMatch.trainIdx - Index of the descriptor in train descriptors
        DMatch.queryIdx - Index of the descriptor in query descriptors
        DMatch.imgIdx - Index of the train image.
        '''

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

        if len(ret)>0:
            ret = np.array(ret)
            #filter
            #estimate the epipolar geometry between the prev and cur image.
            # FundamentalMatrix는 보정되지 않은(캘리브레이션 x) 이미지의 페어 사이의 점 대응에 관한 것
            model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                    FundamentalMatrixTransform,
                                    min_samples=8,
                                    residual_threshold=1, max_trials=50)

            # 노이즈를 제거한다. 여기서 노이즈는 이전 프레임과 현재 프레임 사이의 연관이 확실히 있지 않는 것들
            ret = ret[inliers]
            #change
        #return
        self.last = {'kps' : kps, 'des': des}
        return ret