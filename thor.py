import cv2
import scipy.io
import numpy as np
import copy
from model import HomographyNN


class Thor:
    def __init__(self,
                 img_a_name='',
                 img_b_name=None,
                 patch_a_name='',
                 patch_b_name='',
                 perturbations='',
                 pert_corners=''
                 ):
        self.img_a = cv2.imread(img_a_name)

        self.patch_a = cv2.imread(patch_a_name, cv2.IMREAD_GRAYSCALE)
        self.patch_b = cv2.imread(patch_b_name, cv2.IMREAD_GRAYSCALE)

        self.h_ab, self.corners, self.p_corners = self.gt_homography(perturbations, pert_corners)

        if img_b_name is not None:
            self.img_b = cv2.imread(img_b_name, cv2.IMREAD_GRAYSCALE)
        else:
            b, g, r = cv2.split(self.img_a)
            b = cv2.warpPerspective(b, self.h_ab, b.shape)
            g = cv2.warpPerspective(g, self.h_ab, g.shape)
            r = cv2.warpPerspective(r, self.h_ab, r.shape)
            self.img_b = cv2.merge((b, g, r))

        self.h_sift = None
        self.h_orb = None
        self.h_cnn = None

    def cv_sift_homography_estimate(self):
        """
        estimate the homography using the classical method
        :return: estimation of the homography
        """
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.img_a, None)
        kp2, des2 = sift.detectAndCompute(self.img_b, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        h_ab = None
        if len(good) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            h_ab, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        self.h_sift = h_ab
        return h_ab

    def cv_orb_homography_estimate(self):
        """
        estimate the homography using the classical method
        :return: estimation of the homography
        """
        orb = cv2.ORB_create()

        kp1, des1 = orb.detectAndCompute(self.img_a, None)

        kp2, des2 = orb.detectAndCompute(self.img_b, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        dmatches = sorted(matches, key = lambda x:x.distance)

        # extract the matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in dmatches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in dmatches]).reshape(-1, 1, 2)

        # find homography matrix and do perspective transform
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        self.h_orb = M
        return M

    def cnn_homography_estimate(self, weights_path):
        net = HomographyNN()
        net.build_model()
        net.load_weights(weights_path)

        height, width = self.patch_a.shape

        input = np.zeros([1, height, width, 2])
        input[:, :, :, 0] = self.patch_a
        input[:, :, :, 1] = self.patch_b
        input = input/255

        h = net.predict(input)
        h = np.append(h, 1)
        h = np.reshape(h, [3, 3])
        self.h_cnn = np.linalg.inv(h)

        return h

    def cnn_4points_estimate(self, weights_path):
        net = HomographyNN()
        net.build_model()
        net.load_weights(weights_path)

        height, width = self.patch_a.shape

        input = np.zeros([1, height, width, 2])
        input[:, :, :, 0] = self.patch_a
        input[:, :, :, 1] = self.patch_b
        input = input/255

        perturbations = np.multiply(net.predict(input), 32)

        '''
        perturbations_0 = perturbations
        perturbations[0, 0] = perturbations_0[0, 1]
        perturbations[0, 1] = perturbations_0[0, 0]
        perturbations[0, 2] = perturbations_0[0, 3]
        perturbations[0, 3] = perturbations_0[0, 2]
        perturbations[0, 4] = perturbations_0[0, 5]
        perturbations[0, 5] = perturbations_0[0, 4]
        perturbations[0, 6] = perturbations_0[0, 7]
        perturbations[0, 7] = perturbations_0[0, 6]
        '''

        p_corners = self.corners.astype(np.float32) + np.reshape(perturbations, (4, 2))
        pgt = self.p_corners - self.corners

        corners = np.reshape(self.corners.astype(np.float32), (4, 2))

        h, _ = cv2.findHomography(corners, p_corners, cv2.RANSAC)
        self.h_cnn = h

        return h

    def multiple_project(self, h_s, colors, x0=5, y0=5, sz=50):
        pt0 = np.array([x0, y0, 1]).transpose()
        pt1 = np.array([x0 + sz, y0, 1]).transpose()
        pt2 = np.array([x0 + sz, y0 + sz, 1]).transpose()
        pt3 = np.array([x0, y0 + sz, 1]).transpose()

        pt0_2d = np.delete(pt0, 2)
        pt1_2d = np.delete(pt1, 2)
        pt2_2d = np.delete(pt2, 2)
        pt3_2d = np.delete(pt3, 2)

        pts = np.array([pt0_2d, pt1_2d, pt2_2d, pt3_2d], np.int32)
        pts = pts.reshape((-1, 1, 2))
        #out_a = cv2.polylines(cv2.cvtColor(self.img_a, cv2.COLOR_GRAY2BGR), [pts], True, colors[0])
        out_a = cv2.polylines(self.img_a, [pts], True, colors[0])
        out_b = self.img_b #cv2.cvtColor(self.img_b, cv2.COLOR_GRAY2BGR)

        for i in range(len(h_s)):
            h = h_s[i]
            color = colors[i]

            pt01 = np.matmul(h, pt0)
            pt01 = np.divide(pt01, pt01[2])

            pt11 = np.matmul(h, pt1)
            pt11 = np.divide(pt11, pt11[2])

            pt21 = np.matmul(h, pt2)
            pt21 = np.divide(pt21, pt21[2])

            pt31 = np.matmul(h, pt3)
            pt31 = np.divide(pt31, pt31[2])

            pt01_2d = np.delete(pt01, 2)
            pt11_2d = np.delete(pt11, 2)
            pt21_2d = np.delete(pt21, 2)
            pt31_2d = np.delete(pt31, 2)

            pts1 = np.array([pt01_2d, pt11_2d, pt21_2d, pt31_2d], np.int32)
            pts1 = pts1.reshape((-1, 1, 2))
            out_b = cv2.polylines(out_b, [pts1], True, color)

        return out_a, out_b






    def project(self, h, sz=30, color=(0, 255, 0)):
        pt0 = np.array([5, 5, 1]).transpose()
        pt01 = np.matmul(h, pt0)
        pt01 = np.divide(pt01, pt01[2])

        pt1 = np.array([sz, 5, 1]).transpose()
        pt11 = np.matmul(h, pt1)
        pt11 = np.divide(pt11, pt11[2])

        pt2 = np.array([sz, sz, 1]).transpose()
        pt21 = np.matmul(h, pt2)
        pt21 = np.divide(pt21, pt21[2])

        pt3 = np.array([5, sz, 1]).transpose()
        pt31 = np.matmul(h, pt3)
        pt31 = np.divide(pt31, pt31[2])

        pt0 = np.delete(pt0, 2)
        pt1 = np.delete(pt1, 2)
        pt2 = np.delete(pt2, 2)
        pt3 = np.delete(pt3, 2)

        pt01 = np.delete(pt01, 2)
        pt11 = np.delete(pt11, 2)
        pt21 = np.delete(pt21, 2)
        pt31 = np.delete(pt31, 2)

        pts = np.array([pt0, pt1, pt2, pt3], np.int32)
        pts = pts.reshape((-1, 1, 2))
        out_a = cv2.polylines(cv2.cvtColor(self.img_a, cv2.COLOR_GRAY2BGR), [pts], True, color)

        pts1 = np.array([pt01, pt11, pt21, pt31], np.int32)
        pts1 = pts1.reshape((-1, 1, 2))
        out_b = cv2.polylines(cv2.cvtColor(self.img_b, cv2.COLOR_GRAY2BGR), [pts1], True, color)

        return out_a, out_b

    def project2(self, sz=30):
        color_gt = (0, 255, 0)
        color_est = (0, 255, 255)

        pt0 = np.array([5, 5, 1]).transpose()
        pt01 = np.matmul(self.h_ab, pt0)
        pt01 = np.divide(pt01, pt01[2])
        pt02 = np.matmul(self.h_est, pt0)
        pt02 = np.divide(pt02, pt02[2])

        pt1 = np.array([sz, 5, 1]).transpose()
        pt11 = np.matmul(self.h_ab, pt1)
        pt11 = np.divide(pt11, pt11[2])
        pt12 = np.matmul(self.h_est, pt1)
        pt12 = np.divide(pt12, pt12[2])

        pt2 = np.array([sz, sz, 1]).transpose()
        pt21 = np.matmul(self.h_ab, pt2)
        pt21 = np.divide(pt21, pt21[2])
        pt22 = np.matmul(self.h_est, pt2)
        pt22 = np.divide(pt22, pt22[2])

        pt3 = np.array([5, sz, 1]).transpose()
        pt31 = np.matmul(self.h_ab, pt3)
        pt31 = np.divide(pt31, pt31[2])
        pt32 = np.matmul(self.h_est, pt3)
        pt32 = np.divide(pt32, pt32[2])

        pt0 = np.delete(pt0, 2)
        pt1 = np.delete(pt1, 2)
        pt2 = np.delete(pt2, 2)
        pt3 = np.delete(pt3, 2)

        pt01 = np.delete(pt01, 2)
        pt11 = np.delete(pt11, 2)
        pt21 = np.delete(pt21, 2)
        pt31 = np.delete(pt31, 2)

        pt02 = np.delete(pt02, 2)
        pt12 = np.delete(pt12, 2)
        pt22 = np.delete(pt22, 2)
        pt32 = np.delete(pt32, 2)

        pts = np.array([pt0, pt1, pt2, pt3], np.int32)
        pts = pts.reshape((-1, 1, 2))
        out_a = cv2.polylines(cv2.cvtColor(self.img_a, cv2.COLOR_GRAY2BGR), [pts], True, color_gt)

        pts1 = np.array([pt01, pt11, pt21, pt31], np.int32)
        pts1 = pts1.reshape((-1, 1, 2))
        out_b = cv2.polylines(cv2.cvtColor(self.img_b, cv2.COLOR_GRAY2BGR), [pts1], True, color_gt)

        pts2 = np.array([pt02, pt12, pt22, pt32], np.int32)
        pts2 = pts2.reshape((-1, 1, 2))
        out_b = cv2.polylines(out_b, [pts2], True, color_est)

        return out_a, out_b

    def gt_homography(self, perturbations, pert_corners):
        file = open(perturbations)
        lines = file.readlines()
        file.close()
        ts = []
        for line in lines:
            ts.append(float(line[:-1]))
        offsets = np.asarray(ts)

        file = open(pert_corners)
        lines = file.readlines()
        file.close()
        ts = []
        for line in lines:
            ts.append(float(line[:-1]))
        p_corners = np.asarray(ts)

        # for the sake of clarity
        y_1_offset = int(offsets[0])
        x_1_offset = int(offsets[1])
        y_2_offset = int(offsets[2])
        x_2_offset = int(offsets[3])
        y_3_offset = int(offsets[4])
        x_3_offset = int(offsets[5])
        y_4_offset = int(offsets[6])
        x_4_offset = int(offsets[7])

        y_1_p = int(p_corners[0])
        x_1_p = int(p_corners[1])
        y_2_p = int(p_corners[2])
        x_2_p = int(p_corners[3])
        y_3_p = int(p_corners[4])
        x_3_p = int(p_corners[5])
        y_4_p = int(p_corners[6])
        x_4_p = int(p_corners[7])

        y_1 = y_1_p - y_1_offset
        x_1 = x_1_p - x_1_offset
        y_2 = y_2_p - y_2_offset
        x_2 = x_2_p - x_2_offset
        y_3 = y_3_p - y_3_offset
        x_3 = x_3_p - x_3_offset
        y_4 = y_4_p - y_4_offset
        x_4 = x_4_p - x_4_offset

        pts_img_patch = np.array([[y_1, x_1], [y_2, x_2], [y_3, x_3], [y_4, x_4]])
        pts_img_patch_pert = np.array([[y_1_p, x_1_p], [y_2_p, x_2_p], [y_3_p, x_3_p], [y_4_p, x_4_p]])
        h, _ = cv2.findHomography(pts_img_patch.astype(np.float32), pts_img_patch_pert.astype(np.float32), cv2.RANSAC)

        return h, pts_img_patch, pts_img_patch_pert

    def show_input_images(self):
        i2 = cv2.hconcat([self.img_a, self.img_b])

        cv2.namedWindow('input images', cv2.WINDOW_NORMAL)
        cv2.imshow('input images', i2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()





