import os
import cv2
import random
import argparse
import numpy as np
from tqdm import tqdm


def removeHiddenfile(directory_list):
    if '.' in directory_list:
        directory_list.remove('.')
    if '..' in directory_list:
        directory_list.remove('.')
    if '.DS_Store' in directory_list:
        directory_list.remove('.DS_Store')
    return directory_list


def homographyGeneration(raw_image, image_path, dataPath, pairs_per_img):
    img = cv2.resize(cv2.imread(image_path, 0),(320,240))
    

    random_list = []
    i = 1
    while i < pairs_per_img + 1:
        y_start = random.randint(32,80)
        y_end = y_start + 128
        x_start = random.randint(32,160)
        x_end = x_start + 128

        y_1 = y_start
        x_1 = x_start
        y_2 = y_end
        x_2 = x_start
        y_3 = y_end
        x_3 = x_end
        y_4 = y_start
        x_4 = x_end

        img_patch = img[y_start:y_end, x_start:x_end]  # patch 1

        y_1_offset = random.randint(-32,32)
        x_1_offset = random.randint(-32,32)
        y_2_offset = random.randint(-32,32)
        x_2_offset = random.randint(-32,32)
        y_3_offset = random.randint(-32,32)
        x_3_offset = random.randint(-32,32)
        y_4_offset = random.randint(-32,32)
        x_4_offset = random.randint(-32,32)

        y_1_p = y_1 + y_1_offset
        x_1_p = x_1 + x_1_offset
        y_2_p = y_2 + y_2_offset
        x_2_p = x_2 + x_2_offset
        y_3_p = y_3 + y_3_offset
        x_3_p = x_3 + x_3_offset
        y_4_p = y_4 + y_4_offset
        x_4_p = x_4 + x_4_offset

        pts_img_patch = np.array([[y_1,x_1],[y_2,x_2],[y_3,x_3],[y_4,x_4]]).astype(np.float32)
        pts_img_patch_perturb = np.array([[y_1_p,x_1_p],[y_2_p,x_2_p],[y_3_p,x_3_p],[y_4_p,x_4_p]]).astype(np.float32)
        h,status = cv2.findHomography(pts_img_patch, pts_img_patch_perturb, cv2.RANSAC)

        img_perburb = cv2.warpPerspective(img, h, (320, 240))
        img_perburb_patch = img_perburb[y_start:y_end, x_start:x_end]  # patch 2
        if not [y_1,x_1,y_2,x_2,y_3,x_3,y_4,x_4] in random_list:
            random_list.append([y_1,x_1,y_2,x_2,y_3,x_3,y_4,x_4])
            h_4pt_1 = np.array([y_1_offset,x_1_offset,y_2_offset,x_2_offset,y_3_offset,x_3_offset,y_4_offset,x_4_offset])
            h_4pt_2 = np.array([y_1_p,x_1_p,y_2_p,x_2_p,y_3_p,x_3_p,y_4_p,x_4_p])

            img_patch_path = os.path.join(dataPath, (raw_image.split('.')[0] + '_' + str(i) + '_1' +'.jpg'))
            cv2.imwrite(img_patch_path, img_patch)
            img_perburb_patch_path = os.path.join(dataPath, (raw_image.split('.')[0] + '_' + str(i) + '_2' +'.jpg'))
            cv2.imwrite(img_perburb_patch_path, img_perburb_patch)
            txt_re = os.path.join(dataPath, (raw_image.split('.')[0] + '_' + str(i) + '_re' +'.txt'))
            txt_ab = os.path.join(dataPath, (raw_image.split('.')[0] + '_' + str(i) + '_ab' +'.txt'))
            txt_hm = os.path.join(dataPath, (raw_image.split('.')[0] + '_' + str(i) + '_hm' +'.txt'))
            h = np.reshape(h,(1,9))[0]
            h = h[:-1]
            with open(txt_re, 'ab') as output_file_1, open(txt_ab, 'ab') as output_file_2, open(txt_hm, 'ab') as output_file_3:
                np.savetxt(output_file_1, h_4pt_1)
                np.savetxt(output_file_2, h_4pt_2)
                np.savetxt(output_file_3, h)
            i += 1


def dataCollection_t(rawDataPath, dataPath, img_number,img_per_dir):
    raw_image_list = removeHiddenfile(os.listdir(rawDataPath))
    raw_image_list.sort()
    
    img_per_dir = img_number//len(raw_image_list)
    for i in tqdm(range(len(raw_image_list))): 
        raw_image = raw_image_list[i]
        raw_image_path = os.path.join(rawDataPath, raw_image)
        homographyGeneration(raw_image, raw_image_path, dataPath, img_per_dir)
        
def dataCollection(rawDataPath, dataPath, img_number,img_per_dir):
    raw_image_list = removeHiddenfile(os.listdir(rawDataPath))
    raw_image_list.sort()
    
    img_per_dir = 1#img_number//len(raw_image_list)
    for i in tqdm(range(img_number)): 
        raw_image = raw_image_list[i]
        raw_image_path = os.path.join(rawDataPath, raw_image)
        homographyGeneration(raw_image, raw_image_path, dataPath, img_per_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rawTrainDataPath', type=str, default='F:/datasets/ms-coco/train2014', help='The raw data path.')
    parser.add_argument('--trainDataPath', type=str, default='D:/tesisti/Boem/ar3/train/', help='The train data path')
    parser.add_argument('--trainDataNumber', type=int, default=500000, help='The data size for traininig')

    parser.add_argument('--rawValDataPath', type=str, default='F:/datasets/ms-coco/test2014', help='The raw data path.')
    parser.add_argument('--valDataPath', type=str, default='D:/tesisti/Boem/ar3/val/', help='The val data path')
    parser.add_argument('--valDataNumber', type=int, default=5000, help='The data size for validate tuning')

    parser.add_argument('--rawTestDataPath', type=str, default='F:/datasets/ms-coco/test2014', help='The raw data path.')
    parser.add_argument('--testDataPath', type=str, default='D:/tesisti/Boem/ar3/test/', help='The test data path')
    parser.add_argument('--testDataNumber', type=int, default=5000, help='The data size for testing')

    print('<==================== Loading raw data ===================>\n')
    args = parser.parse_args()
    print('<================= Generating homography =================>\n')

    dataCollection(args.rawValDataPath, args.valDataPath, args.valDataNumber, 4)

    dataCollection(args.rawTestDataPath, args.testDataPath, args.testDataNumber, 1)

    dataCollection_t(args.rawTrainDataPath, args.trainDataPath, args.trainDataNumber, 4)

    print('<=================== Finishing data ===================>')


if __name__ == '__main__':
    main()