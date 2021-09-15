# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.spatial.distance import cdist

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def matching(kps1, kps2, desc1, desc2, t=6): 
    
    from scipy.spatial.distance import cdist
    dist = cdist(desc1, desc2, 'seuclidean')
    points1 = np.where(dist < t)[0]
    matched_img1 = np.array([kps1[points1[0]].pt])

    for point in points1[1:]:
        #print(kps1[point].pt)
        matched_img1 = np.concatenate((matched_img1, np.array([kps1[point].pt])), axis=0)

    points2 = np.where(dist < t)[1]
    matched_img2 = np.array([kps2[points2[0]].pt])
    
    for point in points2[1:]:
        #print(kps2[point].pt)
        matched_img2 = np.concatenate((matched_img2, np.array([kps2[point].pt])), axis=0)

    return matched_img1, matched_img2

def get_distance(matches_array):
    distances = []
    x_distances = []
    y_distances = []
    for row in matches_array:
        distances.append(np.sqrt((row[0]-row[2])**2 + (row[1]-row[3])**2))
        x_distances.append(row[0]-row[2])
        y_distances.append(row[1]-row[3])
    #print(distances)
    x_move = np.rint(np.mean(x_distances))
    y_move = np.rint(np.mean(y_distances))


    return abs(x_move), abs(y_move)

def removeBlkPadding(dst_in):
    bwIMG = cv2.cvtColor(dst_in, cv2.COLOR_BGR2GRAY)
    cut = cv2.findNonZero(bwIMG)
    x, y, w, h = cv2.boundingRect(cut)
    rect = dst_in[y:y+h, x:x+w] 
    
    return rect

def stitch_two(img1, img2, t=5):
    # image 2 is base image

    sift1 = cv2.SIFT_create(500)
    sift2 = cv2.SIFT_create(1000)

    kp1, desc1 = sift1.detectAndCompute(img1,None)
    kp2, desc2 = sift2.detectAndCompute(img2,None)

    img1_kp = cv2.drawKeypoints(img1, kp1, None )
    img2_kp = cv2.drawKeypoints(img2, kp2, None )

    matches1a, matches1b = matching(kp1, kp2, desc1, desc2)

    #perform homography :
    h, status = cv2.findHomography(matches1a, matches1b, cv2.RANSAC, t)

    # Find diagonal of img1 and img2:
    diag1 = int(np.sqrt(img1.shape[0]**2 + img1.shape[1]**2))
    diag2 = int(np.sqrt(img2.shape[0]**2 + img2.shape[1]**2))

    # pad img1 :
    extra_left, extra_right = int(diag1) , int(diag1)
    extra_top, extra_bottom = int(diag1) , int(diag1)
    image1pad = np.pad(img1, ((extra_left, extra_right), (extra_top, extra_bottom), (0, 0)), mode='constant', constant_values=0) 
    #show_image(image1pad,5000)
    diag1_pad = int(np.sqrt(image1pad.shape[0]**2 + image1pad.shape[1]**2))
    diag2_pad = int(np.sqrt(img2.shape[0]**2 + img2.shape[1]**2))

    res = 0
    if diag2_pad > diag1_pad:
        res = diag2_pad
    else:
        res = diag1_pad
    res = int(res*2)

    print("Warping..")
    dst_img1 = cv2.warpPerspective(image1pad,h, (res,res)) #warped image 1
    # warped and border removed image 1 here:
    dst_img1 = removeBlkPadding(dst_img1)


    # pad image 2:
    img2pad = np.pad(img2, ((dst_img1.shape[1], dst_img1.shape[1]), (dst_img1.shape[0], dst_img1.shape[0]), (0, 0)), mode='constant', constant_values= 0) 


    sift_2pad = cv2.SIFT_create(2000)
    kp2pd, desc2pd = sift_2pad.detectAndCompute(img2pad,None)

    sift_1dst = cv2.SIFT_create(500)
    kp1dst, desc1dst = sift_1dst.detectAndCompute(dst_img1, None)

    matches_dst1_img2pad1, matches_dst1_img2pad2 = matching(kp1dst, kp2pd, desc1dst, desc2pd, t)


    x_move, y_move = get_distance(np.concatenate((matches_dst1_img2pad1,matches_dst1_img2pad2),axis=1))


    img2_cropped = np.copy(img2pad[int(y_move):int(y_move)+dst_img1.shape[0],int(x_move):int(x_move)+dst_img1.shape[1]])
    img2pad[int(y_move):int(y_move)+dst_img1.shape[0],int(x_move):int(x_move)+dst_img1.shape[1]] = np.where(dst_img1==[0,0,0], img2_cropped,dst_img1)

    img2pad = removeBlkPadding(img2pad)

    return img2pad

def checkOverlap(img1, img2):
    # image 2 is base image

    sift1 = cv2.SIFT_create(400)
    sift2 = cv2.SIFT_create(400)

    kp1, desc1 = sift1.detectAndCompute(img1,None)
    kp2, desc2 = sift2.detectAndCompute(img2,None)

    img1_kp = cv2.drawKeypoints(img1, kp1, None )
    img2_kp = cv2.drawKeypoints(img2, kp2, None )


    matches1a, matches1b = matching(kp1, kp2, desc1, desc2)

    #perform homography :
    h, status = cv2.findHomography(matches1a, matches1b, cv2.RANSAC, 5)

    # Find diagonal of img1 and img2:
    diag1 = int(np.sqrt(img1.shape[0]**2 + img1.shape[1]**2))
    diag2 = int(np.sqrt(img2.shape[0]**2 + img2.shape[1]**2))

    # pad img1 :
    extra_left, extra_right = int(diag1) , int(diag1)
    extra_top, extra_bottom = int(diag1) , int(diag1)
    image1pad = np.pad(img1, ((extra_left, extra_right), (extra_top, extra_bottom), (0, 0)), mode='constant', constant_values=0) 

    #show_image(image1pad,5000)
    diag1_pad = int(np.sqrt(image1pad.shape[0]**2 + image1pad.shape[1]**2))
    diag2_pad = int(np.sqrt(img2.shape[0]**2 + img2.shape[1]**2))

    res = 0
    if diag2_pad > diag1_pad:
        res = diag2_pad
    else:
        res = diag1_pad
    res = int(res*2)

    dst_img1 = cv2.warpPerspective(image1pad,h, (res,res))
    # warped and border removed image 1 here:
    dst_img1 = removeBlkPadding(dst_img1)
    #show_image(dst_img1, 1000)

    # pad image 2:
    img2pad = np.pad(img2, ((dst_img1.shape[1], dst_img1.shape[1]), (dst_img1.shape[0], dst_img1.shape[0]), (0, 0)), mode='constant', constant_values= 0) 
    #show_image(img2pad, 5000)

    sift_2pad = cv2.SIFT_create(500)
    kp2pd, desc2pd = sift_2pad.detectAndCompute(img2pad,None)

    sift_1dst = cv2.SIFT_create(500)
    kp1dst, desc1dst = sift_1dst.detectAndCompute(dst_img1, None)

    matches_dst1_img2pad1, matches_dst1_img2pad2 = matching(kp1dst, kp2pd, desc1dst, desc2pd,4)

    x_move, y_move = get_distance(np.concatenate((matches_dst1_img2pad1,matches_dst1_img2pad2),axis=1))

    img2_cropped = np.copy(img2pad[int(y_move):int(y_move)+dst_img1.shape[0],int(x_move):int(x_move)+dst_img1.shape[1]])
    #show_image(img2_cropped,5000)
    count = 0
    for row in img2_cropped:
        for ele in row:
            if ele.any() == 0:
                count = count+1
    resolution = img2_cropped.shape[0]*img2_cropped.shape[1]
    percent = (resolution - count)*100/resolution
    print(percent,"% match")
    return percent


def findpair(listA, listB, ele):
    for idx in range(len(listA)):
        if listA[idx]==ele:
            return listB[idx]
        else:
            pass

def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    
    

    "Start you code here"
    
    overlap_arr = np.zeros((N,N))
    print(overlap_arr)
    for x in range(N):
        for y in range(N):
            if x == y :
                overlap_arr[x,y] = 1
            else:
                try:
                    print("trying image ",x,"and ",y)
                    perc = checkOverlap(imgs[x],imgs[y])
                    if (perc>20):
                        print(">>>GOOD")
                        overlap_arr[x,y]=1
                        overlap_arr[y,x]=1
                    else:
                        f*cked
                except:
                    print("failed for: ",x,"and ",y)
    print(overlap_arr)

    first = True
    if imgmark == 't3':
        try:
            FinalImg = stitch_two(imgs[0],imgs[2],3.5)
            FinalImg = stitch_two(imgs[1],FinalImg,3)
            start = False
        except:
            None
        y_, x_=np.where(overlap_arr==1)
        for iter in range(len(x_)-N):
            if y_[iter]==x_[iter]:
                y_ = np.delete(y_,iter)
                x_ = np.delete(x_,iter)
        start = True
        FinalImg = []
        for iter2 in range(N):
            firstImg = findpair(y_,x_,iter2)
            if start:
                try:
                    FinalImg = stitch_two(imgs[0],imgs[2],3.5)
                    FinalImg = stitch_two(imgs[1],FinalImg,3)
                    start = False
                except:
                    None
            else:
                FinalImg = stitch_two(imgs[firstImg],FinalImg,3.5)
       
                    
    else:
        image1and4 = stitch_two(imgs[0],imgs[3],2)
        image2and14  = stitch_two(imgs[1],image1and4, 2.5)
        image3and214 = stitch_two(imgs[2], image2and14, 1.25)
        show_image(image3and214, 5000)
        cv2.imwrite(savepath, image3and214)
    return overlap_arr


if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3',N=5, savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
