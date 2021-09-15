#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def grabcutter(img, mask1):
    mask1 = mask1.astype(np.uint8)
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(img,mask1,None,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask1==2)|(mask1==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    return img

def removeBlkPadding(dst_in):
    bwIMG = cv2.cvtColor(dst_in, cv2.COLOR_BGR2GRAY)
    cut = cv2.findNonZero(bwIMG)
    x, y, w, h = cv2.boundingRect(cut)
    rect = dst_in[y:y+h, x:x+w] 
    
    return rect

def not_matches(ima1,ima2):
    ima1 = cv2.cvtColor(ima1, cv2.COLOR_BGR2GRAY)
    ima2 = cv2.cvtColor(ima2, cv2.COLOR_BGR2GRAY)
    sumImg = np.where(ima1>ima2,0,1)
    return sumImg


def get_distance(matches_array):
    distances = []
    x_distances = []
    y_distances = []
    for row in matches_array:
        distances.append(np.sqrt((row[0]-row[2])**2 + (row[1]-row[3])**2))
        x_distances.append(row[0]-row[2])
        y_distances.append(row[1]-row[3])

    x_move = np.rint(np.mean(x_distances))
    y_move = np.rint(np.mean(y_distances))


    return x_move, y_move

def matching(kps1, kps2, desc1, desc2, t=6): # to be done
    
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
    

def stitch_background(img1, img2, savepath=''):
    
    #show image preview
    show_image(img1, delay=1000)
    show_image(img2, delay=1000)

    sift1 = cv2.SIFT_create(500)
    sift2 = cv2.SIFT_create(500)

    print("Generating features..")
    kp1, desc1 = sift1.detectAndCompute(img1,None)
    kp2, desc2 = sift2.detectAndCompute(img2,None)

    img1_kp = cv2.drawKeypoints(img1, kp1, None )
    img2_kp = cv2.drawKeypoints(img2, kp2, None )

    match1A, match2A = matching(kp1, kp2, desc1, desc2)
    
    #perform homography :
    print("Performing Homography..")
    h, status = cv2.findHomography(match1A, match2A, cv2.RANSAC, 5)

    # Find diagonal of img1 and img2:
    diag1 = int(np.sqrt(img1.shape[0]**2 + img1.shape[1]**2))
    diag2 = int(np.sqrt(img2.shape[0]**2 + img2.shape[1]**2))


    #dst = cv2.warpPerspective(img1, )
    # pad img1 :
    extra_left, extra_right = int((diag1 - img1.shape[1])) , int((diag1-img1.shape[1]))
    extra_top, extra_bottom = int((diag1 - img1.shape[0])) , int((diag1-img1.shape[0]))
    abc = np.pad(img1, ((extra_left, extra_right), (extra_top, extra_bottom), (0, 0)), mode='constant', constant_values=0) 

    diag1_pad = int(np.sqrt(abc.shape[0]**2 + abc.shape[1]**2))
    diag2_pad = int(np.sqrt(img2.shape[0]**2 + img2.shape[1]**2))

    res = 0
    if diag2_pad > diag1_pad:
        res = diag2_pad
    else:
        res = diag1_pad
    res = int(res*2)

    # perform warp persective:
    print("Getting warped image..")
    dst = cv2.warpPerspective(abc,h, (res,res) ) 
    


    # transform
    sift_dst = cv2.SIFT_create()
    kp_dst, desc_dst = sift_dst.detectAndCompute(dst,None)
    imgdst_kp = cv2.drawKeypoints(dst, kp_dst, None )

  
    matches1_dst_img2, matches2_dst_img2 = matching(kp_dst, kp2, desc_dst, desc2, 5)
    
    x_move, y_move = get_distance(np.concatenate((matches1_dst_img2,matches2_dst_img2),axis=1))

    newImg2 = np.pad(img2, ((int(y_move), 0), (int(x_move), 0), (0, 0)), mode='constant', constant_values=0) 

    dst_cropped = np.copy(dst[int(y_move):newImg2.shape[0],int(x_move):newImg2.shape[1]])
    
    mask1 = not_matches(img2, dst_cropped)

    print("removing foreground..")
    cropimg = grabcutter(img2,mask1)
    
    img2_croped = np.copy(img2 - cropimg)

    # adding background from dst
    img2_croped = np.where(img2_croped==[0, 0, 0], dst_cropped, img2_croped)
    img2 = np.where(img2_croped==[0, 0, 0], img2, img2_croped)
    dst[int(y_move):newImg2.shape[0],int(x_move):newImg2.shape[1]] = img2

    dst = removeBlkPadding(dst)
    print("Image preview")
    show_image(dst,6000)
    cv2.imwrite(savepath, dst) 
    print("Done, saved as", savepath)

    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."

    return


if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

