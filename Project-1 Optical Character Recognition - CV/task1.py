"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np
import sys # to be removed

def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args



def ocr(test_img, characters):

    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    characters = enrollment(characters)

    labelDictImg, BoundBoxDict = detection(test_img)
    
    recognDict = recognition(characters, labelDictImg)

    resultList = []

    for key in recognDict:
        newDict = {}
        BBlist = BoundBoxDict[key]
        newDict["bbox"] = BBlist
        newDict["name"] = recognDict[key]
        resultList.append(newDict)

    

    print(resultList)
    print("total bounded boxes :" ,len(resultList))
    return resultList

def enrollment(charList):   # <-- extracts edge features from input charater images
    print(">>> Enrollment begins:")

    # Step 1 : Your Enrollment code should go here.
    # reading the characters and binarizing them:

    for num in range(len(charList)):
        for y1 in range(len(charList[num][1])):
            for idx, val1 in np.ndenumerate(charList[num][1][y1]):
                if val1>127:
                    charList[num][1][y1][idx] = 0
                else:
                    charList[num][1][y1][idx] = 1

    # now removing the borders of the binary image 
    # and applying canny edge detector with thresholds of 100 and 200

    for num in range(len(charList)):
        xx = extract_labels(charList[num][1],build_BB(charList[num][1]),0,False) # removes extra boundary
        for key in xx:
            canny = cv2.Canny(xx[key],100,200)
            charList[num][1] = canny

    print(">>> Enrollment done!")
    # returns nested list of canny edge features and their respective identifier
    return charList


def detection(testImage): # <-- returns a list of labelled characters seperated from single source image and bounding box information
    print(">>> Detection begins: ")
    # Step 2 : Your Detection code should go here.

    # identify the background
    countbg, bg = 0, 0
    for y1 in range(len(testImage)):
        for idx, val1 in np.ndenumerate(testImage[y1]):
            if val1>127:
                testImage[y1][idx] = 0
                countbg = countbg + 1
            else:
                testImage[y1][idx] = 1
    if countbg > (len(testImage) * len(testImage[1])/2):
        bg = 0
    else:
        bg = 1
    #print("bg is :" , bg)

    # make of copy of test_img
    labelledImage = np.copy(testImage)
    labelledImage = labelledImage.astype(np.int32) 
    mylabel = 2

    # CCL implementation: first pass, we label different components:
    for y in range(len(testImage)):
        for x,val in np.ndenumerate(testImage[y]):
            x = x[0]
            if x==0 and (y==0) and (val != bg):
                # start
                labelledImage[y,x] = mylabel
            elif x>0 and (testImage[y,x-1] == val) and (val != bg):
                # horizontal neighbour found
                labelledImage[y, x] = labelledImage[y, x-1]
            elif y>0 and(testImage[y-1, x] == val) and (val != bg):
                # vertical neighbour found
                labelledImage[y, x] = labelledImage[y-1, x]
            elif val != bg:
                # new label introduced
                labelledImage[y, x] = mylabel
                mylabel = mylabel + 1

    # CCL implementation: second pass, join neighoring labels by updating their true values:
    for y in range(len(labelledImage)):
        for x,val in np.ndenumerate(labelledImage[y]):
            x = x[0]
            if y>0 and(testImage[y-1, x] == testImage[y, x]):
                fakelabel = val
                orglabel = labelledImage[y-1, x]
                if fakelabel != orglabel:
                    labelledImage[labelledImage==fakelabel] = orglabel
    #np.savetxt('C:/Users/jmudi/Documents/AA Spring 2021/CV CSE573/project 1/Project1(3)/Project1/data/1.txt', labelledImage, fmt="  %s  ")

    # returns dictionary of [xmin,ymin,xmax,ymax] and unique labels:
    getbb = build_BB(labelledImage) 

    # returns dictionary of [xmin,ymin,width,height] and unique labels:
    getbbDict = formatBBDict(getbb)
    #print(getbbDict)

    # extract each character from the big image and create seperate cropped images:
    labeldict = extract_labels(labelledImage,getbb, bg, False)

    # Display bounded box on the test_img:
    labelledImage = showBB(labelledImage, getbb)
    labelledImage = labelledImage.astype(np.int8)
    #print(type(labelledImage))

    #Show test_img with bounded box for confirmation:
    show_image(labelledImage,20000)

    return labeldict, getbbDict

def recognition(characters, labelleddict): #<-- returns a dictionary of unique key identifier for each label and match value "UNKNOWN" if unmatched
    # Step 3 : Your Recognition code should go here.

    recogDict = {}
    count =0
    # For every labelled image, apply canny edges and perform matching with templates:
    for key in labelleddict:
        # Apply canny edge detector on detected charaters:
        cannylabels= cv2.Canny(labelleddict[key],100,200)
        cannyTemplate = characters[0][1]
        minerror = np.inf
        minnum = np.inf

        # For every character template (from enrollment), resize template according to detected character's bounding box.
        for num in range(len(characters)):
            widthOflabel = len(labelleddict[key][1])
            heightOflabel = len(labelleddict[key])
            cannyTemplate = cv2.resize(characters[num][1],(widthOflabel,heightOflabel))
            compare = cannylabels - cannyTemplate 
            square = np.multiply(compare,compare)
            # Using SSD to perform matching, variable "error" represents the threshold:
            #error = np.sqrt(np.sum(square)/(widthOflabel*heightOflabel))
            error = np.sum(compare)/(widthOflabel*heightOflabel)
            if error<minerror:
                minerror = error
                minnum = num
        recogDict[key] = 'UNKNOWN'
        if minerror < 40 :
            print("error:",minerror," label:",key," template_char:",characters[minnum][0])
            #show_image(compare, 2000)
            recogDict[key] = characters[minnum][0]
            count = count + 1
    print("total:",count)
    return recogDict


def build_BB(labels):
    bb = {} # [xmin,ymin,xmax,ymax]
    print("Building bounding box")
    xmin = np.inf
    ymin = np.inf
    xmax = 0
    ymax = 0

    for y in range(len(labels)):
        for x,val in np.ndenumerate(labels[y]):
            x=x[0]
            if val != 0 :
                if val not in bb :
                    bb[val] = [xmin,ymin,xmax,ymax]
                thisLabel = bb[val]
                if thisLabel[0]>x:
                    bb[val][0] = x
                if thisLabel[2]<x:
                    bb[val][2] = x
                if thisLabel[1]>y:
                    bb[val][1] = y
                if thisLabel[3]<y:
                    bb[val][3] = y
    
    return bb
            
def showBB(labels1, BB):
    labels = np.copy(labels1)
    bbcopy = BB
    for y in range(len(labels)):
        for x,val in np.ndenumerate(labels[y]):
            x=x[0]
            if val in bbcopy:
                for y1 in range(bbcopy[val][1],bbcopy[val][3]+1):
                    for x1 in range(bbcopy[val][0],bbcopy[val][2]+1):
                        labels[bbcopy[val][1],x1] = 385
                        labels[y1,bbcopy[val][2]+1] = 385
                        labels[bbcopy[val][3]+1,x1] = 385
                        labels[y1,bbcopy[val][0]] = 385
                del bbcopy[val]

    return labels


def extract_labels(labelsc, BB, backg, print2file=True):
    bbcopy = BB
    newImageDict = {}
    newImage = np.array(0)
    labels = np.copy(labelsc)
    for key in bbcopy:
        if backg == 0 :
            newImage = np.zeros([bbcopy[key][3]+3-bbcopy[key][1],bbcopy[key][2]+3-bbcopy[key][0]]) - 254
        else:
            newImage = np.zeros([bbcopy[key][3]+3-bbcopy[key][1],bbcopy[key][2]+3-bbcopy[key][0]]) 
        #print(type(newImage))
        for y in range(len(labels)):
            for x,val in np.ndenumerate(labels[y]):
                x=x[0]
                if key == val:
                    if backg == 0 :
                        newImage[y-bbcopy[key][1]+1,x-bbcopy[key][0]+1] = 0
                    else:
                        newImage[y-bbcopy[key][1]+1,x-bbcopy[key][0]+1] = 1
        
        newImage = np.array(newImage * 255, dtype = np.uint8)  
        if print2file :
            cv2.imwrite(str(key)+".jpg", newImage)
        newImageDict[key] = newImage
    return newImageDict

def formatBBDict(BB2):
    # BB1 is  [xmin,ymin,xmax,ymax]
    BB1 = BB2.copy()
    for key in BB1:
        listfromKey = BB1[key]
        BB1[key] = [listfromKey[0], listfromKey[1], listfromKey[2]-listfromKey[0], listfromKey[3]-listfromKey[1] ]

    return BB1


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    np.set_printoptions(threshold=np.inf) # changed to print
    
    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
