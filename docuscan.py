import cv2 
import numpy as np
from matplotlib import pyplot as plt
import math


img = cv2.imread('doc.jpg', 1)
image_bordered = cv2.copyMakeBorder(img, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT) 



gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
kernel = np.ones((3,3), np.uint8)
wiped_img = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel,iterations=5)
ed_img = cv2.Canny(wiped_img, 100, 200)
#image_bordered = cv2.copyMakeBorder(ed_img, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT) 
cv2.imshow('canny',ed_img)
cont_list, _ = cv2.findContours(ed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntsSorted = sorted(cont_list, key= cv2.contourArea, reverse=True)

# if len(cntsSorted) == 0:
#     print("k")
# else:
#     print('l')
# pass    

epsilon = 0.01*cv2.arcLength(cntsSorted[0], True)
big_cont = cv2.approxPolyDP(cntsSorted[0],epsilon,True)
cv2.imshow("img",img)
#finding corners of the doc
extLeft = tuple(big_cont[big_cont[:, :, 0].argmin()][0])
extRight = tuple(big_cont[big_cont[:, :, 0].argmax()][0])
extTop = tuple(big_cont[big_cont[:, :, 1].argmin()][0])
extBot = tuple(big_cont[big_cont[:, :, 1].argmax()][0])



# doc_width1 = (extTop[0] - extLeft[0])
# doc_width2 = (extRight[0] - extBot[0])
doc_width1 = math.dist(extTop,extLeft)
doc_width2 = math.dist(extRight,extBot)
doc_width = round(max(doc_width1,doc_width2))

# doc_length1 = (extBot[1] - extLeft[1])
# doc_length2 = (extRight[1] - extTop[1])
doc_length1 = math.dist(extBot,extLeft)
doc_length2 = math.dist(extRight,extTop)
doc_length = round(max(doc_length1,doc_length2))


pts1 = np.float32([extLeft,extTop,extBot,extRight])
pts2 = np.float32([(0,0),(doc_width,0),(0,doc_length),(doc_width,doc_length)])

p = cv2.getPerspectiveTransform(pts1,pts2)
#wrp = cv2.warpPerspective(img,final_doc,(500,600))

result = cv2.warpPerspective(img, p, (doc_width,doc_length))

#---------- document image was extrated ------------(documents that are not fully in frame don't work. Also doesn't work for images that are already scanned.)
#paste code back after thsi point if the new stuff doesn't work
#the old code is in repository.py
quiet = cv2.fastNlMeansDenoising(result, h= 10)
gray_result = cv2.cvtColor(quiet,cv2.COLOR_BGR2GRAY)
thresholded = cv2.adaptiveThreshold(gray_result,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,55,15)
blob = cv2.GaussianBlur(thresholded,(7,7),0)
canny_text = cv2.Canny(blob,0,255)
cnts, _ = cv2.findContours(canny_text,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
result_copy = result
rect_list = []
for con in cnts:
    x,y,w,h = cv2.boundingRect(con)
    #rect = cv2.rectangle(result_copy,(x,y),(x+w,y+h),(0,255,0,),1)
    rect = [x,y,w,h]
    rect_list.append(rect)
    rect_list.append(rect)
    # rect = cv2.minAreaRect(con)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # img = cv2.drawContours(result_copy,[box],0,(0,255,255),1)

grouped_list, _  = cv2.groupRectangles(rect_list,1)
for r in grouped_list:
    rect = cv2.rectangle(result_copy,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(0,255,0,),1)
    pass
#cv2.drawContours(result,cnts,-1,(0,255,255),1)
cv2.imshow('detected document',result_copy )
cv2.waitKey(0)
cv2.destroyAllWindows