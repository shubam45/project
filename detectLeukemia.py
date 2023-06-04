import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

#Contrast stretching
def contrastStretching(img,r1,r2,a,b,c):
    s1 = a*r1
    s2 = b*(r2-r1)+s1
    imgC = np.zeros((256,256), dtype=np.int32)
    for i in range(0,256):
        for j in range(0,256):
            r = img[i,j]
            if r<r1:
                imgC[i,j] = a*r
            elif r>r1 and r<r2:
                imgC[i,j] = b*(r-r1) +s1
            else: 
                imgC[i,j] = c*(r-r2) + s2

    imgC = imgC.astype(np.uint8)
    return imgC

def histeq(img):
    a = np.zeros((256,),dtype=np.float16)
    b = np.zeros((256,),dtype=np.float16)
    imghist = img

    height,width=img.shape

    #finding histogram
    for i in range(width):
        for j in range(height):
            g = imghist[j,i]
            a[g] = a[g]+1

    #performing histogram equalization
    tmp = 1.0/(height*width)
    b = np.zeros((256,),dtype=np.float16)

    for i in range(256):
        for j in range(i+1):
            b[i] += a[j] * tmp
        b[i] = round(b[i] * 255)

    # b now contains the equalized histogram
    b=b.astype(np.uint8)

    #Re-map values from equalized histogram into the image
    for i in range(width):
        for j in range(height):
            g = imghist[j,i]
            imghist[j,i]= b[g]

    imghist = imghist.astype(np.uint8)
    return imghist

def enhancement(img1,img2):
    imgadd = cv2.add(img1,img2)
    imgsub = cv2.subtract(img1, img2)
    imgfinal= cv2.add(imgadd, imgsub)
    return imgfinal

def thresholding(img,t):
    for i in range(0,256):
        for j in range(0,256):
            if img[i,j]>t:
                img[i,j] =0
            else:
                img[i,j] = 255
    return img

def dilation(img,mask):
    img = img.astype(np.float16)
    dilimg = np.zeros((256,256), dtype=np.float16)
    for i in range(1,255):
        for j in range(1,255):
            imgtemp = img[i-1:i+2, j-1:j+2]
            res = np.multiply(imgtemp,mask)
            dilimg[i,j] = np.amax(res)
    dilimg = dilimg.astype(np.uint8)
    return dilimg

def erosion(img,mask):
    img = img.astype(np.float16)
    eroimg = np.zeros((256,256), dtype=np.float16)
    for i in range(1,255):
        for j in range(1,255):
            imgtemp = img[i-1:i+2, j-1:j+2]
            res=[]
            for k in range(0,3):
                for m in range(0,3):
                    if mask[k][m] ==1:
                        a = imgtemp[k,m]
                        res.append(a)
            eroimg[i,j] = np.amin(res)
    eroimg = eroimg.astype(np.uint8)
    return eroimg
def featureExtraction(img):
 cells=img[:,:,0]
 pixels_to_um = 0.454
 ret1, thresh = cv2.threshold(cells, 0, 255, 
cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 kernel = np.ones((3,3),np.uint8)
 opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
 from skimage.segmentation import clear_border
 opening = clear_border(opening) #Remove edge touching grains
 plt.imshow(opening, cmap='gray') #This is our image to be segmented further using watershed
 sure_bg = cv2.dilate(opening,kernel,iterations=10)
 plt.imshow(sure_bg, cmap='gray') #Dark region is our sure backgroundm
 dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
 plt.imshow(dist_transform, cmap='gray') #Dist transformed img.
 print(dist_transform.max()) #gives about 21.9
 ret2, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
 plt.imshow(sure_fg, cmap='gray')
 sure_fg = np.uint8(sure_fg) #Convert to uint8 from float
 unknown = cv2.subtract(sure_bg,sure_fg)
 plt.imshow(unknown, cmap='gray')
 ret3, markers = cv2.connectedComponents(sure_fg)
 plt.imshow(markers)
 markers = markers+10
# Now, mark the region of unknown with zero
 markers[unknown==255] = 0
 plt.imshow(markers, cmap='jet') #Look at the 3 distinct regions.
#Now we are ready for watershed filling.
 markers = cv2.watershed(img,markers)
#Let us color boundaries in yellow.
#Remember that watershed assigns boundaries a value of -1
 img[markers == -1] = [0,255,255] 
#label2rgb - Return an RG~B image where color-coded labels are painted over the image.
 img2 = color.label2rgb(markers, bg_label=0)
 imr1 = cv2. resize(img, (960, 540))
 imr2 = cv2. resize(img2, (960, 540))
 plt.imshow(img2)
 # cv2.imshow('Overlay on original image', imr1)
 # cv2.imshow('Colored Grains', imr2)
 cv2.waitKey(0)
########################################################################
#############
#Now, time to extract properties of detected cells
#Directly capturing props to pandas dataframe
# 
props = measure.regionprops_table(markers, cells,properties=['label',
 'area', 'equivalent_diameter',
 'mean_intensity', 'solidity', 'orientation',
 'perimeter'])

def edgeDetection(img):
    imgS = img.astype(np.float16)
    sobx=[[-1, -2, -1],
          [0, 0, 0],
          [1, 2, 1]]
    sobx = np.array(sobx, np.float16)
    soby =[[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]
    soby = np.array(soby,np.float16)
    for i in range(1,254):
        for j in range(1,254):
            imgtemp = img[i-1:i+2, j-1:j+2]
            x = np.sum(np.multiply(sobx,imgtemp))
            y = np.sum(np.multiply(soby,imgtemp))
            pixvalue = np.sqrt(x**2 + y**2)
            imgS[i,j] = pixvalue
    imgS = imgS.astype(np.uint8)
    return imgS

def detectCircles(img,openedimg):
    imgcircle = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    detected_circles = cv2.HoughCircles(openedimg,  
                    cv2.HOUGH_GRADIENT, 10, minDist= 10, param2= 30, minRadius = 1, maxRadius = 13) 

    #minDist: Minimum distance between the center (x, y) coordinates of detected circles. If the minDist is too small, multiple circles in the same #neighborhood as the original may be (falsely) detected. If the minDist is too large, then some circles may not be detected at all.

    #param2: Accumulator threshold value for the cv2.HOUGH_GRADIENT method. The smaller the threshold is, the more circles will be detected
    #(including false circles). The larger the threshold is, the more circles will potentially be returned.
    
    # Draw circles that are detected. 
    ctr=0
    if detected_circles is not None: 
    
        # Convert the circle parameters a, b and r to integers. 
        detected_circles = np.uint16(np.around(detected_circles)) 
        
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] #a,b are the coordinates of the center and r is the radius

            # Draw the circumference of the circle. 
            imgcirclefinal = cv2.circle(imgcircle, (a, b), r, (0, 255, 0), 2) 
    
            # Draw a small circle (of radius 1) to show the center. 
            #cv2.circle(img1, (a, b), 1, (255, 0, 0), 3) 
            ctr+=1 
    return imgcirclefinal,ctr