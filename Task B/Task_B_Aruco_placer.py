import numpy as np
import copy
import cv2 as cv
import cv2.aruco as aruco

task = cv.imread("D:\RoboISM\Main Tasks\OpenCV\Task B\CVTask.png")

def rotateMarker(img, angle, scale = 1):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.bitwise_not(img)
    (h, w) = img.shape[:2]
    centre = (w/2, h/2)
    RotMat = cv.getRotationMatrix2D(centre, angle, scale)
    rotatedImg = cv.bitwise_not(cv.warpAffine(img, RotMat, (w, h)))
    return rotatedImg

markers = {}
markers[0] = cv.imread('D:\RoboISM\Main Tasks\OpenCV\Task B\LAMO.jpg')
markers[1] = cv.imread('D:\RoboISM\Main Tasks\OpenCV\Task B\XD.jpg')
markers[2] = cv.imread('D:\RoboISM\Main Tasks\OpenCV\Task B\Ha.jpg')
markers[3] = cv.imread('D:\RoboISM\Main Tasks\OpenCV\Task B\HaHa.jpg')
squareAngles = []  # acc. to aruco ids 1,2,3,4 -> 8,7,4,5
markerAngles = []
imageIds = [8,7,4,5]
contourAreas = []
markerAreas = []
minBoxes = []
boundingBoxesImg = []
croppedMarkers = []

Aruco = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_5X5_50),
                            aruco.DetectorParameters())
for i in range(len(markers)):
    marker = copy.deepcopy(markers[i])
    corners, ids, _ = Aruco.detectMarkers(marker)
    aruco.drawDetectedMarkers(marker, corners, ids, [0,0,255])
    cv.imshow('marker',marker)
    cv.waitKey(0)

img = copy.deepcopy(task)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, [5,5], 0)
canny = cv.Canny(blur, threshold1 = 50, threshold2= 70)

imgContours, hierarchy = cv.findContours(canny, 
                                      mode = cv.RETR_EXTERNAL, 
                                      method= cv.CHAIN_APPROX_SIMPLE)

for i in range(len(imgContours)):
    cnt = cv.drawContours(img, imgContours, i, 
                          [0,0,255], 1, 
                          lineType= cv.LINE_4, 
                          hierarchy= hierarchy)
    minRect = cv.minAreaRect(imgContours[i])
    minBox  = cv.boxPoints(minRect).astype('int')
    boundingBox = cv.boundingRect(imgContours[i])
    cnt = cv.rectangle(cnt, boundingBox[:2], 
                       np.add(boundingBox[:2],
                              boundingBox[2:]),
                         [0,0,255], 2, cv.LINE_4  )
    cnt = cv.drawContours(img, [minBox], -1, 
                    [0,255,0], 2, 
                    lineType= cv.LINE_4)
    cnt = cv.putText(cnt, str(i), 
                     org = minBox[1], 
                     fontFace= cv.FONT_HERSHEY_COMPLEX, 
                     fontScale= 0.7, color= [255,0,0], 
                     thickness= 2, lineType= cv.LINE_4)
    minBoxes.append(minBox)
    boundingBoxesImg.append(boundingBox)
minBoxes = np.array(minBoxes)[imageIds]
boundingBoxesImg = np.array(boundingBoxesImg)[imageIds]

cv.imshow('Hierarchy', cnt)
cv.waitKey(0)

print('Image Bounding Boxes : \n', boundingBoxesImg)

for i in range(len(imageIds)):
    _,_, squareAngle = cv.minAreaRect(imgContours[imageIds[i]])
    cntA = cv.contourArea(imgContours[imageIds[i]]) 
    marker = cv.cvtColor(markers[i], cv.COLOR_BGR2GRAY)
    marker = cv.Canny(marker, 50, 70)
    markerContour,_ = cv.findContours(marker, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    mrkArea = cv.contourArea(markerContour[0])
    _,_, markerAngle = cv.minAreaRect(markerContour[0])
    contourAreas.append(cntA)
    markerAreas.append(mrkArea)
    squareAngles.append(squareAngle)
    markerAngles.append(markerAngle)

print('Contour Areas : ', contourAreas)
print('Marker Areas : ', markerAreas)
print("Contour Angles : ", squareAngles)  
print("Marker Angles : ", markerAngles)

scale = (np.divide(contourAreas, markerAreas))**0.5
correctMarkers = []
boundingBoxesMarker = []
minBoxMarker = []

for i in range(len(markers)):
    correctedImg = rotateMarker(markers[i], markerAngles[i]-squareAngles[i], scale[i])
    correctMarkers.append(correctedImg)
for i in correctMarkers:
    marker = cv.Canny(i, 20, 70)
    markerContour,_ = cv.findContours(marker, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    boundingBox = cv.boundingRect(markerContour[0])
    boundingBoxesMarker.append(boundingBox)

print('Marker Bounding Boxes : \n', boundingBoxesMarker)

for i in range(len(boundingBoxesMarker)):
    x,y,w,h = boundingBoxesMarker[i]
    crop = correctMarkers[i][y-1:y+h+1,x-1:x+w+1]
    croppedMarkers.append(crop)
    marker = cv.Canny(crop, 20, 70)
    cropContours, _ = cv.findContours(marker, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    minRect = cv.minAreaRect(cropContours[0])
    minBox = cv.boxPoints(minRect)
    minBoxMarker.append(minBox)


masks = []
masksInv = []
for i in range(len(croppedMarkers)):
    mask = np.zeros(croppedMarkers[i].shape[:2]).astype('uint8')
    mask = cv.fillPoly(mask, [minBoxMarker[i].astype('int')], [255,255,255], cv.LINE_4)
    masks.append(mask)
    maskInv = cv.bitwise_not(mask)
    masksInv.append(maskInv)

    
# croppedMarkers[0] = croppedMarkers[0][:,:349]
# croppedMarkers[2] = croppedMarkers[2][:274,:274]
# croppedMarkers[3] = croppedMarkers[3][:,:504]
# masks[0] = masks[0][:,:349]
# masks[2] = masks[2][:274,:274]
# masks[3] = masks[3][:,:504]
# masksInv[0] = masksInv[0][:,:349]
# masksInv[2] = masksInv[2][:274,:274]
# masksInv[3] = masksInv[3][:,:504]  

for i in range(len(masks)):
   
    x,y,w,h = boundingBoxesImg[i]
    crop = task[y-1:y+h+1,x-1:x+w+1]
    shape_x, shape_y = crop.shape[:2]
    croppedMarkers[i] = croppedMarkers[i][:shape_x,:shape_y]
    masks[i] = masks[i][:shape_x,:shape_y]
    masksInv[i] = masksInv[i][:shape_x,:shape_y]

    print(crop.shape, croppedMarkers[i].shape, masks[i].shape, masksInv[i].shape)
    mask = copy.deepcopy(masks[i])
    maskInv = copy.deepcopy(masksInv[i])
    cropImg = cv.bitwise_and(crop, crop, mask = maskInv)
    cropMark = cv.bitwise_and(croppedMarkers[i], croppedMarkers[i], mask =  mask)
    place = cv.add((cropImg),cv.cvtColor(cropMark, cv.COLOR_GRAY2BGR))
    task[y-1:y+h+1,x-1:x+w+1] = place

cv.imwrite("D:\RoboISM\Main Tasks\OpenCV\Task B\CVTask_done.png", task)

cv.imshow('Task',task)
cv.waitKey(0)

corners, ids, _ = Aruco.detectMarkers(task)
aruco.drawDetectedMarkers(task, corners, ids, [0,0,255])
cv.imshow('task',task)
cv.waitKey(0)