import cv2 as cv
import numpy as np

ref = cv.imread("D:\RoboISM\Main Tasks\OpenCV\Task A\Reference Grid.png")
task = cv.imread("D:\RoboISM\Main Tasks\OpenCV\Task A\Task.png")

""" Fuction to create Dashed-Red Line  """

def dashed_line(img, pt1, pt2, color,
                 thickness, space_size,
                   dash_size):
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    dist = np.sqrt(np.sum((pt1-pt2)**2))
    cos = (pt2[0]-pt1[0])/dist
    sin = (pt2[1] - pt1[1])/dist
    step_x = (dash_size * cos)
    step_y = (dash_size * sin)
    space_x = (space_size * cos)
    space_y = (space_size * sin)
    i = pt1[0]
    j = pt1[1]
    for itr in range(0,int(dist),space_size+dash_size):
        
        endPoint = (int(i+step_x), int(j+step_y))
        if np.linalg.norm(endPoint - pt1) > dist:
            endPoint = pt2
        img = cv.line(img, (i,j), 
                       endPoint,
                       color= color, 
                       thickness=thickness, 
                       lineType= cv.LINE_4)
        i += int(space_x + step_x)
        j += int(space_y + step_y)
    return img

"""       Coordinates and Ratio       """

print("Reference grid shape: ",ref.shape[:2])

x = []
y = []

#Iterating from mid-point of axes:

for i in range(1, ref.shape[1]):
    if ref[344,i,1] == 255 and ref[344,i-1,1] != 255:
        x.append(i)
for i in range(1,ref.shape[0]):
    if ref[i,344,1] == 255 and ref[i-1,344,1] != 255:
        y.append(i)

print('Grid x-coordinates  : ',x)
print('Grid y-coordinates  : ',y)

x_ratio = np.array(x) / ref.shape[1] 
y_ratio = np.array(y) / ref.shape[0] 

""" Coordinates of Lines on Task Image """

h, w = task.shape[:2]
print('Shape of Task Image : ',task.shape[:2])
task_x = (h * x_ratio).astype('uint16')
task_y = (w * y_ratio).astype('uint16')

print('Task x-coordinates  : ',(task_x))
print('Task y-coordinates  : ',(task_y))

##########################################
"""           LINE PARAMETERS          """
##########################################

thickness = 2      
space_size = 12
dash_size = 20

##########################################

for i in task_y:
    task = cv.line(task,(task_x[0],i), 
                   (task_x[-1],i), 
                   color = (0,255,0), 
                   thickness=thickness, 
                   lineType= cv.LINE_4)

for itr, i in enumerate(task_x):
    if itr == 1 or itr == len(task_x) - 2:
        continue
    task = cv.line(task,(i,task_y[0]), 
                   (i, task_y[-1]), 
                   color = (0,255,0), 
                   thickness=thickness, 
                   lineType= cv.LINE_4)

for i in [4,1]:
    task = cv.line(task, (task_x[i], task_y[2]), 
                    (task_x[i], task_y[-1]), 
                    color = (0,255,0), 
                    thickness = thickness,
                    lineType= cv.LINE_4)
for i in [4,1]:
    task = cv.line(task, (task_x[i], task_y[2]), 
                   (task_x[i], task_y[1]), 
                   color = (0,0,255), 
                   thickness = thickness, 
                   lineType= cv.LINE_4)

dash_pts_ind = [[(1,0),(2,0)],[(1,0),(1,1)],
                [(1,1),(2,1)],[(4,0),(4,1)],
                [(1,3),(2,3)],[(1,4),(2,4)]]
dash_pts = []

for pt_ind in dash_pts_ind:
    dash_pt = []
    for i,j in pt_ind:
        temp = (task_x[i], task_y[j])
        dash_pt.append(temp)
    dash_pts.append(dash_pt)

for pt in dash_pts:
    pt1, pt2 = pt
    task = dashed_line(task, pt1, pt2,
                       color=(0,0,255),
                       thickness= thickness,
                       space_size=space_size,
                       dash_size=dash_size)

cv.imshow('ref',task)
cv.waitKey(0)
# cv.imwrite('D:\RoboISM\Main Tasks\OpenCV\Task A\Grid_Applied.png', task)