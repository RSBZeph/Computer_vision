import numpy as np
import cv2 as cv
import glob

x = 6
y = 9
corners = []
# termination criteria

def click_event(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        corners.append((x, y))
#test
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((x*y,3), np.float32)
objp[:,:2] = np.mgrid[0:y,0:x].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#read photos from folder
images = glob.glob('C:\\Users\\rsbze\\Desktop\\Repos\\Uni\\Computer_vision\\test_images\\*.jpg')

counter = 0
for fname in images:
    counter += 1
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (y,x), None)
    # If found, add object points, image points (after refining them)

    #if ret == false:
    # cv.namedWindow("resize", cv.WINDOW_NORMAL)
    # cv.resizeWindow("resize", 1000, 1000)
    # cv.imshow('resize', img)
    # cv.setMouseCallback('resize', click_event)
    # while len(corners) < 4:
    #     cv.waitKey(10)

    print(corners)

    print("found " + str(counter-1))
    objpoints.append(objp)
    #corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners)
    # Draw and display the corners

    cv.drawChessboardCorners(img, (y,x), corners, None)#ret)
    cv.imshow('resize', img)

    

    cv.waitKey(2500)


cv.destroyAllWindows()


#    cv.namedWindow("ja", cv.WINDOW_NORMAL)
#    cv.resizeWindow("ja", 500, 500)
#    cv.imshow('ja', img)
#    cv.waitKey(500)