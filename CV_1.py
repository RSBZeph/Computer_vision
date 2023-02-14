import numpy as np
import cv2 as cv
import glob

x = 6
y = 9
main_corners = []
# termination criteria

def click_event(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        main_corners.append((x, y))
        print((x,y))

def make_grid():
    corners_grid = []
    c1_c2 = (main_corners[0][0] - main_corners[1][0], main_corners[0][1] - main_corners[1][1])
    c2_c3 = (main_corners[1][0] - main_corners[2][0], main_corners[1][1] - main_corners[2][1])
    c3_c4 = (main_corners[2][0] - main_corners[3][0], main_corners[2][1] - main_corners[3][1])

    print("1 " + str(c1_c2))
    print("2 " + str(c2_c3))
    print("3 " + str(c3_c4))

    # y / 9
    # x / 6
    # for dx in range(x):
    #     for dy in range(y):

#test
criteria = (cv.TERM_CRITERIA_EPS, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((x*y,3), np.float32)
objp[:,:2] = np.mgrid[0:y,0:x].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#read photos from folder
#images = glob.glob('C:\\Users\\rsbze\\Desktop\\Repos\\Uni\\Computer_vision\\test_images\\*.jpg')
images = glob.glob('C:\\Users\\yoran\\Documents\\UU\\GMT\\Jaar1\\P3\\Computer_vision\\ComputerVisionP1\\test_images\\*.jpg')

counter = 0
for fname in images:
    counter += 1
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (y,x), None)
    # If found, add object points, image points (after refining them)
    cv.namedWindow("resize", cv.WINDOW_NORMAL)
    cv.resizeWindow("resize", 800, 800)

    #if ret == false:

    cv.imshow('resize', img)
    cv.setMouseCallback('resize', click_event)
    while len(main_corners) < 4:
        cv.waitKey(10)
    make_grid()
    main_corners.clear()
    #print(corners)

    #print("found " + str(counter-1))
    objpoints.append(objp)
    #corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    #imgpoints.append(corners)
    # Draw and display the corners

    #cv.drawChessboardCorners(img, (y,x), corners, ret)
    #cv.imshow('resize', img)

    

    #cv.waitKey(2500)


cv.destroyAllWindows()


#    cv.namedWindow("ja", cv.WINDOW_NORMAL)
#    cv.resizeWindow("ja", 500, 500)
#    cv.imshow('ja', img)
#    cv.waitKey(500)