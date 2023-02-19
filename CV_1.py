#assignment 1 computer vision
#By Romeo Zeph (6286372), Yoran den Heijer (6242057)
#Sites used:
#   - https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
#   - https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html
#   - https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/

import numpy as np
import cv2 as cv
import glob

x = 6
y = 9
win_size = 800
main_corners = []
live = False

# function called when there is a click on the screen
def click_event(event, x, y, flags, params):
    #saves on of the corners when clicking left
    if event == cv.EVENT_LBUTTONDOWN:
        main_corners.append((x, y))
        print((x,y))

# interpolation for when corners not automatically found
def make_grid():
    # resulting grid
    corners_grid = []
    # diffence of the top two corners and of the bottom two corners
    c2_c1 = (main_corners[1][0] - main_corners[0][0], main_corners[1][1] - main_corners[0][1])
    c4_c3 = (main_corners[3][0] - main_corners[2][0], main_corners[3][1] - main_corners[2][1])
    # how much the next grid point should be (so steps)
    step_x1 = c2_c1[0]/(x-1)
    step_y1 = c2_c1[1]/(x-1)
    step_x2 = c4_c3[0]/(x-1)
    step_y2 = c4_c3[1]/(x-1)
    # top and bottom lists of grid points
    interp1 = []
    interp2 = []
    # filling those lists
    for dx in range(x):
        interp_1x = main_corners[0][0] + dx * step_x1
        interp_1y = main_corners[0][1] + step_y1    
        interp_2x = main_corners[2][0] + dx * step_x2 
        interp_2y = main_corners[2][1] + step_y2 
        interp1.append([interp_1x,interp_1y])
        interp2.append([interp_2x,interp_2y])
    # filling the points between the top and bottom layer and then putting then in result list
    for dx in range(x):
        for dy in range(y):
            step_x3 = (interp2[dx][0] - interp1[dx][0])/(y-1)
            step_y3 = (interp2[dx][1] - interp1[dx][1])/(y-1)

            result_x = interp1[dx][0] + dy * step_x3
            result_y = interp1[dx][1] + dy * step_y3

            result = [result_x,result_y]
            corners_grid.append(result)
    
    return corners_grid

# funtion for drawing axis in online fase
def draw_axis(img, corners, imgpts):
    corners = np.int32(corners).reshape(-1,2)
    corner = tuple(corners[0].ravel())
    imgpts = np.int32(imgpts).reshape(-1,2)
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 8)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 8)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 8)
    return img

# funtion for drawing cube in online fase
def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw pillars of cube
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i].ravel()), tuple(imgpts[j].ravel()),(255, 255, 0),5)
    # draw top and bottom layer of cube
    img = cv.drawContours(img, [imgpts[4:]],-1,(255,255,0),5)
    img = cv.drawContours(img, [imgpts[:4]],-1,(255,255,0),5)
    return img


criteria = (cv.TERM_CRITERIA_EPS, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((x*y,3), np.float32)
objp[:,:2] = np.mgrid[0:y,0:x].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#read photos from folder
images = glob.glob('C:\\Users\\rsbze\\Desktop\\Repos\\Uni\\Computer_vision\\short_test\\*.jpg')
drawimages = glob.glob('C:\\Users\\rsbze\\Desktop\\Repos\\Uni\\Computer_vision\\draw\\*.jpg')
interpolationimages = glob.glob('C:\\Users\\rsbze\\Desktop\\Repos\\Uni\\Computer_vision\\test_interpolation\\*.jpg')
#images = glob.glob('C:\\Users\\yoran\\Documents\\UU\\GMT\\Jaar1\\P3\\Computer_vision\\ComputerVisionP1\\draw\\*.jpg')
#drawimage = glob.glob('C:\\Users\\yoran\\Documents\\UU\\GMT\\Jaar1\\P3\\Computer_vision\\ComputerVisionP1\\draw\\*.jpg')

counter = 0
for fname in drawimages:
    counter += 1
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (y,x), cv.CALIB_CB_FAST_CHECK)
    ret = True
    print(str(counter) + " " + str(ret))
    # make new resized window
    cv.namedWindow("resize", cv.WINDOW_NORMAL)
    cv.resizeWindow("resize", win_size, win_size)
    # if the corners are not automatically found then manually select corners
    if ret == False:
        cv.imshow('resize', img)
        cv.setMouseCallback('resize', click_event)
        while len(main_corners) < 4:
            cv.waitKey(10)
        corners = make_grid()
        main_corners.clear()
        corners = np.float32(corners)

    # add objects to list and apply function to make the corners more accurate
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)

    #Draw and display the corners
    cv.drawChessboardCorners(img, (y,x), corners2, ret)
    cv.imshow('resize', img)

    cv.waitKey(1000)

#calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez("CameraParams", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
# undistortion
img = cv.imread('test1_first.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# print the camera intrinsics
print(str(newcameramtx))

dst = cv.undistort(img, mtx, dist, None, newcameramtx)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

cv.destroyAllWindows()


# online fase

#How axis look like
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
#How the cube looks like
cube = np.float32([[0,0,0], [0,2,0], [2,2,0], [2,0,0], [0,0,-2] ,[0,2,-2] ,[2,2,-2] ,[2,0,-2]])

#live with webcam recording the cube
if live == False:
    for fname in drawimages:
        counter += 1
        cv.namedWindow("img", cv.WINDOW_NORMAL)
        cv.resizeWindow("img", win_size, win_size)
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (y,x), None)
        print(str(counter) + " " + str(ret))
        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            # Find the rotation and translation vectors.
            ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
            # project 3D points to image plane
            imgpts, jac2 = cv.projectPoints(cube, rvecs, tvecs, mtx, dist)
            imgpts2, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
            # draw axis and cube on the picture
            img = draw_cube(img,corners2,imgpts)
            img = draw_axis(img,corners2,imgpts2)

            cv.imshow('img',img)
            cv.waitKey(2000)

#webcam
if live == True:
    cap = cv.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret1, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (y,x), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            # Find the rotation and translation vectors.
            ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
            # project 3D points to image plane
            imgpts, jac2 = cv.projectPoints(cube, rvecs, tvecs, mtx, dist)
            imgpts2, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
            # draw axis and cube on the picture
            frame = draw_axis(frame,corners2,imgpts2)
            frame = draw_cube(frame,corners2,imgpts)

        cv.imshow('Input', frame)

        c = cv.waitKey(1)
        if c == 27:
            break

    cap.release()
cv.destroyAllWindows()
