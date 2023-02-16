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

    c2_c1 = (main_corners[1][0] - main_corners[0][0], main_corners[1][1] - main_corners[0][1])
    c3_c1 = (main_corners[2][0] - main_corners[0][0], main_corners[2][1] - main_corners[0][1])
    c4_c3 = (main_corners[3][0] - main_corners[2][0], main_corners[3][1] - main_corners[2][1])

    # c1_c2 = (main_corners[0][0] - main_corners[1][0], main_corners[0][1] - main_corners[1][1])
    # c2_c3 = (main_corners[1][0] - main_corners[2][0], main_corners[1][1] - main_corners[2][1])
    # c3_c4 = (main_corners[2][0] - main_corners[3][0], main_corners[2][1] - main_corners[3][1])

    print("1 " + str(c2_c1))
    print("2 " + str(c3_c1))
    # print("3 " + str(c3_c4))

    print("corner 4 " + str(main_corners[3]))

    step_x1 = c2_c1[0]/(x-1)
    step_y1 = c2_c1[1]/(x-1)
    step_x2 = c4_c3[0]/(x-1)
    step_y2 = c4_c3[1]/(x-1)
    #step_x2 = c3_c1[0]/(y-1)
    #step_x3 = c4_c3[0]/((x-1))
    
    #step_y = c3_c1[1]/(y-1)
    #step_y2 = c2_c1[1]/(x-1)
    #step_y3 = c4_c3[1]/((y-1))
    interp1 = []
    interp2 = []
    for dx in range(x):
        interp_1x = main_corners[0][0] + dx * step_x1
        interp_1y = main_corners[0][1] + step_y1 #+ dx * step_y2
        interp_2x = main_corners[2][0] + dx * step_x2 
        interp_2y = main_corners[2][1] + step_y2 + dx 
        interp1.append([interp_1x,interp_1y])
        interp2.append([interp_2x,interp_2y])
    #corners_grid.append(interp1)
    #corners_grid.append(interp2)
    print("inter 1 " + str(interp1))
    print("inter 2 " + str(interp2))
    for dx in range(x):
        for dy in range(y):
            step_y3 = (interp2[dx][1] - interp1[dx][1])/(y-1)
            step_x3 = (interp2[dx][0] - interp1[dx][0])/(y-1)
            result_x = interp1[dx][0] + dy * step_x3
            result_y = interp1[dx][1] + dy * step_y3
            result = [result_x,result_y]
            corners_grid.append(result)
    print(corners_grid)

    #for dx in range(x):
    #    for dy in range(y):
    #        result_x = main_corners[0][0] + dx * step_x + dy * step_x2 #+ (dx) * step_x3
    #        result_y = main_corners[0][1] + dy * step_y + dx * step_y2 + (dy) * step_y3
    #        result = [result_x, result_y]
            #corners_grid.append(result)
    
    return corners_grid

#test
criteria = (cv.TERM_CRITERIA_EPS, 54, 0.001)
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
    ret = True
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (y,x), cv.CALIB_CB_FAST_CHECK)
    print(str(counter) + " " + str(ret))
    # If found, add object points, image points (after refining them)
    cv.namedWindow("resize", cv.WINDOW_NORMAL)
    cv.resizeWindow("resize", 500, 500)

    if ret == False:

        cv.imshow('resize', img)
        cv.setMouseCallback('resize', click_event)
        while len(main_corners) < 4:
            cv.waitKey(10)
        corners = make_grid()
        main_corners.clear()
        corners = np.float32(corners)
        print(corners)

    # draw red points
    # for point in corners:
    #     point_x = int(point[0])
    #     point_y = int(point[1])
    #     img[point_x,point_y]=[0,0,255]
    #     print(str(point_x)+" "+ str(point_y))
    # cv.imshow('resize', img)
    # cv.waitKey(0)

    #print("found " + str(counter-1))
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners)
    # # Draw and display the corners

    cv.drawChessboardCorners(img, (y,x), corners2, ret)
    cv.imshow('resize', img)

    cv.waitKey(500)

#calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
img = cv.imread('test1_first.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

cv.destroyAllWindows()


#    cv.namedWindow("ja", cv.WINDOW_NORMAL)
#    cv.resizeWindow("ja", 500, 500)
#    cv.imshow('ja', img)
#    cv.waitKey(500)