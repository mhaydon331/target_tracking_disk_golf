import math
import sys
import cv2
import cv2.aruco
import numpy as np
def main():
    currentFrame = 0

    video_capture = cv2.VideoCapture("g74_Slomo.mp4")  # Open video capture object
    got_image, img = video_capture.read()
    if not got_image:
        print("Cannot read video source")
        sys.exit()
    height = img.shape[0]
    width = img.shape[1]
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    outputVideo = cv2.VideoWriter('output_slo1.mp4', fourcc=fourcc, fps=45,
                                  frameSize=(width, height))

    distances = []
    found = 0
    distance_calculated = 0
    while True:
        currentFrame += 1
        got_image, img = video_capture.read()
        if not got_image:
            break  # End of video; exit the while loop
        #print(img.shape)
        K = np.array([
            (600, 0, 1920),
            (0, 600, 1080),
            (0, 0, 1)
        ]).astype(float)

        # Marker length
        MARKER_LENGTH = 2

        # distortion coefficient
        dist_coeff = None

        # Convert color image to gray image.
        # Get the pattern dictionary for 4x4 markers, with ids 0 through 99.
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

        # Optionally show all markers in the dictionary.
        # for id in range(0, 100):
        #     img = cv2.aruco.drawMarker(dictionary=arucoDict, id=id, sidePixels=200)

        # Detect a marker.  Returns:
        #   corners:   list of detected marker corners; for each marker, corners are clockwise)
        #   ids:   vector of ids for the detected markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            image=img,
            dictionary=arucoDict
        )

        if ids is not None:
            #print(ids)
            cv2.aruco.drawDetectedMarkers(
                image=img, corners=corners, ids=ids, borderColor=(0, 255, 255))

        # function to compute pose from detected corners
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners=corners, markerLength=MARKER_LENGTH,
            cameraMatrix=K, distCoeffs=0
        )
        if rvecs is not None and tvecs is not None:
            # Get the pose of the first detected marker with respect to the camera.
            '''
            rvec_m_c = rvecs[0]  # This is a 1x3 rotation vector
            tm_c = tvecs[0]  # This is a 1x3 translation vector

            # funtcion to draw coordinate axes onto the image, using the detected pose
            cv2.aruco.drawAxis(
                image=img, cameraMatrix=K, distCoeffs=dist_coeff,
                rvec=rvec_m_c, tvec=tm_c, length=MARKER_LENGTH)
            '''

            count6 = 0
            if ids is not None:
                for id in range(len(ids)):
                    rvec_m_c = rvecs[id]
                    tm_c = tvecs[id]
                    cv2.aruco.drawAxis(
                        image=img, cameraMatrix=K, distCoeffs=dist_coeff,
                        rvec=rvec_m_c, tvec=tm_c, length=MARKER_LENGTH)
                    if ids[id][0] == 6 and count6 == 0:
                        found += 1
                        x_sum = corners[id][0][0][0] + corners[id][0][1][0] + corners[id][0][2][0] + corners[id][0][3][0]
                        y_sum = corners[id][0][0][1] + corners[id][0][1][1] + corners[id][0][2][1] + corners[id][0][3][1]

                        x_centerPixel = x_sum * .25
                        y_centerPixel = y_sum * .25
                        #print(x_centerPixel, y_centerPixel)
                        center_coordinates = (int(round(x_centerPixel)), int(round(y_centerPixel)))
                        img = cv2.circle(img, center_coordinates, 20, (0, 0, 255), 40)
                        img = cv2.circle(img, center_coordinates, 40, (255, 255, 255), 20)
                        img = cv2.circle(img, center_coordinates, 60, (0, 0, 255), 20)
                        img = cv2.circle(img, center_coordinates, 80, (255, 255, 255), 20)
                        img = cv2.circle(img, center_coordinates, 100, (0, 0, 255), 20)
                        aruco_perimeter = cv2.arcLength(corners[id], True)
                        width = aruco_perimeter/4
                        distance = (215.9*600)/width
                        R, J = cv2.Rodrigues(rvec_m_c)
                        z = math.atan2(R[1, 0], R[0, 0])
                        if z<0:
                            distance_calculated += 1
                            z = -z
                            opp = math.sin(z) * distance + 711.2
                            adj = math.cos(z) * distance
                            cv2.putText(img,"Distance to target: {}m".format(math.sqrt(opp**2+adj**2)/1000),(40,80)
                                        ,cv2.FONT_HERSHEY_SIMPLEX,2, (0, 255, 255), 3)
                            distances.append(math.sqrt(opp**2+adj**2)/1000)
                        count6 += 1

        img_output = img.copy()

        outputVideo.write(img_output)
        cv2.namedWindow('playMoreDiscGolf', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('playMoreDiscGolf', img)
        cv2.resizeWindow('playMoreDiscGolf', 1500, 800)
        cv2.waitKey(1)
    if len(distances)>0:
        print("Average distance from basket once top of basket in sight: {}m".format(sum(distances)/len(distances)))
    if found > 0:
        print("Target 6 found vs distance calculation preformed: {} vs {} for a ratio of {}".format(found,distance_calculated,distance_calculated/found))
    else:
        print("Target 6 found vs distance calculation preformed: {} vs {} for a ratio of {}".format(found,distance_calculated,0))
    outputVideo.release()
    print(currentFrame)


if __name__ == '__main__':
    main()