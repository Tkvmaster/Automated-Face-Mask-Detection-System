import cv2

cam = cv2.VideoCapture(0)
cv2.namedWindow("Live Cam")

capture_count = 0

while True:
    ret, frame = cam.read()
    if ret:
        cv2.imshow("test", frame)
    
        k = cv2.waitKey(1)
        # Press SPACE for capturing image and ESC for closing video
        if k%256 == 27:
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            windowName = "Captured Image"

            cv2.namedWindow(windowName)

            cv2.destroyWindow('Captured Image')
            img_name = "opencv_frame_{}.png".format(capture_count)
            img_show = frame
            cv2.imwrite(img_name, frame)
            cv2.imshow(windowName, img_show)
            print("{} written!".format(img_name))
            capture_count += 1
    else:
        print("failed to grab frame")
        break

cam.release()

cv2.destroyAllWindows()
