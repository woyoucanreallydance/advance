import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cap = cv2.VideoCapture(r'C:\Users\parya\Desktop\finding lanes\test2.mp4')
ret, image = cap.read()

while ret:
    ret, image = cap.read()
    frame = cv2.resize(image, (640, 480))

    # choosing frame points
    topleft = (235, 257)
    bottomleft = (139, 461)
    topright = (370, 257)
    bottomright = (512, 461)

    cv2.circle(frame, topleft, 5, (0,0,255), -1)
    cv2.circle(frame, bottomleft, 5, (0,0,255), -1)
    cv2.circle(frame, topright, 5, (0,0,255), -1)
    cv2.circle(frame, bottomright, 5, (0,0,255), -1)

    # perspective transformation 
    pts1 = np.float32([topleft, bottomleft, topright, bottomright])
    pts2 = np.float32([[[0,0], [0,480], [640,0], [640,480]]])

    # matrix e lazem baraye taghir 
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640,480))

    #object detection & image thresholding
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
    l_h = 0
    l_s = 0
    l_v = 102
    u_h = 243
    u_s = 149
    u_v = 249

    lower = np.array([l_h,l_s,l_v])
    upper = np.array([u_h,u_s,u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    # histogram 
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis = 0)
    midpoint = int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # sliding window 
    y = 472 
    left_line_x = []
    right_line_x = []

    msk = mask.copy()

    while y > 0 :
        # left threshold 
        img = mask[y-40:y, left_base-50:left_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                left_line_x.append(left_base-50 + cx) 
                left_base = left_base-50 + cx
        # right threshold 
        img = mask[y-40:y, right_base-50:right_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                right_line_x.append(right_base-50 + cx) 
                right_base = right_base-50 + cx        
        cv2.rectangle(msk, (left_base-50, y), (left_base+50, y-40), (255,255,255), 2)
        cv2.rectangle(msk, (right_base-50, y), (right_base+50, y-40), (255,255,255), 2)
        y -= 40


    

    cv2.imshow('original', frame)
    cv2.imshow('bird eye view', transformed_frame)
    cv2.imshow('lane detection & thresholding', mask)
    cv2.imshow('lane detection & sliding windows', msk)



    if cv2.waitKey(10) == ord('q'):
        break

cap.release()    
cv2.destroyAllWindows()



#img = mpimg.imread(r'C:\Users\parya\Desktop\finding lanes\vid image.png')
#plt.imshow(img)
#plt.show()