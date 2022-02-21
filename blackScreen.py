import cv2
import numpy as np
import time

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi',fourcc,20.0,(640,650))


cap = cv2.VideoCapture(0)

time.sleep(1)
bg = 0

for i in range(60):
    ret,bg = cap.read()
    bg = np.flip(bg,axis=1)

    while (cap.isOpened()):
        ret,img = cap.read()

        if not ret:

            break
            img = np.flip(ret,axis=1)


            hsv = cv2.cvtColor(img,CV2.COLOR_BGR2HSV)

            lower_red = np.array([121,34,22])
            upper_red = np.array([21,22,90])
            mask_1 = cv2.inRange(hsv,lower_red,upper_red)

            lower_red = np.array([111,134,92])
            upper_red = np.array([31,92,30])
            mask_2 = cv2.inRange(hsv,lower_red,upper_red)
        
            mask_1 = mask_1 + mask_2


            mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

res_1 = cv2.bitwise_and(img,img,mas = mas_2)


res_2 = cv2.bitwise_and(img,img,mas = mas_1)

final_output = cv2.addWeighted(res_1, 1, res_2, 1, 0)
output_file.write(final_output)
    #Displaying the output to the user
cv2.imshow("magic", final_output) 
cv2.waitKey(1)

cap.release()
out.release()

cv2.destroyAllWindows()

