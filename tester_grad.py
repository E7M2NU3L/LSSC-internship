# tester_grad

import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened:
    print("failed to opened the webcam")
    exit()

count = 0
while count < 3:
    sucess , img = cap.read()

    if not sucess:
        print("failed to open the webcam")
        break

    cv2.imshow("frame",img)

    count += 1

    image_path = f"/home/emman/Desktop/leather projects/gradient_analyser/image_folder/image_{count}.jpg"
    cv2.imwrite(image_path,img)

    print(f"image {count} captued and saved in {image_path} as image_{count}.jpg")
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



