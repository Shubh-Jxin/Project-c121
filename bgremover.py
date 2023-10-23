import cv2
import time
import numpy as np

# Save the output in a file "output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Start the webcam
cap = cv2.VideoCapture(0)

# Allow the webcam to start by making the code sleep for 2 seconds
time.sleep(2)
bg = cv2.imread(r"place.webp", cv2.IMREAD_UNCHANGED)

# Read the captured frame until the camera is open
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    
    # Flip the image for consistency
    img = np.flip(img, axis=1)
    
    # Convert the color from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Generate a mask to detect RED colour
    # These values can also be changed as per the color
    lower_black = np.array([0,0,0]) # BGR Format
    upper_black = np.array([111,111,111]) # BGR Format

    # Generate mask
    mask_1 = cv2.inRange(hsv, lower_black, upper_black)

    # Open and expand the image where there is mask 1 (color)
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))


    # Select only the part that does not have mask one and saving in mask 2
    mask_2 = cv2.bitwise_not(mask_1)

    res_1 = cv2.bitwise_and(img, img, mask=mask_2)

    # Keep only the part of the images with the red color 
    res_2 = cv2.bitwise_and(bg, bg, mask=mask_1)

    # Generate the final output by merging res_1 and res_2 
    final_output = cv2.addWeighted(res_1,1,res_2,1,0)
    output_file.write(final_output)
    
    # Display the output to the user
    cv2.imshow("Avner's Morphy Machine", final_output)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows() 
