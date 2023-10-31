import cv2 as cv
import numpy as np
import os
import serial
import time

def crop_image(image, x1, y1, x2, y2):
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

# Create a directory to save the images
part_name = 'test'
output_directory = 'captured_images/' + part_name
number_pics = 1

os.makedirs(output_directory, exist_ok=True)

cam_port = 0
cam = cv.VideoCapture(cam_port)

ser = serial.Serial(port="COM17", baudrate=9600)
data = 'a'
ser.write(data.encode())
time.sleep(3)
# Capture and save 100 images
for i in range(number_pics):
    data = 'a'
    ser.write(data.encode())
    time.sleep(3)

    result, image = cam.read()
    print(f'Saved Image {i}')
    if result:
        # Crop the image
        height, width, _ = image.shape
        size = 470

        x = (width - size) // 2
        y = (height - size) // 2

        #cv.rectangle(image, (x, y), (x + size, y + size), (0, 255, 0), thickness=2)

        cropped_image = crop_image(image, x,y,x + size, y + size)

        # Save the cropped image
        image_filename = os.path.join(output_directory, f'image_{i}.png')
        cv.imwrite(image_filename, cropped_image)

        
        #cv.imshow('Cropped Image', cropped_image)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
    
    else:
        print(f'Failed to capture Image {i}')
    
# Release the camera
cam.release()
cv.destroyAllWindows()
ser.close()
