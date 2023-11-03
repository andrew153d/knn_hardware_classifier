import cv2
import numpy as np
import os
import serial
import os
import math
import time

class data_point:
    def __init__(self, name):
        self.distance = 9999999999999
        self.attributes = []
        self.name = name

    def load_attributes_from_file(self, path):
        self.attributes.append(int(calculate_part_area(path)))
        self.attributes.append(int(edge_detection(path)))
        self.attributes.append(int(count_edges_in_image(path)/100))

    def load_attributes(self, attributes):
        self.attributes = attributes

    def calculate(self, attributes):
        if(len(attributes) != len(self.attributes)):
            print("length error")
            return -1
        self.distance = 0
        for i in range(0, len(attributes)):
            self.distance += (int(attributes[i])-int(self.attributes[i])) ** 2
        self.distance = math.sqrt(self.distance)

    def __str__(self):
        return f"{self.name}  ->  {self.distance}"

def calculate_part_area(image_path):
    image1 = cv2.imread('captured_images/empty/image_0.png')
    #image2 = cv2.imread('captured_images/M3_screw/image_2.png')
    image2 = cv2.imread(image_path)
    if image1 is None or image2 is None:
        print("One or both images not found.")
    else:
        if image1.shape == image2.shape:
            subtracted_image = cv2.subtract(image1, image2)
            non_black_pixel_count = 0
            for i in range(subtracted_image.shape[0]):
                for j in range(subtracted_image.shape[1]):
                    b, g, r = subtracted_image[i, j]
                    if b > 15 and g > 15 and r > 15:
                        non_black_pixel_count += 1
                        subtracted_image[i, j] = (0, 100, 0)
            
            
            cv2.imwrite('subtracted_image.jpg', subtracted_image)
            return non_black_pixel_count
        else:
            print("Images have different dimensions and cannot be subtracted.")

def calculate_aspect_ratio(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    num_rows, num_columns, _ = image.shape

    # Loop over the columns
    for column_index in range(num_columns):
        for row_index in range(num_rows):
            if (gray[row_index, column_index] > 2):
                cv2.line(gray, (0,column_index), (num_columns, column_index), (255), 2)
                return 0
            #print(f"Pixel at row {row_index}, column {column_index}: {pixel_value}")
    cv2.imwrite('aspect_ratio.png', gray)
    
def edge_detection(image_path):
    try:
        output_path = 'edge_detected.png'
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        median = np.median(image)
        low_threshold = int(max(0, (0) * median))
        high_threshold = int(min(255, (2) * median))
        edges = cv2.Canny(image, low_threshold, high_threshold)
        non_black_pixel_count = 0
        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                b = edges[i, j]
                if b > 15 :
                    non_black_pixel_count += 1
        cv2.imwrite(output_path, edges)
        return non_black_pixel_count
        
    except Exception as e:
        print(f"Error: {e}")

def hough_line_transform(image_path, output_path = 'hough_lines.png', rho=1, theta=np.pi/180, threshold=30):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Perform the Hough Line Transform
    lines = cv2.HoughLines(edges, rho, theta, threshold)
    print(lines)
    if lines is not None:
        print("drawing lines")
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            # Draw the detected lines on the image
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.imwrite(output_path, image)

def count_edges_in_image(image_path, lower_threshold=100, upper_threshold=200):
    # Load the image
    image = cv2.imread(image_path, 0)  # Load the image in grayscale (0)

    # Apply Canny edge detection
    edges = cv2.Canny(image, lower_threshold, upper_threshold)

    # Count the number of edges
    edge_count = edges.sum()
    cv2.imwrite('edge_count.png', image)
    return edge_count

def get_part_descriptors(image_path):
    return_string = str(calculate_part_area(image_path))
    return_string += " "
    return_string += str(edge_detection(image_path))
    return_string += " "
    return_string += str(int(count_edges_in_image(image_path)/100))
    return return_string

def train():
    directory = 'captured_images'
    part_names = []
    for filename in os.scandir(directory):
        if filename.is_dir():
            print(filename.path)
            p = filename.path.split('\\')
            part_names.append(p[len(p)-1])

    with open("output.txt", "w") as file:
        for part_class in part_names:
            if part_class=="test":
                continue
            for image in os.scandir('captured_images/'+part_class):
                if image.is_file():
                    row_string = part_class + " " + get_part_descriptors(str(image.path))
                    file.write(row_string + '\n')
                    print(row_string)
        file.close()

def crop_image(image, x1, y1, x2, y2):
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

def knn(path, n):
    
    d = data_point("test")
    d.load_attributes_from_file(path)
    data = [d]
    line_count = 0
    with open("output.txt", "r") as file:
        for line in file:
            line_count+=1
            line = line.replace("\n","")
            p = data_point(line.split(" ")[0])
            p.load_attributes(line.split(" ")[1:len(line)])
            p.calculate(d.attributes)
            data.append(p)
    if(n==0):
        n=int(math.sqrt(line_count))
    sorted_objects = sorted(data, key=lambda x: x.distance)[0:n]
    sorted_objects = sorted(sorted_objects, key=lambda x: x.name)
    name_dict = {}  # Use a dictionary instead of a list
    for p in range(0, n):
        #print(sorted_objects[p])
        if sorted_objects[p].name in name_dict:
            name_dict[sorted_objects[p].name] += 1
        else:
            name_dict[sorted_objects[p].name] = 1
    sorted_name_dict = dict(sorted(name_dict.items(), key=lambda item: item[1], reverse=True))
    print(sorted_name_dict)



def detect():
    output_directory = 'captured_images/temp'
    os.makedirs(output_directory, exist_ok=True)
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)
    ser = serial.Serial(port="COM17", baudrate=9600)
    data = 'a'
    ser.write(data.encode())
    time.sleep(5)
    print("pucture")
    result, image = cam.read()
    if result:
        data = 'a'
        ser.write(data.encode())
        time.sleep(3)
        # Crop the image
        height, width, _ = image.shape
        size = 470

        x = (width - size) // 2
        y = (height - size) // 2

        #cv.rectangle(image, (x, y), (x + size, y + size), (0, 255, 0), thickness=2)

        cropped_image = crop_image(image, x,y,x + size, y + size)

        # Save the cropped image
        image_filename = os.path.join(output_directory, f'temp.png')
        cv2.imwrite(image_filename, cropped_image)
    cam.release()
    ser.close()
    knn(output_directory+'/temp.png', 0)

detect()
#train()

#knn('captured_images/test/M2x8mm.png', 0)
#edge_detection('captured_images/test/image_1.png')



