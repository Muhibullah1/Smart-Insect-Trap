import os
import re
import csv
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point


def denormalize_yolo_coordinates(box, img_shape):
    box_id, x, y, w, h = map(float, box.split())
    x , y, w, h = x*img_shape[1], y*img_shape[0], w*img_shape[1], h*img_shape[0] 
    w , h = (x+w/2), int(y+h/2)
    x, y = int(x-(w-x)), int(y-(h-y))
    return x, y, w, h

def bbox_center(box):
    c_x = int((box[2] - box[0])/2)
    c_y = int((box[3] - box[1])/2) 
    return (c_x, c_y)

def calculate_box_dimensions(box):
    x, y, w, h = box
    width = (w - x)  # Calculate width
    length = (h - y)  # Calculate length
    return width, length

def filter_Grids(boxes, threshold):
    # Calculate widths and lengths for all boxes
    widths = [calculate_box_dimensions(box)[0] for box in boxes]
    lengths = [calculate_box_dimensions(box)[1] for box in boxes]
    #print('widths : ', widths)
    #print('lengths: ', lengths)
    
    # Calculate median width and length
    median_width = np.median(widths)
    median_length = np.median(lengths)
    
    print(f"Median width: {median_width}, Median length: {median_length}")
    
    # Filter boxes based on width and length thresholds
    filtered_boxes = [box for box in boxes
        if (calculate_box_dimensions(box)[0] >= (median_width - threshold * median_width) and
            calculate_box_dimensions(box)[1] >= (median_length - threshold * median_length))]
    
    # Identify filtered-out boxes
    filtered_out_boxes = [box for box in boxes
        if (calculate_box_dimensions(box)[0] < (median_width - threshold * median_width) or
            calculate_box_dimensions(box)[1] < (median_length - threshold * median_length))]
    
    
    for box in filtered_out_boxes:
        print(f"Filtered out Grids (less than {threshold*100}% of median width or length):")
        width, length = calculate_box_dimensions(box)
        #print(f"Coordinates: {box}, Width: {width}, Length: {length}")
    
    return filtered_boxes

def count_FBs_within_Grids(Grid_box, FB_boxes, img, Already_counted):
    x1, y1, w1, h1 = Grid_box
    count = 0
    for FB_box in FB_boxes:
        x2, y2, w2, h2 = FB_box
        if (x2, y2) in Already_counted:
            continue

        if x1 <= x2 <= w1 and y1 <= y2 <= h1:
            count += 1
            cv2.rectangle(img, (x2, y2), (int((w2)), int((h2))), (250, 0, 0), 2)
            Already_counted.append((x2,y2))
    
    return count, (x1, y1), img, Already_counted

def process_FB_grids_filter(images_dir, Grids_dir, FB_dir, output_dir, csv_output_path, droping_percentage):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(csv_output_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Image', 'Grids', 'Grid_has_FBs', 'FBs', 'FBs in Grids', 'Ratio (FB/Grids)', 'Ratio (FB/Grids_W)'])

        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.JPG')])

        for image_file in tqdm(image_files):
            print(image_file)
            
            image_path = os.path.join(images_dir, image_file)
            output_path = os.path.join(output_dir, image_file)

            try:
                img = cv2.imread(image_path)
                img_shape = img.shape
                image_width, image_height = img_shape[0], img_shape[1]

                Grids_file = os.path.join(Grids_dir, f"{os.path.splitext(image_file)[0]}.txt")
                FB_file = os.path.join(FB_dir, f"{os.path.splitext(image_file)[0]}.txt")
            
                if os.path.exists(Grids_file): # Read Grids data
                    with open(Grids_file, 'r') as f:
                        Grids_data = f.readlines()
                    Grids_boxes = [denormalize_yolo_coordinates(box, img_shape) for box in Grids_data]
                    
                    # Filter out Grids boxes smaller than 20% of the median size
                    Grids_boxes = filter_Grids(Grids_boxes, threshold=droping_percentage)
                else:
                    Grids_boxes = []
                    
                FB_boxes = []  # Read FBs data
                if os.path.exists(FB_file):
                    with open(FB_file, 'r') as f:
                        FB_data = f.readlines()
                    FB_boxes = [denormalize_yolo_coordinates(box, img_shape) for box in FB_data]

                # Process each Grid in the image
                Already_counted, total_FB_in_Grids, Grid_has_FBs = [], 0, 0
                for i, box in enumerate(Grids_boxes):
                    x, y, w, h = box
                    cv2.rectangle(img, (x, y), (int((w)), int((h))), (0, 0, 250), 5)
                    count, (x, y), img, Already_counted = count_FBs_within_Grids(box, FB_boxes, img, Already_counted)
                    if count != 0:
                        Grid_has_FBs += 1
                    total_FB_in_Grids += count
                    text = f'Count: {count}'
                    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                text_total_FB = f'Total FB count => {len(FB_boxes)}'
                cv2.putText(img, text_total_FB, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 250, 50), 2, cv2.LINE_AA)    
                text_total =    f'FBs in grids=> {total_FB_in_Grids}'
                cv2.putText(img, text_total, (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 70), 2, cv2.LINE_AA)                
                                
                num_Grids_boxes = len(Grids_boxes)
                num_FB_boxes = len(FB_boxes)
                ratio_FB_Grids = round(total_FB_in_Grids / num_Grids_boxes, 2) if num_Grids_boxes > 0 else 0
                ratio_Grid_has_FB = round(total_FB_in_Grids / Grid_has_FBs, 2) if Grid_has_FBs > 0 else 0
                
                text_grid_ratio = f'Per grid count=> {ratio_FB_Grids}'
                cv2.putText(img, text_grid_ratio, (15, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 250, 50), 2, cv2.LINE_AA)

                # Write the data to the CSV file
                csv_writer.writerow([image_file, num_Grids_boxes, Grid_has_FBs, num_FB_boxes, total_FB_in_Grids, ratio_FB_Grids, ratio_Grid_has_FB])                               
                cv2.imwrite(output_path, img)

            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                # Write zeros for this image in the CSV
                csv_writer.writerow([image_file, 0, 0, 0, 0, 0, 0])
                continue



def process_images_batch(model_path, confidence, image_folder, name_path, output_csv):
    model = YOLO(model_path)
    results = model(image_folder, save=True, conf=confidence, save_txt=True, show_labels=False, show_conf=False, max_det=5000, name = name_path)
    print('results', results)
    data = []
    
    for result in results:
        # Extract image name and timestamp
        image_path = result.path
        image_name = os.path.basename(image_path)
        match = re.search(r'([\d]{8}-[\d]{6})', image_name)  # Extract timestamp
        time_stamp = match.group(1) if match else "Unknown"
        
        # Count detected objects
        detections = result.boxes.cls  # Get class IDs
        count = len(detections)
        #print('count', count)
        # Extract unique object classes
        #object_classes = list(set(detections.cpu().numpy()))  # Convert to list of unique class IDs
        #object_classes = ', '.join([model.names[int(cls)] for cls in object_classes])  # Map class IDs to names
        
        data.append([image_name, time_stamp, count])#, object_classes])
    
    # Save results to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Time", "Count"])#, "Objects"])
        writer.writerows(data)
    
    print(f"Results saved to {output_csv}")
