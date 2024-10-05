from ultralytics import YOLO
import cv2
import numpy as np
import util
from sort.sort import Sort

# Initialize results dictionary and SORT tracker
results = {}
mot_tracker = Sort()

# Load models
coco_model = YOLO('/Users/daksh/Desktop/ANPR/yolo11s.pt')  # Model for vehicle detection
license_plate_detector = YOLO("/Users/daksh/Desktop/ANPR/runs/detect/train/weights/best.pt")  # Model for license plate detection

# Load video
video_path = "/Users/daksh/Desktop/ANPR/assets/vid1.mp4"
cap = cv2.VideoCapture(video_path)

# Vehicle classes to detect (COCO dataset IDs)
vehicles = [2, 3, 5, 7]  # Car, motorcycle, bus, truck

# Read frames and process
frame_nmr = 5
ret = True

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        
        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Print detected vehicles
        print(f"Frame {frame_nmr}: Detected vehicles: {detections_}")

        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = util.get_car(license_plate, track_ids)

            # Debugging: Print license plate detection
            print(f"License Plate Detection: ({x1}, {y1}), ({x2}, {y2}), Car ID: {car_id}")

            if car_id != -1:
                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                # Process license plate image
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number
                license_plate_text, license_plate_text_score = util.read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }

                # Draw bounding boxes on the original frame
                cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)  # Car bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # License plate bounding box

                # Put the license plate text on the frame
                cv2.putText(frame, license_plate_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Print results for the current frame
        print(f"Frame {frame_nmr} results: {results[frame_nmr]}")

        # Display the frame with bounding boxes
        cv2.imshow('Processed Video', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Before writing results to CSV, print the collected results
print("Final results before writing to CSV:", results)

# Write results to CSV
util.write_csv(results, './test.csv')

# Release resources
cap.release()
cv2.destroyAllWindows()