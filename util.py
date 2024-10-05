import cv2
import pytesseract
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

def read_license_plate(license_plate_img: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
    """
    Reads the text from a given license plate image using Tesseract OCR.
    
    :param license_plate_img: Cropped image of the license plate.
    :return: Tuple of (detected text, confidence score).
    """
    try:
        # OCR processing
        license_plate_text = pytesseract.image_to_string(license_plate_img, config='--psm 8')
        confidence_score = pytesseract.image_to_data(license_plate_img, output_type=pytesseract.Output.DICT)['conf'][0]

        if confidence_score < 0:
            return None, None

        return license_plate_text.strip(), confidence_score
    except Exception as e:
        print(f"Error in read_license_plate: {e}")
        return None, None

def get_car(license_plate: List[float], track_ids: np.ndarray) -> Tuple[int, float, float, float, float, int]:
    """
    Get car information based on detected license plate.

    :param license_plate: Detected license plate bounding box, expected to be a list with at least 5 elements.
    :param track_ids: Numpy array of tracked vehicle IDs and their bounding boxes.
    :return: Tuple of (car bounding box coordinates, car ID).
    """
    if len(license_plate) < 5:
        raise ValueError(f"Expected at least 5 values in license_plate, got {len(license_plate)}: {license_plate}")

    # Unpack the first five values, ignore the rest if any
    x1, y1, x2, y2, *_ = license_plate
    car_id = -1
    car_bbox = (0.0, 0.0, 0.0, 0.0)  # Use float for consistency

    for tracked in track_ids:
        tracked_id = int(tracked[0])  # Get the vehicle ID
        tx1, ty1, tx2, ty2 = tracked[1:5]  # Get the bounding box coordinates

        # Check if the license plate bounding box overlaps with the tracked bounding box
        if tx1 < x1 < tx2 and ty1 < y1 < ty2:
            car_id = tracked_id
            car_bbox = (tx1, ty1, tx2, ty2)
            break

    return (*car_bbox, car_id)

def write_csv(results: Dict[int, Dict[int, Dict[str, Dict]]], output_file: str) -> None:
    """
    Write the detection results to a CSV file.
    
    :param results: Dictionary containing detection results.
    :param output_file: Path to the output CSV file.
    """
    try:
        rows = []
        for frame_id, cars in results.items():
            for car_id, data in cars.items():
                row = {
                    'frame_id': frame_id,
                    'car_id': car_id,
                    'car_bbox': data['car']['bbox'],
                    'license_plate_bbox': data['license_plate']['bbox'],
                    'license_plate_text': data['license_plate']['text'],
                    'license_plate_score': data['license_plate']['bbox_score']
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error in write_csv: {e}")