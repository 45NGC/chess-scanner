import cv2
import numpy as np
import sys
import os

def chessboard_scanner(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image. Please check the path.")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges with Canny
    edges = cv2.Canny(blur, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and shape
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10000:  # Filter small contours
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:  # Look for quadrilaterals
            filtered_contours.append(contour)
    
    if not filtered_contours:
        raise ValueError("Chessboard not found in the image.")
    
    # Select the largest contour
    chessboard_contour = max(filtered_contours, key=cv2.contourArea)
    
    # Get bounding box coordinates
    x, y, w, h = cv2.boundingRect(chessboard_contour)
    
    # Crop the image
    cropped_chessboard = image[y:y+h, x:x+w]
    
    return cropped_chessboard

# Script usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        script_name = os.path.basename(sys.argv[0])
        print(f"Usage: python {script_name} <image_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    try:
        result = chessboard_scanner(input_path)
        cv2.imwrite("cropped_chessboard.png", result)
        print("Cropped chessboard saved as 'cropped_chessboard.png'")
    except Exception as e:
        print(f"Error: {e}")