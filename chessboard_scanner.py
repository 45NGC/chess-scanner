import cv2
import numpy as np
import sys
import os


def chessboard_cropper(image_path):
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
    
    # Resize cropped image to 400x400
    resized_chessboard = cv2.resize(cropped_chessboard, (400, 400), interpolation=cv2.INTER_AREA)
    
    return resized_chessboard






def save_chessboard_squares(chessboard_image, output_dir="squares", target_size=50, margin_ratio=0.1):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the dimensions of the chessboard
    h, w = chessboard_image.shape[:2]
    
    # Calculate the size of each square based on the chessboard dimensions
    square_height = h // 8
    square_width = w // 8
    
    # Calculate margin to remove (percentage of square size)
    margin_h = int(square_height * margin_ratio)
    margin_w = int(square_width * 0.2)  # Use the same ratio for both dimensions
    
    # Iterate over each row and column
    for row in range(8):
        for col in range(8):
            # Calculate the coordinates of the square with margin
            y1 = row * square_height + margin_h
            y2 = (row + 1) * square_height - margin_h
            x1 = col * square_width + margin_w
            x2 = (col + 1) * square_width - margin_w
            
            # Ensure coordinates are within bounds
            y1 = max(0, y1)
            y2 = min(h, y2)
            x1 = max(0, x1)
            x2 = min(w, x2)
            
            # Extract the square (with margins removed)
            square = chessboard_image[y1:y2, x1:x2]
            
            # Resize the square to the target size
            square_resized = cv2.resize(square, (target_size, target_size))
            
            # Save the square image
            square_filename = os.path.join(output_dir, f"square_{row}_{col}.png")
            cv2.imwrite(square_filename, square_resized)
            

if __name__ == "__main__":
    if len(sys.argv) != 2:
        script_name = os.path.basename(sys.argv[0])
        print(f"Usage: python {script_name} <image_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    try:
        # Step 1: Crop the chessboard
        chessboard = chessboard_cropper(input_path)
        cv2.imwrite("cropped_chessboard.png", chessboard)
        print("Cropped chessboard saved as 'cropped_chessboard.png'")
        
        # Step 2: Save all squares
        save_chessboard_squares(chessboard, "squares")
        print("All squares saved in the 'squares' directory.")
    except Exception as e:
        print(f"Error: {e}")