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



def analyse_board(chessboard_image, target_size=50, margin_ratio=0.1, templates_dir="square_templates"):

	# Load templates
	templates = {}
	for filename in os.listdir(templates_dir):
		if filename.endswith(".png") or filename.endswith(".jpg"):
			name = os.path.splitext(filename)[0]
			template = cv2.imread(os.path.join(templates_dir, filename), cv2.IMREAD_GRAYSCALE)
			template = cv2.resize(template, (target_size, target_size))
			_, template_bw = cv2.threshold(template, 128, 255, cv2.THRESH_BINARY)
			templates[name] = template_bw

	board_array = np.zeros((8, 8), dtype=int)

	height, width = chessboard_image.shape[:2]
	square_height = height // 8
	square_width = width // 8
	margin_h = int(square_height * margin_ratio)
	margin_w = int(square_width * 0.2)

	for row in range(8):
		for col in range(8):
			# Square coordinates
			y1 = row * square_height + margin_h
			y2 = (row + 1) * square_height - margin_h
			x1 = col * square_width + margin_w
			x2 = (col + 1) * square_width - margin_w

			square = chessboard_image[y1:y2, x1:x2]

			# Resize square
			square_resized = cv2.resize(square, (target_size, target_size))

			# Convert to grayscale
			gray_square = cv2.cvtColor(square_resized, cv2.COLOR_BGR2GRAY)

			# Convert square to binary (black and white)
			_, square_bw = cv2.threshold(gray_square, 128, 255, cv2.THRESH_BINARY)

			# Compare with templates
			best_match = None
			best_score = float("inf")
			for name, template in templates.items():
				diff = cv2.absdiff(square_bw, template)
				score = np.sum(diff)
				if score < best_score:
					best_score = score
					best_match = name

			
			mapping = {
				"empty_w": 0,
				"empty_b": 0,
				"pawn_ww": 1,
				"pawn_wb": 1,
				"pawn_bb": -1,
				"pawn_bw": -1,
				"rook_ww": 4,
				"rook_wb": 4,
				"rook_bb": -4,
				"rook_bw": -4,
				"queen_ww": 5,
				"queen_wb": 5,
				"queen_bb": -5,
				"queen_bw": -5,
				"bishop_ww": 3,
				"bishop_wb": 3,
				"bishop_bb": -3,
				"bishop_bw": -3,
				"knight_ww": 2,
				"knight_wb": 2,
				"knight_bb": -2,
				"knight_bw": -2,
				"king_ww": 6,
				"king_wb": 6,
				"king_bb": -6,
				"king_bw": -6,
			}

			board_array[row, col] = mapping.get(best_match, 0)

	print(board_array)
	return board_array


def array_to_fen(board: np.ndarray) -> str:
    piece_map = {
        1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
        -1: "p", -2: "n", -3: "b", -4: "r", -5: "q", -6: "k",
        0: None
    }
    
    fen_rows = []
    for row in board:
        fen_row = ""
        empty_count = 0
        for cell in row:
            piece = piece_map[cell]
            if piece is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += piece
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    
    #TODO: add options to control castling (" w KQkq - 0 1")
    fen = "/".join(fen_rows)
    return fen

		
	        

if __name__ == "__main__":
    if len(sys.argv) != 2:
        script_name = os.path.basename(sys.argv[0])
        print(f"Usage: python {script_name} <image_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    try:
        # Step 1: Crop the chessboard
        cropped_chessboard = chessboard_cropper(input_path)
        cv2.imwrite("cropped_chessboard.png", cropped_chessboard)
        print("Cropped chessboard saved as 'cropped_chessboard.png'")
        
        # Step 2: Analyse chessboard 
        piece_positions = analyse_board(cropped_chessboard)

		# Step 3: Convert array to FEN string
        fen = array_to_fen(piece_positions)
        print(fen)
            
    except Exception as e:
        print(f"Error: {e}")