import cv2
import numpy as np

def euclidean_distance(point1, point2):
    distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return distance
def draw_red_circle(image):
    # Draw a big red circle on the image
    height, width, _ = image.shape
    center_coordinates = (width // 2, height // 2)
    radius = 100
    color = (0, 0, 255)  # Red color in BGR format
    thickness = -1  # Negative thickness fills the circle
    cv2.circle(image, center_coordinates, radius, color, thickness)

def order_corner_points(corners):
    sort_corners = [(corner[0][0], corner[0][1]) for corner in corners]
    sort_corners = [list(ele) for ele in sort_corners]
    x, y = [], []
    for i in range(len(sort_corners[:])):
        x.append(sort_corners[i][0])
        y.append(sort_corners[i][1])
    centroid = [sum(x) / len(x), sum(y) / len(y)]

    top_left = None
    top_right = None
    bottom_right = None
    bottom_left = None

    for _, item in enumerate(sort_corners):
        if item[0] < centroid[0]:
            if item[1] < centroid[1]:
                top_left = item
            else:
                bottom_left = item
        elif item[0] > centroid[0]:
            if item[1] < centroid[1]:
                top_right = item
            else:
                bottom_right = item

    ordered_corners = [top_left, top_right, bottom_right, bottom_left]
    return np.array(ordered_corners, dtype="float32")

def find_red_position(center, dimensions):
    x, y = center
    w, h = dimensions

    if x < w / 3:
        col_position = "left"
    elif w / 3 <= x < 2 * w / 3:
        col_position = "middle"
    else:
        col_position = "right"

    if y < h / 3:
        row_position = "top"
    elif h / 3 <= y < 2 * h / 3:
        row_position = "middle"
    else:
        row_position = "bottom"

    position_dict = {
        "top left": 1,
        "top middle": 2,
        "top right": 3,
        "middle left": 4,
        "middle middle": 5,
        "middle right": 6,
        "bottom left": 7,
        "bottom middle": 8,
        "bottom right": 9,
        # Add more positions as needed
    }

    position_key = f"{row_position} {col_position}"
    return position_dict.get(position_key, 0)  # Return 0 for undefined positions


def image_preprocessing(image, corners):
    ordered_corners = order_corner_points(corners)
    top_left, top_right, bottom_right, bottom_left = ordered_corners
    width1 = euclidean_distance(bottom_right, bottom_left)
    width2 = euclidean_distance(top_right, top_left)
    height1 = euclidean_distance(top_right, bottom_right)
    height2 = euclidean_distance(top_left, bottom_right)
    width = max(int(width1), int(width2))
    height = max(int(height1), int(height2))
    dimensions = np.array([[0, 0], [width, 0], [width, width],
                           [0, width]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)
    transformed_image = cv2.warpPerspective(image, matrix, (width, width))
    transformed_image = cv2.resize(transformed_image, (252, 252), interpolation=cv2.INTER_AREA)
    return transformed_image

def get_square_box_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    adaptive_threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    corners = cv2.findContours(adaptive_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corners = corners[0] if len(corners) == 2 else corners[1]
    corners = sorted(corners, key=cv2.contourArea, reverse=True)
    for corner in corners:
        length = cv2.arcLength(corner, True)
        approx = cv2.approxPolyDP(corner, 0.015 * length, True)
        puzzle_image = image_preprocessing(image, approx)

        # Color detection
        hsv_image = cv2.cvtColor(puzzle_image, cv2.COLOR_BGR2HSV)
        
        # Define lower and upper bounds for red color in HSV
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        
        # Define lower and upper bounds for blue color in HSV
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create masks for red and blue
        mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
        mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
        
        # Find contours in the masks
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on the original image
        cv2.drawContours(puzzle_image, contours_red, -1, (0, 0, 255), 2)  # Red color
        cv2.drawContours(puzzle_image, contours_blue, -1, (255, 0, 0), 2)  # Blue color
        
        # Convert puzzle_image to HSV color space for additional color filtering
        hsv_puzzle_image = cv2.cvtColor(puzzle_image, cv2.COLOR_BGR2HSV)
        
        # Additional color filtering (using the filter from the second code snippet)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([150, 255, 255])
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([20, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_blue = cv2.inRange(hsv_puzzle_image, lower_blue, upper_blue)
        mask_red1 = cv2.inRange(hsv_puzzle_image, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_puzzle_image, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_colors = cv2.bitwise_or(mask_blue, mask_red)
        
        # Find contours for colors
        contours_colors, _ = cv2.findContours(mask_colors, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours for colors on the puzzle_image
        cv2.drawContours(puzzle_image, contours_colors, -1, (0, 255, 0), 3)

        # Find the position of the red contours
        center = None
        for contour in contours_colors:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                position = find_red_position(center, puzzle_image.shape[:2])

        return puzzle_image, contours_colors  # Return contours_colors along with the puzzle_image

    return puzzle_image, [] 


import cv2
import numpy as np

# ... (previous code remains unchanged)

def main():
    cap = cv2.VideoCapture(0)
    printed_positions = set()  # Set to store printed positions

    while True:
        ret, frame = cap.read()

        if ret:
            puzzle_image, contours_colors = get_square_box_from_image(frame)

            # Find the positions of the red contours
            red_positions = []
            for contour in contours_colors:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    position = find_red_position(center, puzzle_image.shape[:2])
                    red_positions.append(position)

            cv2.imshow('Original', frame)
            cv2.imshow('Transformed', puzzle_image)

            key = cv2.waitKey(10)  # Increase the waiting time to 10 milliseconds

            if key == 13:  # 13 corresponds to the Enter key
                draw_red_circle(frame)

            if key == ord('p') and red_positions:
                for red_position in red_positions:
                    if red_position not in printed_positions:
                        print(f"{red_position}")
                        printed_positions.add(red_position)

            if key != ord('p'):  # Reset the blacklist if a key other than 'p' is pressed
                printed_positions.clear()

            if key == 27 or key == ord('q'):  # Break the loop on ESC or 'q' key
                break

        else:
            break  # Break the loop if unable to read from the camera

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

    