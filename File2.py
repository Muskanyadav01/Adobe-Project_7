import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_process_image(image_path):
    """Load the image, convert to grayscale, and apply adaptive thresholding."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")

    height, width = image.shape[:2]
    if height > width:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh, image

def extract_contours(thresh):
    """Extract contours from the thresholded image."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the image.")
    largest_contour = max(contours, key=cv2.contourArea)
    points = largest_contour.reshape(-1, 2)
    return points

def classify_shape(points):
    """Classify the shape based on the contour points."""
    approx = cv2.approxPolyDP(points, 0.04 * cv2.arcLength(points, True), True)
    num_vertices = len(approx)
    if num_vertices == 3:
        return "triangle"
    elif num_vertices == 4:
        aspect_ratio = cv2.boundingRect(approx)[2] / cv2.boundingRect(approx)[3]
        if 0.95 <= aspect_ratio <= 1.05:
            return "square"
        else:
            return "rectangle"
    else:
        return "other"

def reflect_point(point, line_angle, centroid):
    """Reflect a point across a line at a given angle."""
    theta = np.radians(line_angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    translated_point = point - centroid
    rotated_point = np.array([
        translated_point[0] * cos_theta + translated_point[1] * sin_theta,
        -translated_point[0] * sin_theta + translated_point[1] * cos_theta
    ])
    reflected_point = np.array([
        rotated_point[0],
        -rotated_point[1]
    ])
    final_point = np.array([
        reflected_point[0] * cos_theta - reflected_point[1] * sin_theta,
        reflected_point[0] * sin_theta + reflected_point[1] * cos_theta
    ]) + centroid

    return final_point

def evaluate_symmetry(points, line_angle, centroid):
    """Evaluate how well the shape is symmetric around a line at the given angle."""
    reflected_points = np.array([reflect_point(point, line_angle, centroid) for point in points])
    distances = np.linalg.norm(points - reflected_points, axis=1)
    return np.mean(distances)

def find_best_symmetry_line(points, angles, centroid):
    """Find the best line of symmetry by minimizing the distance between original and reflected points."""
    best_angle = None
    min_distance = float('inf')

    for angle in angles:
        distance = evaluate_symmetry(points, angle, centroid)
        if distance < min_distance:
            min_distance = distance
            best_angle = angle

    return best_angle

def plot_shape_and_lines(image, points, best_angle, centroid):
    """Plot the original shape and the best symmetry line on the image."""
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.plot(points[:, 0], points[:, 1], 'o-', label='Original Shape')
    plt.plot(np.append(points[:, 0], points[0, 0]),
             np.append(points[:, 1], points[0, 1]), 'r--', label='Closed Shape')

    angle_rad = np.radians(best_angle)
    x_vals = np.array([0, image.shape[1]])
    y_vals = np.tan(angle_rad) * (x_vals - centroid[0]) + centroid[1]
    plt.plot(x_vals, y_vals, 'g--', label=f'Symmetry Line at {best_angle}°')

    plt.xlim([0, image.shape[1]])
    plt.ylim([image.shape[0], 0])

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Shape and Symmetry Line')
    plt.show()


def main(image_path):
    try:
        thresh, image = load_and_process_image(image_path)
        print("Image loaded and processed successfully.")
    except ValueError as e:
        print(e)
        return

    try:
        points = extract_contours(thresh)
        print(f"Contours extracted: {points.shape}")
    except ValueError as e:
        print(e)
        return

    shape_type = classify_shape(points)
    print(f"Detected shape: {shape_type}")

    centroid = np.mean(points, axis=0)
    print(f"Centroid of shape: {centroid}")

    if shape_type == "triangle":
        angles = [90]  # Triangles typically have vertical symmetry
    elif shape_type == "square":
        angles = [0, 90, 180, 270]  # Squares have multiple symmetry axes
    elif shape_type == "rectangle":
        angles = [0, 90]  # Rectangles can have both horizontal and vertical symmetry
    else:
        angles = np.linspace(0, 180, 3600)  # For other shapes, use finer evaluation

    best_angle = find_best_symmetry_line(points, angles, centroid)
    print(f'Best symmetry line is at angle: {best_angle}°')

    plot_shape_and_lines(image, points, best_angle, centroid)

if __name__ == "__main__":
    image_path = r"C:\Users\ASUS\Downloads\Adobe Project\triangle.png"  # Replace with your image path
    main(image_path)
