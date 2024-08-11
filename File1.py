import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from skimage import filters, measure
import cv2
from svgpathtools import svg2paths2


def svg_to_png(svg_path, output_size=(512, 512)):
    paths, attributes, svg_attr = svg2paths2(svg_path)
    image = Image.new('L', output_size, 255)  # Create a white background image
    draw = ImageDraw.Draw(image)

    for path in paths:
        for segment in path:
            # Sample points along the segment
            n_points = 100  # Increase this value for smoother curves
            points = [(segment.point(t).real, segment.point(t).imag) for t in np.linspace(0, 1, n_points)]
            # Draw lines between sampled points
            if len(points) > 1:
                draw.line(points, fill=0, width=2)

    return image


def preprocess_image(image):
    image = np.array(image)
    edges = filters.sobel(image)
    return edges


def find_contours(edges):
    binary_image = edges > np.mean(edges)
    contours = measure.find_contours(binary_image, 0.5)
    return contours


def classify_shape(contour):
    contour = np.array(contour, dtype=np.float32)
    contour = contour[:, np.newaxis, :]
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    num_vertices = len(approx)

    if num_vertices == 3:
        return "Triangle"
    elif num_vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.9 <= aspect_ratio <= 1.1:
            return "Square"
        else:
            return "Rectangle"
    else:
        area = cv2.contourArea(approx)
        perimeter = cv2.arcLength(approx, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > 0.8:
            return "Circle"
        else:
            return classify_irregular_shape(approx)


def classify_irregular_shape(approx):
    # Check if the shape resembles a star
    angles = []
    for i in range(len(approx)):
        pt1 = approx[i % len(approx)]
        pt2 = approx[(i + 1) % len(approx)]
        pt3 = approx[(i + 2) % len(approx)]
        vec1 = pt2 - pt1
        vec2 = pt3 - pt2
        angle = np.arccos(
            np.clip(np.dot(vec1.flatten(), vec2.flatten()) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))
        angles.append(angle)

    if len(angles) >= 10:
        distances = [np.linalg.norm(pt.flatten() - np.mean(approx, axis=0).flatten()) for pt in approx]
        distance_var = np.var(distances)
        if distance_var < 0.1:
            return "Star Shape"

    # Check if the shape is an ellipse
    if len(approx) >= 5:
        ellipse = cv2.fitEllipse(np.array(approx, dtype=np.float32))
        center, axes, angle = ellipse
        (a, b) = axes
        aspect_ratio = b / a if a > b else a / b

        if 0.5 < aspect_ratio < 2.0:
            return "Ellipse"

    if len(approx) > 6:
        return "Irregular Polygon"

    return "Unknown Shape"


def detect_shapes(image):
    edges = preprocess_image(image)
    contours = find_contours(edges)

    shapes = []
    for contour in contours:
        contour_array = np.array(contour, dtype=np.float32)
        if contour_array.ndim == 2 and contour_array.shape[1] == 2:
            if len(contour_array) >= 3:
                try:
                    shape = classify_shape(contour_array)
                    if cv2.contourArea(contour_array) > 100:
                        shapes.append((shape, contour_array))
                except cv2.error as e:
                    print(f"Error calculating contour area: {e}")
        else:
            print(f"Invalid contour format: {contour_array}")

    return shapes


def visualize_shapes(image, shapes):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray', extent=[0, image.size[0], image.size[1], 0])
    ax.set_title('Detected Shapes')
    ax.set_aspect('equal')
    ax.set_xlim(0, image.size[0])
    ax.set_ylim(image.size[1], 0)

    processed_contours = set()

    for shape, contour in shapes:
        contour_tuple = tuple(map(tuple, np.round(contour, decimals=2)))
        if contour_tuple in processed_contours:
            continue

        color = {
            "Circle": 'red',
            "Ellipse": 'green',
            "Rectangle": 'blue',
            "Square": 'magenta',
            "Star Shape": 'black',
            "Irregular Polygon": 'purple',
        }.get(shape, 'orange')

        ax.plot(contour[:, 1], contour[:, 0], color=color, linestyle='-', label=shape)
        processed_contours.add(contour_tuple)

    ax.legend()
    plt.show()


def main(image_path):
    image = svg_to_png(image_path)
    shapes = detect_shapes(image)
    visualize_shapes(image, shapes)


if __name__ == '__main__':
    image_path = r"C:\Users\ASUS\Downloads\Adobe Project\occlusion2.svg"
    main(image_path)
