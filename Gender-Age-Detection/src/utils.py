import cv2
import numpy as np
import os

def ensure_directory(path):
    """Ensure a directory exists, create if it doesn't."""
    if not os.path.exists(path):
        os.makedirs(path)

def load_image(image_path):
    """Load an image from path and handle errors."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return image

def resize_image(image, max_size=800):
    """Resize image while maintaining aspect ratio."""
    height, width = image.shape[:2]
    if height > max_size or width > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))
    return image

def draw_prediction(image, text, position, font_scale=0.8, thickness=2):
    """Draw prediction text with a background for better visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Calculate background rectangle
    padding = 5
    cv2.rectangle(
        image,
        (position[0], position[1] - text_height - 2*padding),
        (position[0] + text_width + 2*padding, position[1]),
        (0, 0, 0),
        -1
    )
    
    # Draw text
    cv2.putText(
        image,
        text,
        (position[0] + padding, position[1] - padding),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )
    
    return image

def save_image(image, output_path):
    """Save image to disk and handle errors."""
    directory = os.path.dirname(output_path)
    if directory:
        ensure_directory(directory)
    
    success = cv2.imwrite(output_path, image)
    if not success:
        raise ValueError(f"Failed to save image to {output_path}")
    
    return output_path 