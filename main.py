import cv2  # OpenCV library for image processing
from ultralytics import YOLO  # YOLO model from the Ultralytics library for object detection
import matplotlib.pyplot as plt  # Matplotlib library to display images
import os  # OS module to interact with the file system (e.g., check if files exist)

# Load the YOLOv5 model (yolov5s.pt is a smaller, faster model for object detection)
model = YOLO("yolov5s.pt")  # This loads the pre-trained YOLOv5 small model

# List of paths to the images you want to run object detection on
image_paths = [
    "images/fruits.jpg",  # Path to the fruits image
    "images/cars.jpg",    # Path to the cars image
    "images/animals.jpg", # Path to the animals image
    "images/people.jpg"   # Path to the people image
]

# Create a grid to display 4 images in 1 row (4 columns)
fig, axes = plt.subplots(1, 4, figsize=(15, 5))  # Adjust the figure size for the plot

# Loop through each image path
for i, image_path in enumerate(image_paths):
    # Check if the image file exists at the given path
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")  # Print an error if the file is not found
        continue  # Skip this image if it's not found

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB because OpenCV loads images in BGR format by default
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run object detection on the image using the YOLO model
    results = model(rgb_image)  # The model predicts objects in the image

    # Get the annotated image with bounding boxes drawn around detected objects
    annotated_image = results[0].plot()  # Plot the detections on the image

    # Display the annotated image in the corresponding subplot
    axes[i].imshow(annotated_image)  # Show the image in the plot
    axes[i].axis("off")  # Hide the axis for a cleaner display
    axes[i].set_title(f"Image {i + 1}")  # Title for each subplot (Image 1, Image 2, etc.)

# After the loop, adjust the layout and display the images
plt.tight_layout()  # Automatically adjust the subplot spacing
plt.show()  # Show the plot with all the images
