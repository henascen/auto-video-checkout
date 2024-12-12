# Create a function that prints the right clicks on an image using opencv
import cv2
import numpy

from pathlib import Path

IMAGE_PATH = Path.cwd() / 'data' / 'images' / 'table_fixed.jpg'
TABLE_IMAGE_PATH = Path.cwd() / 'data' / 'images' / 'nice_table.jpg'

# Load the images
image = cv2.imread(str(IMAGE_PATH))
print(f'Image shape: {image.shape}')
image_table = cv2.imread(str(TABLE_IMAGE_PATH))

# Perspective Points, as X, Y
UP_LEFT_CORNER_ORIG = (384, 840)
UP_LEFT_CORNER_TABLE = (20, 10)

UP_RIGHT_CORNER_ORIG = (1348, 538)
UP_RIGHT_CORNER_TABLE = (380, 10)

DOWN_LEFT_MIDDLE_ORIG = (450, 1084)
DOWN_LEFT_MIDDLE_TABLE = (20, 130)

DOWN_RIGHT_CORNER_ORIG = (1524, 746)
DOWN_RIGHT_CORNER_TABLE = (380, 190)

new_height = DOWN_RIGHT_CORNER_TABLE[1] - UP_RIGHT_CORNER_TABLE[1]
new_width = UP_RIGHT_CORNER_TABLE[0] - UP_LEFT_CORNER_TABLE[0]

array_keypts_orig = numpy.float32(
    [
        numpy.array(UP_LEFT_CORNER_ORIG),
        numpy.array(UP_RIGHT_CORNER_ORIG),
        numpy.array(DOWN_RIGHT_CORNER_ORIG),
        numpy.array(DOWN_LEFT_MIDDLE_ORIG),
    ]
)
array_keypts_table = numpy.float32(
    [
        numpy.array(UP_LEFT_CORNER_TABLE),
        numpy.array(UP_RIGHT_CORNER_TABLE),
        numpy.array(DOWN_RIGHT_CORNER_TABLE),
        numpy.array(DOWN_LEFT_MIDDLE_TABLE),
    ]
)

# Find the perspective matrix to transform the points
persp_matrix = cv2.getPerspectiveTransform(
    array_keypts_orig,
    array_keypts_table
)
# Save the matrix
numpy.save(
     str(Path.cwd() / 'location' / 'transforms' / 'table_persp_mtx_v1.npy'),
     persp_matrix
)
# dst = cv2.warpPerspective(image, persp_matrix, (new_width, new_height))
# h, status = cv2.findHomography(array_keypts_orig, array_keypts_table)

# Draw two points at the front of the table
image = cv2.circle(
    image,
    (UP_LEFT_CORNER_ORIG),
    10,
    (0, 255, 0),
    -1
)
image = cv2.circle(
    image,
    (UP_RIGHT_CORNER_ORIG),
    10,
    (0, 255, 0),
    -1
)
# Draw two points at the end of the table
image = cv2.circle(
    image,
    (DOWN_LEFT_MIDDLE_ORIG),
    10,
    (0, 255, 0),
    -1
)
image = cv2.circle(
    image,
    (DOWN_RIGHT_CORNER_ORIG),
    10,
    (0, 255, 0),
    -1
)

# Resize the image to half the size
image = cv2.resize(image, None, fx=0.5, fy=0.5)

# Global variables to save the clicks in the original image
x_click, y_click = 0, 0
# Create a function to handle mouse clicks
def handle_click(event, x, y, flags, param):
    global x_click, y_click
    if event == cv2.EVENT_LBUTTONUP:
        # Display the coordinates of the click
        print("Clicked at ({}, {})".format(x, y))
        x_click = x
        y_click = y

# Create a window to display the image
cv2.namedWindow("image")
cv2.namedWindow("Table")

# Set the mouse callback function for the window
cv2.setMouseCallback("image", handle_click)

# Display the riginal image in the window
cv2.imshow("image", image)

while True:
    # Transformed point to new image space
    point_trf = cv2.perspectiveTransform(
        numpy.array([[[x_click * 2, y_click * 2]]], dtype=numpy.float32),
        persp_matrix
    )
    image_table = cv2.circle(
        image_table,
        (int(point_trf[0][0][0]), int(point_trf[0][0][1])),
        3,
        (0, 255, 0),
        -1
    )
    # display the image and wait for a keypress
    cv2.imshow("Table", image_table)
    key = cv2.waitKey(1) & 0xFF
    # if the 'c' key is pressed, break from the loop
    if key == ord("c"):
	    break

# Destroy the window
cv2.destroyAllWindows()
