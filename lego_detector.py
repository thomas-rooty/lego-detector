import cv2
import os
import uuid
import numpy as np


def whiten_lego_piece(image):
  """
  Increase the brightness of the Lego piece to make it appear whiter.
  """
  # Convert to HSV for easier brightness adjustment
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  h, s, v = cv2.split(hsv)

  # Increase brightness
  lim = 255 - 50
  v[v > lim] = 255
  v[v <= lim] += 50

  final_hsv = cv2.merge((h, s, v))
  image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
  return image


def process_image(user_image_path):
  img_example = cv2.imread(user_image_path)
  img_bg = cv2.imread('input/background_backlit_B.jpg')

  # Convert to grayscale
  img_bg_gray = cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)
  img_gray = cv2.cvtColor(img_example, cv2.COLOR_BGR2GRAY)

  # Calculate difference between background and example image
  diff_gray = cv2.absdiff(img_bg_gray, img_gray)

  # Gaussian blur to smooth out pixels
  diff_gray_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)

  # Find threshold to convert to binary image using Otsu's method
  ret, img_tresh = cv2.threshold(diff_gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  # Find contours
  arr_cnt, a2 = cv2.findContours(img_tresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Dimensions of the image
  height, width, channels = img_example.shape

  validcontours = []
  contour_index = -1

  # Iterate over each contour
  for i in arr_cnt:
    contour_index += 1
    ca = cv2.contourArea(i)

    # Calculate aspect ratio
    x, y, w, h = cv2.boundingRect(i)
    aspect_ratio = float(w) / h

    # Flag as noise if the contour is on the edge of the image
    edge_noise = x == 0 or y == 0 or (x + w) == width or (y + h) == height

    # Keep contour if area is large enough and not on the edge
    if ca > 1300 and aspect_ratio <= 6 and not edge_noise:
      validcontours.append(contour_index)

  # Check output folder
  if not os.path.exists('output'):
    os.makedirs('output')

  bricks_data = []
  for i in validcontours:
    x, y, w, h = cv2.boundingRect(arr_cnt[i])
    unique_filename = f"crop_{i}_{uuid.uuid4()}.jpg"
    brick_info = {
      "id": i,
      "position": {"x": x, "y": y},
      "size": {"width": w, "height": h},
      "image_path": f"output/{unique_filename}"
    }
    bricks_data.append(brick_info)

    # Crop image
    crop_img = img_example[y:y + h, x:x + w]

    # Whiten the Lego piece
    crop_img = whiten_lego_piece(crop_img)

    # Create a black background
    black_background = np.zeros_like(crop_img)

    # Create a mask from the cropped image
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Combine the cropped image with the black background
    black_background[mask == 255] = crop_img[mask == 255]

    # Save the image with a black background and the lego piece colored whiter
    cv2.imwrite(f"output/{unique_filename}", black_background)

  return bricks_data
