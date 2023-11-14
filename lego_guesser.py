import tensorflow as tf
import numpy as np
import cv2


def guess_legos(images_urls, model):
  global img

  # Define the class indices
  class_indices = {'beam 1m': 0, 'beam 1x2': 1, 'brick 1x1': 2, 'brick 1x2': 3, 'brick 1x3': 4, 'brick 1x4': 5,
                   'brick 2x2': 6, 'brick 2x3': 7, 'brick 2x4': 8, 'brick bow 1x3': 9, 'brick bow 1x4': 10,
                   'brick corner 1x2x2': 11, 'brick d16 w cross': 12, 'bush 2m friction - cross axle': 13,
                   'connector peg w knob': 14, 'cross block fork 2x2': 15, 'curved brick 2 knobs': 16,
                   'flat tile 1x1': 17, 'flat tile 1x2': 18, 'flat tile 2x2': 19, 'flat tile corner 2x2': 20,
                   'flat tile round 2x2': 21, 'lever 2m': 22, 'lever 3m': 23, 'peg with friction': 24, 'plate 1x1': 25,
                   'plate 1x2': 26, 'plate 1x2 with 1 knob': 27, 'plate 1x3': 28, 'plate 2 knobs 2x2': 29,
                   'plate 2x2': 30, 'plate 2x3': 31, 'plate 2x4': 32, 'plate corner 2x2': 33,
                   'roof corner inside tile 2x2': 34, 'roof corner outside tile 2x2': 35, 'roof tile 1x1': 36,
                   'roof tile 1x2': 37, 'roof tile 1x3': 38, 'roof tile 1x4': 39, 'roof tile 2x2': 40,
                   'roof tile 2x3': 41, 'roof tile inside 3x3': 42, 'roof tile outside 3x3': 43, 'round brick 1x1': 44,
                   'technic brick 1x2': 45}
  labels = dict((v, k) for k, v in class_indices.items())

  # Initialize a list to store predictions
  predictions = []

  # Process each image URL
  for url in images_urls:
    img_path = tf.keras.utils.get_file(f"{url.split('/')[-1]}", url)
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    # Predict the label
    pred = model.predict(img)
    pred_class = np.argmax(pred, axis=1)
    pred_label = labels[pred_class[0]]

    # Append the prediction to the list with its URL
    predictions.append({"url": url, "label": pred_label})

  # Return the list of predictions
  return predictions
