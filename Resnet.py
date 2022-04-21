import numpy as np

from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import decode_predictions

import matplotlib.pyplot as plt
import tensorflow as tf
import logging
import time

iterations = 1000

tf.get_logger().setLevel(logging.ERROR)

#Target size of Resnet image
image = load_img("images/leopard.jpg", target_size=(224, 224))
image_np = img_to_array(image)
image_np = np.expand_dims(image_np, axis=0)

model = resnet50.ResNet50(weights='imagenet')

X = resnet50.preprocess_input(image_np.copy())
print('Starting inference')
start = time.time()
for n in range(iterations):
    y = model.predict(X)
    predicted_labels = decode_predictions(y)
    #print(n)
end = time.time()

print('predictions = ', predicted_labels)

plt.imshow(np.uint8(image_np[0]))
plt.show

#average_time = (end-start)/iterations

print('Time of inference was: ', end-start)
print('Average inference time:', (end-start)/iterations)


