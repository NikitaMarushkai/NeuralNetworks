from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.applications.xception import preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import os
import time

model = Xception(weights='imagenet')

images = os.listdir('resources')
times = []

for img_path in images:
    img = image.load_img('resources/' + img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    start_time = time.time()
    preds = model.predict(x)
    times.append(time.time() - start_time)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted for ' + img_path + ": ", decode_predictions(preds, top=3)[0])

print("Average prediction time: %s seconds" % np.mean(times))
