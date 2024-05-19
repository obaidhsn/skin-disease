import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

def predictor(img, csv_path, model_path):
    # Read in the CSV file
    class_df = pd.read_csv(csv_path)
    img_height = int(class_df['height'].iloc[0])
    img_width = int(class_df['width'].iloc[0])
    img_size = (img_width, img_height)
    scale = class_df['scale by'].iloc[0]
    try:
        s = int(scale)
        s2 = 1
        s1 = 0
    except:
        split = scale.split('-')
        s1 = float(split[1])
        s2 = float(split[0].split('*')[1])

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the image
    img = Image.open(img)
    img = img.resize(img_size)
    img = np.array(img)
    img = img * s2 - s1
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)  # Ensure data type is float32

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get class name and probability
    index = np.argmax(output_data)
    probability = output_data[0][index]
    class_name = class_df['class'].iloc[index]
    
    return class_name, probability
