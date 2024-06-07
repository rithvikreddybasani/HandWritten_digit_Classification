

import numpy as np
from PIL import Image
from tensorflow import keras
import streamlit as st


model = keras.models.load_model("mnist_model.h5")


st.title('MNIST Digit Classification App')
st.write('Upload a single image or multiple images of handwritten digits and get predictions!')

uploaded_images = st.file_uploader("Upload your images here", type=["jpg", "png", "jpeg"], accept_multiple_files=True)


if uploaded_images:
    for uploaded_image in uploaded_images:
       
        image = Image.open(uploaded_image)
        
    
        image = image.convert('L')  
        image = image.resize((28, 28))
        
        
        image_array = np.array(image) / 255.0
        
    
        processed_image = image_array.reshape(1, 28, 28)
        
    
        prediction_new_image = model.predict(processed_image)
        highest_index = np.argmax(prediction_new_image)

        
        st.image(uploaded_image, caption='Uploaded Image.', width=200)
        st.write(f'The predicted label for the above image is: {highest_index}')
