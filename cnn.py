import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st

train_dataset_path = r'D:\archive\chest_xray\train'

# Load datasets
batch_size = 64
img_height = 128
img_width = 128

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dataset_path,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names

model = load_model(r'final.h5')

st.title("Chest X-ray Image Classification")

t = st.number_input("Enter an index (0 to 63) to select an image randomly from the dataset:", value=0, min_value=0, max_value=63)
t = int(t)
def normalize_image(image):
    return tf.cast(image, tf.float32) / 255.0

if st.button("Predict"):
    # Predicting the class for the selected image
    predictions = model.predict(train_ds.take(1))
    ind = np.argmax(predictions[t])
    predicted_class = class_names[ind]

    image = list(train_ds.take(1))[0][0][t].numpy().astype("uint8")
    normalize_image(image)
    st.image(image,width=400)

    st.write(f"Predicted class for the selected image: {predicted_class}")

# Displaying probability scores for each class
    st.write("Probability scores:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {predictions[t][i]}")
