import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# Function to predict class of an X-ray image
def predict_xray(image, model, img_height=128, img_width=128, class_names=['NORMAL', 'PNEUMONIA']):
    try:
        # Preprocess the image
        img = image.resize((img_height, img_width))
        # Convert grayscale to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = tf.keras.utils.img_to_array(img)
        img_array = img_array / 255.0  # Normalize pixel values
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        # Make prediction
        predictions = model.predict(img_array, verbose=0)

        # Apply softmax since your model outputs logits
        predictions = tf.nn.softmax(predictions)

        # Get predicted class and confidence
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])

        # Prepare results
        results = {
            'predicted_class': class_names[predicted_class_index],
            'confidence': confidence,
            'probabilities': {
                class_names[i]: float(prob)
                for i, prob in enumerate(predictions[0])
            }
        }
        
        return results

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# Load the model outside the function to avoid reloading it for each prediction
@st.cache_resource
def load_model(model_path='final.h5'):
    return tf.keras.models.load_model(model_path)

# Streamlit App
def main():
    st.title("X-ray Image Classifier")
    st.write("Upload an X-ray image to predict if it's NORMAL or PNEUMONIA.")

    model = load_model()

    # Single image upload
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load image from uploaded file
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        st.write("Classifying...")
        result = predict_xray(image, model)
        
        if result:
            st.write(f"Predicted Class: **{result['predicted_class']}**")
            st.write(f"Confidence: **{result['confidence']:.2%}**")
            st.write("Class Probabilities:")
            for class_name, prob in result['probabilities'].items():
                st.write(f"{class_name}: {prob:.2%}")
            
            # Display image with prediction
            fig, ax = plt.subplots()
            ax.imshow(image)
            ax.set_title(f"Predicted: {result['predicted_class']} ({result['confidence']:.2%})")
            ax.axis('off')
            st.pyplot(fig)

    # Batch image upload
    st.write("Or upload multiple X-ray images for batch prediction.")
    batch_uploaded_files = st.file_uploader("Choose multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if batch_uploaded_files:
        st.write("Batch Processing Results:")
        for uploaded_file in batch_uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Processing: {uploaded_file.name}", use_column_width=True)
            result = predict_xray(image, model)
            
            if result:
                st.write(f"**{uploaded_file.name}**")
                st.write(f"Predicted Class: **{result['predicted_class']}**")
                st.write(f"Confidence: **{result['confidence']:.2%}**")
                st.write("Class Probabilities:")
                for class_name, prob in result['probabilities'].items():
                    st.write(f"{class_name}: {prob:.2%}")

if __name__ == "__main__":
    main()
