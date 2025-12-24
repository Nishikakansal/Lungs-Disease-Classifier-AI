import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, applications


tf.keras.backend.clear_session()

st.set_page_config(
    page_title="Lung Disease Classifier (DenseNet121)",
    page_icon="🫁",
    layout="centered"
)

def build_densenet_model(num_classes=5):
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    
    base_model = applications.DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

@st.cache_resource
def load_model():
    try:
        class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
        num_classes = len(class_names)
        
        if os.path.exists('lung_disease_classifier_densenet121.keras'):
            try:
                model = tf.keras.models.load_model('lung_disease_classifier_densenet121.keras')
                st.info("Successfully loaded full model from lung_disease_classifier_densenet121.keras")
                return model, class_names
            except Exception as e:
                st.warning(f"Could not load full model, trying to load weights: {str(e)}")
                model = build_densenet_model(num_classes)
                try:
                    model.load_weights('lung_disease_classifier_densenet121.keras')
                    st.info("Loaded weights from lung_disease_classifier_densenet121.keras")
                    return model, class_names
                except Exception as e2:
                    st.error(f"Could not load weights either: {str(e2)}")
                    return model, class_names
        elif os.path.exists('best_model.keras'):
            try:
                model = tf.keras.models.load_model('best_model.keras')
                st.info("Successfully loaded full model from best_model.keras")
                return model, class_names
            except Exception as e:
                model = build_densenet_model(num_classes)
                model.load_weights('best_model.keras')
                st.info("Loaded weights from best_model.keras")
                return model, class_names
        else:
            st.warning("No model files found. Using a freshly initialized DenseNet121 model.")
            return build_densenet_model(num_classes), class_names
            
    except Exception as e:
        st.error(f"Error setting up model: {str(e)}")
        return None, None

def preprocess_image(image, target_size=(224, 224)):
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    img_array = cv2.resize(img_array, target_size)
    
    img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict(model, image, class_names):
    try:
        processed_image = preprocess_image(image)
        
        prediction = model.predict(processed_image, verbose=0)
        
        predicted_class_index = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_index] * 100
        predicted_class = class_names[predicted_class_index]
        
        return predicted_class, confidence, prediction[0], class_names
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, 0, None, None

def main():
    st.title("Lung Disease Classifier (DenseNet121)")
    st.write("Upload an X-ray image to classify lung conditions using DenseNet121 architecture")
    
    with st.expander("About the Model"):
        st.write("""
        This application uses a DenseNet121 model fine-tuned on lung X-ray images to classify different lung conditions:
        - Normal
        - Bacterial Pneumonia
        - Viral Pneumonia
        - COVID-19
        - Tuberculosis
        
        DenseNet121 is a powerful convolutional neural network architecture that uses dense connections between layers, 
        which helps with feature reuse, reduces the number of parameters, and mitigates the vanishing gradient problem.
        """)
    
    with st.spinner("Loading DenseNet121 model..."):
        model, class_names = load_model()
    
    if model is not None:
        st.success("Model loaded successfully! You can now upload an image for prediction.")
    
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
        
        if st.button("Predict"):
            if model is None:
                st.error("Model could not be loaded. Please check if the model file exists.")
            else:
                with st.spinner("Analyzing X-ray with DenseNet121..."):
                    try:
                        predicted_class, confidence, all_probs, _ = predict(model, image, class_names)
                        
                        if predicted_class is None:
                            st.error("An error occurred during prediction.")
                        else:
                            st.success(f"Prediction: **{predicted_class}**")
                            st.info(f"Confidence: {confidence:.2f}%")
                            
                            st.subheader("Probability Distribution")
                            fig, ax = plt.subplots(figsize=(10, 5))
                            y_pos = np.arange(len(class_names))
                            ax.barh(y_pos, all_probs * 100)
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(class_names)
                            ax.set_xlabel('Probability (%)')
                            ax.set_title('Prediction Probability by Class')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            st.subheader("Interpretation")
                            if predicted_class == "Normal":
                                st.write("The X-ray appears normal with no significant abnormalities detected.")
                            else:
                                st.write(f"The X-ray shows signs consistent with {predicted_class}. Please consult with a healthcare professional for proper diagnosis.")
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {str(e)}")
                
                st.warning("Disclaimer: This tool is for educational purposes only and should not be used for medical diagnosis. Always consult with a healthcare professional.")

if __name__ == "__main__":
    main()