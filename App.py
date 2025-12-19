import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import io, random, os

st.set_page_config(
    page_title="Dog Breed Classifier 🐾",
    page_icon="🐶",
    layout="centered"
)

st.title("🐾 Dog Breed Classifier")
st.write("Upload or snap a dog image to predict its breed using an EfficientNetB0 model.")

# Load Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "Dogs.keras", compile=False
    )
    return model

model = load_model()
breed_names = np.load("breed_names.npy")

def get_dog_description(csv_path, dog_name):
    df = pd.read_csv(csv_path)
    matching_dogs = df[df['Name'].str.lower() == dog_name.lower()]

    if matching_dogs.empty:
        return f"Sorry, I couldn't find a description for '{dog_name}'."

    dog = matching_dogs.iloc[0].copy()
    
    def children_phrase(value):
        if str(value).lower() == "yes":
            return "great with children"
        if str(value).lower() == "no":
            return "not ideal for families with small children"
        return "generally good with children"

    dog['Good with Children'] = children_phrase(dog['Good with Children'])
    
    description = (
    f"Meet the {dog['Name']}, a {dog['Size']}-sized {dog['Type']} known for {dog['Unique Feature']}. "
    f"They are {dog['Good with Children']} and have a friendliness rating of {dog['Friendly Rating (1-10)']}/10. "
    f"They require about {dog['Exercise Requirements (hrs/day)']} hours of daily exercise to stay happy and healthy. "
    f"Grooming needs are {dog['Grooming Needs']}, and with proper care, they typically live for around {dog['Life Span']} years."
) 
    return description

# Similar Images Section
LABELS_PATH = "labels.csv"
IMAGES_FOLDER = "train"

labels_df = pd.read_csv(LABELS_PATH)
def get_image(predicted_label): 
    def get_similar_images(predicted_label, num_images=3):
        breed_images = labels_df[labels_df['breed'] == predicted_label]['id'].tolist()
    
    # Handle case where there are fewer than 3 images
        num_to_sample = min(num_images, len(breed_images))
    
        if num_to_sample > 0:
        # Randomly select images
            selected_images = random.sample(breed_images, num_to_sample)
            return selected_images
        else:
            return []

# Display similar images
    st.subheader(f"More examples of {predicted_label.replace('_', ' ')}")

    similar_images = get_similar_images(predicted_label)

    if similar_images:
        cols = st.columns(3)
        for idx, img_name in enumerate(similar_images):
            img_path = os.path.join(IMAGES_FOLDER, img_name)
            img_path = img_path + ".jpg"  
            if os.path.exists(img_path):
                with cols[idx]:
                    img = Image.open(img_path)
                    st.image(img)
            else:
                st.warning(f"Image {img_name} not found")
    else:
        st.info(f"No additional images found for {predicted_label}")
# Image Upload
uploaded_file = st.file_uploader("Choose a dog image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    # Preprocess Image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Prediction
    preds = model.predict(img_array)
    predicted_index = np.argmax(preds[0])
    predicted_label = breed_names[predicted_index]
    confidence = np.max(preds[0]) * 100

    # Display Result
    st.subheader(f"🎯 Predicted Breed: {predicted_label}")
    st.progress(int(confidence))
    st.write(f"**Confidence:** {confidence:.2f}%")
    description = get_dog_description('description_updated.csv', predicted_label)
    st.markdown(f"**Description:** {description}")
    get_image(predicted_label)

else:
    st.info("⬆️ Upload an image to begin")

st.markdown("---")


