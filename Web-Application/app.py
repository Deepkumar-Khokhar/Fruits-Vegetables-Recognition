import streamlit as st
import tensorflow as tf
import numpy as np

def model_predict(test_image):
    model = tf.keras.models.load_model('trained_model.h5')
    img = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.array([img_array])


    predictions = model.predict(img_array)

    return np.argmax(predictions)

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Prediction"])

if app_mode == "Home":
    st.title("Home Page")
    st.write("Welcome to the home page")
    image_path = 'Fruits&veg.jpg'
    st.image(image_path, width=700)

elif app_mode == "About":
    st.title("About Page")
    st.subheader("About Dataset")
    st.text("This Dataset contains the images of following fruit or vegetable:")
    st.code("Fruits: banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("Vegetables:  cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

elif app_mode == "Prediction":
    st.title("Prediction Page")
    test_image = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if(st.button("Show Image")):
        st.image(test_image, width=700)
    
    if(st.button("Predict")):
        st.snow()
        st.write("Classifying...")

        with open('label.txt') as f:
            class_labels = f.readlines()
        label = []
        for i in class_labels:
            label.append(i[:-1])
        st.success("Model is predicting... it's a {}".format(label[model_predict(test_image)]))