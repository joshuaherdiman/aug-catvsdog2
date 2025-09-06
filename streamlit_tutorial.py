import streamlit as st
from fastai.vision.all import *

st.title("Cat vs Dog Classifier")
st.text("Built By Joshua")

def is_cat(f):
    return f[0].isupper()


cat_vs_dog_model = load_learner("cat-vs-dog(1).pkl")

def predict(image):
    img = PILImage.create(image)
    pred_class, pred_idx, outputs = cat_vs_dog_model.predict(img)
    likelihood_is_cat = outputs[1].item()
    if likelihood_is_cat > 0.9:
        return "Cat"
    elif likelihood_is_cat < 0.1:
        return "Dog"
    else:
        return "Not sure... try another picture!"

uploaded_file = st.file_uploader("Choose an image to upload...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    prediction = predict(uploaded_file)
    st.write(prediction)

