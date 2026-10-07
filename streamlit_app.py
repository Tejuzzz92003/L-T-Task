import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:5000")

st.set_page_config(
    page_title="Task 15 Image Classifier",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Deep Learning Image Classification")
st.write("Upload a CIFAR-10 style image and get a prediction from the Flask API.")

st.info("Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file:
    st.image(uploaded_file, caption="Selected image", use_container_width=True)

    if st.button("Predict Image", type="primary"):
        try:
            files = {
                "image": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "image/jpeg",
                )
            }

            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                timeout=60,
            )

            data = response.json()

            if response.ok and data.get("success"):
                st.success(
                    f"Prediction: {data['prediction']} "
                    f"({data['confidence']:.2%} confidence)"
                )

                st.subheader("Top 3 predictions")
                for item in data["top_3"]:
                    st.write(
                        f"**{item['class']}** — "
                        f"{item['confidence']:.2%}"
                    )
            else:
                st.error(data.get("error", "Prediction failed."))

        except requests.exceptions.RequestException:
            st.error(
                "Cannot connect to Flask API. "
                "Make sure api.py is running on port 5000."
            )
