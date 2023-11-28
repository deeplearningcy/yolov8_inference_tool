from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO  # Ensure this import works for your setup
import streamlit as st

# Preprocessing function for YOLOv8
def preprocess_image_for_yolov8(image):
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to the input size expected by YOLOv8
        transforms.ToTensor(),  # Convert to tensor
    ])

    # Convert the OpenCV image to PIL image
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Apply the transformations
    tensor_image = transform(image)

    return tensor_image

# Function to process the image with YOLOv8
def YoloV8_preds(image, model):
    # Preprocess the image
    tensor_image = preprocess_image_for_yolov8(image)

    # Perform inference
    results = model.predict(tensor_image.unsqueeze(0), imgsz=640, save=False, conf=0.5, iou=0.75, retina_masks=True)

    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)  # convert back to RGB
        cv2.imwrite('image_1.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))  # OpenCV expects BGR format for saving
    return

# Use st.cache_data to cache model loading
@st.cache_data
def load_model(model_name):
    return YOLO(model_name)

# Streamlit UI
def main():
    st.set_page_config(layout="wide")  # Use a wide page layout

    st.title('Construction Demolition Waste Detection Using YOLOv8 Instance Segmentation')
    st.header('Choose a YOLOv8 model and an image to perform inference')

    with st.sidebar:
        st.write("### Welcome to the Construction Demolition Waste Detection Tool")
        st.write("This interactive web application utilizes the advanced capabilities of YOLOv8 (You Only Look Once, version 8) for detecting and analyzing construction demolition waste in images. It's designed to provide quick, accurate, and automated waste material identification.")

    with st.sidebar:
        st.write("### How to use:")
        st.markdown('Step 1: Choose the YOLOv8 model that best suits your needs from the dropdown menu.')
        st.markdown('Step 2: Upload a JPG image containing construction demolition waste.')
        st.markdown('Step 3: View the processed image with detected waste materials highlighted.')
        st.markdown('Step 4: Optionally, download the processed image for your records.')

    with st.sidebar:
        st.write("### Technical overview:")
        st.write('This application is built using Python, with Streamlit providing the web interface. The image processing and object detection are powered by PyTorch and the Ultralytics YOLO implementation. The app pre-processes each uploaded image to match the input requirements of YOLOv8, including resizing and normalization, '
                 'and then performs object detection to identify various types of construction waste materials.')
    with st.sidebar:
        st.write("### CDW Classes:")
        st.markdown('Concrete')
        st.markdown('Brick')
        st.markdown('Tiles')
        st.markdown('Stones')
        st.markdown('Plaster board')
        st.markdown('Foam')
        st.markdown('Wood')
        st.markdown('Pipes')
        st.markdown('Plastic')
        st.markdown('General Waste')


    model_selector = st.selectbox(
        'Select Inference Model',
        ('YoloV8n-seg.pt', 'YoloV8s-seg.pt', 'YoloV8m-seg.pt', 'YoloV8l-seg.pt', 'YoloV8x-seg.pt'))

    model = load_model(model_selector)

    uploaded_file = st.file_uploader("Upload your file here (jpg format)", type=['jpg'])

    if uploaded_file:
        col1, col2 = st.columns(2)

        with col1:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption='Fig.1: Original Image')

            YoloV8_preds(original_image, model)

        with col2:
            st.image('image_1.jpg', caption='Fig.2: YOLOv8 Predictions')

    if uploaded_file:
        st.markdown("---")
        st.subheader("Additional Options")
        with open('image_1.jpg', 'rb') as file:
            btn = st.download_button(
                label="Download Processed Image",
                data=file,
                file_name="processed_image.jpg",
                mime="image/jpg"
            )

    st.write('### Below you can find snippets of the code used for preprocessing and inference')
    code_1 = '''def preprocess_image_for_yolov8(image):
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to the input size expected by YOLOv8
        transforms.ToTensor(),  # Convert to tensor
    ])

    # Convert the OpenCV image to PIL image
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Apply the transformations
    tensor_image = transform(image)

    return tensor_image'''


    code_2 = '''def YoloV8_preds(image, model):
         # Preprocess the image
        tensor_image = preprocess_image_for_yolov8(image)

        # Perform inference
        results = model.predict(tensor_image.unsqueeze(0), imgsz=640, save=False, conf=0.5, iou=0.75, retina_masks=True)

        for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)  # convert back to RGB
        cv2.imwrite('image_1.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))  # OpenCV expects BGR format for saving
        return'''

    st.code(code_1, language='python')
    st.code(code_2, language='python')

if __name__ == '__main__':
    main()
