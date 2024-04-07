import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

model_path = 'model.pt'
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()  # Set model to evaluation mode

labels = ["Dry", "Normal", "Oily"]

def predict_skin_type(image):
  """
  Preprocesses an image and uses the model to predict skin type.
  """
  # Preprocess the image (resize, normalize, etc.)
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  processed_image = transform(image)
  processed_image = processed_image.unsqueeze(0)
  
  # Make prediction
  with torch.no_grad():
      output = model(processed_image)
      prediction = torch.argmax(output, dim=1).item()
  
  return labels[prediction]

# configure Streamlit app

st.set_page_config(page_title="Skin Type Prediction App")  

with st.sidebar:
  st.header("Instructions", divider='blue')
  st.write("""
      - Upload a close-up image of the face ensuring clear visibility of the skin.
      - Avoid images with excessive makeup or extreme lighting conditions.
      - For best results, use portrait-oriented images with good resolution.
  """)
  st.subheader("Model Information", divider='blue')
  st.write("This app utilizes a fine-tuned ResNet50 model trained on a comprehensive skin dataset.")

st.title("Skin Type Prediction 👩🏻‍⚕️")
uploaded_file = st.file_uploader("Choose an image..", type="jpg")

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption="Uploaded Image", use_column_width=True)

  prediction = predict_skin_type(image)
  st.success(f"Predicted Skin Type: {prediction}")