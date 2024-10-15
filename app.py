import streamlit as st
from transformers import XLNetTokenizer, XLNetModel
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import io
import numpy as np
import os

class TextEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.transformer = XLNetModel.from_pretrained("xlnet-base-cased")
  def forward(self, input_ids, token_type_ids, attention_mask):
    hidden = self.transformer(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask).last_hidden_state
    context = hidden.mean(dim = 1)
    context = context.view(*context.shape, 1, 1)
    return context

class Generator(nn.Module):
  def __init__(self, nz = 100, ngf = 64, nt = 768, nc = 3):
    super().__init__()
    self.layer1 = nn.Sequential(
        nn.ConvTranspose2d(nz+nt, ngf*8, 4, 1, 0, bias = False),
        nn.BatchNorm2d(ngf*8)
    )

    self.layer2 = nn.Sequential(
        nn.Conv2d(ngf*8, ngf*2, 1, 1),
        nn.Dropout2d(inplace = True),
        nn.BatchNorm2d(ngf*2),
        nn.ReLU(os.truncate)
    )

    self.layer3 = nn.Sequential(
        nn.Conv2d(ngf*2, ngf*2, 3,1, 1),
        nn.Dropout2d(inplace = True),
        nn.BatchNorm2d(ngf*2),
        nn.ReLU(os.truncate)
    )

    self.layer4 = nn.Sequential(
        nn.Conv2d(ngf*2, ngf*8, 3,1, 1),
        nn.Dropout2d(inplace = True),
        nn.BatchNorm2d(ngf*8),
        nn.ReLU(os.truncate)
    )

    self.layer5 = nn.Sequential(
        nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias = False),
        nn.BatchNorm2d(ngf*4),
        nn.ReLU(True)
    )

    self.layer6 = nn.Sequential(
        nn.Conv2d(ngf*4, ngf, 1, 1),
        nn.Dropout2d(inplace = True),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True)
    )

    self.layer7 = nn.Sequential(
        nn.Conv2d(ngf, ngf, 3, 1, 1),
        nn.Dropout2d(inplace = True),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True)
    )

    self.layer8 = nn.Sequential(
        nn.Conv2d(ngf, ngf*4, 3, 1, 1),
        nn.Dropout2d(inplace = True),
        nn.BatchNorm2d(ngf*4),
        nn.ReLU(True)
    )

    self.layer9 = nn.Sequential(
        nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias = False),
        nn.BatchNorm2d(ngf*2),
        nn.ReLU(True)
    )

    self.layer10 = nn.Sequential(
        nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias = False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True)
    )

    self.layer11 = nn.Sequential(
        nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias = False),
        nn.Tanh()
    )
  def forward(self, noise, encoded_text):
    # encoded_text = encoded_text.view(noise.shape[0], -1)  #please delete this
    # batch_size = noise.size(0)
    # encoded_text = encoded_text.view(batch_size, -1, 1, 1)
    x = torch.cat([noise,encoded_text],dim=1)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)
    x = self.layer7(x)
    x = self.layer8(x)
    x = self.layer9(x)
    x = self.layer10(x)
    x = self.layer11(x)
    return x



model_path = "./checkpoint.pth"
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
text_encoder = XLNetModel.from_pretrained('xlnet-base-cased')
model = Generator()
model_state_dict = torch.load(model_path, map_location="cpu")
generator = model_state_dict['models']['generator']
model.load_state_dict(generator)

text_encoder.to("cpu")
model.to("cpu")

model.eval()

# Streamlit app design
st.set_page_config(page_title="Flower Image Generator", layout="wide")

# Sidebar for user input
st.sidebar.title("Flower Image Generator")
text_input = st.sidebar.text_area("Enter a flower-related description", "A beautiful red rose")

# Main section
st.title("Flower Image Generator")
st.write("Enter a description of a flower to generate an image based on your input!")


def generate_image(enc_text):
    noise = torch.randn((1, 100, 1, 1), device="cpu")
    with torch.no_grad():
        generated_image = model(noise, enc_text).detach().squeeze().cpu()
    gen_image_np = generated_image.numpy()
    gen_image_np = np.transpose(gen_image_np, (1, 2, 0))  # Change from CHW to HWC
    gen_image_np = (gen_image_np - gen_image_np.min()) / (gen_image_np.max() - gen_image_np.min())  # Normalize to [0, 1]
    gen_image_np = (gen_image_np * 255).astype(np.uint8)  
    return gen_image_np

def encode_text(text):
    text_encoder = TextEncoder()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    encoded_text = text_encoder(**inputs)
    # inputs = {key: value.to("cpu") for key, value in inputs.items()}
    return encoded_text

# When the user clicks 'Generate'
if st.sidebar.button("Generate Image"):
    if text_input:
        # Encode the input text
        encoded_text = encode_text(text_input)

        # Generate image from encoded text
        gen_image = generate_image(encoded_text)

        # Display generated image
        st.image(gen_image, caption="Generated Flower Image")

    else:
        st.error("Please enter a text prompt to generate a flower image.")
