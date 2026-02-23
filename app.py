import streamlit as st
import torch as nn
from model import Generator
import numpy as np 

@st.cache_resource
def get_device_and_model():
    device = nn.device("cuda" if nn.cuda.is_available() else "cpu")
    Z_DIM = 100        
    CHANNELS_IMG = 3    
    FEATURES_GEN = 64   
    
    model = Generator(z_dim=Z_DIM, img_channels=CHANNELS_IMG, gen_features=FEATURES_GEN) 
    model.load_state_dict(nn.load("generator_new.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = get_device_and_model()
st.title("Anime Face Generator")
st.write(f"Hardware Status: **{device}**")


if st.button('Generate Image'):
    random_seed = np.random.randint(0, 10**6)
    nn.manual_seed(random_seed)
    noise = nn.randn(1, 100, 1, 1, device=device) 
    
    with nn.no_grad():
        generated_tensor = model(noise)

        img_array = generated_tensor.squeeze().cpu().detach().numpy()
        img_array = (img_array + 1) / 2.0
        img_array = np.transpose(img_array, (1, 2, 0)) 
        
    st.image(img_array, caption=f"Generated from seed {random_seed}", use_container_width=True)