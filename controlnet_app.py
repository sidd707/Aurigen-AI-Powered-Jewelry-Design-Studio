import streamlit as st
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from PIL import Image
import os
import logging
import time
import cv2
import numpy as np

import warnings

warnings.filterwarnings("ignore")

#######################
# Page configuration
st.set_page_config(
    page_title="Aurigen - Jewelry Design Studio",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Setup logging
logging.basicConfig(level=logging.INFO)

#######################
# CSS Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&display=swap');
    
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
        font-family: 'Playfair Display', serif;
        font-size: 22px;
    }
    
    [data-testid="block-container"] {
        background-color: #2a2a2a;
        padding: 2rem 3rem;
        border-radius: 10px;
        margin-top: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    h1 { font-size: 48px !important; }
    button { font-size: 20px; }
    .stTextArea, .stTextInput { font-size: 20px; }
    .stAlert { font-size: 20px; }
    </style>
    """,
    unsafe_allow_html=True,
)

#######################
# Model Setup
@st.cache_resource
def load_model():
    checkpoint_dir = "fine-tuned-weights"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ControlNet and Pipeline
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to(device)

    # Load fine-tuned weights
    unet_checkpoint_path = os.path.join(checkpoint_dir, "unet_epoch_3.pth")
    if os.path.exists(unet_checkpoint_path):
        pipe.unet.load_state_dict(torch.load(unet_checkpoint_path, map_location=device))
    return pipe

pipe = load_model()

#######################
# Image Processing
def process_image(image_input, detect_edges=True):
    if image_input is None:
        # Create a white background image as default
        image = Image.new("RGB", (1024, 1024), (255, 255, 255))
    elif hasattr(image_input, 'read'):
        # It's a file-like object from st.file_uploader
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        # It's already a PIL Image (e.g., from refinement)
        image = image_input.convert("RGB")
    else:
        raise ValueError("Invalid image input")

    if detect_edges:
        # Apply Canny edge detection
        image_np = np.array(image)
        canny = cv2.Canny(image_np, 100, 200)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        image = Image.fromarray(canny)
    # Resize to 1024x1024
    image = image.resize((1024, 1024))
    return image

#######################
# Sidebar Controls
with st.sidebar:
    st.title("💎 Aurigen Studio")

    # Generation Parameters
    prompt = st.text_area(
        "Design Description",
        "A luxurious diamond necklace with intricate gold filigree, studio lighting, 8k resolution",
    )
    negative_prompt = st.text_area(
        "Exclusions",
        "blurry, low quality, plastic, unrealistic proportions, poor lighting",
    )

    # ControlNet Settings
    st.markdown("---")
    st.markdown("**Design Guidance**")
    reference_image = st.file_uploader(
        "Upload Sketch/Reference", type=["png", "jpg", "jpeg"]
    )
    conditioning_scale = st.slider("Guidance Strength", 0.0, 2.0, 1.2)
    detect_edges = st.checkbox("Auto-detect Edges", True)

    # Generation Settings
    st.markdown("---")
    num_images = st.slider("Number of Designs", 1, 4, 2)
    num_inference_steps = st.slider("Refinement Steps", 20, 100, 45)
    guidance_scale = st.slider("Prompt Adherence", 1.0, 20.0, 7.5)

#######################
# Main Interface
col = st.columns((1, 6, 1))
with col[1]:
    st.markdown("## Craft Your Jewelry Masterpiece")

    # Generation Controls
    col1, col2 = st.columns([1, 3])
    with col1:
        generate_btn = st.button("✨ Generate Designs", type="primary")
    with col2:
        if "generated_images" in st.session_state:
            selected_refinement = st.selectbox(
                "Select Design to Enhance",
                options=list(range(len(st.session_state.generated_images))),
                format_func=lambda x: f"Design {x+1}",
            )

    # Image Display and Refinement
    if "generated_images" in st.session_state:
        st.markdown("---")
        cols = st.columns(2)
        for idx, img in enumerate(st.session_state.generated_images):
            with cols[idx % 2]:
                st.image(img, use_container_width=True, caption=f"Design {idx+1}")

                # Refinement Options
                with st.expander(f"Enhance Design {idx+1}"):
                    refine_prompt = st.text_input(
                        "Modification Request",
                        key=f"refine_{idx}",
                        placeholder="Make the gems more emerald-colored",
                    )
                    if st.button("Apply Changes", key=f"btn_refine_{idx}"):
                        with st.spinner("Refining design..."):
                            try:
                                # Process existing image for ControlNet
                                control_image = process_image(img, detect_edges)

                                # Run refinement
                                refined = pipe(
                                    prompt=refine_prompt or prompt,
                                    negative_prompt=negative_prompt,
                                    image=control_image,
                                    num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale,
                                    controlnet_conditioning_scale=conditioning_scale,
                                ).images[0]

                                # Update the design
                                st.session_state.generated_images[idx] = refined
                                st.rerun()
                            except Exception as e:
                                st.error(f"Refinement failed: {str(e)}")

    # Initial Generation
    if generate_btn:
        start_time = time.time()
        with st.spinner("Crafting your designs..."):
            try:
                control_image = process_image(reference_image, detect_edges)

                images = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=control_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images,
                    controlnet_conditioning_scale=conditioning_scale,
                ).images

                st.session_state.generated_images = images
                st.success(
                    f"Created {len(images)} designs in {time.time()-start_time:.1f}s"
                )
                st.rerun()

            except Exception as e:
                st.error(f"Generation failed: {str(e)}")

    # Instructions Section
    with st.expander("🛠️ Studio Guide", expanded=True):
        st.markdown(
            """
        **Design Studio Features:**
        - 🖼️ **Reference Upload**: Add sketches or inspiration images (defaults to white background if none uploaded)
        - 🎛️ **Precision Control**: Adjust guidance strength and edge detection
        - ✨ **Iterative Refinement**: Modify specific designs after generation
        - 💎 **Material Focus**: Use terms like "24k gold", "flawless diamonds", "vintage engraving"
        
        **Pro Tips:**
        1. Start with broad concepts, then refine details
        2. Use "enhance" for color/material adjustments
        3. Combine reference images with textual descriptions
        4. Experiment with guidance strength (0.8-1.5 recommended)
        """
        )

#######################
# System Information
st.sidebar.markdown("---")
st.sidebar.markdown("**System Status**")
st.sidebar.write(f"Device: {'GPU 🔥' if torch.cuda.is_available() else 'CPU ⚙️'}")
st.sidebar.write(f"Model: SDXL 1.0 + ControlNet")
st.sidebar.write(f"Precision: {pipe.dtype}")