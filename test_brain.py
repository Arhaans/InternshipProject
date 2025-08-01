from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
import torch
import os
import sys  # For sys.exit()

# --- Configuration ---
MODEL_ID = "google/medgemma-4b-it"
LOCAL_IMAGE_FILENAME = "axial_slice.png"
FALLBACK_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
MAX_NEW_TOKENS = 256

# --- 1. Load Model and Processor ---
print("Loading model and processor...")
try:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading model or processor: {e}")
    print("Please ensure you have 'transformers', 'accelerate', 'pillow', 'requests', and 'torch' installed.")
    sys.exit(1)

# --- 2. Load the Image ---
image = None
if os.path.exists(LOCAL_IMAGE_FILENAME):
    print(f"Opening local image: {LOCAL_IMAGE_FILENAME}")
    try:
        image = Image.open(LOCAL_IMAGE_FILENAME).convert("RGB")
    except Exception as e:
        print(f"Error opening local image '{LOCAL_IMAGE_FILENAME}': {e}")
        print("Attempting to download fallback image.")
if image is None:
    print(f"Local image '{LOCAL_IMAGE_FILENAME}' not found or could not be opened. Downloading fallback image.")
    try:
        response = requests.get(FALLBACK_IMAGE_URL, headers={"User-Agent": "Mozilla/5.0"}, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")
        print("Sample image downloaded and opened successfully.")
    except Exception as e:
        print(f"Error downloading or processing fallback image: {e}")
        sys.exit(1)

# --- 3. Construct Messages in Chat Format ---
messages = [
    {
        "role": "system",
        "content": [{
            "type": "text",
            "text": (
                "You are a board‐certified neuroradiologist with deep expertise in brain imaging. "
                "Your task is to examine the provided axial MRI slice of the brain and deliver a concise, "
                "clinically oriented report. Highlight any anatomical findings, signal abnormalities, volumetric "
                "changes (e.g., medial temporal lobe atrophy), white‐matter lesions, or other markers. "
                "Based on these imaging features, give your best‐supported assumptions about the patient's risk or "
                "early indicators of Alzheimer’s disease or other neurodegenerative processes. "
                "Use precise medical terminology and, where appropriate, reference standardized scoring or grading."
            )
        }]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please review this axial slice image and provide your findings, assumptions, and any Alzheimer’s‐related assessment."},
            {"type": "image", "image": image}
        ]
    }
]

# --- 4. Preprocess Inputs ---
print("Preparing inputs...")
try:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    print("Inputs prepared.")
except Exception as e:
    print(f"Error preparing inputs: {e}")
    sys.exit(1)

input_len = inputs["input_ids"].shape[-1]

# --- 5. Generate Output ---
print("Generating report...")
try:
    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id
        )
        new_tokens = gen[0, input_len:]
        report = processor.decode(new_tokens, skip_special_tokens=True).strip()
    print("Generation complete.")
except Exception as e:
    print(f"Error during generation: {e}")
    sys.exit(1)

# --- 6. Display Result ---
print("\n--- Neuroradiology Report ---")
if report:
    print(report)
else:
    print("No meaningful output generated. Try increasing MAX_NEW_TOKENS or adjusting sampling parameters.")
