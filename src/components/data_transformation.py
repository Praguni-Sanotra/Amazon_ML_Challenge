import os
import cv2
import pytesseract
import pandas as pd
from tqdm import tqdm
from src.utils import download_images

# Constants and configurations
IMAGE_DIR = "images"  # Directory to save downloaded images
ALLOWED_UNITS = {
    'item_weight': ['gram', 'kilogram', 'ounce', 'pound'],
    'item_volume': ['millilitre', 'litre', 'fluid_ounce'],
}

def download_images_wrapper(image_links):
    """
    Download images and save them to IMAGE_DIR.
    """
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    
    for i, link in enumerate(tqdm(image_links, desc="Downloading images")):
        try:
            image_path = os.path.join(IMAGE_DIR, f"image_{i}.jpg")
            download_images(link, image_path)
        except Exception as e:
            print(f"Failed to download image {link}: {e}")

def extract_text_from_image(image_path):
    """
    Extract text from an image using OCR.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        extracted_text = pytesseract.image_to_string(gray_image)
        return extracted_text
    except Exception as e:
        print(f"Error in extracting text from {image_path}: {e}")
        return ""

def parse_entity_value(extracted_text, entity_name):
    """
    Parse the extracted text to find the relevant entity value and unit.
    """
    allowed_units = ALLOWED_UNITS.get(entity_name, [])
    words = extracted_text.split()
    
    for i, word in enumerate(words):
        try:
            value = float(word)
            if i + 1 < len(words) and words[i + 1] in allowed_units:
                return f"{value} {words[i + 1]}"
        except ValueError:
            continue
    
    return ""

def transform_text(input_csv, output_csv):
    """
    Process the input CSV to extract and transform text data from images.
    """
    data = pd.read_csv(input_csv)
    download_images_wrapper(data['image_link'].tolist())
    
    predictions = []
    
    for idx, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing images"):
        image_path = os.path.join(IMAGE_DIR, f"image_{idx}.jpg")
        extracted_text = extract_text_from_image(image_path)
        entity_value = parse_entity_value(extracted_text, row['entity_name'])
        predictions.append(entity_value)
    
    output_df = pd.DataFrame({
        'index': data['index'],
        'prediction': predictions
    })
    output_df.to_csv(output_csv, index=False)
    print(f"Output saved to {output_csv}")

def transform_image(image_path):
    """
    Extract text from a single image for other purposes.
    """
    return extract_text_from_image(image_path)