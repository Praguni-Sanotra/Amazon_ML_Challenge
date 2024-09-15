import os
import re
import requests
import pandas as pd
import multiprocessing
import time
from pathlib import Path
from functools import partial
import urllib
from PIL import Image

# Attempt to import tqdm; handle the case where it's not installed
try:
    from tqdm import tqdm
except ImportError:
    print("Module 'tqdm' is not installed. Install it by running 'pip install tqdm'")
    tqdm = None  # Set tqdm to None to avoid breaking code if not installed

# Add the correct parent directory of both utils.py and constants.py
import sys
from pathlib import Path

# Ensure the directory containing both files is in the path
src_path = Path(__file__).resolve().parent
sys.path.insert(0, str(src_path))

import constants  # Absolute import, assuming constants.py is in the same directory

# Function to correct common unit mistakes
def common_mistake(unit):
    if unit in constants.ALLOWED_UNITS:
        return unit
    if unit.replace('ter', 'tre') in constants.ALLOWED_UNITS:
        return unit.replace('ter', 'tre')
    if unit.replace('feet', 'foot') in constants.ALLOWED_UNITS:
        return unit.replace('feet', 'foot')
    return unit

# Function to parse a string with value and unit
def parse_string(s):
    s_stripped = "" if s is None or str(s) == 'nan' else s.strip()
    if s_stripped == "":
        return None, None
    pattern = re.compile(r'^-?\d+(\.\d+)?\s+[a-zA-Z\s]+$')
    if not pattern.match(s_stripped):
        raise ValueError(f"Invalid format in {s}")
    parts = s_stripped.split(maxsplit=1)
    number = float(parts[0])
    unit = common_mistake(parts[1])
    if unit not in constants.ALLOWED_UNITS:
        raise ValueError(f"Invalid unit [{unit}] found in {s}. Allowed units: {constants.ALLOWED_UNITS}")
    return number, unit

# Function to create a placeholder image in case of download failure
def create_placeholder_image(image_save_path):
    try:
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        placeholder_image.save(image_save_path)
    except Exception as e:
        print(f"Error creating placeholder image: {e}")

# Function to download a single image with retries
def download_image(image_link, save_folder, retries=3, delay=3):
    if not isinstance(image_link, str):
        return

    filename = Path(urllib.parse.urlparse(image_link).path).name
    image_save_path = os.path.join(save_folder, filename)

    if os.path.exists(image_save_path):
        return  # Skip downloading if the image already exists

    for _ in range(retries):
        try:
            urllib.request.urlretrieve(image_link, image_save_path)
            return
        except Exception as e:
            print(f"Error downloading {image_link}: {e}. Retrying...")
            time.sleep(delay)
    
    # Create a black placeholder image if download fails after retries
    create_placeholder_image(image_save_path)

# Function to download multiple images, with multiprocessing option
def download_images(image_links, download_folder, allow_multiprocessing=True):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    download_image_partial = partial(
        download_image, save_folder=download_folder, retries=3, delay=3)

    if allow_multiprocessing:
        if tqdm is not None:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                list(tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)))
                pool.close()
                pool.join()
        else:
            print("tqdm not available. Downloading without progress bar.")
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                pool.map(download_image_partial, image_links)
                pool.close()
                pool.join()
    else:
        for image_link in (tqdm(image_links, total=len(image_links)) if tqdm is not None else image_links):
            download_image(image_link, save_folder=download_folder, retries=3, delay=3)
