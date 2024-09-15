import os
import sys
import pandas as pd
from tqdm import tqdm
from src.utils import download_images
from src.constants import allowed_units
import logging

# Ensure src directory is in the system path for imports
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Ensure the 'artifacts' directory exists
os.makedirs('artifacts', exist_ok=True)

# Setup logging for artifacts
logging.basicConfig(filename='artifacts/data_injection.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# Define data injection class
class DataInjection:
    def __init__(self, test_data_path, image_download_folder, artifacts_folder):
        self.test_data_path = test_data_path
        self.image_download_folder = image_download_folder
        self.artifacts_folder = artifacts_folder

        # Create necessary directories
        os.makedirs(self.image_download_folder, exist_ok=True)
        os.makedirs(self.artifacts_folder, exist_ok=True)

    def load_data(self):
        """
        Load the test data into a pandas DataFrame.
        """
        logging.info("Loading test data...")
        print("Loading test data...")
        self.test_df = pd.read_csv(self.test_data_path)
        self.test_df['prediction'] = ""  # Initialize prediction column
        print(f"Test data loaded with {len(self.test_df)} rows.")
        logging.info(f"Test data loaded with {len(self.test_df)} rows.")
        
        # Save initial data as artifact
        initial_data_path = os.path.join(self.artifacts_folder, 'initial_test_data.csv')
        self.test_df.to_csv(initial_data_path, index=False)
        logging.info(f"Initial test data saved as {initial_data_path}.")
        return self.test_df
    
    def download_images(self):
        """
        Download images from the URLs present in the test dataset.
        """
        print(f"Downloading images to {self.image_download_folder}...")
        logging.info(f"Downloading images to {self.image_download_folder}...")

        failed_downloads = []

        for idx, row in tqdm(self.test_df.iterrows(), total=len(self.test_df), desc="Downloading Images"):
            image_url = row['image_link']
            index = row['index']

            # Download the image using the download_images function
            try:
                image_path = download_images([image_url], self.image_download_folder)
                self.test_df.at[idx, 'image_path'] = image_path  # Store image path for future use
            except Exception as e:
                print(f"Failed to download image for index {index}: {e}")
                logging.error(f"Failed to download image for index {index}: {e}")
                self.test_df.at[idx, 'image_path'] = None  # Mark failed downloads
                failed_downloads.append(index)

        # Save image paths as artifact
        image_paths_data_path = os.path.join(self.artifacts_folder, 'image_paths_data.csv')
        self.test_df.to_csv(image_paths_data_path, index=False)
        logging.info(f"Image paths data saved as {image_paths_data_path}.")
        if failed_downloads:
            logging.warning(f"Failed to download images for indices: {failed_downloads}")

    def preprocess_data(self):
        """
        Preprocess the test data and image paths for prediction.
        You can add more preprocessing steps here as needed (e.g., resizing images).
        """
        print("Preprocessing data...")
        logging.info("Preprocessing data...")
        # Add more preprocessing if required, such as feature extraction from images

        # Save preprocessed data as artifact
        preprocessed_data_path = os.path.join(self.artifacts_folder, 'preprocessed_test_data.csv')
        self.test_df.to_csv(preprocessed_data_path, index=False)
        logging.info(f"Preprocessed test data saved as {preprocessed_data_path}.")
        
    def inject_data(self):
        """
        Perform data loading, image downloading, and preprocessing.
        """
        self.load_data()
        self.download_images()
        self.preprocess_data()
        print("Data injection complete.")
        logging.info("Data injection complete.")
        return self.test_df

if __name__ == "__main__":
    # Define paths
    test_data_path = '/Users/pragunisanotra/Desktop/Amazon ML Challenge/notebook/dataset/test.csv'
    image_download_folder = '/Users/pragunisanotra/Desktop/Amazon ML Challenge/images'
    artifacts_folder = 'artifacts'  # Folder for storing artifacts

    # Initialize DataInjection
    data_injector = DataInjection(test_data_path, image_download_folder, artifacts_folder)
    
    # Inject data
    test_data = data_injector.inject_data()

    # Optionally, save the final processed data as a final artifact
    output_data_path = 'artifacts/final_processed_test_data.csv'
    test_data.to_csv(output_data_path, index=False)
    print(f"Final processed test data saved to {output_data_path}")
    logging.info(f"Final processed test data saved to {output_data_path}")