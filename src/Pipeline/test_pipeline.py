import os
import sys

# Correct import statement
from src.components.data_ingestion import DataInjection  # Ensure the class name is correct within data_ingestion.py

def main():
    # Define paths
    test_data_path = '/Users/pragunisanotra/Desktop/Amazon ML Challenge/notebook/dataset/test.csv'
    image_download_folder = '/Users/pragunisanotra/Desktop/Amazon ML Challenge/images'
    
    # Validate paths
    if not os.path.exists(test_data_path):
        print(f"Error: Test data path '{test_data_path}' does not exist.")
        sys.exit(1)
    
    if not os.path.isdir(image_download_folder):
        print(f"Creating image download folder at '{image_download_folder}'.")
        os.makedirs(image_download_folder, exist_ok=True)

    # Initialize DataInjection
    print("Initializing Data Injection...")
    data_injector = DataInjection(test_data_path, image_download_folder)
    
    # Inject data
    print("Starting data injection process...")
    test_data = data_injector.inject_data()

    # Optionally, save the processed data
    output_data_path = 'output/processed_test_data.csv'
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    test_data.to_csv(output_data_path, index=False)
    print(f"Processed test data saved to {output_data_path}")

if __name__ == "_main_":
    # Ensure correct entry point for script execution
    main()