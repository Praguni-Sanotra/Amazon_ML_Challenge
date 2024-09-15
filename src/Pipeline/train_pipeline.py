import os
from src.components.data_transformation import transform_text

# Constants for file paths
INPUT_CSV_PATH = "dataset/train.csv"  # Path to the training dataset CSV file
OUTPUT_CSV_PATH = "output/train_output.csv"  # Path to save the processed output

def main():
    """
    Main function to run the training pipeline.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run the data transformation process for training data
    transform_text(INPUT_CSV_PATH, OUTPUT_CSV_PATH)

    print(f"Training pipeline completed. Output saved to: {OUTPUT_CSV_PATH}")

if __name__ == "_main_":
    main()