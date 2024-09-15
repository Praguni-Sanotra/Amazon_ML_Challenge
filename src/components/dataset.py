import sys
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Print debug information
print("Python path:", sys.path)
print("Current working directory:", os.getcwd())
print("Project root:", project_root)

# Try to import from utils directly
try:
    from src.utils import parse_string  # Adjust this to your actual project structure
    print("Successfully imported parse_string from src.utils")
except ImportError as e:
    print(f"Failed to import parse_string from src.utils: {e}")
    # Define a dummy parse_string function as a fallback
    def parse_string(s):
        print("Warning: Using dummy parse_string function")
        return 0, "unit"

# Try to import constants
try:
    import src.constants as constants  # Adjust this to your actual project structure
    print("Successfully imported constants from src.constants")
except ImportError as e:
    print(f"Failed to import constants: {e}")
    # Define dummy constants as fallback
    class constants:
        allowed_units = ["unit"]

class EntityExtractionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Verify if the file exists before loading it
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        
        # Verify if the image directory exists
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.basename(self.data.iloc[idx]['image_link'])
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError as e:
            print(f"Image not found: {img_path}. Skipping this record.")
            return None
        
        if self.transform:
            image = self.transform(image)
        
        entity_name = self.data.iloc[idx]['entity_name']
        entity_value = self.data.iloc[idx]['entity_value']
        
        # Parse the entity value
        try:
            number, unit = parse_string(entity_value)
            parsed_value = torch.tensor([number, list(constants.allowed_units).index(unit)])
        except (ValueError, IndexError) as e:
            print(f"Error parsing entity value: {entity_value}. Error: {e}")
            parsed_value = torch.tensor([-1, -1])  # Use a special token for invalid/empty values
        
        return {
            'image': image,
            'entity_name': entity_name,
            'entity_value': entity_value,
            'parsed_value': parsed_value,
            'index': self.data.iloc[idx]['index']
        }

def get_dataloader(csv_file, img_dir, batch_size=32, shuffle=True, num_workers=4):
    dataset = EntityExtractionDataset(csv_file, img_dir)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

if __name__ == "__main__":
    # Use actual paths to your CSV file and image directory
    csv_file = '/Users/pragunisanotra/Desktop/Amazon ML Challenge/notebook/dataset/train.csv'  # Update this path
    img_dir = '/Users/pragunisanotra/Desktop/Amazon ML Challenge/data/images'  # Update this path
    
    print("Creating DataLoader...")
    dataloader = get_dataloader(csv_file, img_dir)
    
    for batch in dataloader:
        print(batch)  # Debug: print each batch to verify correctness
    print("Dataset module loaded successfully")
