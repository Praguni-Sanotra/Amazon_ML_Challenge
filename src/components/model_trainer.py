import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from src.utils import download_images  # Assuming this function is available for downloading images
from src.constants import allowed_units  # Assuming ALLOWED_UNITS is defined in constants.py

# Define the image size and batch size
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

def load_data(csv_file):
    data = pd.read_csv(csv_file)
    return data

def prepare_images(data, image_dir):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    image_paths = []
    for idx, row in data.iterrows():
        index_value = row.get('index') if 'index' in data.columns else row.get('ID')
        if index_value is None:
            raise KeyError("Neither 'index' nor 'ID' column found in the dataset.")

        image_url = row['image_link']
        image_path = os.path.join(image_dir, f"{index_value}.jpg")
        download_images(image_url, image_path)
        image_paths.append(image_path)
    
    return image_paths

def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(allowed_units), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_images, val_images):
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_dataframe(
        dataframe=train_images,
        x_col='file_path',
        y_col='entity_value',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_gen = datagen.flow_from_dataframe(
        dataframe=val_images,
        x_col='file_path',
        y_col='entity_value',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    model.fit(train_gen, validation_data=val_gen, epochs=10)
    return model

def generate_predictions(model, test_data, image_dir):
    predictions = []
    for _, row in test_data.iterrows():
        index_value = row.get('index') if 'index' in test_data.columns else row.get('ID')
        if index_value is None:
            raise KeyError("Neither 'index' nor 'ID' column found in the dataset.")

        image_path = os.path.join(image_dir, f"{index_value}.jpg")
        if os.path.exists(image_path):
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(img_array)
            
            predicted_unit = allowed_units[np.argmax(pred[0])]
            predicted_value = np.random.uniform(1, 100)
            predictions.append(f"{predicted_value:.2f} {predicted_unit}")
        else:
            predictions.append("")
    
    output_df = pd.DataFrame({
        'index': test_data['index'] if 'index' in test_data.columns else test_data['ID'],
        'prediction': predictions
    })
    return output_df

def save_predictions(output_df, output_file):
    output_df.to_csv(output_file, index=False)

if __name__ == '_main_':
    parser = argparse.ArgumentParser(description='Model Trainer')
    parser.add_argument('--train_csv', type=str, help='Path to train CSV file', required=True)
    parser.add_argument('--test_csv', type=str, help='Path to test CSV file', required=True)
    parser.add_argument('--output_file', type=str, help='Path to save predictions', required=True)
    parser.add_argument('--image_dir', type=str, default='images', help='Directory to save images')
    
    args = parser.parse_args()

    # Load the training and test data
    train_data = load_data(args.train_csv)
    test_data = load_data(args.test_csv)

    # Prepare images
    train_data['file_path'] = prepare_images(train_data, args.image_dir)
    test_data['file_path'] = prepare_images(test_data, args.image_dir)

    # Split data into train and validation sets
    train_images, val_images = train_test_split(train_data, test_size=0.2, random_state=42)

    # Build and train the model
    input_shape = IMAGE_SIZE + (3,)
    model = build_model(input_shape)
    model = train_model(model, train_images, val_images)

    # Generate predictions for the test dataset
    output_df = generate_predictions(model, test_data, args.image_dir)

    # Save predictions to CSV
    save_predictions(output_df, args.output_file)