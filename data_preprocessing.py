import pandas as pd
import numpy as np
import cv2

def load_data(file_path):
    """
    Load data from a pickle file. 
    """
    return pd.read_pickle(file_path)

def resize_wafer_map(wafer_map, target_size=(32, 32)):
    """
    Resize the wafer map to the target size using cubic interpolation and normalize to [-1, 1].
    """
    wafer_map_array = np.array(wafer_map)
    resized_wafer_map = cv2.resize(wafer_map_array, target_size, interpolation=cv2.INTER_CUBIC)
    normalized_wafer_map = (resized_wafer_map / 127.5) - 1  # Normalize to [-1, 1]
    return normalized_wafer_map

def find_dim(wafer_map):
    """
    Find the dimensions of the wafer map.
    """
    return wafer_map.shape

def preprocess_data(data):
    # Drop unnecessary columns
    columns_to_drop = ['waferIndex', 'dieSize', 'lotName']
    data = data.drop(columns=columns_to_drop)

    # Create mappings for failureType and trainTestLabel
    mapping_type = {'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3, 'Loc': 4,
                    'Random': 5, 'Scratch': 6, 'Near-full': 7, 'none': 8}
    mapping_traintest = {'Training': 1, 'Test': 2}
    
    # Drop rows with failure types not in the mapping
    data = data[data['failureType'].isin(mapping_type.keys())]

    # Apply the mapping for failureType and trainTestLabel
    data['failureType'] = data['failureType'].map(mapping_type)
    data['trainTestLabel'] = data['trainTestLabel'].map(mapping_traintest)

    # Resize and normalize wafer maps
    data['waferMap'] = data['waferMap'].apply(resize_wafer_map)
    
    return data

def save_data(data, file_path):
    """
    Save data to a pickle file.
    """
    data.to_pickle(file_path)

def print_column_names(data, dataset_name):
    print(f"\nColumns in {dataset_name} dataset:")
    print(data.columns.tolist())

# Example usage
if __name__ == "__main__":
    input_file = "/scratch/general/vast/u1475870/wafer_project/data/WM811K.pkl"
    output_file = "/scratch/general/vast/u1475870/wafer_project/data/WM811K_preprocessed.pkl"

    # Load data
    df = load_data(input_file)

    # Preprocess data
    preprocessed_df = preprocess_data(df)

    # Print column names for preprocessed data
    print_column_names(preprocessed_df, "preprocessed")

    # Save preprocessed data
    save_data(preprocessed_df, output_file)

    # Extract and save data for training and testing
    training_data = preprocessed_df[preprocessed_df['trainTestLabel'] == 1]
    testing_data = preprocessed_df[preprocessed_df['trainTestLabel'] == 2]

    # Add print statements to show the number of samples
    print(f"\nNumber of training samples: {len(training_data)}")
    print(f"Number of testing samples: {len(testing_data)}")

    # Add print statements to show the distribution of failure types in each set
    print("\nDistribution of failure types in training data:")
    print(training_data['failureType'].value_counts())
    print("\nDistribution of failure types in testing data:")
    print(testing_data['failureType'].value_counts())

    # Drop the trainTestLabel column
    training_data = training_data.drop(columns=['trainTestLabel'])
    testing_data = testing_data.drop(columns=['trainTestLabel'])

    # Print column names for training and testing data
    print_column_names(training_data, "training")
    print_column_names(testing_data, "testing")

    # Save training and testing data
    save_data(training_data, "/scratch/general/vast/u1475870/wafer_project/data/WM811K_training.pkl")
    save_data(testing_data, "/scratch/general/vast/u1475870/wafer_project/data/WM811K_testing.pkl")

    # Print the shape of waferMap and failureType columns
    print("\nShape of waferMap:")
    print(preprocessed_df['waferMap'].iloc[0].shape)
    print("\nUnique values in failureType:")
    print(preprocessed_df['failureType'].unique())

    # Print the head of the training dataframe
    print("\nHead of the training dataframe:")
    print(training_data.head())