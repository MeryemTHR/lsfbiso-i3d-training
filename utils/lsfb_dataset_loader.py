import pandas as pd
import os
import json

def load_lsfb_dataset(data_dir, verbose=True):
    '''
    Read the LSFB dataset using instances.csv and train/test JSON splits.

    PARAMETERS:
      data_dir : The path to LSFB corpus folder containing instances.csv and metadata/splits/train.json
      verbose : Whether to print loading messages

    OUTPUT:
      dataset : A dataframe containing 4 columns (label, label_nbr, path, subset)
    '''
    # Load instances.csv
    csv_path = os.path.join(data_dir, 'instances.csv')
    dataset = pd.read_csv(csv_path)

    # Load train/test splits
    metadata_splits = os.path.join(data_dir, 'metadata', 'splits')
    with open(os.path.join(metadata_splits, 'train.json'), 'r') as f:
        train_files = json.load(f)
    with open(os.path.join(metadata_splits, 'test.json'), 'r') as f:
        test_files = json.load(f)

    # Add 'subset' column based on filenames
    subset_column = []
    for video_name in dataset['video']:
        if video_name in train_files:
            subset_column.append('train')
        elif video_name in test_files:
            subset_column.append('test')
        else:
            subset_column.append('unknown')  # (if needed)

    dataset['subset'] = subset_column

    # Build full path for each video
    dataset['path'] = dataset['video'].apply(lambda x: os.path.join(data_dir, 'videos', x))

    # Keep only necessary columns
    dataset = dataset[['label', 'label_nbr', 'path', 'subset']]

    if verbose:
        print(f"Dataset loaded: {len(dataset)} samples.")

    return dataset
