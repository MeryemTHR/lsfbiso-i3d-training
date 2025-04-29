import pandas as pd
import os
import json

def load_lsfb_dataset(data_dir, verbose=True):
    '''
    Read the LSFB dataset using instances.csv (id) and train/test JSON splits.

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

    # Add 'subset' column based on 'id'
    subset_column = []
    for video_name in dataset['id']:
        filename = video_name + '.mp4'
        if filename in train_files:
            subset_column.append('train')
        elif filename in test_files:
            subset_column.append('test')
        else:
            subset_column.append('unknown')

    dataset['subset'] = subset_column

    # Build full path for each video
    dataset['path'] = dataset['id'].apply(lambda x: os.path.join(data_dir, 'videos', x + '.mp4'))

    # Create numeric labels
    label_map = {label: idx for idx, label in enumerate(dataset['sign'].unique())}
    dataset['label_nbr'] = dataset['sign'].map(label_map)

    # Keep only necessary columns
    dataset = dataset[['sign', 'label_nbr', 'path', 'subset']]
    dataset.rename(columns={'sign': 'label'}, inplace=True)

    if verbose:
        print(f"Dataset loaded: {len(dataset)} samples.")

    return dataset
