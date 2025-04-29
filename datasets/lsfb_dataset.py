from torch.utils.data import Dataset
from typing import Tuple, Dict
import torch
import cv2
import numpy as np
import random
import os


class LsfbDataset(Dataset):
    """Load the LSFB videos based on a DataFrame containing their paths."""

    def __init__(
        self,
        data,
        padding="loop",
        sequence_label=False,
        one_hot=False,
        transform=None,
        labels=None,
    ):
        """
        PARAMETERS:
        - data: Pandas DataFrame with columns ['label', 'label_nbr', 'path', 'subset']
        - padding: "loop" (default) or "zero" padding to make videos have the same number of frames
        - sequence_label: If True, returns a label for each frame instead of one label per video
        - one_hot: If True, labels are returned as one-hot encoded
        - transform: Transformations to apply on the video frames (resize, normalization, etc.)
        - labels: Optional dictionary mapping class indices to labels
        """
        self.data = data.copy()  # Create a copy to avoid modifying the original
        self.padding = padding
        self.sequence_label = sequence_label
        self.transform = transform
        self.one_hot = one_hot

        # More robust path validation
        valid_videos = []
        total = len(self.data)
        print(f"Validating {total} videos...")
        
        for i, row in self.data.iterrows():
            path = row.get("path", "")
            # Skip if path is missing or invalid
            if not isinstance(path, str) or not path:
                continue
            try:
                if os.path.exists(path) and os.path.getsize(path) > 1000:
                    valid_videos.append(i)
            except (OSError, IOError) as e:
                print(f"Error checking file {path}: {str(e)}")
        
        # Safety check - ensure we have videos
        if len(valid_videos) == 0:
            raise ValueError(f"No valid videos found in the dataset. Check your paths and file permissions.")
            
        # Filter to valid videos only
        self.data = self.data.loc[valid_videos].reset_index(drop=True)
        print(f"Kept {len(self.data)} valid videos out of {total} total.")

        if labels is None:
            self.labels = self._get_label_mapping()
        else:
            self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Handle index out of range
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.data)} items")
            
        item = self.data.iloc[idx]

        # Load video frames
        try:
            capture = cv2.VideoCapture(item["path"])
            video = self.extract_frames(capture)
            
            # Check if video is empty
            if len(video) == 0:
                # Create a dummy frame with zeros as a fallback
                print(f"Warning: Empty video found at {item['path']}")
                video = np.zeros((1, 224, 224, 3), dtype=np.float32)
                
            video_len = len(video)

            # Apply transformations
            if self.transform:
                try:
                    video = self.transform(video)
                except Exception as e:
                    print(f"Error transforming video {item['path']}: {str(e)}")
                    # Return a dummy transformed video
                    video = np.zeros((3, 64, 224, 224), dtype=np.float32)
                    video = torch.from_numpy(video)

            # Ensure the video tensor is float32, not float64
            if isinstance(video, torch.Tensor):
                video = video.float()
            elif isinstance(video, np.ndarray):
                video = torch.from_numpy(video).float()

            # Adjust labels if sequence labeling
            y = int(item["label_nbr"])

            if self.sequence_label:
                if self.padding == "zero":
                    pad_nbr = list(self.labels.keys())[list(self.labels.values()).index("SEQUENCE-PADDING")]
                    pad_len = len(video) - video_len
                    y = np.array([y] * video_len + [pad_nbr] * pad_len)
                elif self.padding == "loop":
                    y = np.array([y] * len(video))

            if self.one_hot:
                nbr_labels = len(self.labels)
                if isinstance(y, int):
                    tmp = np.zeros(nbr_labels)
                    tmp[y] = 1
                else:
                    tmp = np.zeros((nbr_labels, len(video)))
                    for idx, label in enumerate(y):
                        tmp[label][idx] = 1
                y = tmp
                
            return video, y
            
        except Exception as e:
            print(f"Error processing video at index {idx}, path {item.get('path', 'unknown')}: {str(e)}")
            # Return a dummy sample as fallback
            dummy_video = torch.zeros((3, 64, 224, 224), dtype=torch.float32)
            dummy_label = 0 if not self.one_hot else torch.zeros(len(self.labels), dtype=torch.float32)
            return dummy_video, dummy_label

    def _get_label_mapping(self) -> Dict[int, str]:
        """Create a mapping from label index to label name."""
        if len(self.data) == 0:
            return {0: "UNKNOWN"}  # Return a dummy mapping for empty datasets
            
        try:
            labels = self.data.label.unique()

            mapping = {}
            for label in labels:
                subset = self.data[self.data["label"] == label]
                if len(subset) > 0:  # Ensure the subset is not empty
                    class_number = subset["label_nbr"].iloc[0]
                    mapping[class_number] = label

            if self.padding == "zero" and self.sequence_label:
                nbr_class = len(mapping)
                mapping[nbr_class] = "SEQUENCE-PADDING"
                
            if not mapping:  # If mapping is empty, provide a default
                mapping = {0: "UNKNOWN"}
                
            return mapping
        except Exception as e:
            print(f"Error creating label mapping: {str(e)}")
            return {0: "UNKNOWN"}  # Return a dummy mapping on error

    def extract_frames(self, capture: cv2.VideoCapture):
        """Extract all frames from a video and normalize them between 0 and 1."""
        frame_array = []
        success, frame = capture.read()

        frame_count = 0
        while success:
            frame_count += 1
            # Convert BGR (OpenCV) to RGB
            b, g, r = cv2.split(frame)
            frame = cv2.merge([r, g, b])
            # Use float32 explicitly
            frame_array.append((frame / 255.0).astype(np.float32))
            success, frame = capture.read()

            # Safety limit: max 150 frames (~5 seconds)
            if frame_count > 150:
                break

        return np.array(frame_array, dtype=np.float32)  # Ensure float32 type
