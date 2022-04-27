import torch
from torch.utils.data import Dataset

import numpy as np

import os

class CustomDataSet(Dataset):
    
    def __init__(self, data_root_dir, data, transforms=None):
        #self.root_dir = args.dataset_root
        self.data_root_dir = data_root_dir
        self.transforms = transforms
        self.annotations_frame = data

    def __len__(self):
      return len(self.annotations_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient_id = self.annotations_frame.iloc[idx, 0]
        exam_id = self.annotations_frame.iloc[idx, 1]
        exam_location = self.annotations_frame.iloc[idx, 2]
        video_score = self.annotations_frame.iloc[idx, 7]
        frame_id = self.annotations_frame.iloc[idx, 8]
        frame_score = self.annotations_frame.iloc[idx, 9]

        frame_path = os.path.join(self.data_root_dir, str(patient_id), str(exam_id), str(exam_location), '{}.npy'.format(frame_id))
        frame = np.load(frame_path)

        # info
        info = {
        'patient_id': patient_id,
        'exam_id': exam_id,
        'exam_location': exam_location,
        'video_score': video_score,
        'frame_id': frame_id,
        'frame_score': frame_score
        }

        if self.transforms:
            frame = self.transforms(frame)

        label = torch.tensor(frame_score, dtype=torch.long)

        return frame, label, info