import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path

# Takes sentence data from the Zuco dataset (json) and loads it into a pandas DataFrame
def load_zuco_dataframe(file_path="../portable_data/sentence_content.json"):
    """
    Load Zuco sentence data from JSON into a clean pandas DataFrame.
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            sentence_data = json.load(f)
        
        rows = []
        for subject, tasks in sentence_data.items():
            for task_type, sentences in tasks.items():
                for idx, sentence in sentences.items():
                    rows.append({
                        'subject': subject,
                        'task': task_type,
                        'index': int(idx),
                        'sentence': sentence
                    })

        return pd.DataFrame(rows)
    
    except Exception as e:
        print(f"Error loading sentence data: {e}")
        return pd.DataFrame()
    


class ZucoSentenceDataset(Dataset):
    """Dataset for Zuco sentences with NR and TSR indices"""
    
    def __init__(self, csv_path, transform=None):
        """
        Initialize the dataset from a CSV file
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file with 'sentence', 'NR_index', and 'TSR_index' columns
        transform : callable, optional
            Optional transform to be applied to each sample
        """
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        row = self.data.iloc[idx]
        
        sample = {
            'sentence': row['sentence'],
            'NR_index': row['NR_index'],
            'TSR_index': row['TSR_index'],
            'in_NR': row['NR_index'] != -100,
            'in_TSR': row['TSR_index'] != -100
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

def get_zuco_sentence_dataloader(csv_path, batch_size=32, shuffle=True, transform=None):
    """
    Create a DataLoader for the Zuco sentences dataset
    """
    dataset = ZucoSentenceDataset(csv_path, transform=transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2  # Can safely use multiple workers now
    )

# Example tokenizer transform
class TokenizerTransform:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, sample):
        # Tokenize the sentence
        tokenized = self.tokenizer(
            sample['sentence'],
            return_tensors='pt',
            padding='max_length',
            truncation=False
        )
         #max_length=128
        
        # Remove the batch dimension
        for key in tokenized:
            if isinstance(tokenized[key], torch.Tensor):
                tokenized[key] = tokenized[key].squeeze(0)
        
        # Add tokenized data to the sample
        sample['tokens'] = tokenized
        
        return sample
