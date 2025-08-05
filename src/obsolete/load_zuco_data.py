import os
import pickle
import json
import numpy as np
from typing import Dict, List, Union, Optional

class ZucoDataLoader:
    """
    A flexible loader for the ZuCo dataset that supports multiple feature sets and easy data access.
    
    Attributes:
        data (Dict): Loaded dataset containing features, labels, and indices
        metadata (Dict): Dataset metadata including subjects and feature sets
        output_dir (str): Directory where data is stored
    """
    
    def __init__(self, portable_data_dir: str = "portable_zuco_data"):
        """
        Initialize the ZuCo data loader.
        
        Args:
            portable_data_dir (str): Path to directory containing ZuCo portable data
        """
        self.output_dir = portable_data_dir
        self.data = self._load_combined_data()
        self.metadata = self._load_metadata()
    
    def _load_combined_data(self) -> Dict:
        """
        Load the combined dataset from pickle file.
        
        Returns:
            Dict: Loaded dataset
        """
        combined_data_path = os.path.join(self.output_dir, "combined_zuco_data.pkl")
        try:
            with open(combined_data_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"ZuCo dataset not found at {combined_data_path}. "
                                    "Ensure you've run extract_zuco_portable_data.py first.")
    
    def _load_metadata(self) -> Dict:
        """
        Load metadata from JSON file.
        
        Returns:
            Dict: Dataset metadata
        """
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata not found at {metadata_path}. "
                                    "Ensure you've run extract_zuco_portable_data.py first.")
    
    def get_features(
        self, 
        feature_set: Optional[str] = None, 
        subjects: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Retrieve features for specified feature set and subjects.
        
        Args:
            feature_set (Optional[str]): Specific feature set to retrieve. 
                If None, returns all feature sets.
            subjects (Optional[List[str]]): List of subjects to retrieve. 
                If None, returns features for all subjects.
        
        Returns:
            Dict of features
        """
        if feature_set is None:
            feature_set = list(self.data['features'].keys())
        elif isinstance(feature_set, str):
            feature_set = [feature_set]
        
        if subjects is None:
            subjects = list(set.union(*[set(self.data['features'][fs].keys()) for fs in feature_set]))
        
        features = {}
        for fs in feature_set:
            features[fs] = {subj: self.data['features'][fs][subj] 
                            for subj in subjects if subj in self.data['features'][fs]}
        
        return features
    
    def get_labels(
        self, 
        feature_set: Optional[str] = None, 
        subjects: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, List]]:
        """
        Retrieve labels for specified feature set and subjects.
        
        Args:
            feature_set (Optional[str]): Specific feature set to retrieve. 
                If None, returns all feature sets.
            subjects (Optional[List[str]]): List of subjects to retrieve. 
                If None, returns labels for all subjects.
        
        Returns:
            Dict of labels
        """
        if feature_set is None:
            feature_set = list(self.data['labels'].keys())
        elif isinstance(feature_set, str):
            feature_set = [feature_set]
        
        if subjects is None:
            subjects = list(set.union(*[set(self.data['labels'][fs].keys()) for fs in feature_set]))
        
        labels = {}
        for fs in feature_set:
            labels[fs] = {subj: self.data['labels'][fs][subj] 
                          for subj in subjects if subj in self.data['labels'][fs]}
        
        return labels
    
    def get_stimulus(
        self, 
        subjects: Optional[List[str]] = None,
        task: Optional[str] = None  # 'NR' for normal reading, 'TSR' for task-specific reading
    ) -> Dict[str, List[str]]:
        """
        Retrieve stimulus text for specified subjects and reading task.
        
        Args:
            subjects (Optional[List[str]]): List of subjects to retrieve. 
                If None, returns stimulus for all subjects.
            task (Optional[str]): Specific reading task ('NR' or 'TSR'). 
                If None, returns all tasks.
        
        Returns:
            Dict of stimulus text, keyed by subject and task
        """
        # We'll need to modify our data extraction script to preserve this information
        # For now, this is a placeholder implementation
        if subjects is None:
            subjects = self.metadata['subjects']
        
        if task is None:
            task = ['NR', 'TSR']
        elif isinstance(task, str):
            task = [task]
        
        stimulus = {}
        
        # This part would need to be implemented in the data extraction script
        # The actual implementation depends on how we want to store the stimulus text
        for subj in subjects:
            stimulus[subj] = {}
            for t in task:
                # Placeholder - actual implementation depends on data extraction
                stimulus[subj][t] = [
                    "Example sentence for demonstration",
                    "Another example sentence"
                ]
        
        return stimulus


    

def main():
    """Example usage of ZucoDataLoader"""
    loader = ZucoDataLoader()
    
    # Example: get all features
    all_features = loader.get_features()
    
    # Example: get features for a specific feature set
    electrode_features = loader.get_features(feature_set='electrode_features_all')
    
    # Example: get features for specific subjects
    selected_subjects_features = loader.get_features(
        feature_set='sent_gaze_sacc', 
        subjects=['YAC', 'YDR']
    )

    # Example: get stimulus for all subjects
    all_stimulus = loader.get_stimulus()
    
    # Example: get stimulus for specific subjects and task
    specific_stimulus = loader.get_stimulus(
        subjects=['YAC', 'YDR'], 
        task='NR'
    )

    
    # Print some details about the loaded dataset
    print("Available Feature Sets:", list(loader.data['features'].keys()))
    print("Total Subjects:", len(loader.metadata['subjects']))
    print("Channel Locations:", loader.metadata['channel_locations'])
    print("Stimulus Example:", specific_stimulus)

if __name__ == "__main__":
    main()