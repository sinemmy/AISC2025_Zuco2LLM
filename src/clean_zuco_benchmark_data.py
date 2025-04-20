"""
Post-processing tool for ZuCo data that identifies problematic sentences/words and 
creates cleaned versions of the data files.

This is likely specific to the Zuco benchmark dataset (from zuco benchmark) and not for 
general osf downloaded data - which is from the files online seem to suggest it may be in better shape
Once the basics are done, it will be best 

There are issues with the matfiles from the zuco benchmark such as different tasks and 
subjects having different numbers of sentences, and some sentences being empty or having
missing EEG features. This script aims to clean up those issues by removing or marking
problematic sentences and words.

This script:

1. Loads each pickle file containing the extracted ZuCo data
2. Identifies problematic sentences (those without any usable words)
3. For remaining sentences, it identifies and marks problematic words
4. Creates a cleaned version of the data by:
    Removing sentences with no usable words
    Keeping only words with valid boundaries (essential for sliding window analysis)
    Marking remaining issues on words that are kept


Optionally creates backup files of the original data
Updates both individual files and the combined all_data.pkl file
Generates a comprehensive report of what was removed/modified



Usage:
python clean_zuco_benchmark_data.py path/to/extracted/data

This script creates filtered versions of the ZuCo data, removing or marking problematic 
elements, and generates a report of what was removed.
"""
"""
Post-processing tool for ZuCo data that identifies problematic sentences/words and 
creates cleaned versions of the data files.

Usage:
python clean_zuco_data.py path/to/extracted/data

This script creates filtered versions of the ZuCo data, removing or marking problematic 
elements, and generates a report of what was removed.
"""
import os
import pickle
import json
import argparse
from pathlib import Path
import copy
from tqdm import tqdm

def clean_extracted_data(data_dir, create_backup=True):
    """
    Clean extracted ZuCo data by identifying and handling problematic sentences and words.
    
    Args:
        data_dir: Directory containing the extracted pickle files
        create_backup: Whether to create backup copies of the original files
    
    Returns:
        dict: Summary of problematic data
    """
    problematic_data = {
        "bad_sentences": {},  # Subject_Task -> [sentence_indices]
        "stats": {
            "total_sentences": 0,
            "removed_sentences": 0,
            "total_words": 0,
            "removed_words": 0
        }
    }
    
    # Find all individual subject pickle files
    pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl') and '_' in f and not f.startswith('all_')]
    
    print(f"Found {len(pkl_files)} subject-task files to clean")
    
    all_cleaned_data = {}
    
    for pkl_file in tqdm(pkl_files, desc="Processing files"):
        filepath = os.path.join(data_dir, pkl_file)
        
        try:
            with open(filepath, 'rb') as f:
                sentences = pickle.load(f)
            
            # Extract subject_task key from filename
            subject_task_key = os.path.splitext(pkl_file)[0]
            problematic_data["bad_sentences"][subject_task_key] = []
            
            problematic_data["stats"]["total_sentences"] += len(sentences)
            
            # Create a clean copy
            clean_sentences = []
            
            # Analyze each sentence
            for sentence in sentences:
                sentence_idx = sentence.get('sentence_idx', -1)
                words = sentence.get('words', [])
                
                # Skip completely empty sentences
                if not words:
                    problematic_data["bad_sentences"][subject_task_key].append(sentence_idx)
                    problematic_data["stats"]["removed_sentences"] += 1
                    continue
                
                problematic_data["stats"]["total_words"] += len(words)
                
                # Filter out words without boundaries
                clean_words = []
                for word_obj in words:
                    # Check if word has boundaries (essential for sliding window)
                    if 'boundaries' in word_obj and word_obj['boundaries']:
                        clean_words.append(word_obj)
                    else:
                        problematic_data["stats"]["removed_words"] += 1
                
                # Only keep sentences with usable words
                if clean_words:
                    clean_sentence = copy.deepcopy(sentence)
                    clean_sentence['words'] = clean_words
                    clean_sentences.append(clean_sentence)
                else:
                    problematic_data["bad_sentences"][subject_task_key].append(sentence_idx)
                    problematic_data["stats"]["removed_sentences"] += 1
            
            # Create backup if requested
            if create_backup and clean_sentences != sentences:
                backup_path = os.path.join(data_dir, f"{subject_task_key}_backup.pkl")
                with open(backup_path, 'wb') as f:
                    pickle.dump(sentences, f, protocol=4)
                print(f"Created backup: {backup_path}")
            
            # Save the cleaned data
            if clean_sentences:
                with open(filepath, 'wb') as f:
                    pickle.dump(clean_sentences, f, protocol=4)
                print(f"Saved cleaned data: {filepath}")
                
                # Store for all_data.pkl update
                all_cleaned_data[subject_task_key] = clean_sentences
            else:
                print(f"Warning: No valid sentences in {subject_task_key}")
            
        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")
    
    # Update the all_data.pkl file if it exists
    all_data_path = os.path.join(data_dir, "all_data.pkl")
    if os.path.exists(all_data_path):
        try:
            # Create backup
            if create_backup:
                backup_path = os.path.join(data_dir, "all_data_backup.pkl")
                if not os.path.exists(backup_path):
                    import shutil
                    shutil.copy2(all_data_path, backup_path)
                    print(f"Created backup: {backup_path}")
            
            # Save updated all_data.pkl
            with open(all_data_path, 'wb') as f:
                pickle.dump(all_cleaned_data, f, protocol=4)
            print(f"Saved cleaned data: {all_data_path}")
        except Exception as e:
            print(f"Error updating all_data.pkl: {e}")
    
    # Write report to a file
    report_path = os.path.join(data_dir, "data_cleaning_report.json")
    with open(report_path, 'w') as f:
        json.dump(problematic_data, f, indent=2)
        
    print(f"Saved data cleaning report to {report_path}")
    print(f"Summary:")
    print(f"  Total sentences: {problematic_data['stats']['total_sentences']}")
    print(f"  Removed sentences: {problematic_data['stats']['removed_sentences']}")
    print(f"  Total words: {problematic_data['stats']['total_words']}")
    print(f"  Removed words: {problematic_data['stats']['removed_words']}")
    
    return problematic_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean extracted ZuCo data by handling problematic elements.')
    parser.add_argument('data_dir', help='Path to directory containing extracted ZuCo data')
    parser.add_argument('--no-backup', dest='create_backup', action='store_false', 
                      help='Skip creating backup files of the original data')
    parser.set_defaults(create_backup=True)
    
    args = parser.parse_args()
    
    clean_extracted_data(args.data_dir, args.create_backup)