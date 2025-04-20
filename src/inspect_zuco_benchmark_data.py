"""
Diagnostic script to examine ZuCo data structure before cleaning.
Shows detailed statistics about word structure in the extracted data.

Usage:
python inspect_zuco_benchmark_data.py path/to/extracted/data
"""
import os
import pickle
import json
import argparse
from collections import Counter

def inspect_word_structure(data_dir):
    """
    Inspect the structure of words in the extracted ZuCo data.
    
    Args:
        data_dir: Directory containing the extracted pickle files
    """
    # Find all individual subject pickle files
    pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl') and '_' in f and not f.startswith('all_')]
    
    print(f"Found {len(pkl_files)} subject-task files to inspect")
    
    # Track structure statistics
    stats = {
        "total_sentences": 0,
        "total_words": 0,
        "files_examined": 0,
        "word_keys": Counter(),
        "boundary_keys": Counter(),
        "eeg_keys": Counter(),
        "empty_boundary_count": 0
    }
    
    example_words = {}
    
    # Examine a limited number of files to avoid long runtime
    for pkl_file in pkl_files[:5]:  # Examine 5 files
        filepath = os.path.join(data_dir, pkl_file)
        subject_task_key = os.path.splitext(pkl_file)[0]
        
        try:
            print(f"Examining {pkl_file}...")
            with open(filepath, 'rb') as f:
                sentences = pickle.load(f)
            
            stats["total_sentences"] += len(sentences)
            stats["files_examined"] += 1
            
            # Sample 5 sentences from this file for deeper inspection
            sample_sentences = sentences[:5] if len(sentences) >= 5 else sentences
            
            # Examine the structure of words in these sentences
            for sentence in sample_sentences:
                words = sentence.get('words', [])
                stats["total_words"] += len(words)
                
                # Examine the structure of each word
                for word_idx, word_obj in enumerate(words[:10]):  # Look at first 10 words
                    # Track top-level keys in word objects
                    for key in word_obj.keys():
                        stats["word_keys"][key] += 1
                    
                    # Look specifically at boundary structure
                    if 'boundaries' in word_obj:
                        for key in word_obj['boundaries'].keys():
                            stats["boundary_keys"][key] += 1
                        
                        # Check if boundaries are empty
                        if not word_obj['boundaries']:
                            stats["empty_boundary_count"] += 1
                    
                    # Look at EEG features structure
                    if 'eeg_features' in word_obj:
                        for key in word_obj['eeg_features'].keys():
                            stats["eeg_keys"][key] += 1
                    
                    # Save an example word for inspection
                    if word_idx == 0 and subject_task_key not in example_words:
                        example_words[subject_task_key] = word_obj
        
        except Exception as e:
            print(f"Error examining {pkl_file}: {e}")
    
    # Print summary
    print("\nSummary of data structure:")
    print(f"Files examined: {stats['files_examined']}")
    print(f"Total sentences: {stats['total_sentences']}")
    print(f"Total words: {stats['total_words']}")
    print(f"Empty boundary count: {stats['empty_boundary_count']}")
    
    print("\nCommon word keys:")
    for key, count in stats["word_keys"].most_common():
        print(f"  {key}: {count}")
    
    print("\nBoundary keys:")
    for key, count in stats["boundary_keys"].most_common():
        print(f"  {key}: {count}")
    
    print("\nEEG feature keys:")
    for key, count in stats["eeg_keys"].most_common():
        print(f"  {key}: {count}")
    
    # Save example words to a file for detailed inspection
    example_path = os.path.join(data_dir, "example_words.json")
    try:
        with open(example_path, 'w') as f:
            json.dump(example_words, f, indent=2, default=lambda x: str(x) if not isinstance(x, (dict, list, str, int, float, bool, type(None))) else x)
    except Exception as e:
        print(f"Error saving example words: {e}")
        # Try a simpler version
        with open(example_path, 'w') as f:
            simple_examples = {k: str(v) for k, v in example_words.items()}
            json.dump(simple_examples, f, indent=2)
    
    print(f"\nExample words saved to {example_path}")
    
    # Save stats to a file
    stats_path = os.path.join(data_dir, "data_structure_stats.json")
    with open(stats_path, 'w') as f:
        # Convert Counter objects to regular dictionaries for JSON serialization
        serializable_stats = {
            "total_sentences": stats["total_sentences"],
            "total_words": stats["total_words"],
            "files_examined": stats["files_examined"],
            "word_keys": dict(stats["word_keys"]),
            "boundary_keys": dict(stats["boundary_keys"]),
            "eeg_keys": dict(stats["eeg_keys"]),
            "empty_boundary_count": stats["empty_boundary_count"]
        }
        json.dump(serializable_stats, f, indent=2)
    
    print(f"Data structure statistics saved to {stats_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect ZuCo data structure before cleaning.')
    parser.add_argument('data_dir', help='Path to directory containing extracted ZuCo data')
    
    args = parser.parse_args()
    
    inspect_word_structure(args.data_dir)