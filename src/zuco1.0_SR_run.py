#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Dataloader for ZuCo 1.0 data set (SR)


# In[16]:


import os
import numpy as np
from scipy.io import loadmat
import torch
from transformer_lens import HookedTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.multioutput import MultiOutputRegressor

import os
import pickle
import numpy as np
from pathlib import Path


class ZucoDataLoader:
    def __init__(self, data_dir='../zuco_data/zuco1.0/task1-SR/Matlab files'):
        self.data_dir = data_dir
        self.subject_files = self._get_subject_files()

    def _get_subject_files(self):
        subject_files = {}
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".mat"):
                subject_id = file_name.split('.')[0]
                subject_files[subject_id] = os.path.join(self.data_dir, file_name)
        return subject_files

    def get_subject_ids(self):
        return list(self.subject_files.keys())

    def load_subject_data(self, subject_id):
        file_path = self.subject_files[subject_id]
        print(f"Loading data from {file_path}")
        data = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        return data

    def extract_word_level_data(self, subject_id):
        """Extract word-level EEG data with sentence context"""
        data = self.load_subject_data(subject_id)
        sentences = data['sentenceData']

        # Store word-level data
        word_data = []

        for sent_idx, sentence in enumerate(sentences):
            try:
                # Check if words is iterable
                if not hasattr(sentence, 'word'):
                    print(f"Sentence {sent_idx} has no word attribute")
                    continue

                words = sentence.word

                # Handle case where words is not iterable (e.g., a float)
                if not hasattr(words, '__iter__'):
                    print(f"Sentence {sent_idx} words is not iterable: {type(words)}")
                    continue

                sentence_text = sentence.content if hasattr(sentence, 'content') else ""

                for word_idx, word in enumerate(words):
                    # Extract EEG features
                    eeg_features = {}
                    word_text = word.content if hasattr(word, 'content') else ""

                    # Extract each frequency band
                    for feature in ['FFD', 'TRT', 'GD', 'GPT']:
                        for band in ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']:
                            feature_name = feature + band
                            if hasattr(word, feature_name):
                                eeg_features[feature_name] = getattr(word, feature_name)

                    word_data.append({
                        'word': word_text,
                        'word_idx': word_idx,
                        'sentence_id': sent_idx,
                        'sentence': sentence_text,
                        'eeg_features': eeg_features
                    })
            except (AttributeError, IndexError, TypeError) as e:
                print(f"Error processing sentence {sent_idx}: {e}")
                continue

        return word_data

class EmbeddingGenerator:
    def __init__(self, model_name='gpt2-medium'):
        """Initialize with TransformerLens"""
        print(f"Loading model {model_name}...")
        self.model = HookedTransformer.from_pretrained(model_name)
        self.model.eval()

    def extract_embeddings_sliding_window(self, word_data):
        """
        Generate contextual embeddings using sliding window approach (Goldstein method)
        """
        # Group by sentence
        sentences = {}
        for word in word_data:
            sent_id = word['sentence_id']
            if sent_id not in sentences:
                sentences[sent_id] = {
                    'text': word['sentence'],
                    'words': []
                }
            sentences[sent_id]['words'].append(word)

        # Process each sentence
        embeddings = []

        for sent_id, sent_info in tqdm(sentences.items(), desc="Extracting embeddings"):
            sent_text = sent_info['text']
            words = sent_info['words']

            # Sort words by position
            words.sort(key=lambda x: x['word_idx'])

            # Process each word with its preceding context
            for i, word in enumerate(words):
                # Build context window (all words up to and including current)
                word_tokens = [w['word'] for w in words[:i+1]]
                context = " ".join(word_tokens)

                # Get activations for this context
                _, cache = self.model.run_with_cache(context)

                # Extract final layer activation for last token
                # This follows Goldstein's methodology
                final_layer_activations = cache['blocks.23.hook_resid_post'][0]
                word_embedding = final_layer_activations[-1].detach().cpu().numpy()

                embeddings.append({
                    'word': word['word'],
                    'sentence_id': sent_id,
                    'word_idx': word['word_idx'],
                    'embedding': word_embedding,
                    'eeg_features': word['eeg_features']
                })

        return embeddings

class BrainEmbeddingMapper:
    def __init__(self):
        """Linear mapper between embeddings and EEG"""
        self.models = {}

    def train_mapper(self, embeddings, feature_name='FFD_t1', n_splits=5):
        """Train a linear mapping between embeddings and EEG features"""
        # First, check dimensions of the feature
        sample_shapes = {}
        for item in embeddings:
            if feature_name in item['eeg_features']:
                feature = item['eeg_features'][feature_name]
                if hasattr(feature, 'shape'):
                    shape = feature.shape
                    if shape not in sample_shapes:
                        sample_shapes[shape] = 0
                    sample_shapes[shape] += 1

        print(f"Found {len(sample_shapes)} different shapes for {feature_name}")
        for shape, count in sample_shapes.items():
            print(f"  Shape {shape}: {count} samples")

        # Choose most common shape with non-zero dimensions
        valid_shapes = {shape: count for shape, count in sample_shapes.items() 
                    if shape and shape[0] > 0}

        if not valid_shapes:
            print(f"No valid shapes found for feature {feature_name}")
            return None

        target_shape = max(valid_shapes.items(), key=lambda x: x[1])[0]
        print(f"Using shape {target_shape} for training")

        # Filter to samples with consistent dimensions
        valid_embeddings = []
        valid_features = []

        for item in embeddings:
            if feature_name in item['eeg_features']:
                feature = item['eeg_features'][feature_name]
                if hasattr(feature, 'shape') and feature.shape == target_shape:
                    if not np.isnan(feature).any():
                        valid_embeddings.append(item['embedding'])
                        valid_features.append(feature)

        if len(valid_embeddings) < 10:  # Minimum samples for training
            print(f"Not enough valid samples after filtering")
            return None

        print(f"Training with {len(valid_embeddings)} samples")

        # Add regularization to handle ill-conditioned matrices
        alpha = 10.0  # Increase regularization strength

        # Convert to numpy arrays
        X = np.array(valid_embeddings)
        y = np.array(valid_features)

        # Cross-validation
        kf = KFold(n_splits=min(n_splits, len(X)), shuffle=True, random_state=42)
        results = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train model
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Calculate correlation for each electrode
            correlations = []
            for i in range(y_test.shape[1]):
                if np.std(y_test[:, i]) > 0 and np.std(y_pred[:, i]) > 0:
                    corr = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
                    correlations.append(corr)

            results.append({
                'correlations': correlations,
                'mean_correlation': np.mean(correlations)
            })

        self.models[feature_name] = {
            'model': model,
            'results': results
        }

        return results

    def train_multifeature_mapper(self, embeddings, features=None, n_splits=5):
        """Train a linear mapping with multiple features at once"""
        # Get all available features if none specified
        if not features:
            all_features = set()
            for item in embeddings:
                all_features.update(item['eeg_features'].keys())
            features = sorted(list(all_features))

        print(f"Training with {len(features)} features")

        # Find most common electrode count across features
        feature_shapes = {}
        for feature in features:
            shapes = {}
            for item in embeddings:
                if feature in item['eeg_features']:
                    arr = item['eeg_features'][feature]
                    if hasattr(arr, 'shape') and len(arr.shape) > 0:
                        shape = arr.shape
                        if shape not in shapes:
                            shapes[shape] = 0
                        shapes[shape] += 1

            if shapes:
                feature_shapes[feature] = max(shapes.items(), key=lambda x: x[1])[0]

        if not feature_shapes:
            print("No valid features found")
            return None

        # Filter to features with same electrode count
        valid_features = []
        target_shape = (105,)  # Standard electrode count

        for feature, shape in feature_shapes.items():
            if shape == target_shape:
                valid_features.append(feature)

        if not valid_features:
            print("No features with consistent electrode count")
            return None

        print(f"Using {len(valid_features)} features with {target_shape[0]} electrodes")

        # Collect valid samples
        valid_data = []

        for item in embeddings:
            sample = {
                'embedding': item['embedding'],
                'targets': {}
            }

            has_valid_data = False
            for feature in valid_features:
                if feature in item['eeg_features']:
                    arr = item['eeg_features'][feature]
                    if hasattr(arr, 'shape') and arr.shape == target_shape:
                        if not np.isnan(arr).any():
                            sample['targets'][feature] = arr
                            has_valid_data = True

            if has_valid_data:
                valid_data.append(sample)

        print(f"Found {len(valid_data)} samples with valid data")

        if len(valid_data) < 100:
            print("Not enough valid samples for training")
            return None

        # Create model for each feature
        results = {}

        for feature in valid_features:
            # Get samples with this feature
            feature_samples = []
            feature_targets = []

            for sample in valid_data:
                if feature in sample['targets']:
                    feature_samples.append(sample['embedding'])
                    feature_targets.append(sample['targets'][feature])

            if len(feature_samples) < 100:
                print(f"Skipping feature {feature}: not enough samples")
                continue

            X = np.array(feature_samples)
            y = np.array(feature_targets)

            print(f"Training model for {feature} with {len(X)} samples")

            # Cross-validation
            kf = KFold(n_splits=min(n_splits, len(X)), shuffle=True, random_state=42)
            feature_results = []

            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Train model with increased regularization
                model = Ridge(alpha=50.0)
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)

                # Calculate correlation for each electrode
                correlations = []
                for i in range(y_test.shape[1]):
                    if np.std(y_test[:, i]) > 0 and np.std(y_pred[:, i]) > 0:
                        corr = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
                        correlations.append(corr)

                feature_results.append({
                    'correlations': correlations,
                    'mean_correlation': np.mean(correlations)
                })

            results[feature] = {
                'model': model,
                'results': feature_results
            }

        # Store models
        self.models.update(results)

        return results


    def train_all_features(self, embeddings, n_splits=5):
        """Train linear mappings for all available EEG features"""
        # Find all features that appear in the data
        all_features = set()
        for item in embeddings:
            all_features.update(item['eeg_features'].keys())

        print(f"Found {len(all_features)} features in the data")

        # Train a model for each feature
        results_by_feature = {}

        for feature_name in all_features:
            print(f"\nTraining model for {feature_name}")
            results = self.train_mapper(embeddings, feature_name=feature_name, n_splits=n_splits)

            if results:
                mean_corr = np.mean([fold['mean_correlation'] for fold in results])
                print(f"Mean correlation: {mean_corr:.4f}")
                results_by_feature[feature_name] = results

        return results_by_feature


    def extract_steering_vector(self, feature_name='FFD_t1', method='weighted', threshold=0.1):
        """
        Extract a steering vector using different methods:
        - 'weighted': Weight electrodes by correlation strength
        - 'top_n': Use only top N electrodes 
        - 'threshold': Use electrodes with correlation above threshold
        """
        if feature_name not in self.models:
            print(f"No model trained for feature {feature_name}")
            return None

        feature_data = self.models[feature_name]

        # Handle both single-feature and multi-feature formats
        if isinstance(feature_data, dict) and 'model' in feature_data:
            model = feature_data['model']
            results = feature_data['results']
        else:
            model = feature_data  # Original format
            results = self.models[feature_name]['results']

        weights = model.coef_.T  # [embedding_dim, n_electrodes]

        # Calculate correlation strength per electrode
        correlation_means = []
        for result in results:
            correlation_means.append(np.array(result['correlations']))
        electrode_correlations = np.mean(np.stack(correlation_means), axis=0)

        # Select electrodes based on method
        if method == 'weighted':
            # Weight each electrode by its correlation strength
            electrode_weights = np.abs(electrode_correlations)
            electrode_weights = electrode_weights / np.sum(electrode_weights)
            steering_vector = np.zeros(weights.shape[0])

            for i, weight in enumerate(electrode_weights):
                if not np.isnan(weight):
                    steering_vector += weight * weights[:, i]

        elif method == 'top_n':
            n_electrodes = 10  # Default to top 10
            # Get top N electrodes by absolute correlation
            top_indices = np.argsort(np.abs(electrode_correlations))[-n_electrodes:]
            steering_vector = np.mean(weights[:, top_indices], axis=1)

        elif method == 'threshold':
            # Use electrodes above correlation threshold
            mask = np.abs(electrode_correlations) > threshold
            if not np.any(mask):
                print(f"No electrodes above threshold {threshold}")
                return None
            steering_vector = np.mean(weights[:, mask], axis=1)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Normalize
        steering_vector = steering_vector / np.linalg.norm(steering_vector)
        return steering_vector  

    def extract_combined_steering_vector(self, method='weighted', threshold=0.1):
        """
        Extract a steering vector that combines information across multiple features,
        weighting each feature by its overall prediction performance.
        """
        # Find all available features
        available_features = [f for f in self.models.keys()]
        if not available_features:
            print("No models trained for any features")
            return None

        print(f"Combining steering vectors from features: {available_features}")

        # Get individual steering vectors for each feature
        feature_vectors = {}
        feature_scores = {}

        for feature in available_features:
            # Extract steering vector using existing method
            vector = self.extract_steering_vector(
                feature_name=feature,
                method=method,
                threshold=threshold
            )

            if vector is not None:
                feature_vectors[feature] = vector

                # Get average correlation score for this feature
                feature_data = self.models[feature]
                if isinstance(feature_data, dict) and 'results' in feature_data:
                    mean_corr = np.mean([np.mean(r['correlations']) for r in feature_data['results']])
                else:
                    mean_corr = np.mean([np.mean(r['correlations']) for r in feature_data['results']])

                feature_scores[feature] = mean_corr

        if not feature_vectors:
            print("No valid steering vectors extracted")
            return None

        # Weight features by their scores
        total_score = sum(feature_scores.values())
        weights = {f: score/total_score for f, score in feature_scores.items()}

        # Combine vectors (they should all have the same dimensionality)
        dim = len(next(iter(feature_vectors.values())))
        combined_vector = np.zeros(dim)

        for feature, vector in feature_vectors.items():
            combined_vector += weights[feature] * vector

        # Normalize
        combined_vector = combined_vector / np.linalg.norm(combined_vector)

        return combined_vector




# In[3]:


# 1. Initialize the data loader
zuco_loader = ZucoDataLoader(data_dir='../zuco_data/zuco1.0/task1-SR/Matlab files')

# 2. Extract and save word-level data for all subjects
all_subjects_data = {}
subject_data_path = Path('saved_data/subject_word_data.pkl')
subject_data_path.parent.mkdir(parents=True, exist_ok=True)

if subject_data_path.exists():
    print(f"Loading subject data from {subject_data_path}")
    with open(subject_data_path, 'rb') as f:
        all_subjects_data = pickle.load(f)
else:
    for subject_id in zuco_loader.get_subject_ids():
        word_data = zuco_loader.extract_word_level_data(subject_id)
        all_subjects_data[subject_id] = word_data
        print(f"Extracted {len(word_data)} words from subject {subject_id}")

    # Save the results
    with open(subject_data_path, 'wb') as f:
        pickle.dump(all_subjects_data, f)
    print(f"Saved subject data to {subject_data_path}")

# 3. Generate embeddings with checkpoints
embeddings_path = Path('saved_data/embeddings.pkl')

if embeddings_path.exists():
    print(f"Loading embeddings from {embeddings_path}")
    with open(embeddings_path, 'rb') as f:
        sentence_embeddings = pickle.load(f)
else:
    # Gather unique sentences across all subjects
    unique_sentences = {}
    for subject_id, word_data in all_subjects_data.items():
        for word in word_data:
            sent_id = word['sentence_id']
            if sent_id not in unique_sentences:
                unique_sentences[sent_id] = word['sentence']

    # Initialize embeddings generator
    embedding_gen = EmbeddingGenerator(model_name='gpt2-medium')
    sentence_embeddings = {}

    # Generate embeddings with checkpoints
    checkpoint_path = Path('saved_data/embeddings_checkpoint.pkl')

    try:
        # If checkpoint exists, load it
        if checkpoint_path.exists():
            with open(checkpoint_path, 'rb') as f:
                sentence_embeddings = pickle.load(f)
            print(f"Loaded checkpoint with {len(sentence_embeddings)} sentences")

        # Process remaining sentences
        remaining_sentences = {k: v for k, v in unique_sentences.items() 
                               if k not in sentence_embeddings}

        for i, (sent_id, sentence) in enumerate(remaining_sentences.items()):
            # Create dummy word data structure for the embeddings function
            words = sentence.split()
            dummy_words = [{'word': word, 'word_idx': i, 'sentence_id': sent_id, 
                           'sentence': sentence, 'eeg_features': {}} 
                           for i, word in enumerate(words)]

            embeddings = embedding_gen.extract_embeddings_sliding_window(dummy_words)
            sentence_embeddings[sent_id] = {e['word_idx']: e['embedding'] for e in embeddings}

            # Save checkpoint every 50 sentences
            if (i + 1) % 50 == 0:
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(sentence_embeddings, f)
                print(f"Saved checkpoint after {i+1}/{len(remaining_sentences)} sentences")

        # Save final embeddings
        with open(embeddings_path, 'wb') as f:
            pickle.dump(sentence_embeddings, f)
        print(f"Saved embeddings to {embeddings_path}")

        # Remove checkpoint file
        if checkpoint_path.exists():
            os.remove(checkpoint_path)

    except Exception as e:
        # Save checkpoint on error
        print(f"Error during embedding generation: {e}")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(sentence_embeddings, f)
        print(f"Saved checkpoint to {checkpoint_path}")
        raise e


# # linear mapping

# In[4]:


# 4. Linear mapping with save points
results_path = Path('saved_data/mapping_results.pkl')

if results_path.exists():
    print(f"Loading mapping results from {results_path}")
    with open(results_path, 'rb') as f:
        results_by_subject = pickle.load(f)
else:
    mapper = BrainEmbeddingMapper()
    results_by_subject = {}

    for subject_id, word_data in all_subjects_data.items():
        subject_result_path = Path(f'saved_data/subject_{subject_id}_results.pkl')

        if subject_result_path.exists():
            with open(subject_result_path, 'rb') as f:
                results = pickle.load(f)
            results_by_subject[subject_id] = results
            print(f"Loaded results for subject {subject_id}")
            continue

        # Combine word data with embeddings
        combined_data = []
        for word in word_data:
            sent_id = word['sentence_id']
            word_idx = word['word_idx']

            if sent_id in sentence_embeddings and word_idx in sentence_embeddings[sent_id]:
                combined_data.append({
                    'word': word['word'],
                    'sentence_id': sent_id,
                    'word_idx': word_idx,
                    'embedding': sentence_embeddings[sent_id][word_idx],
                    'eeg_features': word['eeg_features']
                })

        # Train linear mapping for this subject
        # results = mapper.train_mapper(combined_data, feature_name='FFD_t1')
        results = mapper.train_multifeature_mapper(combined_data, n_splits=5)
        results_by_subject[subject_id] = results

        # Save subject results
        with open(subject_result_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved results for subject {subject_id}")

        # Print results for this subject
        if results:
            # New structure is a dictionary by feature
            feature_means = []
            for feature, feature_results in results.items():
                feature_mean = np.mean([fold['mean_correlation'] for fold in feature_results['results']])
                feature_means.append(feature_mean)
                print(f"Feature {feature}: Mean correlation = {feature_mean:.4f}")

            overall_mean = np.mean(feature_means)
            print(f"Subject {subject_id}: Overall mean correlation = {overall_mean:.4f}")

    # Save all results
    with open(results_path, 'wb') as f:
        pickle.dump(results_by_subject, f)
    print(f"Saved all mapping results to {results_path}")


# # VISUALIZE (by Subject)

# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path




# In[6]:


def visualize_results(results_by_subject):
    """Visualize results from our mapping approach"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from pathlib import Path

    # Create output directory
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)

    # Prepare data
    data = []

    for subject_id, subject_results in results_by_subject.items():
        # Handle the case where results is a list of fold results (original mapper)
        if isinstance(subject_results, list):
            feature = 'FFD_t1'  # Default feature name
            for fold_idx, fold in enumerate(subject_results):
                mean_corr = fold['mean_correlation']
                for i, corr in enumerate(fold['correlations']):
                    data.append({
                        'Subject': subject_id,
                        'Feature': feature,
                        'Fold': fold_idx,
                        'Electrode': i,
                        'Correlation': corr
                    })
        # Handle dictionary of features (multi-feature mapper)
        elif isinstance(subject_results, dict):
            for feature, feature_data in subject_results.items():
                if 'results' in feature_data:
                    for fold_idx, fold in enumerate(feature_data['results']):
                        mean_corr = fold['mean_correlation']
                        for i, corr in enumerate(fold['correlations']):
                            data.append({
                                'Subject': subject_id,
                                'Feature': feature,
                                'Fold': fold_idx,
                                'Electrode': i,
                                'Correlation': corr
                            })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Subject performance
    plt.figure(figsize=(12, 6))
    subject_means = df.groupby('Subject')['Correlation'].mean().sort_values(ascending=False)

    plt.bar(subject_means.index, subject_means.values)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('Subject ID')
    plt.ylabel('Mean Correlation')
    plt.title('Mean Correlation by Subject')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'subject_correlations.png')
    plt.close()

    # If multiple features are present
    if len(df['Feature'].unique()) > 1:
        # Feature performance
        plt.figure(figsize=(12, 6))
        feature_means = df.groupby('Feature')['Correlation'].mean().sort_values(ascending=False)

        plt.bar(feature_means.index, feature_means.values)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel('EEG Feature')
        plt.ylabel('Mean Correlation')
        plt.title('Mean Correlation by EEG Feature')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_correlations.png')
        plt.close()

    # Distribution of correlations across electrodes
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Subject', y='Correlation', data=df)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Distribution of Electrode Correlations by Subject')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'electrode_correlations.png')
    plt.close()

    # Cross-subject consistency
    plt.figure(figsize=(10, 8))
    subjects = sorted(df['Subject'].unique())
    corr_matrix = np.zeros((len(subjects), len(subjects)))
    np.fill_diagonal(corr_matrix, 1.0)  # Self-correlation = 1

    for i, subj1 in enumerate(subjects):
        for j, subj2 in enumerate(subjects):
            if i < j:  # Only calculate once for each pair
                s1 = df[df['Subject'] == subj1].groupby('Electrode')['Correlation'].mean()
                s2 = df[df['Subject'] == subj2].groupby('Electrode')['Correlation'].mean()

                # Find common electrodes
                common_elec = list(set(s1.index) & set(s2.index))
                if common_elec:
                    corr = np.corrcoef(s1[common_elec], s2[common_elec])[0, 1]
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr  # Matrix is symmetric

    sns.heatmap(corr_matrix, xticklabels=subjects, yticklabels=subjects, 
               cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt='.2f')
    plt.title('Cross-subject Consistency')
    plt.tight_layout()
    plt.savefig(output_dir / 'subject_consistency.png')
    plt.close()

    return df


# In[7]:


viz_data = visualize_results(results_by_subject)


# # steering

# In[ ]:





# In[8]:


# Testing steering vectors
def test_steering(mapper, embedding_gen, best_subject):
    # Get steering vectors for different features
    feature_to_test = 'FFD_g1'  # Gamma band during first fixation

    # Extract different steering vectors with different methods
    weighted_vector = mapper.extract_combined_steering_vector(
        feature_name=feature_to_test, method='weighted')

    top_n_vector = mapper.extract_combined_steering_vector(
        feature_name=feature_to_test, method='top_n', n_electrodes=10)

    threshold_vector = mapper.extract_combined_steering_vector(
        feature_name=feature_to_test, method='threshold', threshold=0.15)

    # Test prompts
    test_prompts = [
        "The scientists discovered a new",
        "In the mountains, the hikers found",
        "The president announced that the government will"
    ]

    # Test different steering vectors and scales
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")

        # Generate without steering
        print("No steering:")
        text = generate_with_steering(
            embedding_gen.model, prompt, weighted_vector, scale=0.0)
        print(text)

        # Generate with different steering vectors
        for vector_name, vector in [
            ("Weighted", weighted_vector),
            ("Top-N", top_n_vector),
            ("Threshold", threshold_vector)
        ]:
            for scale in [0.5, 1.0, 2.0]:
                print(f"\n{vector_name} steering (scale={scale}):")
                text = generate_with_steering(
                    embedding_gen.model, prompt, vector, scale=scale)
                print(text)


# In[9]:


def apply_steering_vector(embedding_model, text, steering_vector, scale=1.0):
    # Check if steering vector is None
    if steering_vector is None:
        print("Warning: Steering vector is None. Proceeding without steering.")
        # Just run the model normally without hooks
        return embedding_model(text)

    # Define the hook function
    def steering_hook(acts, hook):
        # Apply to final layer activations
        if acts.shape[-1] == len(steering_vector):
            # Project activations onto steering direction and amplify
            projection = torch.matmul(
                acts, 
                torch.tensor(steering_vector, dtype=acts.dtype, device=acts.device)
            )

            # Apply steering by adding scaled projection
            return acts + scale * projection.unsqueeze(-1) * torch.tensor(
                steering_vector, dtype=acts.dtype, device=acts.device
            )
        return acts

    # Run model with hook
    output = embedding_model.run_with_hooks(
        text,
        fwd_hooks=[("blocks.23.hook_resid_post", steering_hook)]
    )

    return output



# In[ ]:


def test_steering_vector(mapper, results_by_subject, embedding_gen):
    # Check the models dictionary
    if hasattr(mapper, 'models') and isinstance(mapper.models, dict):
        print(f"Available features: {list(mapper.models.keys())}")

        # Try to extract steering vector for a standard feature
        feature_name = "FFD_t1"  # Standard feature to try
        print(f"Attempting to extract steering vector for feature: {feature_name}")

        # Extract steering vector using the existing method
        steering_vector = mapper.extract_steering_vector(
            feature_name=feature_name,
            method='weighted',
            threshold=0.1
        )

        if steering_vector is None:
            print(f"Failed to extract steering vector for feature {feature_name}")
            return None

        # Test the steering vector
        print(f"Testing with: 'The experiment results indicated that'")
        sentence = "The experiment results indicated that"

        # Generate without steering for comparison
        original_output = embedding_gen.model.generate(sentence, max_new_tokens=20)
        print(f"Original: {original_output}")

        return steering_vector
    else:
        print("Mapper does not have expected 'models' attribute")
        return None


# In[17]:


test_steering_vector(mapper, results_by_subject, embedding_gen)


#