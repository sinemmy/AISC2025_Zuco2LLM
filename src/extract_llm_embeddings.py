import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer

class TransformerLensEmbeddingExtractor:
    """
    A class to extract contextual embeddings using the TransformerLens library
    for the Zuco dataset. Designed for easier steering and intervention later.
    """
    
    def __init__(self, model_name='gpt2-medium', device=None):
        """
        Initialize the embedding extractor with TransformerLens.
        
        Args:
            model_name: Name of the transformer model to use
            device: Device to run the model on (cpu, cuda, etc.)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model {model_name} using TransformerLens on {self.device}...")
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=self.device
        )
        
        # Get tokenizer from the model
        self.tokenizer = self.model.tokenizer
            
        self.model.eval()  # Set to evaluation mode
    
    def get_word_boundaries(self, sentence, word_boundaries=None):
        """
        Get the token indices for each word in the sentence.
        
        Args:
            sentence: The input sentence
            word_boundaries: Optional pre-defined word boundaries
            
        Returns:
            List of (start_idx, end_idx) tuples for each word
        """
        if word_boundaries is not None:
            return word_boundaries
        
        # Default: use simple whitespace tokenization
        words = sentence.split()
        
        # Tokenize the full sentence
        tokens = self.tokenizer.encode(sentence)
        
        # Get the token indices for each word
        word_token_indices = []
        current_idx = 0  # TransformerLens may handle special tokens differently
        
        for word in words:
            # Tokenize the word separately to determine how many tokens it produces
            word_tokens = self.tokenizer.encode(" " + word if current_idx > 0 else word)
            
            # Handle the first token specially as it may include BOS token
            if current_idx == 0 and self.tokenizer.bos_token_id is not None:
                word_tokens = word_tokens[1:]  # Skip BOS token
                
            n_tokens = len(word_tokens)
            
            # Store the range of token indices for this word
            word_token_indices.append((current_idx, current_idx + n_tokens))
            current_idx += n_tokens
            
        return word_token_indices
    
    def extract_layer_activations(self, text, layers=None):
        """
        Extract all activations for a given text from specified layers.
        
        Args:
            text: The input text
            layers: List of layer indices to extract (None = all layers)
            
        Returns:
            Dictionary mapping layer names to activations
        """
        # Determine which layers to extract
        if layers is None:
            n_layers = self.model.cfg.n_layers
            layers = list(range(n_layers))
            
        # Initialize cache to store activations
        cache = {}
        
        # Define hook functions for each layer
        hooks = []
        for layer_idx in layers:
            # Hook for MLP outputs
            def create_hook_fn(layer_idx):
                def hook_fn(act, hook):
                    cache[f"blocks.{layer_idx}.mlp"] = act.detach().cpu()
                return hook_fn
            
            hooks.append((f"blocks.{layer_idx}.mlp.hook_post", create_hook_fn(layer_idx)))
            
            # Hook for attention outputs
            def create_attn_hook_fn(layer_idx):
                def hook_fn(act, hook):
                    cache[f"blocks.{layer_idx}.attn"] = act.detach().cpu()
                return hook_fn
                
            hooks.append((f"blocks.{layer_idx}.attn.hook_result", create_attn_hook_fn(layer_idx)))
            
        # Hook for residual stream (final layer output)
        def final_hook_fn(act, hook):
            cache["final"] = act.detach().cpu()
        
        hooks.append(("ln_final.hook_normalized", final_hook_fn))
        
        # Run the model with hooks
        self.model.run_with_hooks(
            text,
            fwd_hooks=hooks
        )
        
        return cache
    
    def extract_embeddings_from_sentence(self, sentence, layers=None, word_boundaries=None):
        """
        Extract contextual embeddings for each word in a sentence using TransformerLens.
        
        Args:
            sentence: The input sentence
            layers: List of layer indices to extract (None = all layers)
            word_boundaries: Optional pre-defined word boundaries
            
        Returns:
            Dictionary mapping layer indices to dictionaries of word embeddings
            {layer_idx: {word_idx: embedding}}
        """
        # Get all activations
        cache = self.extract_layer_activations(sentence, layers)
        
        # Get word boundaries
        word_bounds = self.get_word_boundaries(sentence, word_boundaries)
        
        # Initialize result dictionary
        result = {}
        for layer_key in cache.keys():
            if "blocks" in layer_key:
                layer_idx = int(layer_key.split(".")[1])
                if layer_idx not in result:
                    result[layer_idx] = {}
            elif layer_key == "final":
                result["final"] = {}
        
        # Extract embeddings for each word at each layer
        for layer_key, activations in cache.items():
            if "blocks" in layer_key:
                layer_idx = int(layer_key.split(".")[1])
                layer_type = layer_key.split(".")[2]  # "mlp" or "attn"
                
                # Skip if we only want specific layers
                if layers is not None and layer_idx not in layers:
                    continue
                
                # Extract word embeddings
                for word_idx, (start_idx, end_idx) in enumerate(word_bounds):
                    # Get the embeddings for this word (average across its tokens)
                    word_activations = activations[0, start_idx:end_idx]  # [0] for batch dimension
                    
                    if len(word_activations) > 0:
                        word_embedding = torch.mean(word_activations, dim=0).numpy()
                        # Store by layer type
                        if f"{layer_type}_{layer_idx}" not in result:
                            result[f"{layer_type}_{layer_idx}"] = {}
                        result[f"{layer_type}_{layer_idx}"][word_idx] = word_embedding
            
            elif layer_key == "final":
                # Extract word embeddings from final layer
                for word_idx, (start_idx, end_idx) in enumerate(word_bounds):
                    word_activations = activations[0, start_idx:end_idx]
                    if len(word_activations) > 0:
                        word_embedding = torch.mean(word_activations, dim=0).numpy()
                        result["final"][word_idx] = word_embedding
        
        return result
    
    def extract_embeddings_with_sliding_window(self, sentence, layers=None):
        """
        Extract contextual embeddings using a sliding window approach,
        following Goldstein's methodology where each word's embedding 
        is influenced by all preceding context.
        
        Args:
            sentence: The input sentence
            layers: List of layer indices to extract
            
        Returns:
            Dictionary mapping layer indices to dictionaries of word embeddings
        """
        # Determine which layers to extract
        if layers is None:
            n_layers = self.model.cfg.n_layers
            layers = list(range(n_layers))
            
        # Split the sentence into words
        words = sentence.split()
        
        # Initialize result dictionary with the same structure as the previous method
        result = {}
        for layer_idx in layers:
            result[f"mlp_{layer_idx}"] = {}
            result[f"attn_{layer_idx}"] = {}
        result["final"] = {}
        
        # Process each word with its preceding context
        for word_idx, _ in enumerate(words):
            # Build context from all preceding words up to and including current word
            context = ' '.join(words[:word_idx+1])
            
            # Extract activations for this context
            cache = self.extract_layer_activations(context, layers)
            
            # For each layer, get the embedding for the last token
            for layer_key, activations in cache.items():
                if "blocks" in layer_key:
                    layer_idx = int(layer_key.split(".")[1])
                    layer_type = layer_key.split(".")[2]  # "mlp" or "attn"
                    
                    # Skip if we only want specific layers
                    if layers is not None and layer_idx not in layers:
                        continue
                    
                    # The embedding for the current word is from the last token(s)
                    # Here we'll take the last token's embedding, following Goldstein's approach
                    last_token_embedding = activations[0, -1].numpy()
                    result[f"{layer_type}_{layer_idx}"][word_idx] = last_token_embedding
                    
                elif layer_key == "final":
                    # Extract final layer embedding
                    last_token_embedding = activations[0, -1].numpy()
                    result["final"][word_idx] = last_token_embedding
        
        return result

def extract_embeddings_for_zuco(csv_path, model_name='gpt2-medium', layers=None, 
                               sliding_window=True, batch_size=16, device=None,
                               save_path=None):
    """
    Extract embeddings for all sentences in the Zuco dataset using TransformerLens.
    
    Args:
        csv_path: Path to the CSV file with sentences and task indices
        model_name: Name of the transformer model to use
        layers: List of layer indices to extract
        sliding_window: Whether to use sliding window approach (Goldstein method)
        batch_size: Batch size for processing
        device: Device to run the model on
        save_path: Optional path to save embeddings incrementally
        
    Returns:
        Dictionary mapping sentence_id to extracted embeddings
    """
    # Load sentences from CSV
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    # Create embedding extractor
    extractor = TransformerLensEmbeddingExtractor(model_name=model_name, device=device)
    
    # Initialize results
    results = {}
    
    # Process sentences
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing sentences"):
        sentence = row['sentence']
        nr_index = row['NR_index']
        tsr_index = row['TSR_index']
        
        # Use a unique sentence ID combining task and index
        # -100 means the sentence is not present in that task
        if nr_index != -100:
            sentence_id = f"NR_{nr_index}"
        elif tsr_index != -100:
            sentence_id = f"TSR_{tsr_index}"
        else:
            sentence_id = f"sentence_{idx}"
        
        # Extract embeddings based on approach
        if sliding_window:
            embeddings = extractor.extract_embeddings_with_sliding_window(
                sentence, layers=layers)
        else:
            embeddings = extractor.extract_embeddings_from_sentence(
                sentence, layers=layers)
        
        # Store results
        results[sentence_id] = {
            'sentence': sentence,
            'NR_index': nr_index,
            'TSR_index': tsr_index,
            'embeddings': embeddings
        }
        
        # Save incrementally if path is provided
        if save_path and (idx + 1) % 100 == 0:
            temp_save_path = Path(save_path).with_suffix(f".checkpoint_{idx+1}.pt")
            torch.save(results, temp_save_path)
            print(f"Checkpoint saved to {temp_save_path}")
    
    # Final save
    if save_path:
        final_save_path = Path(save_path)
        torch.save(results, final_save_path)
        print(f"Final embeddings saved to {final_save_path}")
    
    return results

def save_embeddings(embeddings, output_path):
    """
    Save extracted embeddings to disk.
    
    Args:
        embeddings: Dictionary of embeddings
        output_path: Path to save embeddings to
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save using torch.save for efficiency
    torch.save(embeddings, output_path)
    print(f"Embeddings saved to {output_path}")

def load_embeddings(input_path):
    """
    Load embeddings from disk.
    
    Args:
        input_path: Path to load embeddings from
        
    Returns:
        Dictionary of embeddings
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    embeddings = torch.load(input_path)
    print(f"Embeddings loaded from {input_path}")
    return embeddings

def get_layer_activations(model, text, layer=0, neuron_index=None):
    """
    Helper function to get activations at a specific layer, 
    similar to the one used in your test notebook.
    
    Args:
        model: TransformerLens model
        text: Input text
        layer: Layer index
        neuron_index: Optional specific neuron index
        
    Returns:
        Activations from the specified layer
    """
    cache = {}

    def caching_hook(act, hook):
        if neuron_index is not None:
            cache["activation"] = act[:, :, neuron_index]
        else: 
            cache["activation"] = act

    model.run_with_hooks(
        text, fwd_hooks=[(f"blocks.{layer}.mlp.hook_post", caching_hook)]
    )
    return utils.to_numpy(cache["activation"])

def extract_embeddings_with_dataloader(csv_path, model_name='gpt2-medium', layers=None, 
                                      sliding_window=True, batch_size=16, device=None):
    """
    Extract embeddings using your custom DataLoader with TransformerLens.
    
    Args:
        csv_path: Path to the CSV file
        model_name: Name of the transformer model to use
        layers: List of layer indices to extract
        sliding_window: Whether to use sliding window approach
        batch_size: Batch size for processing
        device: Device to run the model on
        
    Returns:
        Dictionary mapping sentence_id to extracted embeddings
    """
    from src.load_zuco_sentences import get_zuco_sentence_dataloader, TokenizerTransform
    from transformers import GPT2Tokenizer
    
    # Create tokenizer and transform
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    transform = TokenizerTransform(tokenizer)
    
    # Create dataloader
    dataloader = get_zuco_sentence_dataloader(
        csv_path=csv_path,
        transform=transform,
        batch_size=batch_size,
        shuffle=False  # Important: keep order for mapping
    )
    
    # Create embedding extractor
    extractor = TransformerLensEmbeddingExtractor(model_name=model_name, device=device)
    
    # Determine which layers to extract
    if layers is None:
        n_layers = extractor.model.cfg.n_layers
        layers = list(range(n_layers))
    
    # Initialize results
    results = {}
    
    # Process batches
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        sentences = batch['sentence']
        nr_indices = batch['NR_index']
        tsr_indices = batch['TSR_index']
        
        # Process each sentence in the batch individually
        # (TransformerLens hooks are easier to manage with single sentences)
        for i, sentence in enumerate(sentences):
            nr_index = nr_indices[i].item()
            tsr_index = tsr_indices[i].item()
            
            # Use a unique sentence ID
            if nr_index != -100:
                sentence_id = f"NR_{nr_index}"
            elif tsr_index != -100:
                sentence_id = f"TSR_{tsr_index}"
            else:
                sentence_id = f"sentence_{batch_idx}_{i}"
            
            # Extract embeddings based on approach
            if sliding_window:
                embeddings = extractor.extract_embeddings_with_sliding_window(
                    sentence, layers=layers)
            else:
                embeddings = extractor.extract_embeddings_from_sentence(
                    sentence, layers=layers)
            
            # Store results
            results[sentence_id] = {
                'sentence': sentence,
                'NR_index': nr_index,
                'TSR_index': tsr_index,
                'embeddings': embeddings
            }
    
    return results


# Example usage
if __name__ == "__main__":
    # Example: Extract embeddings for all sentences in Zuco dataset
    csv_path = "zuco_unique_sentences_with_task_indices.csv"
    model_name = "gpt2-medium"
    layers = [0, 6, 12, 23]  # Example: first, middle, and last layers
    save_path = "embeddings/zuco_gpt2_medium_embeddings.pt"
    
    # Extract embeddings with sliding window (Goldstein approach)
    embeddings = extract_embeddings_for_zuco(
        csv_path=csv_path,
        model_name=model_name,
        layers=layers,
        sliding_window=True,
        save_path=save_path
    )
    
    print(f"Extracted embeddings for {len(embeddings)} sentences")