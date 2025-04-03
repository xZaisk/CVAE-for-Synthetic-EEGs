import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

# Directory containing the dataset files
data_dir = "bci_iv_2a_data"  # Update this if your directory name is different

def extract_trials_from_continuous(X, trial_indices, trial_length=750):
    """
    Extract fixed-length trials from continuous EEG data.
    
    Args:
        X: Continuous EEG data
        trial_indices: Start indices for each trial
        trial_length: Length of each trial in samples (3 seconds at 250 Hz = 750 samples)
        
    Returns:
        Extracted trials, shape (n_trials, n_channels, trial_length)
    """
    # Debug prints
    print(f"X type: {type(X)}")
    print(f"X shape: {X.shape if hasattr(X, 'shape') else 'No shape attribute'}")
    
    # Make sure X is properly formatted as a numpy array
    if not isinstance(X, np.ndarray):
        try:
            X = np.array(X)
            print(f"Converted X to numpy array, new shape: {X.shape}")
        except:
            print("Failed to convert X to numpy array")
            return np.array([])
    
    # If X is a 1D array, reshape to 2D (assuming a single channel)
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
        print(f"Reshaped X to 2D, new shape: {X.shape}")
    
    # Make sure trial_indices is a 1D array
    if isinstance(trial_indices, np.ndarray) and len(trial_indices.shape) > 1:
        trial_indices = trial_indices.flatten()
    
    n_trials = len(trial_indices)
    
    # Make sure X has at least 2 dimensions
    if len(X.shape) < 2:
        print(f"Error: X has fewer than 2 dimensions: {X.shape}")
        return np.array([])
    
    n_channels = X.shape[1]
    trials = np.zeros((n_trials, n_channels, trial_length))
    
    for i, start_idx in enumerate(trial_indices):
        # Convert to int - this is critical!
        start_idx = int(start_idx)
        
        # Ensure we don't go beyond the data's end
        if start_idx + trial_length <= X.shape[0]:
            # Extract and transpose to get (channel, time) format
            trials[i] = X[start_idx:start_idx+trial_length, :].T
    
    return trials

def load_and_preprocess_data(data_dir):
    """
    Load and preprocess the BCI Competition IV 2a dataset.
    
    Args:
        data_dir: Directory containing the .mat files
        
    Returns:
        X_all: Extracted EEG trials
        y_all: Class labels (0-3)
    """
    X_all = []
    y_all = []
    
    # Process each subject
    for subject in range(1, 10):
        print(f"Processing Subject {subject}...")
        
        # Load data files
        train_file = os.path.join(data_dir, f"A0{subject}T.mat")
        eval_file = os.path.join(data_dir, f"A0{subject}E.mat")
        
        # Process training data
        if os.path.exists(train_file):
            try:
                # Load the .mat file
                mat_data = scipy.io.loadmat(train_file)
                
                # Access the data more carefully
                data_struct = mat_data['data'][0]
                
                # Look at one session in detail to understand its structure
                if len(data_struct) > 3:  # Only if we have session 3
                    session_data = data_struct[3][0, 0]
                    print("\nDetailed analysis of session 3:")
                    for field_name in session_data.dtype.names:
                        field_data = session_data[field_name]
                        if isinstance(field_data, np.ndarray):
                            if field_data.dtype.kind == 'O':  # Object array
                                print(f"  {field_name}: Object array of shape {field_data.shape}")
                                if field_data.size > 0:
                                    first_item = field_data[0]
                                    print(f"    First item type: {type(first_item)}")
                                    print(f"    First item shape: {first_item.shape if hasattr(first_item, 'shape') else 'No shape'}")
                            else:
                                print(f"  {field_name}: Array of shape {field_data.shape}, dtype {field_data.dtype}")
                        else:
                            print(f"  {field_name}: Not an array, type {type(field_data)}")
                
                # Loop through all elements in the data structure
                for idx in range(len(data_struct)):
                    try:
                        # Access session data
                        session_data = data_struct[idx][0, 0]
                        
                        # Check if this session has trials and labels
                        field_names = session_data.dtype.names
                        
                        if 'trial' in field_names and 'y' in field_names:
                            # Get trial indices
                            trial_indices = session_data['trial']
                            
                            # Skip if trial_indices is empty
                            if trial_indices.size == 0:
                                continue
                            
                            # Get class labels
                            labels = session_data['y']
                            
                            # Skip if labels is empty
                            if labels.size == 0:
                                continue
                            
                            # Get EEG data - be extra careful here
                            X_field = session_data['X']
                            print(f"X_field type: {type(X_field)}, shape: {X_field.shape if hasattr(X_field, 'shape') else 'No shape'}")
                            
                            # Handle X field which might be wrapped in an object array
                            if X_field.dtype.kind == 'O' and X_field.size > 0:
                                X_continuous = X_field[0]
                                print(f"Extracted X_continuous from object array, type: {type(X_continuous)}")
                                if hasattr(X_continuous, 'shape'):
                                    print(f"X_continuous shape: {X_continuous.shape}")
                            else:
                                X_continuous = X_field
                            
                            # Make sure X_continuous is a numpy array
                            if not isinstance(X_continuous, np.ndarray):
                                try:
                                    X_continuous = np.array(X_continuous)
                                    print(f"Converted X_continuous to numpy array, shape: {X_continuous.shape}")
                                except:
                                    print("Failed to convert X_continuous to numpy array")
                                    continue
                            
                            # Extract trials
                            trials = extract_trials_from_continuous(X_continuous, trial_indices)
                            
                            # Skip if no trials were extracted
                            if len(trials) == 0:
                                continue
                            
                            # Flatten labels and convert to 0-3
                            if labels.dtype.kind == 'O' and labels.size > 0:
                                labels = labels[0]
                            
                            labels = labels.flatten() - 1
                            
                            # Ensure we have the same number of trials and labels
                            min_len = min(len(trials), len(labels))
                            if min_len == 0:
                                continue
                                
                            trials = trials[:min_len]
                            labels = labels[:min_len]
                            
                            print(f"Extracted {len(trials)} trials from session {idx}")
                            X_all.append(trials)
                            y_all.append(labels)
                    except Exception as e:
                        print(f"Error processing session {idx} for subject {subject}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                        
            except Exception as e:
                print(f"Error processing file {train_file}: {e}")
                continue
        
        # Process evaluation data (same approach)
        if os.path.exists(eval_file):
            try:
                eval_data = scipy.io.loadmat(eval_file)
                data_struct = eval_data['data'][0]
                
                for idx in range(len(data_struct)):
                    try:
                        session_data = data_struct[idx][0, 0]
                        field_names = session_data.dtype.names
                        
                        if 'trial' in field_names and 'y' in field_names:
                            # Get trial indices
                            trial_indices = session_data['trial']
                            
                            # Skip if trial_indices is empty
                            if trial_indices.size == 0:
                                continue
                            
                            # Get class labels
                            labels = session_data['y']
                            
                            # Skip if labels is empty
                            if labels.size == 0:
                                continue
                            
                            # Get EEG data - be extra careful here
                            X_field = session_data['X']
                            
                            # Handle X field which might be wrapped in an object array
                            if X_field.dtype.kind == 'O' and X_field.size > 0:
                                X_continuous = X_field[0]
                            else:
                                X_continuous = X_field
                            
                            # Make sure X_continuous is a numpy array
                            if not isinstance(X_continuous, np.ndarray):
                                try:
                                    X_continuous = np.array(X_continuous)
                                except:
                                    print("Failed to convert X_continuous to numpy array")
                                    continue
                            
                            # Extract trials
                            trials = extract_trials_from_continuous(X_continuous, trial_indices)
                            
                            # Skip if no trials were extracted
                            if len(trials) == 0:
                                continue
                            
                            # Flatten labels and convert to 0-3
                            if labels.dtype.kind == 'O' and labels.size > 0:
                                labels = labels[0]
                            
                            labels = labels.flatten() - 1
                            
                            # Ensure we have the same number of trials and labels
                            min_len = min(len(trials), len(labels))
                            if min_len == 0:
                                continue
                                
                            trials = trials[:min_len]
                            labels = labels[:min_len]
                            
                            print(f"Extracted {len(trials)} trials from session {idx} (eval)")
                            X_all.append(trials)
                            y_all.append(labels)
                    except Exception as e:
                        print(f"Error processing session {idx} for subject {subject} (eval): {e}")
                        continue
            except Exception as e:
                print(f"Error processing file {eval_file}: {e}")
                continue
    
    # Combine all data
    if len(X_all) > 0:
        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)
        
    return X_all, y_all

def normalize_data(X):
    """
    Normalize EEG data using z-score normalization.
    
    Args:
        X: EEG data, shape (n_trials, n_channels, n_times)
        
    Returns:
        Normalized EEG data
    """
    X_norm = np.zeros_like(X)
    
    for i in range(X.shape[0]):  # For each trial
        for c in range(X.shape[1]):  # For each channel
            channel_data = X[i, c, :]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            X_norm[i, c, :] = (channel_data - mean) / (std + 1e-10)
    
    return X_norm

def visualize_data(X, y, num_samples=1):
    """
    Visualize EEG trials.
    
    Args:
        X: EEG data, shape (n_trials, n_channels, n_times)
        y: Class labels
        num_samples: Number of samples to visualize per class
    """
    class_names = ["Left Hand", "Right Hand", "Feet", "Tongue"]
    
    for class_idx in range(4):
        # Find trials of this class
        class_indices = np.where(y == class_idx)[0]
        if len(class_indices) == 0:
            continue
            
        # Select a subset of trials
        selected_indices = class_indices[:num_samples]
        
        for idx in selected_indices:
            plt.figure(figsize=(15, 8))
            plt.suptitle(f"Class: {class_names[class_idx]}, Trial Index: {idx}")
            
            # Plot first few EEG channels
            num_channels_to_plot = min(6, X.shape[1])
            for ch in range(num_channels_to_plot):
                plt.subplot(num_channels_to_plot, 1, ch+1)
                plt.plot(X[idx, ch, :])
                plt.ylabel(f"Ch {ch}")
                
                if ch == num_channels_to_plot - 1:
                    plt.xlabel("Time Samples")
            
            plt.tight_layout()
            plt.show()

# Main execution function
def main():
    print("Loading and preprocessing BCI IV 2a dataset...")
    
    try:
        # Load and preprocess data
        X, y = load_and_preprocess_data(data_dir)
        
        if len(X) == 0:
            print("No trials were extracted. Please check the data structure.")
            return
            
        print(f"Dataset shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Class distribution: {np.bincount(y.astype(int))}")
        
        # Normalize the data
        X_norm = normalize_data(X)
        
        # Visualize examples
        visualize_data(X_norm, y, num_samples=1)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Reshape for CVAE input
        n_train_trials, n_channels, n_times = X_train.shape
        X_train_flat = X_train.reshape(n_train_trials, -1)
        
        n_test_trials = X_test.shape[0]
        X_test_flat = X_test.reshape(n_test_trials, -1)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_flat)
        y_train_tensor = torch.LongTensor(y_train.astype(int))
        
        X_test_tensor = torch.FloatTensor(X_test_flat)
        y_test_tensor = torch.LongTensor(y_test.astype(int))
        
        # Create DataLoaders
        batch_size = 32
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print("Data preparation complete!")
        
        # Save preprocessed data
        np.savez('eeg_data_preprocessed.npz', 
                 X_train=X_train, y_train=y_train, 
                 X_test=X_test, y_test=y_test)
        
        print("Preprocessed data saved to 'eeg_data_preprocessed.npz'")
        
        # Return loaders for model training
        return train_loader, test_loader
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()