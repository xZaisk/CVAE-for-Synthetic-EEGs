import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy import signal

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Parameters
latent_dim = 64  # Size of latent space
n_epochs = 50    # Number of training epochs
batch_size = 64  # Batch size
learning_rate = 1e-3  # Learning rate
lambda_l1 = 0.01      # Weight for L1 regularization
lambda_smooth = 0.05  # Weight for smoothness regularization
early_stopping_patience = 5  # Number of epochs to wait before early stopping

# Flag to use shorter sequences
use_shorter_sequences = True
sequence_length_factor = 0.5  # 50% of original length

# Number of cross-validation folds
n_folds = 5

# Load preprocessed data
data = np.load('eeg_data_preprocessed.npz')
X_full, y_full = data['X_train'], data['y_train']
X_test_full, y_test_full = data['X_test'], data['y_test']

# Get dimensions
n_trials, n_channels, n_times_original = X_full.shape

# Implement shorter sequences if enabled
if use_shorter_sequences:
    n_times = int(n_times_original * sequence_length_factor)
    print(f"Using shorter sequences: {n_times} time points instead of {n_times_original}")
    
    # Trim sequences to shorter length (take middle section)
    start_idx = (n_times_original - n_times) // 2
    end_idx = start_idx + n_times
    
    X_full = X_full[:, :, start_idx:end_idx]
    X_test_full = X_test_full[:, :, start_idx:end_idx]
else:
    n_times = n_times_original

# Update input dimension based on possibly shortened sequences
input_dim = n_channels * n_times
n_classes = 4  # Number of motor imagery classes

# Flatten data
X_full_flat = X_full.reshape(X_full.shape[0], -1)
X_test_flat = X_test_full.reshape(X_test_full.shape[0], -1)

# Function to create one-hot encoding
def one_hot_encode(labels, n_classes):
    one_hot = torch.zeros(labels.size(0), n_classes)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot

# CVAE Model with dropout
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, n_classes, dropout_rate=0.2):
        super(CVAE, self).__init__()
        
        # Condition dimensions (one-hot encoded class)
        self.condition_dim = n_classes
        
        # Input dimensions
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.encoder_fc1 = nn.Linear(input_dim + self.condition_dim, 512)
        self.encoder_dropout1 = nn.Dropout(dropout_rate)
        self.encoder_fc2 = nn.Linear(512, 256)
        self.encoder_dropout2 = nn.Dropout(dropout_rate)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder layers
        self.decoder_fc1 = nn.Linear(latent_dim + self.condition_dim, 256)
        self.decoder_dropout1 = nn.Dropout(dropout_rate)
        self.decoder_fc2 = nn.Linear(256, 512)
        self.decoder_dropout2 = nn.Dropout(dropout_rate)
        self.decoder_output = nn.Linear(512, input_dim)
        
    def encode(self, x, c):
        # Concatenate inputs and conditions
        x_c = torch.cat([x, c], dim=1)
        
        # Pass through encoder layers with dropout
        h = F.relu(self.encoder_fc1(x_c))
        h = self.encoder_dropout1(h)
        h = F.relu(self.encoder_fc2(h))
        h = self.encoder_dropout2(h)
        
        # Get mean and log variance of latent distribution
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # Apply reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z
    
    def decode(self, z, c):
        # Concatenate latent vector and condition
        z_c = torch.cat([z, c], dim=1)
        
        # Pass through decoder layers with dropout
        h = F.relu(self.decoder_fc1(z_c))
        h = self.decoder_dropout1(h)
        h = F.relu(self.decoder_fc2(h))
        h = self.decoder_dropout2(h)
        
        # Output reconstruction
        recon_x = self.decoder_output(h)
        
        return recon_x
    
    def forward(self, x, c):
        # Encode input
        mu, logvar = self.encode(x, c)
        
        # Sample from latent space
        z = self.reparameterize(mu, logvar)
        
        # Decode and reconstruct
        recon_x = self.decode(z, c)
        
        return recon_x, mu, logvar
    
    def generate(self, n_samples, class_label):
        """
        Generate synthetic EEG signals for a specific class
        
        Args:
            n_samples: Number of samples to generate
            class_label: Class label (0-3)
            
        Returns:
            Synthetic EEG signals
        """
        # Create one-hot encoded class condition
        device = next(self.parameters()).device
        c = torch.zeros(n_samples, self.condition_dim, device=device)
        c[:, class_label] = 1
        
        # Sample from latent space
        z = torch.randn(n_samples, self.latent_dim, device=device)
        
        # Decode to generate synthetic data
        with torch.no_grad():
            gen_x = self.decode(z, c)
        
        return gen_x

# Smoothness penalty function
def smoothness_penalty(x, n_channels, n_times):
    """Calculate temporal smoothness penalty"""
    # Reshape to (batch_size, n_channels, n_times)
    x_reshaped = x.view(-1, n_channels, n_times)
    
    # Calculate temporal differences
    temp_diff = x_reshaped[:, :, 1:] - x_reshaped[:, :, :-1]
    
    # Return sum of squared differences
    return torch.sum(temp_diff**2)

# Enhanced loss function with multiple regularization terms
def loss_function(recon_x, x, mu, logvar, n_channels, n_times, lambda_l1=0.01, lambda_smooth=0.05):
    # Reconstruction loss (mean squared error)
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    
    # Kullback-Leibler divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # L1 regularization for sparsity
    L1_loss = lambda_l1 * torch.sum(torch.abs(recon_x))
    
    # Smoothness penalty
    smooth_loss = lambda_smooth * smoothness_penalty(recon_x, n_channels, n_times)
    
    # Log individual loss components for monitoring
    total_loss = MSE + KLD + L1_loss + smooth_loss
    
    if torch.isnan(total_loss):
        print(f"NaN detected! MSE: {MSE}, KLD: {KLD}, L1: {L1_loss}, Smooth: {smooth_loss}")
    
    return total_loss, {
        'mse': MSE.item(),
        'kld': KLD.item(),
        'l1': L1_loss.item(),
        'smooth': smooth_loss.item()
    }

# Training function - updated to track component losses
def train(model, train_loader, optimizer, epoch, n_channels, n_times):
    model.train()
    train_loss = 0
    mse_loss_sum = 0
    kld_loss_sum = 0
    l1_loss_sum = 0
    smooth_loss_sum = 0
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        
        # Create one-hot encoded conditions
        conditions = one_hot_encode(labels, n_classes).to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(data, conditions)
        
        # Calculate loss with L1 regularization
        loss, loss_components = loss_function(
            recon_batch, data, mu, logvar, 
            n_channels, n_times, lambda_l1, lambda_smooth
        )
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Accumulate losses
        train_loss += loss.item()
        mse_loss_sum += loss_components['mse']
        kld_loss_sum += loss_components['kld']
        l1_loss_sum += loss_components['l1']
        smooth_loss_sum += loss_components['smooth']
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    
    # Calculate average losses
    avg_loss = train_loss / len(train_loader.dataset)
    avg_mse = mse_loss_sum / len(train_loader.dataset)
    avg_kld = kld_loss_sum / len(train_loader.dataset)
    avg_l1 = l1_loss_sum / len(train_loader.dataset)
    avg_smooth = smooth_loss_sum / len(train_loader.dataset)
    
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, '
          f'KLD: {avg_kld:.4f}, L1: {avg_l1:.4f}, Smooth: {avg_smooth:.4f}')
    
    # Return average loss and components
    return avg_loss, {
        'mse': avg_mse,
        'kld': avg_kld,
        'l1': avg_l1,
        'smooth': avg_smooth
    }

# Testing function - updated to track component losses
def test(model, test_loader, n_channels, n_times):
    model.eval()
    test_loss = 0
    mse_loss_sum = 0
    kld_loss_sum = 0
    l1_loss_sum = 0
    smooth_loss_sum = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            
            # Create one-hot encoded conditions
            conditions = one_hot_encode(labels, n_classes).to(device)
            
            # Forward pass
            recon_batch, mu, logvar = model(data, conditions)
            
            # Calculate loss with L1 regularization
            loss, loss_components = loss_function(
                recon_batch, data, mu, logvar,
                n_channels, n_times, lambda_l1, lambda_smooth
            )
            
            # Accumulate losses
            test_loss += loss.item()
            mse_loss_sum += loss_components['mse']
            kld_loss_sum += loss_components['kld']
            l1_loss_sum += loss_components['l1']
            smooth_loss_sum += loss_components['smooth']
    
    # Calculate average losses
    avg_loss = test_loss / len(test_loader.dataset)
    avg_mse = mse_loss_sum / len(test_loader.dataset)
    avg_kld = kld_loss_sum / len(test_loader.dataset)
    avg_l1 = l1_loss_sum / len(test_loader.dataset)
    avg_smooth = smooth_loss_sum / len(test_loader.dataset)
    
    print(f'====> Test set loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, '
          f'KLD: {avg_kld:.4f}, L1: {avg_l1:.4f}, Smooth: {avg_smooth:.4f}')
    
    # Return average loss and components
    return avg_loss, {
        'mse': avg_mse,
        'kld': avg_kld,
        'l1': avg_l1,
        'smooth': avg_smooth
    }

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prepare for cross-validation
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store fold results
fold_train_losses = []
fold_test_losses = []
fold_models = []

# Cross-validation loop
for fold, (train_indices, val_indices) in enumerate(kf.split(X_full_flat)):
    print(f"\n--- Fold {fold+1}/{n_folds} ---")
    
    # Split data for this fold
    X_train_fold = X_full_flat[train_indices]
    y_train_fold = y_full[train_indices]
    X_val_fold = X_full_flat[val_indices]
    y_val_fold = y_full[val_indices]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_fold)
    y_train_tensor = torch.LongTensor(y_train_fold)
    X_val_tensor = torch.FloatTensor(X_val_fold)
    y_val_tensor = torch.LongTensor(y_val_fold)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and optimizer for this fold
    model = CVAE(input_dim, latent_dim, n_classes, dropout_rate=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    # Training loop for this fold
    train_losses = []
    val_losses = []
    loss_components_train = {'mse': [], 'kld': [], 'l1': [], 'smooth': []}
    loss_components_val = {'mse': [], 'kld': [], 'l1': [], 'smooth': []}
    
    best_model_state = None
    best_val_loss = float('inf')
    
    for epoch in range(1, n_epochs + 1):
        # Train the model
        train_loss, train_components = train(model, train_loader, optimizer, epoch, n_channels, n_times)
        val_loss, val_components = test(model, val_loader, n_channels, n_times)
        
        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Record loss components
        for key in loss_components_train:
            if key in train_components:
                loss_components_train[key].append(train_components[key])
            if key in val_components:
                loss_components_val[key].append(val_components[key])
        
        # Check for best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Save the best model for this fold
    model.load_state_dict(best_model_state)
    fold_models.append(model)
    
    # Save loss history for this fold
    fold_train_losses.append(train_losses)
    fold_test_losses.append(val_losses)
    
    # Plot loss curves for this fold
    plt.figure(figsize=(15, 10))
    
    # Plot overall loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.title(f'Fold {fold+1} - CVAE Total Loss')
    
    # Plot MSE component
    plt.subplot(2, 2, 2)
    plt.plot(loss_components_train['mse'], label='Train MSE')
    plt.plot(loss_components_val['mse'], label='Val MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Reconstruction Loss (MSE)')
    
    # Plot KLD component
    plt.subplot(2, 2, 3)
    plt.plot(loss_components_train['kld'], label='Train KLD')
    plt.plot(loss_components_val['kld'], label='Val KLD')
    plt.xlabel('Epoch')
    plt.ylabel('KLD Loss')
    plt.legend()
    plt.title('KL Divergence Loss')
    
    # Plot L1 + Smooth components
    plt.subplot(2, 2, 4)
    plt.plot(loss_components_train['l1'], label='Train L1')
    plt.plot(loss_components_val['l1'], label='Val L1')
    plt.plot(loss_components_train['smooth'], label='Train Smooth')
    plt.plot(loss_components_val['smooth'], label='Val Smooth')
    plt.xlabel('Epoch')
    plt.ylabel('Regularization Loss')
    plt.legend()
    plt.title('Regularization Losses')
    
    plt.tight_layout()
    plt.savefig(f'cvae_loss_curves_fold_{fold+1}.png')
    plt.close()
    
    # Save fold model
    torch.save({
        'fold': fold+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'loss_components_train': loss_components_train,
        'loss_components_val': loss_components_val,
    }, f'cvae_model_fold_{fold+1}.pt')

# Evaluate models on the test set
print("\n--- Final Evaluation on Test Set ---")

# Convert test data to tensors
X_test_tensor = torch.FloatTensor(X_test_flat)
y_test_tensor = torch.LongTensor(y_test_full)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate each fold's model
fold_test_metrics = []
for fold, model in enumerate(fold_models):
    print(f"Evaluating model from fold {fold+1}")
    test_loss, test_components = test(model, test_loader, n_channels, n_times)
    fold_test_metrics.append((test_loss, test_components))

# Find best fold model (lowest test loss)
best_fold = np.argmin([metrics[0] for metrics in fold_test_metrics])
print(f"\nBest model is from fold {best_fold+1} with test loss: {fold_test_metrics[best_fold][0]:.4f}")

# Use the best model for generation
best_model = fold_models[best_fold]

# Generate synthetic samples for each class
n_samples = 5

plt.figure(figsize=(15, 10))

for class_idx in range(n_classes):
    # Generate samples
    gen_samples = best_model.generate(n_samples, class_idx)
    
    # Reshape back to (channels, time)
    gen_samples = gen_samples.cpu().numpy().reshape(n_samples, n_channels, n_times)
    
    # Plot
    for i in range(n_samples):
        plt.subplot(n_classes, n_samples, class_idx * n_samples + i + 1)
        plt.plot(gen_samples[i, 0, :])  # Plot first channel
        if i == 0:
            class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
            plt.title(f'Class: {class_names[class_idx]}')
        plt.axis('off')

plt.tight_layout()
plt.savefig('cvae_generated_samples_best_model.png')

# Compare original and generated samples
plt.figure(figsize=(15, 10))

for class_idx in range(n_classes):
    # Get original samples for this class
    orig_indices = np.where(y_test_full == class_idx)[0][:n_samples]
    orig_samples = X_test_full[orig_indices]
    
    # Generate samples
    gen_samples = best_model.generate(n_samples, class_idx)
    gen_samples = gen_samples.cpu().numpy().reshape(n_samples, n_channels, n_times)
    
    # Plot original vs generated for each sample
    for i in range(n_samples):
        # Original sample
        plt.subplot(n_classes, 2*n_samples, 2*class_idx*n_samples + 2*i + 1)
        plt.plot(orig_samples[i, 0, :])  # Plot first channel
        if i == 0:
            plt.title(f'Original - {class_names[class_idx]}')
        plt.axis('off')
        
        # Generated sample
        plt.subplot(n_classes, 2*n_samples, 2*class_idx*n_samples + 2*i + 2)
        plt.plot(gen_samples[i, 0, :])  # Plot first channel
        if i == 0:
            plt.title(f'Generated - {class_names[class_idx]}')
        plt.axis('off')

plt.tight_layout()
plt.savefig('cvae_original_vs_generated_best_model.png')

# Additional analysis - Frequency domain comparison
plt.figure(figsize=(15, 10))

for class_idx in range(n_classes):
    # Get original samples for this class
    orig_indices = np.where(y_test_full == class_idx)[0][:1]  # Just use one sample
    orig_sample = X_test_full[orig_indices][0, 0, :]  # First sample, first channel
    
    # Generate a sample
    gen_sample = best_model.generate(1, class_idx)
    gen_sample = gen_sample.cpu().numpy().reshape(1, n_channels, n_times)[0, 0, :]
    
    # Calculate power spectrum
    f_orig, Pxx_orig = signal.welch(orig_sample, fs=250, nperseg=min(256, len(orig_sample)))
    f_gen, Pxx_gen = signal.welch(gen_sample, fs=250, nperseg=min(256, len(gen_sample)))
    
    # Plot time domain
    plt.subplot(n_classes, 3, 3*class_idx + 1)
    plt.plot(orig_sample)
    plt.title(f'Original - {class_names[class_idx]}')
    plt.axis('tight')
    
    plt.subplot(n_classes, 3, 3*class_idx + 2)
    plt.plot(gen_sample)
    plt.title(f'Generated - {class_names[class_idx]}')
    plt.axis('tight')
    
    # Plot frequency domain
    plt.subplot(n_classes, 3, 3*class_idx + 3)
    plt.semilogy(f_orig, Pxx_orig, label='Original')
    plt.semilogy(f_gen, Pxx_gen, label='Generated')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V^2/Hz]')
    plt.title('Power Spectrum')
    plt.legend()

plt.tight_layout()
plt.savefig('cvae_spectral_analysis_best_model.png')

# Save best model
torch.save({
    'model_state_dict': best_model.state_dict(),
    'fold': best_fold + 1,
    'sequence_length': n_times,
    'original_sequence_length': n_times_original,
    'use_shorter_sequences': use_shorter_sequences,
    'input_dim': input_dim,
    'latent_dim': latent_dim,
    'n_classes': n_classes,
    'lambda_l1': lambda_l1,
    'lambda_smooth': lambda_smooth,
}, 'cvae_best_model.pt')

print("Training complete! Best model saved.")
