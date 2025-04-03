import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Parameters
latent_dim = 64  # Size of latent space
n_epochs = 100   # Number of training epochs
batch_size = 64  # Batch size
learning_rate = 1e-3  # Learning rate

# Load preprocessed data
data = np.load('eeg_data_preprocessed.npz')
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

# Get dimensions
n_trials, n_channels, n_times = X_train.shape
input_dim = n_channels * n_times  # Flattened input dimension
n_classes = 4  # Number of motor imagery classes

# Reshape data (flatten channels and time)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_flat)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_flat)
y_test_tensor = torch.LongTensor(y_test)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Function to create one-hot encoding
def one_hot_encode(labels, n_classes):
    one_hot = torch.zeros(labels.size(0), n_classes)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot

# CVAE Model
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, n_classes):
        super(CVAE, self).__init__()
        
        # Condition dimensions (one-hot encoded class)
        self.condition_dim = n_classes
        
        # Input dimensions
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.encoder_fc1 = nn.Linear(input_dim + self.condition_dim, 512)
        self.encoder_fc2 = nn.Linear(512, 256)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder layers
        self.decoder_fc1 = nn.Linear(latent_dim + self.condition_dim, 256)
        self.decoder_fc2 = nn.Linear(256, 512)
        self.decoder_output = nn.Linear(512, input_dim)
        
    def encode(self, x, c):
        # Concatenate inputs and conditions
        x_c = torch.cat([x, c], dim=1)
        
        # Pass through encoder layers
        h = F.relu(self.encoder_fc1(x_c))
        h = F.relu(self.encoder_fc2(h))
        
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
        
        # Pass through decoder layers
        h = F.relu(self.decoder_fc1(z_c))
        h = F.relu(self.decoder_fc2(h))
        
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

# Loss function
def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (mean squared error)
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    
    # Kullback-Leibler divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return MSE + KLD

# Initialize model, optimizer and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CVAE(input_dim, latent_dim, n_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        
        # Create one-hot encoded conditions
        conditions = one_hot_encode(labels, n_classes).to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(data, conditions)
        
        # Calculate loss
        loss = loss_function(recon_batch, data, mu, logvar)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Accumulate loss
        train_loss += loss.item()
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    
    # Return average loss
    return train_loss / len(train_loader.dataset)

# Testing function
def test(model, test_loader):
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            
            # Create one-hot encoded conditions
            conditions = one_hot_encode(labels, n_classes).to(device)
            
            # Forward pass
            recon_batch, mu, logvar = model(data, conditions)
            
            # Calculate loss
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    
    # Return average loss
    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    return test_loss

# Training loop
train_losses = []
test_losses = []

for epoch in range(1, n_epochs + 1):
    train_loss = train(model, train_loader, optimizer, epoch)
    test_loss = test(model, test_loader)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    
    # Save model checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
    }, f'cvae_checkpoint_epoch_{epoch}.pt')
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('CVAE Training and Testing Loss')
    plt.savefig('cvae_loss_curve.png')
    plt.close()

# Generate synthetic samples for each class
n_samples = 5
n_channels = X_train.shape[1]
n_times = X_train.shape[2]

plt.figure(figsize=(15, 10))

for class_idx in range(n_classes):
    # Generate samples
    gen_samples = model.generate(n_samples, class_idx)
    
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
plt.savefig('cvae_generated_samples.png')
plt.show()

# Save model
torch.save(model.state_dict(), 'cvae_final_model.pt')