import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# --- Global Settings ---
# random_seed: Initializes random number generators to ensure reproducible results.
# Running the script with the same seed keeps random operations, such as data splitting and weight initialization, consistent.
random_seed = 42
# batch_size: Number of samples processed in one training or evaluation iteration.
# A larger batch size can speed up training but requires more GPU memory.
batch_size = 32
# device: Determines whether computations run on the GPU (CUDA) or CPU.
# This script assumes CUDA is available and explicitly selects it.
device = torch.device('cuda')

# Set the PyTorch seed on all devices (CPU and CUDA) for reproducibility.
torch.manual_seed(random_seed)

# --- Model Definition ---
# Define a MobileNet class derived from nn.Module, the standard PyTorch model base class.
class MobileNet(nn.Module):
    def __init__(self, num_classes):
        # Initialize the base nn.Module class.
        super().__init__()
        # Use a predefined MobileNetV2 model with weights trained on ImageNet.
        # MobileNetV2 is a lightweight architecture designed specifically for mobile devices.
        # IMAGENET1K_V1 selects a specific, stable version of the pretrained weights.
        # Transfer learning reuses knowledge learned from ImageNet for waste classification.
        # MobileNetV2 uses depthwise separable convolutions and inverted residuals,
        # significantly reducing the parameter count while maintaining good accuracy.
        self.network = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # MobileNetV2 originally classifies 1,000 ImageNet classes.
        # Replace its final classifier with one output per waste class.
        # In MobileNetV2, classifier[1] is the final linear layer.
        # num_ftrs stores the number of inputs to the original classifier.
        num_ftrs = self.network.classifier[1].in_features
        # Replace the classifier with a linear layer that outputs num_classes logits.
        self.network.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, xb):
        # Define how data flows through the model.
        return self.network(xb)

# --- Training and Evaluation Functions ---
# Disable gradient calculation during evaluation to improve speed and reduce memory use.
@torch.no_grad()
def evaluate(model, val_loader):
    # Switch layers such as Dropout and BatchNorm to evaluation behavior.
    model.eval()
    # Store loss and accuracy for each batch.
    batch_losses = []
    batch_accs = []
    
    # Iterate over validation batches.
    for images, labels in val_loader:
        # Move images and labels to the selected device.
        images, labels = images.to(device), labels.to(device)
        
        # Run a forward pass to obtain logits.
        out = model(images)
        # Calculate the standard multiclass cross-entropy loss.
        loss = F.cross_entropy(out, labels)
        
        # Select the class with the highest logit for each sample.
        _, preds = torch.max(out, dim=1)
        # Compare predictions with labels to calculate batch accuracy.
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        
        # Detach the loss before storing it to avoid retaining the computation graph.
        batch_losses.append(loss.detach())
        batch_accs.append(acc)
            
    # Calculate mean loss and accuracy for the validation epoch.
    epoch_loss = torch.stack(batch_losses).mean().item()
    epoch_acc = torch.stack(batch_accs).mean().item()
    return {'val_loss': epoch_loss, 'val_acc': epoch_acc}

# Main training function.
# epochs: Number of training epochs.
# lr: Learning rate.
# model: Model to train.
# train_loader: Training DataLoader.
# val_loader: Validation DataLoader.
# opt_func: Optimizer class (Adam by default).
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    # Store metrics for every epoch.
    history = []
    # Initialize the optimizer with all trainable model parameters.
    optimizer = opt_func(model.parameters(), lr)
    
    # Iterate through the requested number of epochs.
    for epoch in range(epochs):
        # Switch layers such as Dropout and BatchNorm to training behavior.
        model.train()
        # Store the training loss for each batch in this epoch.
        train_losses = []
        
        # Display training progress with tqdm.
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training')
        # Iterate over training batches.
        for images, labels in train_progress:
            # Move data to the selected device.
            images, labels = images.to(device), labels.to(device)
            
            # --- Step 1: Forward pass ---
            out = model(images)
            
            # --- Step 2: Calculate loss ---
            loss = F.cross_entropy(out, labels)
            # Store a detached CPU copy to avoid retaining GPU computation history.
            train_losses.append(loss.detach().cpu())
            
            # --- Step 3: Backward pass ---
            loss.backward()
            
            # --- Step 4: Update weights ---
            optimizer.step()
            
            # --- Step 5: Clear accumulated gradients ---
            optimizer.zero_grad()
            
            # Show the current loss in the progress bar.
            train_progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # --- Validation phase after each epoch ---
        result = evaluate(model, val_loader)
        # Calculate the mean training loss for the completed epoch.
        result['train_loss'] = torch.stack(train_losses).mean().item()
        
        # Print metrics for the current epoch.
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))
        # Add the epoch metrics to the history.
        history.append(result)
        
    # Return the training history.
    return history

# --- Plotting Function ---
def plot_results(history):
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność Walidacyjna')
    plt.title('Dokładność vs. Liczba Epok (MobileNetV2)')
    plt.grid(True)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx', label='Treningowa')
    plt.plot(val_losses, '-rx', label='Walidacyjna')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()
    plt.title('Strata vs. Liczba Epok (MobileNetV2)')
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('./models/pytorch_mobilenet', exist_ok=True)
    plt.savefig('./models/pytorch_mobilenet/pytorch_mobilenet_results.png', dpi=150, bbox_inches='tight')
    plt.show()

# --- Main Script ---
if __name__ == "__main__":
    print("=== STARTING TRAINING - MobileNetV2 (Simplified, CUDA-Only, 60/10/30 Split) ===")
    print(f"Device: {device}")

    data_dir = "data/garbage_classification"
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory '{data_dir}' was not found.")
        exit()
        
    # MobileNetV2 transforms with a 224x224 input size.
    transformations = transforms.Compose([
        transforms.Resize((224, 224)),  # MobileNetV2 uses 224x224 images by default.
        transforms.ToTensor()
    ])
    
    print("Loading dataset...")
    try:
        dataset = ImageFolder(data_dir, transform=transformations)
    except FileNotFoundError:
        print(f"Error: Could not load the dataset from '{data_dir}'. Check the path.")
        exit()
        
    print(f"Classes found: {dataset.classes}")
    print(f"Total samples: {len(dataset)}")
    
    total_size = len(dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    print(f"Data split - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers=0, pin_memory=True)
    
    model = MobileNet(len(dataset.classes)).to(device)
    
    print("Evaluation before training:")
    initial_result = evaluate(model, val_dl) 
    print(f"Initial val_loss: {initial_result['val_loss']:.4f}, val_acc: {initial_result['val_acc']:.4f}")
    
    num_epochs = 8
    opt_func = torch.optim.Adam
    lr = 2e-4 
    
    print(f"\nTraining parameters:")
    print(f"- Model: MobileNetV2")
    print(f"- Epochs: {num_epochs}")
    print(f"- Optimizer: Adam")
    print(f"- Learning rate: {lr}")
    print(f"- Batch size: {batch_size} (train), {batch_size*2} (val)")
    print(f"- Image size: 224x224 (standard for mobile models)")
    
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    
    results = {
        'approach': 'PyTorch MobileNetV2 (CUDA-Only, Split 60/10/30)',
        'model': 'MobileNetV2',
        'epochs': num_epochs,
        'learning_rate': lr,
        'batch_size': batch_size,
        'optimizer': 'Adam',
        'input_size': '224x224',
        'final_val_acc': history[-1]['val_acc'],
        'final_val_loss': history[-1]['val_loss'],
        'final_train_loss': history[-1]['train_loss'],
        'history': history,
        'data_split': {'train': len(train_ds), 'val': len(val_ds), 'test': len(test_ds)}
    }
    
    os.makedirs('./models/pytorch_mobilenet', exist_ok=True)
    results_path = './models/pytorch_mobilenet/pytorch_mobilenet_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    plot_results(history)
    
    print(f"\n=== FINAL RESULTS - MobileNetV2 ===")
    print(f"Final validation accuracy: {history[-1]['val_acc']:.4f} ({history[-1]['val_acc']*100:.2f}%)")
    print(f"Final validation loss: {history[-1]['val_loss']:.4f}")
    print(f"Final training loss: {history[-1]['train_loss']:.4f}")
    
    model_path = './models/pytorch_mobilenet/pytorch_mobilenet_model.pth'
    torch.save(model.state_dict(), model_path)
