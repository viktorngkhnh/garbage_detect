import os
import torch
from torch.utils.data import Subset
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
# CUDA is preferred, while CPU remains available as a slower fallback.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 224
normalization_mean = [0.485, 0.456, 0.406]
normalization_std = [0.229, 0.224, 0.225]

# Set the PyTorch seed on all devices (CPU and CUDA) for reproducibility.
torch.manual_seed(random_seed)

# --- Model Definition ---
# Define a ResNet class derived from nn.Module, the standard PyTorch model base class.
class ResNet(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze_backbone=True):
        # Initialize the base nn.Module class.
        super().__init__()
        # Use a predefined ResNet50 model with weights trained on ImageNet.
        # models.ResNet50_Weights.IMAGENET1K_V1 selects a specific, stable version of the weights.
        # Transfer learning reuses knowledge learned from ImageNet for waste classification.
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.network = models.resnet50(weights=weights)

        if freeze_backbone:
            for parameter in self.network.parameters():
                parameter.requires_grad = False
        
        # ResNet50 originally classifies 1,000 ImageNet classes.
        # Replace its final fully connected layer with one output per waste class.
        # num_ftrs stores the number of inputs to the original fully connected layer.
        num_ftrs = self.network.fc.in_features
        # Replace the fc layer with a linear layer that outputs num_classes logits.
        self.network.fc = nn.Linear(num_ftrs, num_classes)
    
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
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
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
        batch_samples = labels.size(0)
        total_loss += loss.item() * batch_samples
        total_correct += torch.sum(preds == labels).item()
        total_samples += batch_samples
            
    # Calculate mean loss and accuracy for the validation epoch.
    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples
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
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = opt_func(trainable_parameters, lr)
    best_state = None
    best_val_acc = -1.0
    
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

        if result['val_acc'] > best_val_acc:
            best_val_acc = result['val_acc']
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        
    # Return the training history.
    return history, best_state, best_val_acc


def stratified_split_indices(dataset, train_ratio=0.6, val_ratio=0.1):
    generator = torch.Generator().manual_seed(random_seed)
    train_indices = []
    val_indices = []
    test_indices = []

    for class_index in range(len(dataset.classes)):
        class_indices = [
            index for index, target in enumerate(dataset.targets)
            if target == class_index
        ]
        permutation = torch.randperm(len(class_indices), generator=generator).tolist()
        shuffled_indices = [class_indices[index] for index in permutation]
        train_count = int(train_ratio * len(shuffled_indices))
        val_count = int(val_ratio * len(shuffled_indices))

        train_indices.extend(shuffled_indices[:train_count])
        val_indices.extend(shuffled_indices[train_count:train_count + val_count])
        test_indices.extend(shuffled_indices[train_count + val_count:])

    return train_indices, val_indices, test_indices

# --- Plotting Function ---
def plot_results(history):
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność Walidacyjna')
    plt.title('Dokładność vs. Liczba Epok')
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
    plt.title('Strata vs. Liczba Epok')
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('./models/pytorch_resnet', exist_ok=True)
    plt.savefig('./models/pytorch_resnet/pytorch_replica_results_simplified.png', dpi=150, bbox_inches='tight')
    plt.close()

# --- Main Script ---
if __name__ == "__main__":
    print("=== STARTING RESNET50 TRANSFER LEARNING (60/10/30 Split) ===")
    print(f"Device: {device}")

    data_dir = "data/garbage_classification"
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory '{data_dir}' was not found.")
        exit()
        
    train_transformations = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_std)
    ])
    evaluation_transformations = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_std)
    ])
    
    print("Loading dataset...")
    try:
        dataset = ImageFolder(data_dir)
    except FileNotFoundError:
        print(f"Error: Could not load the dataset from '{data_dir}'. Check the path.")
        exit()
        
    print(f"Classes found: {dataset.classes}")
    print(f"Total samples: {len(dataset)}")
    
    train_indices, val_indices, test_indices = stratified_split_indices(dataset)

    train_ds = Subset(ImageFolder(data_dir, transform=train_transformations), train_indices)
    val_ds = Subset(ImageFolder(data_dir, transform=evaluation_transformations), val_indices)
    test_ds = Subset(ImageFolder(data_dir, transform=evaluation_transformations), test_indices)
    print(f"Data split - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    pin_memory = device.type == 'cuda'
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers=0, pin_memory=pin_memory)
    
    model = ResNet(len(dataset.classes)).to(device)
    
    print("Evaluation before training:")
    initial_result = evaluate(model, val_dl) 
    print(f"Initial val_loss: {initial_result['val_loss']:.4f}, val_acc: {initial_result['val_acc']:.4f}")
    
    num_epochs = 8
    opt_func = torch.optim.Adam
    lr = 1e-3
    
    print(f"\nTraining parameters:")
    print(f"- Epochs: {num_epochs}")
    print(f"- Optimizer: Adam")
    print(f"- Learning rate: {lr}")
    print(f"- Batch size: {batch_size} (train), {batch_size*2} (val)\n")
    
    history, best_state, best_val_acc = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    model.load_state_dict(best_state)
    
    results = {
        'approach': 'PyTorch transfer learning (frozen backbone, Split 60/10/30)',
        'model': 'ResNet50',
        'epochs': num_epochs,
        'learning_rate': lr,
        'batch_size': batch_size,
        'optimizer': 'Adam',
        'best_val_acc': best_val_acc,
        'class_names': dataset.classes,
        'input_size': input_size,
        'final_val_acc': history[-1]['val_acc'],
        'final_val_loss': history[-1]['val_loss'],
        'final_train_loss': history[-1]['train_loss'],
        'history': history,
        'data_split': {'train': len(train_ds), 'val': len(val_ds), 'test': len(test_ds)}
    }
    
    os.makedirs('./models/pytorch_resnet', exist_ok=True)
    results_path = './models/pytorch_resnet/pytorch_replica_results_simplified.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    plot_results(history)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Final validation accuracy: {history[-1]['val_acc']:.4f} ({history[-1]['val_acc']*100:.2f}%)")
    print(f"Final validation loss: {history[-1]['val_loss']:.4f}")
    print(f"Final training loss: {history[-1]['train_loss']:.4f}")

    test_dl = DataLoader(test_ds, batch_size*2, num_workers=0, pin_memory=pin_memory)
    final_test_result = evaluate(model, test_dl)
    print(f"Final test accuracy: {final_test_result['val_acc']:.4f}")
    
    model_path = './models/pytorch_resnet/waste_classifier_resnet50.pth'
    checkpoint = {
        'format_version': 1,
        'architecture': 'resnet50',
        'state_dict': model.state_dict(),
        'class_names': dataset.classes,
        'input_size': input_size,
        'normalization_mean': normalization_mean,
        'normalization_std': normalization_std,
        'best_val_acc': best_val_acc,
        'test_acc': final_test_result['val_acc']
    }
    torch.save(checkpoint, model_path)
    print(f"\nModel saved as '{model_path}'")
    print(f"Results saved to '{results_path}'")