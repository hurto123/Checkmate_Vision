"""
Supervised Learning Training Script for ChessNet
Trains the neural network to imitate Grandmaster moves.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.model import ChessNet
from brain.data_utils import ChessDataset

def train_epoch(model, dataloader, optimizer, policy_criterion, value_criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    
    for batch_idx, (X, policy_target, value_target) in enumerate(dataloader):
        X = X.to(device)
        policy_target = policy_target.to(device)
        value_target = value_target.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        policy_out, value_out = model(X)
        
        # Policy loss (Cross Entropy)
        policy_loss = policy_criterion(policy_out, policy_target)
        
        # Value loss (MSE)
        value_loss = value_criterion(value_out, value_target)
        
        # Combined loss
        loss = policy_loss + value_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        
    n_batches = len(dataloader)
    return total_loss/n_batches, total_policy_loss/n_batches, total_value_loss/n_batches


def validate(model, dataloader, policy_criterion, value_criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, policy_target, value_target in dataloader:
            X = X.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device).unsqueeze(1)
            
            policy_out, value_out = model(X)
            
            policy_loss = policy_criterion(policy_out, policy_target)
            value_loss = value_criterion(value_out, value_target)
            loss = policy_loss + value_loss
            
            total_loss += loss.item()
            
            # Calculate policy accuracy (top-1)
            _, predicted = policy_out.max(1)
            correct += predicted.eq(policy_target).sum().item()
            total += policy_target.size(0)
    
    n_batches = len(dataloader)
    accuracy = 100. * correct / total
    return total_loss/n_batches, accuracy


def train(data_dir, output_dir, epochs=50, batch_size=256, lr=0.001, 
          num_res_blocks=10, num_channels=128):
    """
    Main training function.
    
    Args:
        data_dir: Directory containing X.npy, policy.npy, value.npy
        output_dir: Directory to save model checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        num_res_blocks: Number of residual blocks in ChessNet
        num_channels: Number of channels in residual blocks
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset from {data_dir}")
    dataset = ChessDataset(data_dir)
    print(f"Dataset size: {len(dataset)} positions")
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = ChessNet(num_res_blocks=num_res_blocks, num_channels=num_channels)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    best_val_acc = 0
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(1, epochs + 1):
        train_loss, train_p_loss, train_v_loss = train_epoch(
            model, train_loader, optimizer, policy_criterion, value_criterion, device
        )
        
        val_loss, val_acc = validate(model, val_loader, policy_criterion, value_criterion, device)
        
        scheduler.step()
        
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} (P: {train_p_loss:.4f}, V: {train_v_loss:.4f}) | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"  -> Saved best model (acc: {val_acc:.2f}%)")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    print("\n" + "="*60)
    print(f"Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    print("="*60)
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ChessNet with Supervised Learning')
    parser.add_argument('--data', type=str, default='dataset', help='Data directory')
    parser.add_argument('--output', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--res-blocks', type=int, default=10, help='Number of residual blocks')
    parser.add_argument('--channels', type=int, default=128, help='Number of channels')
    
    args = parser.parse_args()
    
    # Check if data exists
    if not os.path.exists(os.path.join(args.data, 'X.npy')):
        print("ERROR: Dataset not found!")
        print("First run: python -m brain.data_utils to create dataset from PGN")
        print("Or download a PGN file and use create_dataset() function")
        exit(1)
    
    train(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_res_blocks=args.res_blocks,
        num_channels=args.channels
    )
