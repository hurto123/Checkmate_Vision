import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = F.relu(x)
        return x

class ChessNet(nn.Module):
    def __init__(self, num_res_blocks=10, num_channels=256):
        super(ChessNet, self).__init__()
        
        # Initial Block
        # Input: 119 channels (history + color + castling rights, etc.) -> Standard AlphaZero representation
        # For simplicity in this demo, we might use fewer channels (e.g., 12 for just piece positions)
        # Let's assume standard 8x8 input with N channels.
        self.input_channels = 12 # 6 pieces * 2 colors
        
        self.conv_input = nn.Conv2d(self.input_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Residual Tower
        self.res_tower = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )
        
        # Policy Head (Move Probabilities)
        # Output: 8x8 * 73 (AlphaZero move planes) = 4672 possible moves
        self.conv_policy = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * 8 * 8, 4672) # Simplified for standard chess moves
        
        # Value Head (Win/Loss/Draw prediction)
        self.conv_value = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(8 * 8, 256)
        self.fc_value2 = nn.Linear(256, 1)

    def forward(self, x):
        # Input Layer
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = F.relu(x)
        
        # Residual Tower
        x = self.res_tower(x)
        
        # Policy Head
        p = self.conv_policy(x)
        p = self.bn_policy(p)
        p = F.relu(p)
        p = p.reshape(p.size(0), -1) # Flatten
        p = self.fc_policy(p)
        # No Softmax here if using CrossEntropyLoss, but for inference we'll need it.
        
        # Value Head
        v = self.conv_value(x)
        v = self.bn_value(v)
        v = F.relu(v)
        v = v.reshape(v.size(0), -1)
        v = self.fc_value1(v)
        v = F.relu(v)
        v = self.fc_value2(v)
        v = torch.tanh(v) # Output between -1 (Loss) and 1 (Win)
        
        return p, v

if __name__ == "__main__":
    # Test with random input
    # Batch size 1, 12 channels, 8x8 board
    sample_input = torch.randn(1, 12, 8, 8)
    model = ChessNet(num_res_blocks=5, num_channels=64) # Smaller model for testing
    policy, value = model(sample_input)
    
    print(f"Policy Output Shape: {policy.shape}") # Should be [1, 4672]
    print(f"Value Output Shape: {value.shape}")   # Should be [1, 1]
    print("ChessNet is ready!")
