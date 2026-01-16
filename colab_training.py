# Chess AI Training on Google Colab
# รันบน Colab เพื่อใช้ GPU ฟรี

"""
วิธีใช้:
1. ไปที่ https://colab.research.google.com/
2. File -> Upload notebook -> เลือกไฟล์นี้
3. Runtime -> Change runtime type -> GPU
4. รัน cells ทั้งหมด
"""

# ========================
# Cell 1: ติดตั้ง Dependencies
# ========================
# !pip install torch torchvision python-chess tqdm numpy

# ========================
# Cell 2: Import Libraries
# ========================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import chess
import random
import numpy as np
from collections import deque
from datetime import datetime
from tqdm import tqdm

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========================
# Cell 3: Chess Neural Network Model
# ========================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return F.relu(x)


class ChessNet(nn.Module):
    def __init__(self, num_res_blocks=10, num_channels=128):
        super().__init__()
        
        # Input: 12 channels (6 piece types x 2 colors)
        self.conv_input = nn.Conv2d(12, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4672)  # All possible moves
        
        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        for block in self.res_blocks:
            x = block(x)
        
        # Policy
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 8 * 8)
        policy = self.policy_fc(policy)
        
        # Value
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 64)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value


# ========================
# Cell 4: Board to Tensor Conversion
# ========================
def board_to_tensor(board):
    """Convert chess board to 12x8x8 tensor"""
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    piece_map = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = square // 8
            col = square % 8
            channel = piece_map[(piece.piece_type, piece.color)]
            tensor[channel, row, col] = 1.0
    
    return tensor


# ========================
# Cell 5: Smart AI Player
# ========================
class SmartAI:
    def __init__(self, model, exploration=0.1):
        self.model = model
        self.exploration = exploration
        self.device = next(model.parameters()).device
        
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
    
    def get_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        if random.random() < self.exploration:
            return random.choice(legal_moves)
        
        self.model.eval()
        best_move = None
        best_value = float('-inf')
        
        for move in legal_moves:
            board.push(move)
            state = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                _, value = self.model(state)
            
            board.pop()
            move_value = -value.item()
            
            if move_value > best_value:
                best_value = move_value
                best_move = move
        
        return best_move if best_move else random.choice(legal_moves)


# ========================
# Cell 6: Self-Play Training System
# ========================
class SelfPlayTrainer:
    def __init__(self, num_res_blocks=10, num_channels=128):
        self.model = ChessNet(num_res_blocks=num_res_blocks, num_channels=num_channels)
        self.model = self.model.to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.buffer = deque(maxlen=100000)
        
        self.stats = {'games': 0, 'white_wins': 0, 'black_wins': 0, 'draws': 0}
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def play_game(self, max_moves=150):
        board = chess.Board()
        game_data = []
        
        ai = SmartAI(self.model, exploration=0.3)
        
        while not board.is_game_over() and board.fullmove_number <= max_moves:
            state = board_to_tensor(board)
            turn = board.turn
            
            move = ai.get_move(board)
            if move is None:
                break
            
            game_data.append({'state': state, 'turn': turn})
            board.push(move)
        
        # Determine result
        if board.is_checkmate():
            result = -1 if board.turn else 1
            if board.turn:
                self.stats['black_wins'] += 1
            else:
                self.stats['white_wins'] += 1
        else:
            result = 0
            self.stats['draws'] += 1
        
        # Add to buffer
        for item in game_data:
            value = result if item['turn'] == chess.WHITE else -result
            self.buffer.append((
                torch.from_numpy(item['state']),
                value
            ))
        
        self.stats['games'] += 1
        return result, board.fullmove_number
    
    def train_batch(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return 0
        
        self.model.train()
        
        batch = random.sample(self.buffer, batch_size)
        states, values = zip(*batch)
        
        states = torch.stack(states).to(device)
        values = torch.tensor(values, dtype=torch.float32).to(device).unsqueeze(1)
        
        self.optimizer.zero_grad()
        
        _, value_out = self.model(states)
        loss = nn.MSELoss()(value_out, values)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_games=1000, train_every=10, batch_size=128, save_every=100):
        print(f"\n{'='*60}")
        print(f"  🧠 SELF-PLAY TRAINING ON GPU")
        print(f"{'='*60}")
        print(f"  Games: {num_games}")
        print(f"  Device: {device}")
        print(f"{'='*60}\n")
        
        total_loss = 0
        loss_count = 0
        
        for game_num in tqdm(range(1, num_games + 1), desc="Training"):
            result, moves = self.play_game()
            
            if game_num % train_every == 0 and len(self.buffer) >= batch_size:
                loss = self.train_batch(batch_size)
                total_loss += loss
                loss_count += 1
            
            if game_num % save_every == 0:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'stats': self.stats,
                }, f'/content/chess_model_game_{game_num}.pt')
                print(f"\n  💾 Saved checkpoint at game {game_num}")
        
        # Final save
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stats': self.stats,
        }, '/content/chess_model_final.pt')
        
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"  📊 TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"  Total Games:  {self.stats['games']}")
        print(f"  White Wins:   {self.stats['white_wins']}")
        print(f"  Black Wins:   {self.stats['black_wins']}")
        print(f"  Draws:        {self.stats['draws']}")
        print(f"  Avg Loss:     {avg_loss:.6f}")
        print(f"{'='*60}")


# ========================
# Cell 7: Run Training
# ========================
if __name__ == "__main__":
    # สร้าง Trainer
    trainer = SelfPlayTrainer(num_res_blocks=10, num_channels=128)
    
    # Train 1000 เกม
    trainer.train(
        num_games=1000,
        train_every=10,
        batch_size=128,
        save_every=200
    )
    
    print("\n✅ Training complete!")
    print("📁 Download file: /content/chess_model_final.pt")


# ========================
# Cell 8: Download Model (รันหลัง training เสร็จ)
# ========================
# from google.colab import files
# files.download('/content/chess_model_final.pt')
