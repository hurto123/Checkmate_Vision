"""
Chess Data Utils - Convert PGN games to training data for ChessNet
"""

import chess
import chess.pgn
import numpy as np
import torch
import os
from tqdm import tqdm

# Piece to channel mapping (12 channels total)
PIECE_TO_CHANNEL = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def board_to_tensor(board):
    """
    Convert a chess.Board to a 12x8x8 tensor.
    Channels 0-5: White pieces (P, N, B, R, Q, K)
    Channels 6-11: Black pieces (p, n, b, r, q, k)
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            
            channel = PIECE_TO_CHANNEL[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6  # Black pieces in channels 6-11
                
            tensor[channel, rank, file] = 1.0
            
    return tensor


def move_to_index(move, board):
    """
    Convert a chess move to a policy index (0-4671).
    AlphaZero uses 73 planes for move encoding.
    Simplified version: from_square * 64 + to_square (4096 possible)
    We'll use a simpler encoding for this demo.
    """
    from_sq = move.from_square
    to_sq = move.to_square
    
    # Simple encoding: from * 64 + to
    # This gives 4096 possible moves (enough for basic moves)
    # Promotion moves need special handling
    promotion_offset = 0
    if move.promotion:
        # Add offset for promotions (N=1, B=2, R=3, Q=4) - pawn is 0
        promo_pieces = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
        promotion_offset = promo_pieces.get(move.promotion, 0) * 64
        
    return from_sq * 64 + to_sq + promotion_offset


def parse_pgn_file(pgn_path, max_games=None):
    """
    Parse a PGN file and extract (board_tensor, move_index, result) tuples.
    
    result: 1.0 for white win, -1.0 for black win, 0.0 for draw
    """
    data = []
    
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
        game_count = 0
        
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
                
            if max_games and game_count >= max_games:
                break
            
            # Get result
            result_str = game.headers.get('Result', '*')
            if result_str == '1-0':
                result = 1.0
            elif result_str == '0-1':
                result = -1.0
            elif result_str == '1/2-1/2':
                result = 0.0
            else:
                continue  # Skip incomplete games
            
            # Replay game and collect positions
            board = game.board()
            for move in game.mainline_moves():
                # Convert current position to tensor
                tensor = board_to_tensor(board)
                
                # Convert move to index
                move_idx = move_to_index(move, board)
                
                # Value from perspective of current player
                value = result if board.turn == chess.WHITE else -result
                
                data.append((tensor, move_idx, value))
                
                # Make the move
                board.push(move)
            
            game_count += 1
            
            if game_count % 100 == 0:
                print(f"Processed {game_count} games, {len(data)} positions")
    
    print(f"Total: {game_count} games, {len(data)} positions")
    return data


def create_dataset(pgn_path, output_dir, max_games=10000):
    """
    Create training dataset from PGN file.
    Saves: X.npy (board tensors), policy.npy (move indices), value.npy (results)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Parsing PGN: {pgn_path}")
    data = parse_pgn_file(pgn_path, max_games)
    
    if len(data) == 0:
        print("No data collected!")
        return
    
    # Split into arrays
    X = np.array([d[0] for d in data], dtype=np.float32)
    policy = np.array([d[1] for d in data], dtype=np.int64)
    value = np.array([d[2] for d in data], dtype=np.float32)
    
    # Save
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'policy.npy'), policy)
    np.save(os.path.join(output_dir, 'value.npy'), value)
    
    print(f"Saved dataset to {output_dir}")
    print(f"  X.npy: {X.shape}")
    print(f"  policy.npy: {policy.shape}")
    print(f"  value.npy: {value.shape}")


class ChessDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for chess positions."""
    
    def __init__(self, data_dir):
        self.X = np.load(os.path.join(data_dir, 'X.npy'))
        self.policy = np.load(os.path.join(data_dir, 'policy.npy'))
        self.value = np.load(os.path.join(data_dir, 'value.npy'))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.policy[idx], dtype=torch.long),
            torch.tensor(self.value[idx], dtype=torch.float32)
        )


if __name__ == "__main__":
    # Example usage
    print("Chess Data Utils")
    print("-" * 40)
    print("To create dataset from PGN file:")
    print("  from data_utils import create_dataset")
    print("  create_dataset('games.pgn', 'dataset/', max_games=10000)")
    
    # Quick test
    board = chess.Board()
    tensor = board_to_tensor(board)
    print(f"\nTest board_to_tensor: {tensor.shape}")  # Should be (12, 8, 8)
