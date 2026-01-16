"""
Self-Play Training Script for ChessNet (Reinforcement Learning)
Implements AlphaZero-style self-play training.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.model import ChessNet
from brain.mcts import MCTS
from brain.data_utils import board_to_tensor, move_to_index


class ReplayBuffer:
    """Experience replay buffer for self-play games."""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, policy, value):
        """Store a training example."""
        self.buffer.append((state, policy, value))
        
    def sample(self, batch_size):
        """Sample a random batch."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)
        return (
            torch.stack(states),
            torch.stack(policies),
            torch.tensor(values, dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


def play_game(model, mcts, device, temperature=1.0):
    """
    Play a single game of self-play.
    Returns list of (state, policy, value) tuples.
    """
    board = chess.Board()
    game_history = []
    
    while not board.is_game_over():
        # Get board tensor
        state_tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
        
        # Run MCTS to get move probabilities
        # For simplicity, we'll use visit counts as policy
        root = mcts.search(board)
        
        # Build policy from MCTS visit counts (placeholder - simplified)
        policy = np.zeros(4672)  # Simplified action space
        
        # Get best move from MCTS
        best_move = root  # MCTS.search returns the best move directly in our implementation
        
        # Store state (we'll update value later based on game outcome)
        game_history.append({
            'state': board_to_tensor(board),
            'turn': board.turn
        })
        
        # Make move
        board.push(best_move)
        
        # Limit game length
        if board.fullmove_number > 150:
            break
    
    # Determine game outcome
    if board.is_checkmate():
        # The player who just moved won
        winner = not board.turn  # Opposite of current turn
        result = 1.0 if winner == chess.WHITE else -1.0
    else:
        result = 0.0  # Draw
    
    # Create training examples with final values
    training_data = []
    for item in game_history:
        # Value from perspective of the player to move at that position
        value = result if item['turn'] == chess.WHITE else -result
        
        # Create dummy policy (in full implementation, this comes from MCTS)
        policy = torch.zeros(4672)
        
        training_data.append((
            torch.from_numpy(item['state']),
            policy,
            value
        ))
    
    return training_data, result


def train_step(model, optimizer, states, policies, values, device):
    """Single training step."""
    model.train()
    
    states = states.to(device)
    policy_targets = policies.to(device)
    value_targets = values.to(device).unsqueeze(1)
    
    optimizer.zero_grad()
    
    policy_out, value_out = model(states)
    
    # Policy loss (cross entropy with soft targets)
    policy_loss = nn.CrossEntropyLoss()(policy_out, policy_targets.argmax(dim=1))
    
    # Value loss (MSE)
    value_loss = nn.MSELoss()(value_out, value_targets)
    
    loss = policy_loss + value_loss
    loss.backward()
    optimizer.step()
    
    return loss.item(), policy_loss.item(), value_loss.item()


def self_play_train(output_dir, num_iterations=100, games_per_iter=10, 
                    train_steps=100, batch_size=32, mcts_sims=100):
    """
    Main self-play training loop.
    
    Args:
        output_dir: Directory to save checkpoints
        num_iterations: Number of training iterations
        games_per_iter: Number of self-play games per iteration
        train_steps: Number of training steps per iteration
        batch_size: Batch size for training
        mcts_sims: Number of MCTS simulations per move
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model (smaller for faster self-play)
    model = ChessNet(num_res_blocks=5, num_channels=64)
    model = model.to(device)
    
    # Try to load existing model
    model_path = os.path.join(output_dir, 'selfplay_best.pt')
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    replay_buffer = ReplayBuffer(capacity=100000)
    mcts = MCTS(model, num_simulations=mcts_sims)
    
    print("\n" + "="*60)
    print("Starting Self-Play Training")
    print(f"Iterations: {num_iterations}")
    print(f"Games per iteration: {games_per_iter}")
    print(f"MCTS simulations: {mcts_sims}")
    print("="*60 + "\n")
    
    total_games = 0
    wins = {'white': 0, 'black': 0, 'draw': 0}
    
    for iteration in range(1, num_iterations + 1):
        print(f"\n--- Iteration {iteration}/{num_iterations} ---")
        
        # Self-play phase
        print("Playing games...")
        for game_idx in range(games_per_iter):
            game_data, result = play_game(model, mcts, device)
            
            for state, policy, value in game_data:
                replay_buffer.push(state, policy, value)
            
            total_games += 1
            
            if result == 1.0:
                wins['white'] += 1
            elif result == -1.0:
                wins['black'] += 1
            else:
                wins['draw'] += 1
        
        print(f"Buffer size: {len(replay_buffer)}")
        print(f"Win stats - W: {wins['white']}, B: {wins['black']}, D: {wins['draw']}")
        
        # Training phase
        if len(replay_buffer) >= batch_size:
            print("Training...")
            total_loss = 0
            
            for step in range(train_steps):
                states, policies, values = replay_buffer.sample(batch_size)
                loss, p_loss, v_loss = train_step(
                    model, optimizer, states, policies, values, device
                )
                total_loss += loss
            
            avg_loss = total_loss / train_steps
            print(f"Avg training loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if iteration % 10 == 0:
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_games': total_games,
            }, os.path.join(output_dir, f'selfplay_iter_{iteration}.pt'))
            
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_games': total_games,
            }, model_path)
            
            print(f"Saved checkpoint (iteration {iteration})")
    
    print("\n" + "="*60)
    print("Self-Play Training Complete!")
    print(f"Total games played: {total_games}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Self-Play Training for ChessNet')
    parser.add_argument('--output', type=str, default='selfplay_checkpoints', 
                        help='Output directory')
    parser.add_argument('--iterations', type=int, default=100, 
                        help='Number of iterations')
    parser.add_argument('--games', type=int, default=10, 
                        help='Games per iteration')
    parser.add_argument('--mcts-sims', type=int, default=50, 
                        help='MCTS simulations per move')
    
    args = parser.parse_args()
    
    self_play_train(
        output_dir=args.output,
        num_iterations=args.iterations,
        games_per_iter=args.games,
        mcts_sims=args.mcts_sims
    )
