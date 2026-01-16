import math
import numpy as np
import torch
import chess

class Node:
    def __init__(self, state, parent=None, action_taken=None, prior=0):
        self.state = state  # chess.Board object
        self.parent = parent
        self.action_taken = action_taken
        self.children = []
        
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior  # Probability from Policy Network
        
    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct=1.0):
        # Upper Confidence Bound applied to Trees
        # Q + U
        q_value = self.value()
        u_value = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return q_value + u_value

class MCTS:
    def __init__(self, model, c_puct=1.0, num_simulations=800):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def search(self, board):
        root = Node(board, prior=1.0)
        
        for _ in range(self.num_simulations):
            node = root
            
            # Selection
            while node.is_expanded():
                node = self.select_child(node)
                
            # Expansion & Evaluation
            value, policy = self.evaluate(node.state)
            self.expand(node, policy)
            
            # Backpropagation
            self.backpropagate(node, value)
            
        # Select best move based on visit count
        # In competitive play, choose max visit. In training, sample from distribution.
        best_child = max(root.children, key=lambda child: child.visit_count)
        
        move = best_child.action_taken
        return move

    def select_child(self, node):
        # Select child with highest UCB score
        return max(node.children, key=lambda child: child.ucb_score(self.c_puct))

    def evaluate(self, board):
        # Convert board to tensor for model
        # Note: This requires a proper board_to_tensor function
        # For prototype, we return random/dummy values or need to implement the converter
        
        # Checking terminal state first
        if board.is_game_over():
            result = board.result()
            if result == '1-0': return 1, {}
            elif result == '0-1': return -1, {}
            return 0, {}

        # Inference (Placeholder)
        # tensor = board_to_tensor(board)
        # policy_logits, value = self.model(tensor)
        
        # Dummy return for structure
        value = 0.1 # Placeholder
        policy = {move: 1.0/board.legal_moves.count() for move in board.legal_moves} 
        
        return value, policy

    def expand(self, node, policy):
        for move, prob in policy.items():
            new_state = node.state.copy()
            new_state.push(move)
            child = Node(new_state, parent=node, action_taken=move, prior=prob)
            node.children.append(child)

    def backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            # Value is from perspective of current player.
            # If node.state.turn is White, and value is high for White, we add it.
            # But MCTS usually flips value for opponent.
            node.value_sum += value 
            node = node.parent
            value = -value # Switch perspective

if __name__ == "__main__":
    # Test MCTS structure (Mock)
    print("Initializing MCTS Test...")
    board = chess.Board()
    mcts = MCTS(model=None, num_simulations=10) # 10 sims for speed
    best_move = mcts.search(board)
    print(f"MCTS Suggested Move: {best_move}")
