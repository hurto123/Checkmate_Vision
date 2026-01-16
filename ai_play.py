"""
AI vs AI Self-Play System
ให้ AI เล่นหมากรุกกันเอง พร้อมแสดงผลกระดาน
"""

import chess
import time
import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Simple AI that picks random legal moves (for initial training)
class RandomAI:
    def __init__(self, name="RandomAI"):
        self.name = name
    
    def get_move(self, board):
        """Pick a random legal move"""
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves)


# Simple heuristic AI (slightly smarter)
class SimpleAI:
    def __init__(self, name="SimpleAI"):
        self.name = name
        # Piece values
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
    
    def evaluate_board(self, board):
        """Simple material count evaluation"""
        if board.is_checkmate():
            return -9999 if board.turn else 9999
        
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        return score
    
    def get_move(self, board):
        """Pick the best move based on simple evaluation"""
        best_move = None
        best_score = float('-inf') if board.turn else float('inf')
        
        for move in board.legal_moves:
            board.push(move)
            score = self.evaluate_board(board)
            board.pop()
            
            if board.turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
        
        return best_move if best_move else random.choice(list(board.legal_moves))


def print_board(board, clear=True):
    """Print the chess board in a nice format"""
    if clear:
        os.system('cls' if os.name == 'nt' else 'clear')
    
    print("\n  ╔═══╤═══╤═══╤═══╤═══╤═══╤═══╤═══╗")
    
    for rank in range(7, -1, -1):
        print(f"{rank + 1} ║", end="")
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            
            if piece:
                symbol = piece.symbol()
                # Use Unicode chess symbols
                unicode_pieces = {
                    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
                    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
                }
                display = unicode_pieces.get(symbol, symbol)
            else:
                display = ' '
            
            print(f" {display} ", end="")
            if file < 7:
                print("│", end="")
        
        print("║")
        if rank > 0:
            print("  ╟───┼───┼───┼───┼───┼───┼───┼───╢")
    
    print("  ╚═══╧═══╧═══╧═══╧═══╧═══╧═══╧═══╝")
    print("    a   b   c   d   e   f   g   h\n")


def play_game(white_ai, black_ai, show_board=True, delay=0.5, max_moves=200):
    """
    Play a single game between two AIs.
    
    Returns: 
        result: '1-0' (white wins), '0-1' (black wins), '1/2-1/2' (draw)
        moves: list of moves played
    """
    board = chess.Board()
    moves = []
    
    if show_board:
        print(f"\n{'='*50}")
        print(f"  {white_ai.name} (White) vs {black_ai.name} (Black)")
        print(f"{'='*50}")
        print_board(board)
    
    move_count = 0
    
    while not board.is_game_over() and move_count < max_moves:
        # Get current player's AI
        current_ai = white_ai if board.turn == chess.WHITE else black_ai
        
        # Get move
        move = current_ai.get_move(board)
        
        # Make move
        san_move = board.san(move)
        board.push(move)
        moves.append(san_move)
        move_count += 1
        
        if show_board:
            turn = "White" if not board.turn else "Black"
            print(f"Move {move_count}: {turn} plays {san_move}")
            print_board(board, clear=False)
            time.sleep(delay)
    
    # Determine result
    if board.is_checkmate():
        result = '0-1' if board.turn else '1-0'
        winner = black_ai.name if board.turn else white_ai.name
        if show_board:
            print(f"\n🏆 CHECKMATE! {winner} wins!")
    elif board.is_stalemate():
        result = '1/2-1/2'
        if show_board:
            print("\n🤝 STALEMATE! Draw.")
    elif board.is_insufficient_material():
        result = '1/2-1/2'
        if show_board:
            print("\n🤝 INSUFFICIENT MATERIAL! Draw.")
    elif move_count >= max_moves:
        result = '1/2-1/2'
        if show_board:
            print(f"\n🤝 MAX MOVES ({max_moves}) REACHED! Draw.")
    else:
        result = '1/2-1/2'
        if show_board:
            print("\n🤝 Draw.")
    
    return result, moves


def run_tournament(num_games=10, show_games=False, delay=0.1):
    """
    Run multiple games between AIs.
    
    Args:
        num_games: Number of games to play
        show_games: Whether to show board during games
        delay: Delay between moves (seconds) when showing
    """
    print("\n" + "="*60)
    print("  🏁 AI CHESS TOURNAMENT")
    print("="*60)
    
    # Create AIs
    ai1 = SimpleAI("SimpleAI-1")
    ai2 = SimpleAI("SimpleAI-2")
    
    stats = {
        ai1.name: {'wins': 0, 'losses': 0, 'draws': 0},
        ai2.name: {'wins': 0, 'losses': 0, 'draws': 0}
    }
    
    all_games = []
    
    for game_num in range(1, num_games + 1):
        print(f"\n--- Game {game_num}/{num_games} ---")
        
        # Alternate colors
        if game_num % 2 == 1:
            white, black = ai1, ai2
        else:
            white, black = ai2, ai1
        
        result, moves = play_game(white, black, show_board=show_games, delay=delay)
        
        # Update stats
        if result == '1-0':
            stats[white.name]['wins'] += 1
            stats[black.name]['losses'] += 1
            print(f"  Result: {white.name} wins as White")
        elif result == '0-1':
            stats[black.name]['wins'] += 1
            stats[white.name]['losses'] += 1
            print(f"  Result: {black.name} wins as Black")
        else:
            stats[white.name]['draws'] += 1
            stats[black.name]['draws'] += 1
            print(f"  Result: Draw")
        
        print(f"  Moves: {len(moves)}")
        
        all_games.append({
            'white': white.name,
            'black': black.name,
            'result': result,
            'moves': moves
        })
    
    # Print final stats
    print("\n" + "="*60)
    print("  📊 TOURNAMENT RESULTS")
    print("="*60)
    
    for name, s in stats.items():
        total = s['wins'] + s['losses'] + s['draws']
        win_rate = (s['wins'] / total * 100) if total > 0 else 0
        print(f"\n  {name}:")
        print(f"    Wins:   {s['wins']}")
        print(f"    Losses: {s['losses']}")
        print(f"    Draws:  {s['draws']}")
        print(f"    Win Rate: {win_rate:.1f}%")
    
    print("\n" + "="*60)
    
    return all_games, stats


def save_games_to_pgn(games, filename="games_database.pgn"):
    """
    Save games to PGN file for database and future training.
    """
    import datetime
    
    with open(filename, 'a', encoding='utf-8') as f:
        for i, game in enumerate(games):
            f.write(f'[Event "AI Self-Play"]\n')
            f.write(f'[Site "Checkmate Vision"]\n')
            f.write(f'[Date "{datetime.datetime.now().strftime("%Y.%m.%d")}"]\n')
            f.write(f'[Round "{i+1}"]\n')
            f.write(f'[White "{game["white"]}"]\n')
            f.write(f'[Black "{game["black"]}"]\n')
            f.write(f'[Result "{game["result"]}"]\n')
            f.write('\n')
            
            # Write moves
            moves_str = ""
            for j, move in enumerate(game['moves']):
                if j % 2 == 0:
                    moves_str += f"{j//2 + 1}. "
                moves_str += f"{move} "
            
            moves_str += game['result']
            f.write(moves_str + '\n\n')
    
    print(f"\n📁 Games saved to: {filename}")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AI vs AI Chess')
    parser.add_argument('--games', type=int, default=10, help='Number of games')
    parser.add_argument('--show', action='store_true', help='Show games visually')
    parser.add_argument('--delay', type=float, default=0.3, help='Delay between moves')
    parser.add_argument('--single', action='store_true', help='Play single game with display')
    parser.add_argument('--save', type=str, default=None, help='Save games to PGN file')
    
    args = parser.parse_args()
    
    if args.single:
        # Play single visual game
        ai1 = SimpleAI("White-AI")
        ai2 = SimpleAI("Black-AI")
        play_game(ai1, ai2, show_board=True, delay=args.delay)
    else:
        # Run tournament
        games, stats = run_tournament(num_games=args.games, show_games=args.show, delay=args.delay)
        
        # Save to PGN if requested
        if args.save:
            save_games_to_pgn(games, args.save)
        else:
            # Default: always save
            save_games_to_pgn(games, "games_database.pgn")

