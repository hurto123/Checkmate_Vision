"""
Generate PGN Games for Training
Use Stockfish to play against itself and save games to PGN.
This creates a large dataset of "Valid/Legal" games for the AI to learn rules from.
"""

import chess
import chess.engine
import chess.pgn
import os
import random
from datetime import datetime

# Path to Stockfish
STOCKFISH_PATH = "stockfish/stockfish.exe"

def generate_pgn(output_file="generated_rules.pgn", num_games=100, move_time=0.01):
    print(f"♟️ Generating {num_games} games using Stockfish...")
    
    stockfish_binary = STOCKFISH_PATH
    
    # Check engine
    if not os.path.exists(stockfish_binary):
        # Try common paths
        paths = ["stockfish/stockfish.exe", "stockfish.exe", "C:/stockfish/stockfish.exe"]
        found = False
        for p in paths:
            if os.path.exists(p):
                stockfish_binary = p
                found = True
                break
        if not found:
            print("❌ Stockfish not found! configure STOCKFISH_PATH.")
            return

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_binary)
        engine.configure({"Skill Level": 5}) # Skill 5 is enough for legal/decent moves
    except Exception as e:
        print(f"❌ Error starting engine: {e}")
        return

    full_pgn = open(output_file, "w", encoding="utf-8")

    try:
        for i in range(num_games):
            board = chess.Board()
            game = chess.pgn.Game()
            game.headers["Event"] = "Stockfish Self-Play Rule Learning"
            game.headers["Site"] = "Local"
            game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            game.headers["Round"] = str(i+1)
            game.headers["White"] = f"Stockfish (Skill 5)"
            game.headers["Black"] = f"Stockfish (Skill 5)"
            
            node = game
            
            while not board.is_game_over() and board.fullmove_number <= 100:
                result = engine.play(board, chess.engine.Limit(time=move_time))
                board.push(result.move)
                node = node.add_variation(result.move)
            
            if board.is_checkmate():
                game.headers["Result"] = "1-0" if board.turn == chess.BLACK else "0-1"
            elif board.is_stalemate() or board.is_insufficient_material():
                game.headers["Result"] = "1/2-1/2"
            else:
                game.headers["Result"] = "*" # Interrupted or draw
                
            print(game, file=full_pgn, end="\n\n")
            print(f"  Generated Game {i+1}/{num_games} ({board.result()})")
            
            # Flush periodically
            if (i+1) % 10 == 0:
                full_pgn.flush()
                
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted.")
    finally:
        engine.quit()
        full_pgn.close()
        print(f"✅ Saved games to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate PGN games using Stockfish self-play")
    parser.add_argument('--games', type=int, default=100, help='Number of games to generate')
    parser.add_argument('--output', type=str, default='generated_rules.pgn', help='Output PGN file')
    parser.add_argument('--skill', type=int, default=5, help='Stockfish skill level (0-20)')
    args = parser.parse_args()
    
    generate_pgn(output_file=args.output, num_games=args.games)
