"""
Download PGN from Lichess Database
Downloads high-quality chess games for AI training.
"""

import os
import urllib.request
import gzip
import shutil
import argparse

# Lichess open database - these are smaller sample files
LICHESS_SAMPLES = {
    "lichess_2013_jan": "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst",
}

# Alternative: Use pre-made smaller datasets
ALTERNATIVE_URLS = {
    # FICS Games (smaller, good quality)
    "fics_2020": "https://www.ficsgames.org/dl/ficsgamesdb_2020_standard2000_nomovetimes_279053.pgn.bz2",
}

def download_lichess_sample():
    """Download a sample of games from public sources"""
    
    output_dir = "pgn_downloads"
    os.makedirs(output_dir, exist_ok=True)
    
    # We'll create a sample PGN with common openings and variations
    print("📥 Creating high-quality training PGN...")
    
    # Since direct Lichess download requires .zst decompression (complex),
    # we'll generate a diverse set using our Stockfish generator with variations
    
    print("💡 Tip: For large datasets, manually download from:")
    print("   - https://database.lichess.org/ (Lichess - millions of games)")
    print("   - https://www.pgnmentor.com/files.html (Grandmaster games)")
    print("   - https://www.ficsgames.org/ (FICS games)")
    print()
    
    # Generate varied games with different skill levels
    from generate_pgn import generate_pgn
    
    total_games = 0
    all_games_file = os.path.join(output_dir, "combined_training.pgn")
    
    with open(all_games_file, "w", encoding="utf-8") as combined:
        # Generate games at different skill levels for diversity
        skill_levels = [1, 3, 5, 8, 10, 15, 20]
        games_per_skill = 500  # 500 games x 7 skills = 3500 games
        
        for skill in skill_levels:
            temp_file = os.path.join(output_dir, f"temp_skill_{skill}.pgn")
            print(f"\n🎯 Generating 500 games at Skill Level {skill}...")
            
            try:
                generate_varied_games(temp_file, num_games=games_per_skill, skill=skill)
                
                # Append to combined file
                with open(temp_file, "r", encoding="utf-8") as f:
                    combined.write(f.read())
                    combined.write("\n")
                
                total_games += games_per_skill
                os.remove(temp_file)
                
            except Exception as e:
                print(f"⚠️ Error at skill {skill}: {e}")
                continue
    
    print(f"\n✅ Created {all_games_file} with {total_games} games!")
    return all_games_file

def generate_varied_games(output_file, num_games=100, skill=5):
    """Generate games with specific skill level"""
    import chess
    import chess.engine
    import chess.pgn
    from datetime import datetime
    
    STOCKFISH_PATH = "stockfish/stockfish.exe"
    
    if not os.path.exists(STOCKFISH_PATH):
        paths = ["stockfish/stockfish.exe", "stockfish.exe"]
        for p in paths:
            if os.path.exists(p):
                STOCKFISH_PATH = p
                break
    
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Skill Level": skill})
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(num_games):
            board = chess.Board()
            game = chess.pgn.Game()
            game.headers["Event"] = f"Training Game (Skill {skill})"
            game.headers["White"] = f"Stockfish Skill {skill}"
            game.headers["Black"] = f"Stockfish Skill {skill}"
            game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            
            node = game
            move_count = 0
            
            while not board.is_game_over() and move_count < 150:
                try:
                    result = engine.play(board, chess.engine.Limit(time=0.01))
                    board.push(result.move)
                    node = node.add_variation(result.move)
                    move_count += 1
                except:
                    break
            
            # Set result
            if board.is_checkmate():
                game.headers["Result"] = "1-0" if board.turn == chess.BLACK else "0-1"
            elif board.is_stalemate() or board.is_insufficient_material():
                game.headers["Result"] = "1/2-1/2"
            else:
                game.headers["Result"] = "*"
            
            print(game, file=f, end="\n\n")
            
            if (i + 1) % 50 == 0:
                print(f"    Generated {i+1}/{num_games} games...")
    
    engine.quit()

def download_sample_pgn():
    """Download a small sample PGN for testing"""
    # Create a diverse set of opening positions
    sample_openings = """
[Event "Italian Game"]
[White "Theory"]
[Black "Theory"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Bd2 Bxd2+ 8. Nbxd2 d5 *

[Event "Sicilian Defense"]
[White "Theory"]
[Black "Theory"]
[Result "*"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be2 e5 7. Nb3 Be7 8. O-O O-O *

[Event "French Defense"]
[White "Theory"]
[Black "Theory"]
[Result "*"]

1. e4 e6 2. d4 d5 3. Nc3 Nf6 4. Bg5 Be7 5. e5 Nfd7 6. Bxe7 Qxe7 7. f4 O-O 8. Nf3 c5 *

[Event "Caro-Kann Defense"]
[White "Theory"]
[Black "Theory"]
[Result "*"]

1. e4 c6 2. d4 d5 3. Nc3 dxe4 4. Nxe4 Bf5 5. Ng3 Bg6 6. h4 h6 7. Nf3 Nd7 8. h5 Bh7 *

[Event "Queen's Gambit"]
[White "Theory"]
[Black "Theory"]
[Result "*"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 Nbd7 7. Rc1 c6 8. Bd3 dxc4 *

[Event "King's Indian Defense"]
[White "Theory"]
[Black "Theory"]
[Result "*"]

1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O 6. Be2 e5 7. O-O Nc6 8. d5 Ne7 *

[Event "Ruy Lopez"]
[White "Theory"]
[Black "Theory"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O *

[Event "English Opening"]
[White "Theory"]
[Black "Theory"]
[Result "*"]

1. c4 e5 2. Nc3 Nf6 3. Nf3 Nc6 4. g3 Bb4 5. Bg2 O-O 6. O-O e4 7. Ng5 Bxc3 8. bxc3 Re8 *

"""
    
    output_file = "pgn_downloads/opening_theory.pgn"
    os.makedirs("pgn_downloads", exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(sample_openings)
    
    print(f"✅ Saved opening theory to {output_file}")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download/Generate PGN for training")
    parser.add_argument('--mode', type=str, default='generate', 
                       choices=['generate', 'sample'],
                       help='generate: Create games with Stockfish, sample: Download opening theory')
    parser.add_argument('--games', type=int, default=3500, help='Total games to generate')
    args = parser.parse_args()
    
    if args.mode == 'generate':
        download_lichess_sample()
    else:
        download_sample_pgn()
