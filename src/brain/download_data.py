"""
Download sample PGN files for training.
Sources: Lichess & KingBase (Free databases)
"""

import os
import urllib.request
import gzip
import shutil

SAMPLE_URLS = {
    # Small sample for testing (Lichess rated games)
    'lichess_sample': 'https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst',
}

def download_lichess_sample(output_dir='pgn_data'):
    """
    Downloads a small sample of Lichess games.
    Note: Official Lichess database uses .zst compression.
    For simplicity, we'll provide instructions instead.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("PGN Dataset Download Instructions")
    print("="*60)
    print()
    print("Option 1: Lichess Database (Recommended for testing)")
    print("-" * 40)
    print("1. Go to: https://database.lichess.org/")
    print("2. Download a small monthly file (e.g., 2013-01)")
    print("3. Extract the .pgn file")
    print(f"4. Place it in: {os.path.abspath(output_dir)}")
    print()
    print("Option 2: KingBase (High-quality GM games)")
    print("-" * 40)
    print("1. Go to: https://kingbase-chess.net/")
    print("2. Download the PGN archive")
    print("3. Extract and place in the same folder")
    print()
    print("Option 3: FICS Games Database")
    print("-" * 40)
    print("1. Go to: https://www.ficsgames.org/download.html")
    print("2. Download games from specific year/month")
    print()
    print("After downloading, run:")
    print("  python -c \"from src.brain.data_utils import create_dataset; create_dataset('pgn_data/your_file.pgn', 'dataset/')\"")
    print()


def create_sample_pgn(output_path='pgn_data/sample.pgn'):
    """
    Create a minimal sample PGN file for testing the pipeline.
    Contains 3 famous games.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sample_games = '''[Event "Sample Game 1"]
[Site "Test"]
[Date "2024.01.01"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0

[Event "Sample Game 2"]
[Site "Test"]
[Date "2024.01.02"]
[White "Player3"]
[Black "Player4"]
[Result "0-1"]

1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. e3 O-O 5. Bd3 d5 6. Nf3 c5 7. O-O Nc6 8. a3 Bxc3 9. bxc3 dxc4 10. Bxc4 Qc7 0-1

[Event "Sample Game 3"]
[Site "Test"]
[Date "2024.01.03"]
[White "Player5"]
[Black "Player6"]
[Result "1/2-1/2"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be2 e5 7. Nb3 Be7 8. O-O O-O 9. Be3 Be6 10. f3 Nbd7 1/2-1/2
'''
    
    with open(output_path, 'w') as f:
        f.write(sample_games)
    
    print(f"Created sample PGN at: {output_path}")
    print("This is just for testing. Use real games for actual training!")
    return output_path


if __name__ == "__main__":
    print("Creating sample PGN for testing...")
    path = create_sample_pgn()
    
    print("\nNow creating dataset from sample...")
    from data_utils import create_dataset
    create_dataset(path, 'dataset/', max_games=100)
    
    print("\n" + "="*60)
    print("Sample dataset created! You can now run training:")
    print("  python src/brain/train_sl.py --epochs 5")
    print("="*60)
