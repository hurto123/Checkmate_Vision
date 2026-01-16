# 🏆 Checkmate Vision

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

**Checkmate Vision** is an AI-powered chess assistant that combines computer vision (YOLO + OpenCV) with deep learning (ChessNet + MCTS) to analyze chessboards in real-time and suggest optimal moves.

---

## ✨ Features

- 📷 **Real-time Board Detection** - Uses YOLO to detect chess pieces from camera
- 🧠 **Deep Learning Brain** - ChessNet neural network with Residual Blocks
- 🔍 **MCTS Search** - Monte Carlo Tree Search for move evaluation
- 🎯 **Multiple Training Modes** - Supervised Learning, Self-Play, Imitation Learning
- ⚡ **Stockfish Integration** - Train against Stockfish at various skill levels

---

## 📁 Project Structure

```
Checkmate_Vision/
├── src/
│   ├── brain/           # Neural network & MCTS
│   │   ├── model.py     # ChessNet architecture
│   │   ├── mcts.py      # Monte Carlo Tree Search
│   │   └── train_*.py   # Training scripts
│   └── vision/          # Computer vision modules
│       ├── detector.py  # YOLO piece detection
│       └── board_mapper.py
├── main.py              # Main entry point
├── train_sl_enhanced.py # Supervised Learning training
├── train_selfplay.py    # Self-play training
├── train_imitation.py   # Imitation learning
└── ai_play.py           # Play against AI
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/Checkmate_Vision.git
cd Checkmate_Vision

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📦 Dependencies

| Package | Description |
|---------|-------------|
| `torch` | Deep learning framework |
| `torchvision` | PyTorch computer vision |
| `ultralytics` | YOLO object detection |
| `opencv-python` | Computer vision library |
| `python-chess` | Chess game library |
| `numpy` | Numerical computing |
| `tqdm` | Progress bars |

---

## 🎮 Usage

### 1. Training the AI

#### Supervised Learning (Recommended First Step)
```bash
python train_sl_enhanced.py --epochs 200 --batch-size 256
```

#### Self-Play Training
```bash
python train_selfplay.py --games 1000
```

#### Training vs Stockfish
```bash
python train_vs_stockfish.py --skill 10
```

### 2. Play Against AI
```bash
python ai_play.py
```

### 3. Real-time Vision Mode
```bash
python main.py
```

---

## 🧠 Model Architecture

**ChessNet** uses a ResNet-style architecture:

```
Input: 20 x 8 x 8 (encoded board state)
    ↓
Convolutional Input Block
    ↓
10 Residual Blocks (128 channels)
    ↓
┌─────────────┬─────────────┐
│ Policy Head │ Value Head  │
│ (4672 moves)│ (-1 to +1)  │
└─────────────┴─────────────┘
```

---

## 📊 Training Results

After 200 epochs of Supervised Learning:
- Policy Accuracy: ~45%
- Value Loss: ~0.3
- Training on 100k+ positions from Stockfish games

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [python-chess](https://python-chess.readthedocs.io/) - Chess library
- [Stockfish](https://stockfishchess.org/) - Chess engine for training data
- [Ultralytics YOLO](https://ultralytics.com/) - Object detection
- AlphaZero paper for inspiration

---

<p align="center">
  Made with ❤️ for chess lovers
</p>
