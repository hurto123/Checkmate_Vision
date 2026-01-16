# 🏆 Checkmate Vision

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

**Checkmate Vision** คือระบบ AI ผู้ช่วยเล่นหมากรุกที่ผสมผสาน Computer Vision (YOLO + OpenCV) กับ Deep Learning (ChessNet + MCTS) เพื่อวิเคราะห์กระดานหมากรุกแบบ Real-time และแนะนำการเดินที่ดีที่สุด

---

## ✨ คุณสมบัติ

- 📷 **ตรวจจับกระดานแบบ Real-time** - ใช้ YOLO ตรวจจับตัวหมากจากกล้อง
- 🧠 **สมอง Deep Learning** - โครงข่ายประสาทเทียม ChessNet พร้อม Residual Blocks
- 🔍 **MCTS Search** - การค้นหาแบบ Monte Carlo Tree Search
- 🎯 **โหมดการฝึกหลากหลาย** - Supervised Learning, Self-Play, Imitation Learning
- ⚡ **เชื่อมต่อ Stockfish** - ฝึกแข่งกับ Stockfish ในระดับความยากต่างๆ

---

## 📁 โครงสร้างโปรเจค

```
Checkmate_Vision/
├── src/
│   ├── brain/           # โครงข่ายประสาทเทียม & MCTS
│   │   ├── model.py     # สถาปัตยกรรม ChessNet
│   │   ├── mcts.py      # Monte Carlo Tree Search
│   │   └── train_*.py   # สคริปต์การฝึก
│   └── vision/          # โมดูล Computer Vision
│       ├── detector.py  # ตรวจจับหมากด้วย YOLO
│       └── board_mapper.py
├── main.py              # จุดเริ่มต้นหลัก
├── train_sl_enhanced.py # ฝึก Supervised Learning
├── train_selfplay.py    # ฝึกแบบ Self-play
├── train_imitation.py   # ฝึก Imitation learning
└── ai_play.py           # เล่นกับ AI
```

---

## 🚀 การติดตั้ง

### ความต้องการเบื้องต้น
- Python 3.8 ขึ้นไป
- CUDA (ไม่จำเป็น, สำหรับใช้ GPU)

### ขั้นตอนการติดตั้ง

```bash
# Clone repository
git clone https://github.com/yourusername/Checkmate_Vision.git
cd Checkmate_Vision

# สร้าง Virtual Environment
python -m venv .venv

# เปิดใช้งาน Virtual Environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# ติดตั้ง Dependencies
pip install -r requirements.txt
```

---

## 📦 Dependencies

| แพ็คเกจ | คำอธิบาย |
|---------|----------|
| `torch` | Framework สำหรับ Deep Learning |
| `torchvision` | PyTorch Computer Vision |
| `ultralytics` | YOLO Object Detection |
| `opencv-python` | ไลบรารี Computer Vision |
| `python-chess` | ไลบรารีเกมหมากรุก |
| `numpy` | การคำนวณเชิงตัวเลข |
| `tqdm` | แสดง Progress Bar |

---

## 🎮 วิธีใช้งาน

### 1. ฝึก AI

#### Supervised Learning (แนะนำให้เริ่มจากขั้นนี้)
```bash
python train_sl_enhanced.py --epochs 200 --batch-size 256
```

#### ฝึกแบบ Self-Play
```bash
python train_selfplay.py --games 1000
```

#### ฝึกแข่งกับ Stockfish
```bash
python train_vs_stockfish.py --skill 10
```

### 2. เล่นกับ AI
```bash
python ai_play.py
```

### 3. โหมด Vision แบบ Real-time
```bash
python main.py
```

---

## 🧠 สถาปัตยกรรม Model

**ChessNet** ใช้สถาปัตยกรรมแบบ ResNet:

```
Input: 20 x 8 x 8 (encoded board state)
    ↓
Convolutional Input Block
    ↓
10 Residual Blocks (128 channels)
    ↓
┌─────────────┬─────────────┐
│ Policy Head │ Value Head  │
│ (4672 moves)│ (-1 ถึง +1) │
└─────────────┴─────────────┘
```

---

## 📊 ผลการฝึก

หลังฝึก Supervised Learning 200 epochs:
- Policy Accuracy: ~45%
- Value Loss: ~0.3
- ฝึกจากตำแหน่งมากกว่า 100,000 ตำแหน่งจากเกม Stockfish

---

## 🤝 การมีส่วนร่วม

ยินดีรับ Contributions! สามารถส่ง Pull Request ได้เลย

1. Fork โปรเจค
2. สร้าง Feature Branch (`git checkout -b feature/FeatureName`)
3. Commit การเปลี่ยนแปลง (`git commit -m 'เพิ่ม Feature ใหม่'`)
4. Push ไปยัง Branch (`git push origin feature/FeatureName`)
5. เปิด Pull Request

---

## 📄 License

โปรเจคนี้ใช้ MIT License - ดูรายละเอียดที่ไฟล์ [LICENSE](LICENSE)

---

## 🙏 ขอบคุณ

- [python-chess](https://python-chess.readthedocs.io/) - ไลบรารีหมากรุก
- [Stockfish](https://stockfishchess.org/) - Chess Engine สำหรับสร้างข้อมูลฝึก
- [Ultralytics YOLO](https://ultralytics.com/) - Object Detection
- บทความ AlphaZero สำหรับแรงบันดาลใจ

---

<p align="center">
  สร้างด้วย ❤️ สำหรับคนรักหมากรุก
</p>
