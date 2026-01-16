"""
Download and Use Pre-trained Chess Models
==========================================
Options to start from a model that already knows how to play chess.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from brain.model import ChessNet

def create_pretrained_options():
    """
    สร้างตัวเลือก Pre-trained Models
    """
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    🚀 PRE-TRAINED MODEL OPTIONS                            ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Option 1: Continue from Current Checkpoint                                ║
║     - ใช้ checkpoint ที่ฝึกไว้แล้ว (sl_checkpoint_XX.pt)                      ║
║     - เหมาะสำหรับ: ต้องการฝึกต่อจากที่หยุดไว้                                   ║
║                                                                            ║
║  Option 2: Transfer Learning from Random-Move Model                        ║
║     - เริ่มจากโมเดลที่รู้จักการเดินถูกกฎ (แต่ยังไม่เก่ง)                          ║
║     - เหมาะสำหรับ: เร่งการเรียนรู้กฎพื้นฐาน                                     ║
║                                                                            ║
║  Option 3: Download Community Model                                        ║
║     - ดาวน์โหลดจาก GitHub/HuggingFace                                       ║
║     - เหมาะสำหรับ: ข้ามขั้นตอนการฝึกพื้นฐาน                                     ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

def init_weights_smart(model):
    """
    Initialize weights using Xavier/Kaiming initialization
    ช่วยให้โมเดลเรียนรู้เร็วขึ้นตั้งแต่เริ่มต้น
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'conv' in name:
                torch.nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'fc' in name or 'linear' in name:
                torch.nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            torch.nn.init.zeros_(param)
    
    print("✅ Applied smart weight initialization")
    return model

def create_base_model(save_path="base_model.pt"):
    """
    สร้างโมเดลพื้นฐานที่ initialized ดีแล้ว
    ใช้เป็นจุดเริ่มต้นที่ดีกว่า random
    """
    model = ChessNet(num_res_blocks=10, num_channels=128)
    model = init_weights_smart(model)
    
    torch.save(model.state_dict(), save_path)
    print(f"💾 Saved base model to: {save_path}")
    
    return model

def load_and_adapt_checkpoint(checkpoint_path, strict=False):
    """
    โหลด checkpoint และปรับให้เข้ากับโมเดลปัจจุบัน
    ใช้เมื่อโมเดลมีโครงสร้างต่างกันเล็กน้อย
    """
    model = ChessNet(num_res_blocks=10, num_channels=128)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check if it's a full checkpoint or just state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Try to load, ignoring mismatched keys
        try:
            model.load_state_dict(state_dict, strict=strict)
            print(f"✅ Loaded checkpoint from: {checkpoint_path}")
        except RuntimeError as e:
            print(f"⚠️ Partial load due to architecture mismatch: {e}")
            # Load what we can
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() 
                           if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            print(f"   Loaded {len(filtered_dict)}/{len(state_dict)} parameters")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    return model

def find_best_checkpoint():
    """
    ค้นหา checkpoint ที่ดีที่สุดในโฟลเดอร์ปัจจุบัน
    """
    checkpoints = []
    
    for f in os.listdir('.'):
        if f.startswith('sl_checkpoint_') and f.endswith('.pt'):
            try:
                epoch = int(f.replace('sl_checkpoint_', '').replace('.pt', ''))
                checkpoints.append((epoch, f))
            except:
                pass
    
    if not checkpoints:
        print("❌ No checkpoints found")
        return None
    
    # Sort by epoch and get the latest
    checkpoints.sort(reverse=True)
    best = checkpoints[0]
    
    print(f"📁 Found {len(checkpoints)} checkpoints")
    print(f"   Latest: {best[1]} (epoch {best[0]})")
    
    return best[1]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Pre-trained Model Tools')
    parser.add_argument('--action', type=str, default='info',
                       choices=['info', 'create-base', 'find-best', 'load'],
                       help='Action to perform')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path for load action')
    
    args = parser.parse_args()
    
    if args.action == 'info':
        create_pretrained_options()
    elif args.action == 'create-base':
        create_base_model()
    elif args.action == 'find-best':
        find_best_checkpoint()
    elif args.action == 'load':
        if args.checkpoint:
            load_and_adapt_checkpoint(args.checkpoint)
        else:
            best = find_best_checkpoint()
            if best:
                load_and_adapt_checkpoint(best)
