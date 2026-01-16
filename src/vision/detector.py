from ultralytics import YOLO
import cv2
import numpy as np

# Mapping from class index to piece symbol (adjust based on your training labels)
# Standard: Uppercase = White, Lowercase = Black
CLASS_TO_SYMBOL = {
    'white-king': 'K',
    'white-queen': 'Q',
    'white-rook': 'R',
    'white-bishop': 'B',
    'white-knight': 'N',
    'white-pawn': 'P',
    'black-king': 'k',
    'black-queen': 'q',
    'black-rook': 'r',
    'black-bishop': 'b',
    'black-knight': 'n',
    'black-pawn': 'p',
}

class ChessPieceDetector:
    """
    Detects chess pieces using YOLOv11.
    """
    
    def __init__(self, model_path="best.pt", conf_threshold=0.5):
        """
        model_path: Path to trained YOLOv11 weights
        conf_threshold: Minimum confidence to consider a detection
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names  # {0: 'white-king', 1: 'white-queen', ...}
        
    def detect(self, frame):
        """
        Run inference on a frame.
        Returns list of detections: [(class_name, confidence, (x1, y1, x2, y2)), ...]
        """
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
                
            cls_id = int(box.cls[0])
            cls_name = self.class_names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            detections.append({
                'class': cls_name,
                'confidence': conf,
                'bbox': (int(x1), int(y1), int(x2), int(y2))
            })
            
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on frame.
        """
        display = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det['class']
            conf = det['confidence']
            
            # Color based on piece color
            color = (0, 255, 0) if 'white' in cls_name else (0, 0, 255)
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(display, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return display
    
    def get_piece_centers(self, detections):
        """
        Get center point of each detection.
        Returns: [(class_name, (cx, cy)), ...]
        """
        centers = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            centers.append((det['class'], (cx, cy)))
        return centers


def map_detections_to_grid(detections, board_mapper):
    """
    Given detections and a calibrated board mapper, map each piece to a square.
    Returns: 8x8 grid where each cell is None or piece symbol
    """
    # Initialize empty board
    grid = [[None for _ in range(8)] for _ in range(8)]
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        # Use bottom-center of bounding box (base of piece)
        cx = (x1 + x2) // 2
        cy = y2  # Bottom of box
        
        # Transform point to warped space
        point = np.array([[[cx, cy]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, board_mapper.transform_matrix)
        tx, ty = transformed[0][0]
        
        # Convert to grid coordinates
        file_idx, rank_idx = board_mapper.pixel_to_square(tx, ty)
        
        # Check bounds
        if 0 <= file_idx < 8 and 0 <= rank_idx < 8:
            symbol = CLASS_TO_SYMBOL.get(det['class'], '?')
            grid[7 - rank_idx][file_idx] = symbol  # Row 0 = rank 8
            
    return grid


def grid_to_fen(grid):
    """
    Convert 8x8 grid to FEN position string.
    """
    fen_rows = []
    
    for row in grid:
        fen_row = ""
        empty_count = 0
        
        for cell in row:
            if cell is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
                
        if empty_count > 0:
            fen_row += str(empty_count)
            
        fen_rows.append(fen_row)
    
    # Join rows with '/'
    position = "/".join(fen_rows)
    
    # Add default game state (white to move, all castling rights, no en passant)
    # In a real system, you'd track these separately
    fen = f"{position} w KQkq - 0 1"
    
    return fen


if __name__ == "__main__":
    # Test detector (requires trained model)
    print("ChessPieceDetector module loaded.")
    print("To use: train a YOLO model with train_yolo.py first.")
    print("Then instantiate: detector = ChessPieceDetector('runs/detect/train/weights/best.pt')")
