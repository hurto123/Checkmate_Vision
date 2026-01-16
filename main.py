"""
Checkmate Vision - Main Entry Point
Connects the Vision (YOLO + OpenCV) with the Brain (ChessNet + MCTS)
"""

import cv2
import chess
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from vision.board_mapper import BoardMapper, interactive_calibration
from vision.detector import ChessPieceDetector, map_detections_to_grid, grid_to_fen
# from brain.model import ChessNet
# from brain.mcts import MCTS

def draw_move_arrow(frame, from_sq, to_sq, board_mapper):
    """
    Draw an arrow showing the suggested move on the original frame.
    """
    # Get square centers in warped space
    from_file, from_rank = chess.square_file(from_sq), chess.square_rank(from_sq)
    to_file, to_rank = chess.square_file(to_sq), chess.square_rank(to_sq)
    
    from_center = board_mapper.get_square_center(from_file, from_rank)
    to_center = board_mapper.get_square_center(to_file, to_rank)
    
    # Transform back to original image space
    import numpy as np
    pts = np.array([[from_center], [to_center]], dtype=np.float32)
    original_pts = cv2.perspectiveTransform(pts, board_mapper.inv_transform_matrix)
    
    from_pt = tuple(map(int, original_pts[0][0]))
    to_pt = tuple(map(int, original_pts[1][0]))
    
    cv2.arrowedLine(frame, from_pt, to_pt, (0, 255, 255), 3, tipLength=0.3)
    
    return frame


def main():
    print("=" * 50)
    print("  CHECKMATE VISION - AI Chess Assistant")
    print("=" * 50)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera!")
        return
    
    # Step 1: Calibrate board
    print("\n[STEP 1] Board Calibration")
    print("Click 4 corners of the chessboard: TL -> TR -> BR -> BL")
    print("Press Enter to confirm, 'r' to reset, 'q' to quit")
    
    board_mapper = interactive_calibration(cap)
    
    if board_mapper.transform_matrix is None:
        print("Calibration failed or cancelled.")
        cap.release()
        return
    
    # Step 2: Load YOLO model (if available)
    print("\n[STEP 2] Loading YOLO Detector...")
    detector = None
    model_path = "runs/detect/train/weights/best.pt"
    
    if os.path.exists(model_path):
        try:
            detector = ChessPieceDetector(model_path)
            print(f"Loaded model from: {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    else:
        print(f"Model not found at: {model_path}")
        print("Running in demo mode (no piece detection)")
    
    # Step 3: Load Chess Brain (if available)
    # brain = None
    # try:
    #     brain = ChessNet()
    #     brain.load_state_dict(torch.load("brain_weights.pt"))
    #     mcts = MCTS(brain)
    # except:
    #     print("Brain not loaded. Running without AI suggestions.")
    
    # Main loop
    print("\n[RUNNING] Press 'q' to quit, 's' to scan board")
    print("-" * 50)
    
    current_fen = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        
        # Show warped view in corner
        warped = board_mapper.warp_image(frame)
        warped_small = cv2.resize(warped, (200, 200))
        display[10:210, 10:210] = warped_small
        
        # Draw border around mini-map
        cv2.rectangle(display, (10, 10), (210, 210), (255, 255, 255), 2)
        cv2.putText(display, "Board View", (15, 225), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # If detector is loaded, run detection
        if detector is not None:
            detections = detector.detect(frame)
            display = detector.draw_detections(display, detections)
            
            # Show detection count
            cv2.putText(display, f"Pieces: {len(detections)}", (10, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show current FEN if available
        if current_fen:
            cv2.putText(display, f"FEN: {current_fen[:40]}...", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Checkmate Vision", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s') and detector is not None:
            # Scan board and generate FEN
            print("\nScanning board...")
            detections = detector.detect(frame)
            grid = map_detections_to_grid(detections, board_mapper)
            current_fen = grid_to_fen(grid)
            print(f"Detected FEN: {current_fen}")
            
            # Validate with python-chess
            try:
                board = chess.Board(current_fen)
                print(f"Valid position! {len(list(board.legal_moves))} legal moves.")
                print(board)
            except Exception as e:
                print(f"Invalid FEN: {e}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nCheckmate Vision ended.")


if __name__ == "__main__":
    main()
