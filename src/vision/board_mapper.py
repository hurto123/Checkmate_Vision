import cv2
import numpy as np

class BoardMapper:
    """
    Maps camera image to 8x8 chessboard grid using perspective transform.
    """
    
    def __init__(self):
        self.transform_matrix = None
        self.inv_transform_matrix = None
        self.board_corners = None  # Order: top-left, top-right, bottom-right, bottom-left
        self.square_size = 64  # Pixels per square in warped image
        self.warped_size = self.square_size * 8  # 512x512 pixels
        
    def calibrate(self, corners):
        """
        Set the 4 corners of the chessboard in the camera image.
        corners: np.array of shape (4, 2) in order [TL, TR, BR, BL]
        """
        self.board_corners = np.float32(corners)
        
        # Destination points for a perfect 8x8 grid
        dst_corners = np.float32([
            [0, 0],
            [self.warped_size, 0],
            [self.warped_size, self.warped_size],
            [0, self.warped_size]
        ])
        
        self.transform_matrix = cv2.getPerspectiveTransform(self.board_corners, dst_corners)
        self.inv_transform_matrix = cv2.getPerspectiveTransform(dst_corners, self.board_corners)
        
        print("Board calibrated successfully!")
        return True
        
    def warp_image(self, frame):
        """
        Apply perspective transform to get top-down view of the board.
        """
        if self.transform_matrix is None:
            raise ValueError("Board not calibrated! Call calibrate() first.")
            
        warped = cv2.warpPerspective(frame, self.transform_matrix, 
                                      (self.warped_size, self.warped_size))
        return warped
    
    def pixel_to_square(self, x, y):
        """
        Convert pixel coordinates in warped image to chess square (file, rank).
        Returns: (file, rank) where file is 0-7 (a-h) and rank is 0-7 (1-8)
        """
        file_idx = int(x // self.square_size)
        rank_idx = 7 - int(y // self.square_size)  # Flip because row 0 is rank 8
        return (file_idx, rank_idx)
    
    def square_to_notation(self, file_idx, rank_idx):
        """
        Convert (file, rank) indices to chess notation (e.g., 'e4').
        """
        file_char = chr(ord('a') + file_idx)
        rank_char = str(rank_idx + 1)
        return file_char + rank_char
    
    def get_square_center(self, file_idx, rank_idx):
        """
        Get the center pixel coordinates of a square in the warped image.
        """
        x = (file_idx + 0.5) * self.square_size
        y = (7 - rank_idx + 0.5) * self.square_size
        return (int(x), int(y))

    def draw_grid(self, warped_frame):
        """
        Draw 8x8 grid overlay on the warped image for visualization.
        """
        overlay = warped_frame.copy()
        
        # Draw vertical lines
        for i in range(9):
            x = i * self.square_size
            cv2.line(overlay, (x, 0), (x, self.warped_size), (0, 255, 0), 1)
            
        # Draw horizontal lines
        for i in range(9):
            y = i * self.square_size
            cv2.line(overlay, (0, y), (self.warped_size, y), (0, 255, 0), 1)
            
        # Label squares
        for file_idx in range(8):
            for rank_idx in range(8):
                center = self.get_square_center(file_idx, rank_idx)
                notation = self.square_to_notation(file_idx, rank_idx)
                cv2.putText(overlay, notation, (center[0]-10, center[1]+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
        return overlay


def interactive_calibration(cap):
    """
    Allow user to click 4 corners of the board to calibrate.
    Returns: BoardMapper object with calibration set.
    """
    mapper = BoardMapper()
    corners = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal corners
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append([x, y])
            print(f"Corner {len(corners)}: ({x}, {y})")
    
    print("Click 4 corners of the chessboard in order: TL, TR, BR, BL")
    print("Press 'r' to reset, 'q' to quit, Enter to confirm")
    
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        display = frame.copy()
        
        # Draw clicked corners
        for i, corner in enumerate(corners):
            cv2.circle(display, tuple(corner), 5, (0, 0, 255), -1)
            cv2.putText(display, str(i+1), (corner[0]+10, corner[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw lines between corners
        if len(corners) >= 2:
            for i in range(len(corners) - 1):
                cv2.line(display, tuple(corners[i]), tuple(corners[i+1]), (0, 255, 0), 2)
            if len(corners) == 4:
                cv2.line(display, tuple(corners[3]), tuple(corners[0]), (0, 255, 0), 2)
        
        cv2.putText(display, f"Corners: {len(corners)}/4", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Calibration", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            corners = []
            print("Reset corners")
        elif key == ord('q'):
            break
        elif key == 13 and len(corners) == 4:  # Enter key
            mapper.calibrate(np.array(corners))
            break
    
    cv2.destroyWindow("Calibration")
    return mapper


if __name__ == "__main__":
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    mapper = interactive_calibration(cap)
    
    if mapper.transform_matrix is not None:
        print("Showing warped view. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            warped = mapper.warp_image(frame)
            grid_view = mapper.draw_grid(warped)
            
            cv2.imshow("Original", frame)
            cv2.imshow("Warped Grid", grid_view)
            
            if cv2.waitKey(1) == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
