import cv2

def test_camera():
    print("Opening camera...")
    # Try index 0 first, then 1 if 0 fails
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Camera 0 not found. Trying index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open any webcam.")
            return

    print("Camera opened successfully. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break
            
        cv2.imshow('Checkmate Vision - Camera Test', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()
