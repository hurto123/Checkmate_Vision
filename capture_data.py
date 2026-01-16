import cv2
import os
import time

def capture_dataset(output_dir="dataset/raw_images", cam_index=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Open camera
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {cam_index}")
        return

    print("--- Checkmate Vision Data Collector ---")
    print(f"Saving images to: {output_dir}")
    print("Controls:")
    print("  's' : Save current frame")
    print("  'q' : Quit")
    
    # Get existing file count to avoid overwriting
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("chess_") and f.endswith(".jpg")]
    count = len(existing_files)
    print(f"Starting count from: {count}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break

        # Display info on screen
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Count: {count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Data Collector', display_frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"{output_dir}/chess_{count:04d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            count += 1
            # Flash effect
            cv2.imshow('Data Collector', 255 - display_frame)
            cv2.waitKey(50)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_dataset()
