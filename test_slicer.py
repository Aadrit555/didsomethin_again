import os
import cv2
import numpy as np
from ingestion.adaptive_slicer import AdaptiveSlicer

def create_dummy_video(path, duration_sec=5, fps=30):
    """
    Creates a dummy video with a scene cut in the middle.
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    # Scene 1: Black frames
    for _ in range(int(duration_sec * fps / 2)):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        out.write(frame)
    
    # Scene 2: White frames (Scene Cut)
    for _ in range(int(duration_sec * fps / 2)):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        out.write(frame)
        
    out.release()

def test_slicer():
    video_path = "test_video.avi"
    output_dir = "test_output_frames"
    
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
        os.rmdir(output_dir)

    create_dummy_video(video_path)
    
    slicer = AdaptiveSlicer()
    slicer.process_video(video_path, output_dir=output_dir)
    
    print(f"\nFiles in {output_dir}:")
    files = os.listdir(output_dir)
    for f in files:
        print(f" - {f}")
    
    # Cleanup
    # os.remove(video_path)

if __name__ == "__main__":
    test_slicer()
