import cv2
import numpy as np
import os
from PIL import Image

class AdaptiveSlicer:
    def __init__(self, threshold_multiplier=3.0, min_threshold=10.0, motion_persistence=3):
        """
        Initializes the Adaptive Slicer.
        
        :param threshold_multiplier: How many standard deviations above the mean to trigger a scene cut.
        :param min_threshold: Minimum MAD spike to consider a change significant.
        :param motion_persistence: Number of frames MAD must stay high to classify as "High Motion".
        """
        self.threshold_multiplier = threshold_multiplier
        self.min_threshold = min_threshold
        self.motion_persistence = motion_persistence
        self.mad_history = []
        self.history_size = 100 # Rolling window for dynamic thresholding

    def calculate_mad(self, frame1, frame2):
        """
        Calculates the Mean Absolute Difference between two frames.
        """
        # Convert to grayscale for faster processing if not already
        if len(frame1.shape) == 3:
            frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            frame1_gray = frame1
            frame2_gray = frame2

        # Resize for faster math if frames are large
        # h, w = frame1_gray.shape
        # if w > 640:
        #     frame1_gray = cv2.resize(frame1_gray, (640, int(h * (640 / w))))
        #     frame2_gray = cv2.resize(frame2_gray, (640, int(h * (640 / w))))

        diff = cv2.absdiff(frame1_gray, frame2_gray)
        mad = np.mean(diff)
        return mad

    def get_dynamic_threshold(self):
        if not self.mad_history:
            return self.min_threshold
        
        mean_mad = np.mean(self.mad_history)
        std_mad = np.std(self.mad_history)
        
        # Adaptive threshold: mean + k * std
        threshold = mean_mad + (self.threshold_multiplier * std_mad)
        return max(threshold, self.min_threshold)

    def format_timestamp(self, seconds):
        """
        Formats seconds into MM_SS_MS string.
        Example: 00_04_250.jpg
        """
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{minutes:02d}_{secs:02d}_{millis:03d}"

    def process_video(self, video_path, output_dir="extracted_frames"):
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        prev_frame = None
        extracted_data = []
        count = 0
        motion_counter = 0
        segment_id = 0
        
        print(f"Adaptive Slicing (VideoRAG Style): {video_path}...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_secs = count / fps
            
            if prev_frame is not None:
                mad = self.calculate_mad(prev_frame, frame)
                threshold = self.get_dynamic_threshold()

                is_scene_cut = mad > threshold
                
                # Update history
                self.mad_history.append(mad)
                if len(self.mad_history) > self.history_size:
                    self.mad_history.pop(0)

                # High Motion logic
                # If MAD stays above 80% of threshold for motion_persistence frames
                if mad > threshold * 0.8:
                    motion_counter += 1
                else:
                    motion_counter = 0
                
                is_high_motion = motion_counter >= self.motion_persistence

                if is_scene_cut or is_high_motion:
                    # Increment segment_id on scene cuts to logically group frames
                    if is_scene_cut:
                        segment_id += 1

                    ts_str = self.format_timestamp(timestamp_secs)
                    frame_filename = f"{ts_str}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    
                    # Prevent overwriting if multiple frames hit the same millisecond bucket
                    if os.path.exists(frame_path):
                         frame_path = os.path.join(output_dir, f"{ts_str}_{count}.jpg")

                    cv2.imwrite(frame_path, frame)

                    # For internal metadata tracking (segment_id needed for KG)
                    extracted_data.append({
                        "timestamp": timestamp_secs,
                        "frame_path": os.path.abspath(frame_path),
                        "segment_id": segment_id,
                        "type": "scene_cut" if is_scene_cut else "high_motion"
                    })

            prev_frame = frame.copy()
            count += 1

        cap.release()
        print(f"Extracted {len(extracted_data)} keyframes.")
        return extracted_data

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        slicer = AdaptiveSlicer()
        slicer.process_video(sys.argv[1])
