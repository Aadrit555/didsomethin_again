import numpy as np
import sys

# Robust NumPy 2.x compatibility monkeypatch
for attr in ['object', 'bool', 'int', 'float', 'str', 'complex']:
    if not hasattr(np, attr):
        setattr(np, attr, getattr(__builtins__, attr) if hasattr(__builtins__, attr) else getattr(np, f"{attr}_") if hasattr(np, f"{attr}_") else None)

# Specifically for Keras/tf2onnx
if not hasattr(np, 'object'): np.object = object

import os
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from ingestion.adaptive_slicer import AdaptiveSlicer


class VideoIngestor:
    def __init__(self, model_name='clip-ViT-B-32'):
        print(f"Loading CLIP model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.frames_dir = "extracted_frames"
        os.makedirs(self.frames_dir, exist_ok=True)
        self.slicer = AdaptiveSlicer()

    def extract_keyframes(self, video_path):
        """
        Extracts frames from video using Adaptive Slicing (Scene Cuts & High Motion).
        """
        print(f"Extracting frames from {video_path} using Adaptive Slicer...")
        return self.slicer.process_video(video_path, output_dir=self.frames_dir)

    def generate_segment_summaries(self, extracted_data):
        """
        Generates semantic summaries for each segment (Hierarchical Context Encoding).
        For now, it picks the mid-frame of each segment as the representative visual.
        """
        print("Generating hierarchical segment summaries...")
        segments = {}
        for data in extracted_data:
            sid = data['segment_id']
            if sid not in segments:
                segments[sid] = []
            segments[sid].append(data)
        
        summaries = []
        for sid, frames in segments.items():
            # Pick a representative frame (middle one)
            rep_frame = frames[len(frames)//2]
            
            summary = (
                f"Segment {sid}: Visual sequence containing {len(frames)} frames "
                f"starting at {frames[0]['timestamp']:.2f}s. ({rep_frame['type']})"
            )
            
            summaries.append({
                "segment_id": sid,
                "summary": summary,
                "representative_frame": rep_frame['frame_path'],
                "start_time": frames[0]['timestamp'],
                "end_time": frames[-1]['timestamp']
            })
        return summaries

    def encode_frames(self, extracted_data):
        """
        Generates embeddings for each extracted frame using CLIP.
        """
        if not extracted_data:
            print("No frames were extracted.")
            return []
            
        print(f"Generating embeddings for {len(extracted_data)} frames...")
        images = [Image.open(data['frame_path']) for data in extracted_data]
        embeddings = self.model.encode(images, show_progress_bar=True)
        
        for i, data in enumerate(extracted_data):
            data['embedding'] = embeddings[i].tolist()
            
        return extracted_data

    def process_video(self, video_path, kg_path="video_kg.json"):
        from reasoning.graph_grounding import VideoKnowledgeGraph
        data = self.extract_keyframes(video_path)
        data_with_embeddings = self.encode_frames(data)
        summaries = self.generate_segment_summaries(data)
        
        # Channel 1: Build the Knowledge Graph
        kg = VideoKnowledgeGraph()
        kg.build_graph(summaries)
        kg.save_graph(kg_path)
        
        return {
            "frames": data_with_embeddings,
            "summaries": summaries,
            "kg_path": os.path.abspath(kg_path)
        }

if __name__ == "__main__":
    # Test stub
    pass
