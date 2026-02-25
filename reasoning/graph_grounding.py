import json
import os

class VideoKnowledgeGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def build_graph(self, segment_summaries):
        """
        Builds a temporal knowledge graph where each node is a segment summary.
        Links are established chronologically.
        """
        self.nodes = []
        self.edges = []
        
        for i, summary in enumerate(segment_summaries):
            node_id = f"node_{summary['segment_id']}"
            node = {
                "id": node_id,
                "label": summary['summary'],
                "start_time": summary['start_time'],
                "end_time": summary['end_time'],
                "representative_frame": summary['representative_frame']
            }
            self.nodes.append(node)
            
            # Temporal linking: node n -> node n+1
            if i > 0:
                self.edges.append({
                    "source": self.nodes[i-1]['id'],
                    "target": node_id,
                    "type": "FOLLOWS"
                })

        print(f"Built Knowledge Graph with {len(self.nodes)} nodes and {len(self.edges)} edges.")

    def search_graph(self, query):
        """
        Simple keyword-based search over nodes.
        In a full VideoRAG, this would use semantic retrieval.
        Using substring matching for better robustness in this demo.
        """
        results = []
        q = query.lower()
        for node in self.nodes:
            if q in node['label'].lower():
                results.append(node)
        return results

    def save_graph(self, path="video_kg.json"):
        with open(path, "w") as f:
            json.dump({"nodes": self.nodes, "edges": self.edges}, f, indent=4)

    def load_graph(self, path="video_kg.json"):
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                self.nodes = data['nodes']
                self.edges = data['edges']
