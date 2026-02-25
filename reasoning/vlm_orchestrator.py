from reasoning.graph_grounding import VideoKnowledgeGraph


class VLMOrchestrator:
    """
    Simplified orchestrator: for now we do NOT call a heavy vision model.
    We answer using the textual knowledge graph nodes and basic metadata only,
    so the app works even when Ollama / VLM is unavailable.
    """

    def __init__(self, model_name=None, ollama_url=None):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.kg = VideoKnowledgeGraph()

    def generate_answer(self, query, context_frames, knowledge_graph_path=None):
        """
        Lightweight textual reasoning based on the video knowledge graph
        and timestamps from retrieved frames. No external VLM calls.
        """
        if knowledge_graph_path:
            self.kg.load_graph(knowledge_graph_path)

        grounding_results = self.kg.search_graph(query)

        if not grounding_results and not context_frames:
            return "I couldn't find any segment in the video that clearly matches this question."

        # If we have KG nodes, treat their labels as the primary explanation.
        if grounding_results:
            top_nodes = grounding_results[:3]
            sentences = []
            for node in top_nodes:
                label = node.get("label", "")
                start = node.get("start_time", 0.0)
                end = node.get("end_time", 0.0)
                if label:
                    sentences.append(
                        f"Between {start:.2f}s and {end:.2f}s the video contains: {label}"
                    )

            if sentences:
                return " ".join(sentences)

        # Fallback: only have timestamps from retrieved frames.
        if context_frames:
            unique_ts = sorted({frame["timestamp"] for frame in context_frames})
            ts_str = ", ".join(f"{t:.2f}s" for t in unique_ts[:8])
            return (
                "I retrieved several relevant moments in the video around these times: "
                f"{ts_str}. However, without a vision model I can't reliably describe "
                "the exact visual content at those timestamps."
            )

        # Absolute fallback – should rarely be hit.
        return "I was not able to derive a meaningful answer from the stored video structure."


if __name__ == "__main__":
    # Test stub
    pass
