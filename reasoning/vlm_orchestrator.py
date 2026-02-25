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
            return "No relevant segments were found in the video for this question."

        parts = []

        if grounding_results:
            parts.append("From the video graph I found these relevant segments:")
            for node in grounding_results[:5]:
                label = node.get("label", "")
                start = node.get("start_time", 0.0)
                end = node.get("end_time", 0.0)
                parts.append(f"- {label} (from {start:.2f}s to {end:.2f}s)")

        if context_frames:
            unique_ts = sorted({frame["timestamp"] for frame in context_frames})
            ts_str = ", ".join(f"{t:.2f}s" for t in unique_ts[:8])
            parts.append(
                f"I also used frames around these times as evidence: {ts_str}."
            )

        parts.append(
            "Given only this structural information (without a vision model), "
            "this is the most grounded description I can provide."
        )

        return "\n".join(parts)


if __name__ == "__main__":
    # Test stub
    pass
