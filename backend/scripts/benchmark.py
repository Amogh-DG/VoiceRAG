import argparse
import asyncio
import time
from backend.embeddings.wrapper import EmbeddingModelWrapper
from backend.retrieval.vector_db import MockVectorDB
from backend.generation.generator import LLMGenerator
from backend.harness.orchestrator import PipelineOrchestrator

async def run_benchmark(query: str, runs: int):
    """Profiles the latency and outputs of the RAG pipeline.
    
    TODO:
    - Implement loading a dataset of evaluation questions.
    - Track metrics like TTFT (Time to First Token), latency, and accuracy.
    - Export results to JSON/CSV for further analytics.
    """
    print(f"Running benchmark with query: '{query}' ({runs} runs)...")
    
    # Instantiate modules
    embedder = EmbeddingModelWrapper()
    vector_db = MockVectorDB()
    generator = LLMGenerator()
    orchestrator = PipelineOrchestrator(embedder, vector_db, generator)
    
    # Run the benchmark runs
    latencies = []
    responses = []
    
    for i in range(runs):
        start_time = time.perf_counter()
        response = await orchestrator.run_pipeline(query)
        end_time = time.perf_counter()
        
        latency = end_time - start_time
        latencies.append(latency)
        responses.append(response)
        
        print(f"Run {i+1}/{runs}: Latency = {latency:.4f}s")
        
    avg_latency = sum(latencies) / len(latencies)
    print(f"\n--- Benchmark Results ---")
    print(f"Average Latency: {avg_latency:.4f}s")
    print(f"Min Latency: {min(latencies):.4f}s")
    print(f"Max Latency: {max(latencies):.4f}s")
    print(f"Sample response answer: '{responses[0].answer}'")

def main():
    parser = argparse.ArgumentParser(description="Profile RAG pipeline latency and performance.")
    parser.add_argument("--query", type=str, default="What is retrieval augmented generation?", help="Query to run")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark iterations")
    args = parser.parse_args()
    
    asyncio.run(run_benchmark(args.query, args.runs))

if __name__ == "__main__":
    main()
