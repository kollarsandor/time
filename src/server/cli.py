import argparse
import sys
import uvicorn
from src.server.config import ServerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="glm-server",
        description="GLM-4.7-FP8 Inference Server",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    serve_parser = subparsers.add_parser("serve", help="Start the inference server")
    serve_parser.add_argument("--model-dir", type=str, default="./model", help="Model directory")
    serve_parser.add_argument("--engine", type=str, default="./build/engine.so", help="Engine shared library path")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=5000, help="Port to bind")
    serve_parser.add_argument("--max-concurrency", type=int, default=64, help="Maximum concurrent requests")
    serve_parser.add_argument("--max-batch-tokens", type=int, default=32768, help="Maximum batch tokens")
    serve_parser.add_argument("--max-seq-len", type=int, default=4096, help="Maximum sequence length")
    serve_parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum new tokens per request")
    serve_parser.add_argument("--max-prompt-tokens", type=int, default=2048, help="Maximum prompt tokens")
    serve_parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs")
    serve_parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    serve_parser.add_argument("--queue-size", type=int, default=256, help="Request queue size")
    serve_parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn workers")
    serve_parser.add_argument("--mock", action="store_true", help="Use mock engine for testing")

    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    benchmark_parser.add_argument("--model-dir", type=str, default="./model", help="Model directory")
    benchmark_parser.add_argument("--engine", type=str, default="./build/engine.so", help="Engine shared library path")
    benchmark_parser.add_argument("--duration", type=int, default=60, help="Benchmark duration in seconds")
    benchmark_parser.add_argument("--concurrency", type=int, default=32, help="Number of concurrent requests")
    benchmark_parser.add_argument("--prompt-len", type=int, default=128, help="Prompt length in tokens")
    benchmark_parser.add_argument("--output-len", type=int, default=128, help="Output length in tokens")

    return parser.parse_args()


def run_serve(args: argparse.Namespace):
    config = ServerConfig(
        model_dir=args.model_dir,
        engine_path=args.engine,
        host=args.host,
        port=args.port,
        max_concurrency=args.max_concurrency,
        max_batch_tokens=args.max_batch_tokens,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        num_gpus=args.num_gpus,
        log_level=args.log_level,
        queue_size=args.queue_size,
    )
    from src.server.app import create_app
    app = create_app(config, use_mock=args.mock)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=args.workers,
        log_level=config.log_level.lower(),
        access_log=True,
    )


def run_benchmark(args: argparse.Namespace):
    import asyncio
    import time
    import aiohttp

    async def benchmark():
        base_url = f"http://localhost:5000"
        prompt = "Hello, how are you? " * (args.prompt_len // 5)
        payload = {
            "model": "glm-4.7-fp8",
            "prompt": prompt,
            "max_tokens": args.output_len,
            "temperature": 0.7,
            "stream": False,
        }
        start_time = time.time()
        completed = 0
        total_tokens = 0
        errors = 0

        async def make_request(session):
            nonlocal completed, total_tokens, errors
            try:
                async with session.post(f"{base_url}/v1/completions", json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        completed += 1
                        total_tokens += data.get("usage", {}).get("completion_tokens", 0)
                    else:
                        errors += 1
            except Exception:
                errors += 1

        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < args.duration:
                tasks = [make_request(session) for _ in range(args.concurrency)]
                await asyncio.gather(*tasks)
                elapsed = time.time() - start_time
                tps = total_tokens / elapsed if elapsed > 0 else 0
                print(f"\rElapsed: {elapsed:.1f}s | Completed: {completed} | Tokens: {total_tokens} | TPS: {tps:.1f} | Errors: {errors}", end="")
            print()
            elapsed = time.time() - start_time
            print(f"\n=== Benchmark Results ===")
            print(f"Duration: {elapsed:.1f}s")
            print(f"Completed requests: {completed}")
            print(f"Total tokens: {total_tokens}")
            print(f"Tokens per second: {total_tokens / elapsed:.1f}")
            print(f"Requests per second: {completed / elapsed:.2f}")
            print(f"Errors: {errors}")

    print(f"Running benchmark for {args.duration}s with {args.concurrency} concurrent requests...")
    print(f"Prompt length: {args.prompt_len}, Output length: {args.output_len}")
    asyncio.run(benchmark())


def main():
    args = parse_args()
    if args.command == "serve":
        run_serve(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    else:
        print("Usage: glm-server {serve,benchmark} [options]")
        print("Run 'glm-server serve --help' for serve options")
        print("Run 'glm-server benchmark --help' for benchmark options")
        sys.exit(1)


if __name__ == "__main__":
    main()
