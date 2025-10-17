
import argparse, httpx, asyncio

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--host", default="http://localhost:8000")
    args = p.parse_args()

    async with httpx.AsyncClient(timeout=60) as cx:
        r = await cx.post(f"{args.host}/v1/generate", json={
            "prompt": args.prompt,
            "adapter_id": args.adapter,
            "max_tokens": 32
        })
        r.raise_for_status()
        print(r.json())

if __name__ == "__main__":
    asyncio.run(main())
