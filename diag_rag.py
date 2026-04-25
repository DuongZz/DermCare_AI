import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from app.services.rag_service import rag_service

import sys

async def main():
    print("--- RAG DIAGNOSTIC TEST ---")
    question = sys.argv[1] if len(sys.argv) > 1 else "Vảy nến là gì?"
    try:
        print(f"Querying: {question}")
        result = await rag_service.query(question)
        print("\nSUCCESS!")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources: {result['sources']}")
    except Exception as e:
        print(f"\nFAILURE: {e}")

if __name__ == "__main__":
    asyncio.run(main())
