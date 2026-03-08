# AI Personalized Tutor POC

## Overview
This POC demonstrates how AI can bridge the gap in education by transforming static course materials into an interactive, personalized tutoring experience using Retrieval-Augmented Generation (RAG).

## Tech Stack
- Python 3.9+
- LangChain for orchestration
- OpenAI API for LLM reasoning
- ChromaDB for vector storage of course material

## Setup
1. `pip install langchain langchain-openai chromadb`
2. Set your `OPENAI_API_KEY` in your environment variables.

## Usage
Run `python main.py`. It ingests a sample text file, creates a vector index, and allows you to ask targeted questions as a student.