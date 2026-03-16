#!/usr/bin/env python3
"""Setup script for RAG System."""

import os
import sys
from pathlib import Path


def create_directories():
    """Create necessary directories."""
    dirs = [
        "data/pdf_reports",
        "data/debug/data_01_parsed_reports",
        "data/debug/data_02_merged_reports",
        "data/databases/chunked_reports",
        "data/databases/vector_dbs",
        "data/uploads",
        "temp"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")


def check_env():
    """Check environment variables."""
    env_path = Path(".env")
    
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
        print("⚠️  Created .env file. Please add your OpenAI API key.")
    else:
        with open(env_path) as f:
            content = f.read()
            if "your_openai_api_key" in content:
                print("⚠️  Please add your OpenAI API key to .env file")
            else:
                print("✅ OpenAI API key configured")


def check_pdfs():
    """Check for PDF files."""
    pdf_dir = Path("data/pdf_reports")
    pdfs = list(pdf_dir.glob("*.pdf"))
    
    if pdfs:
        print(f"✅ Found {len(pdfs)} PDF files")
    else:
        print("⚠️  No PDF files found in data/pdf_reports/")
        print("   Copy your PDF files to this directory")


def main():
    print("=" * 50)
    print("RAG System Setup")
    print("=" * 50)
    
    create_directories()
    check_env()
    check_pdfs()
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Add your OpenAI API key to .env")
    print("2. Copy PDF files to data/pdf_reports/")
    print("3. Run: python main.py webui")
    print("4. Open http://localhost:5000 in your browser")


if __name__ == "__main__":
    main()
