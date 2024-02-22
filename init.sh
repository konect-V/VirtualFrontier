#!/usr/bin/bash

echo "Download pip deps..."
pip install discord.py TTS llama-cpp-python huggingface-hub torch

echo "Download llama"
huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q5_0.gguf --local-dir . --local-dir-use-symlinks False
mv llama-2-7b.Q5_0.gguf llama_model.gguf