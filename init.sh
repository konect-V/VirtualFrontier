#!/usr/bin/bash

echo "Download pip deps..."
pip install discord.py TTS huggingface-hub torch langchain langchain-community pydub gtts langdetect fs

if ! [ -f llm_model_fast.gguf ]; then
    echo "Download gpt4all-falcon-newbpe-q4_0.ggu as llm_model_fast.gguf"
    wget https://gpt4all.io/models/gguf/gpt4all-falcon-newbpe-q4_0.gguf -O llm_model_fast.gguf
fi 

if ! [ -f llm_model_large.gguf ]; then
    echo "Download wizardlm-13b-v1.2.Q4_0.gguf as llm_model_large.gguf"
    wget https://gpt4all.io/models/gguf/wizardlm-13b-v1.2.Q4_0.gguf -O llm_model_large.gguf
fi

python3 main.py