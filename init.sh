#!/usr/bin/bash

echo "Downloading deps ..."
sudo apt install ffmpeg libavcodec-extra-53

echo "Downloading pip deps ..."
pip install discord.py TTS huggingface-hub torch langchain langchain-community pydub gtts langdetect fs llama-cpp-python

if ! [ -f llm_model_fast.gguf ]; then
    echo "Downloading fast model ..."
    wget -q https://gpt4all.io/models/gguf/gpt4all-falcon-newbpe-q4_0.gguf -O llm_model_fast.gguf
fi 

if ! [ -f llm_model_large.gguf ]; then
    echo "Downloading large model ..."
    wget -q https://gpt4all.io/models/gguf/nous-hermes-llama2-13b.Q4_0.gguf -O llm_model_large.gguf
fi

if ! [ -d tts_model ]; then
    sudo apt install git-lfs
    git lfs install
    mkdir tts_model
    cd tts_model
    GIT_LFS_SKIP_SMUDGE=0 git clone https://huggingface.co/coqui/XTTS-v2
fi
