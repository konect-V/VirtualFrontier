#!/usr/bin/python3

import sys
import mmap
import time
import typing
import asyncio
import discord
import threading
import functools
from TTS.api import TTS
from llama_cpp import Llama
import torch

# Functions

async def send_message(channel, message):
  await channel.send(message)

async def send_file(channel, message, file_path):
  await channel.send(message, file=discord.File(file_path))

def generate_sound(channel, text):
    print(text)
    try:
      tts.tts_to_file(text=text, speaker_wav="input.wav", language="fr", file_path="output.wav")
      asyncio.run_coroutine_threadsafe(send_file(channel, text, "output.wav"), client.loop)
    except:
      asyncio.run_coroutine_threadsafe(send_message(channel, "Audio generation error of : " + text), client.loop)
     


def generate_answer(message):
    question = message.content.replace(message.content[0], "", 1)

    asyncio.run_coroutine_threadsafe(send_message(message.channel, "Generating answer to : " + question), client.loop)

    try:
      last_answer = ""
      prompt = "Q:" + question + " A:"
      answer = "" 
      request = prompt
      while (not ("Q:" in last_answer)) and (not ("\n" in last_answer)) and last_answer != ".":
        if last_answer != "" and last_answer != ".":
          thread = threading.Thread(target=generate_sound, args=(message.channel, last_answer,))
          thread.start()
           
        output = llm(request, max_tokens=250, stop=["."], echo=False)
        last_answer = output['choices'][0]['text'].encode('utf-8', errors='ignore').decode('utf-8') + "."
        request = last_answer
        answer = answer + last_answer

      if("Q:" in last_answer):
        last_answer = last_answer[0:last_answer.index("Q:")]

      if("\n" in last_answer):
        last_answer = last_answer[0:last_answer.index("\n")]

      if last_answer != "" and last_answer != ".":
        thread = threading.Thread(target=generate_sound, args=(message.channel, last_answer,))
        thread.start()
      
      answer = answer[len(prompt):]

      if(("Q:" in answer)):
        answer = answer[0:answer.index("Q:")]

      if(("\n" in answer)):
        answer = answer[0:answer.index("\n")]

      asyncio.run_coroutine_threadsafe(send_message(message.channel, answer), client.loop)
    except:
      asyncio.run_coroutine_threadsafe(send_message(message.channel, "Text generation error for the following prompt : " + question), client.loop)


# Main

llm = Llama(
    model_path="./llama_model.gguf",
    n_gpu_layers=32,
    n_ctx=2048
)

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'Logged in as `{client.user}`')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('?'):
        thread = threading.Thread(target=generate_answer, args=(message,))
        thread.start()

client.run("MTIwOTg4MjgxMDE0ODcyMDY4MA.Grrxks.64RWwzLlg0AbVXdRGydzSYiJAapDvkot4znYGs")
