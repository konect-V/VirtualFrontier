#!/usr/bin/python3

import os
import re
import sys
import mmap
import time
import torch
import typing
import asyncio
import discord
import threading
import functools
from gtts import gTTS 
from TTS.api import TTS
from langdetect import detect
from pydub import AudioSegment
from discord import app_commands
from langchain.llms import LlamaCpp
from llama_cpp import Any, Dict, List
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage
from langchain.callbacks.manager import CallbackManager
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.base import BaseCallbackHandler

#global variable
lock_audio = threading.Lock()
lock_answer = threading.Lock()
current_answer = ""
current_question = ""
current_sentence_index = 0
current_audio_sentence_index = 0
current_original_message = None
last_time_send_to_discord = 0

# Functions
async def edit_message(original_message, message):
    await original_message.edit(content=message)

async def edit_message_file(original_message, message, file_path):
    await original_message.edit(content=message)
    if os.path.exists(file_path):
        await original_message.add_files(discord.File(file_path))

def generate_audio(voice_str, audio_sentence_index):
    global current_audio_sentence_index

    output_path = "output_tmp.mp3"
    if not os.path.exists("output.mp3"):
        output_path = "output.mp3"

    try:
        language = detect(voice_str)
        gTTS(text=voice_str, lang=language, slow=False).save(output_path)
    except:
        pass

    # Concatenate the two mp3
    if output_path == "output_tmp.mp3":
        sound_first = AudioSegment.from_mp3("output.mp3")
        sound_second = AudioSegment.from_mp3("output_tmp.mp3")
        combined_sounds = sound_first + sound_second
        combined_sounds.export("output.mp3", format="mp3")

    current_audio_sentence_index = audio_sentence_index


def generate_answer(llm, original_message, question):
    global current_answer, current_question, current_original_message, current_sentence_index, current_audio_sentence_index

    lock_answer.acquire(blocking=True, timeout=-1)

    try:
        pre_prompt = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'. "
        prompt = pre_prompt + question + " Assistant: "

        # Reset variables
        if os.path.exists("output.mp3"):
            os.remove("output.mp3")
        
        current_answer = ""
        current_sentence_index = 0
        current_question = question
        current_original_message = original_message
        current_audio_sentence_index = 0

        llm(prompt, temperature=0.7, top_p=0.1, top_k=40, repeat_penalty=1.176, max_tokens=512)

        while current_sentence_index != current_audio_sentence_index:
            time.sleep(1)
        
        sentences = re.split(r'(\.|\?|!|:)', current_answer)
        voice_str = sentences[len(sentences) - 1]
        if voice_str != "\"" and voice_str != "":
            current_sentence_index += 1
            thread = threading.Thread(target=generate_audio, args=(voice_str, current_sentence_index,))
            thread.start()
            while current_sentence_index != current_audio_sentence_index:
                time.sleep(1)

        asyncio.run_coroutine_threadsafe(edit_message_file(current_original_message, current_question + "\n" + current_answer, "output.mp3"), client.loop)
    except:
        asyncio.run_coroutine_threadsafe(edit_message(original_message, "Text generation error for the following prompt : " + question), client.loop)

    lock_answer.release()

# Class
class StreamingLLMToDiscord(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        return
    
    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any,) -> None:
        return
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        global current_answer, current_question, current_sentence_index, current_original_message, last_time_send_to_discord

        current_answer = current_answer + token
        sentences = re.split(r'(\.|\?|!|:)', current_answer)

        voice_str = ""

        for i in range(len(sentences) - current_sentence_index - 1):
            voice_str = voice_str + sentences[i + current_sentence_index]
        
        if voice_str != "":
            current_sentence_index = len(sentences) - 1
            thread = threading.Thread(target=generate_audio, args=(voice_str, current_sentence_index,))
            thread.start()


        if(time.time() - last_time_send_to_discord > 1):
            last_time_send_to_discord = time.time()
            asyncio.run_coroutine_threadsafe(edit_message_file(current_original_message, current_question + "\n" + current_answer, "output.mp3"), client.loop)
        
        return
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        return
    
    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        return 
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        return
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        return
    
    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        return
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        return
    
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        return

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        return
    
    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        return
    
    def on_text(self, text: str, **kwargs: Any) -> None:
        return
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        return


# Main
callback_manager = CallbackManager([StreamingLLMToDiscord()])

print("Loading llm fast...")
llm_fast = LlamaCpp(
    model_path="./llm_model_fast.gguf",
    n_gpu_layers=32,
    n_ctx=2048,
    callback_manager=callback_manager, 
    verbose=True
)

print("Loading llm large...")
llm_large = LlamaCpp(
    model_path="./llm_model_large.gguf",
    n_gpu_layers=32,
    n_ctx=2048,
    callback_manager=callback_manager, 
    verbose=True
)

print("Loading Discord")
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


@tree.command(name="fast-ask", description="ask a question to AI")
@app_commands.describe(question="The question")
async def slash_command(interaction: discord.Interaction, question: str = None): 
    await interaction.response.send_message("Generating answer to : " + question)
    original_message = await interaction.original_response()

    thread = threading.Thread(target=generate_answer, args=(llm_fast, original_message, question,))
    thread.start()

@tree.command(name="smart-ask", description="ask a question to AI, with smart answer")
@app_commands.describe(question="The question")
async def slash_command(interaction: discord.Interaction, question: str = None):    
    await interaction.response.send_message("Generating answer to : " + question)
    original_message = await interaction.original_response()

    thread = threading.Thread(target=generate_answer, args=(llm_large, original_message, question,))
    thread.start()

@client.event
async def on_ready():
    await tree.sync()
    print(f'Logged in as `{client.user}`')

client.run("MTIwOTg4MjgxMDE0ODcyMDY4MA.Grrxks.64RWwzLlg0AbVXdRGydzSYiJAapDvkot4znYGs")
