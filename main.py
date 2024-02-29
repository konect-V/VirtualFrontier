#!/usr/bin/python3

import fs
import os
import queue
import re
import sys
import mmap
import time
from gtts import gTTS
import torch
import typing
import shutil
import asyncio
import discord
import threading
import functools
from TTS.api import TTS
from fs.tempfs import TempFS
from langdetect import detect
from pydub import AudioSegment
from discord import app_commands
from typing import Any, Dict, List
from langchain_community.llms import LlamaCpp
from TTS.utils.generic_utils import get_user_data_dir
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage
from langchain.callbacks.manager import CallbackManager
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.base import BaseCallbackHandler



#global variable
lock_audio = threading.Lock()
tmpfspath = ""
voice_str_list = []
current_answer = ""
current_question = ""
answer_queu = queue.Queue()
use_natural_tts = os.getenv("USE_NATURAL_TTS", 'False').lower() in ('true', '1', 't')
audio_generate_index = 0
current_sentence_index = 0
current_voice_channel = None
connected_voice_channels = []
last_time_send_to_discord = 0
audio_generate_index_read = 0
current_original_message = None
audio_generate_queu = queue.Queue()

# Functions
async def edit_message(original_message, message):
    await original_message.edit(content=message)

async def edit_message_file(original_message, message, file_path):
    await original_message.edit(content=message)
    if os.path.exists(file_path):
        await original_message.add_files(discord.File(file_path))

def generate_audio_worker():
    global current_original_message, current_voice_channel, audio_generate_queu, audio_generate_index

    while True:
        item = audio_generate_queu.get()
        if item is None:
            time.sleep(0.1)
            pass

        voice_str = item
        read_audio_path = tmpfspath + "/output_tmp_" + str(audio_generate_index)
        output_path = read_audio_path

        if not os.path.exists(tmpfspath + "/output.wav"):
            output_path = tmpfspath + "/output.wav"

        try:
            language = detect(voice_str)
        except:
            language = "en"

        try:
            if use_natural_tts:
                read_audio_path += ".wav"
                tts.tts_to_file(text=voice_str, speaker_wav="input.wav", language=language, file_path=output_path)
                # Concatenate the two wav
                if output_path != tmpfspath + "/output.wav":
                    sound_first = AudioSegment.from_wav(tmpfspath + "/output.wav")
                    sound_second = AudioSegment.from_wav(read_audio_path)
                    combined_sounds = sound_first + sound_second
                    combined_sounds.export(tmpfspath + "/output.wav", format="wav")
                else:
                    shutil.copyfile(output_path, read_audio_path)
            else:
                # Use gTTS for faster process
                read_audio_path += ".mp3"
                gTTS(text=voice_str, lang=language).save(read_audio_path)
                # Concatenate the two wav
                if output_path != tmpfspath + "/output.wav":
                    sound_first = AudioSegment.from_wav(tmpfspath + "/output.wav")
                    sound_second = AudioSegment.from_mp3(read_audio_path)
                    combined_sounds = sound_first + sound_second
                    combined_sounds.export(tmpfspath + "/output.wav", format="wav")
                else:
                    AudioSegment.from_mp3(read_audio_path).export(output_path, format="wav")
        except:
            pass

        audio_generate_index += 1

        audio_generate_queu.task_done()


def play_audio_worker():
    global audio_generate_index_read, current_original_message, current_voice_channel, audio_play_queu, audio_generate_index, audio_generate_queu, voice_str_list

    while True:
        if current_voice_channel != None:
            if audio_generate_index_read < audio_generate_index:
                read_audio_path = tmpfspath + "/output_tmp_" + str(audio_generate_index_read)
                if use_natural_tts:
                    read_audio_path += ".wav"
                else:
                    read_audio_path += ".mp3"

                while current_voice_channel.is_playing():
                    time.sleep(0.01)
                
                try:
                    current_voice_channel.play(discord.FFmpegPCMAudio(source=read_audio_path))
                except:
                    pass
                
                audio_generate_index_read += 1
            else:
                if (not audio_generate_queu.empty()) and (not current_voice_channel.is_playing()):
                    try:
                        voice_str = voice_str_list[audio_generate_index_read]
                        read_audio_path = tmpfspath + "/output_tmp_fast_" + str(audio_generate_index_read) + ".mp3"
                        try:
                            language = detect(voice_str)
                        except:
                            language = "en"
                        gTTS(text=voice_str, lang=language).save(read_audio_path)
                        current_voice_channel.play(discord.FFmpegPCMAudio(source=read_audio_path))
                        audio_generate_index_read += 1
                    except:
                        pass
                time.sleep(0.01)
        else:
            if audio_generate_index_read < audio_generate_index:
                audio_generate_index_read += 1



def generate_audio(voice_str):
    global audio_generate_queu
    voice_str = voice_str.replace("\"", "")
    voice_str_list.append(voice_str)
    audio_generate_queu.put(voice_str)

def generate_answer_worker():
    global current_answer, current_question, current_original_message, current_sentence_index, audio_generate_queu, audio_generate_index, current_voice_channel, connected_voice_channels
    
    while True:
        item = answer_queu.get()

        if item == None:
            time.sleep(0.1)
            pass

        llm = item[0]
        question = item[1]
        original_message = item[2]

        # Update voice channel
        current_voice_channel = None

        for voice_channel in connected_voice_channels:
            if voice_channel.guild == original_message.guild:
                current_voice_channel = voice_channel
                break

        try:
            pre_prompt = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'. User: "
            prompt = pre_prompt + question + "\n\n Assistant: "

            # Reset variables
            if os.path.exists(tmpfspath + "/output.wav"):
                os.remove(tmpfspath + "/output.wav")
            
            current_answer = ""
            current_sentence_index = 0
            current_question = question
            current_original_message = original_message

            llm.invoke(prompt, temperature=0.1, top_p=0.1, top_k=40, repeat_penalty=1.176, max_tokens=2048)

            try:
                sentences = re.split(r'(\.|\?|!|:|,)', current_answer)
                voice_str = sentences[len(sentences) - 1]
                if voice_str != "":
                    generate_audio(voice_str)
            except:
                pass

            asyncio.run_coroutine_threadsafe(edit_message(current_original_message, "> " + current_question + "\n" + current_answer + "\n *(End of text generation)* \n *(Finishing to generate audio...)* "), client.loop)
            
            audio_generate_queu.join()

            asyncio.run_coroutine_threadsafe(edit_message_file(current_original_message, "> " + current_question + "\n" + current_answer + "\n *(End generation)* ", tmpfspath + "/output.wav"), client.loop)
        except:
            asyncio.run_coroutine_threadsafe(edit_message(original_message, "Text generation error for the following prompt : " + question), client.loop)

        while audio_generate_index_read < audio_generate_index:
            time.sleep(0.01)
                
        answer_queu.task_done()
    

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
            generate_audio(voice_str)
            current_sentence_index = len(sentences) - 1


        if(time.time() - last_time_send_to_discord > 1):
            last_time_send_to_discord = time.time()
            asyncio.run_coroutine_threadsafe(edit_message(current_original_message, "> " + current_question + "\n" + current_answer), client.loop)
        
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

print("Loading ramfs...")
mem_fs = TempFS(identifier='virtualfrontiertmp', temp_dir='.', auto_clean=True)

dirs = os.listdir('.')
for dir in dirs:
    if "virtualfrontiertmp" in dir:
        tmpfspath = dir
        break


callback_manager = CallbackManager([StreamingLLMToDiscord()])

print("Loading llm fast...")
llm_fast = LlamaCpp(
    model_path="./llm_model_fast.gguf",
    n_gpu_layers=32,
    n_ctx=4096,
    callback_manager=callback_manager, 
    verbose=True
)

print("Loading llm large...")
llm_large = LlamaCpp(
    model_path="./llm_model_large.gguf",
    n_gpu_layers=32,
    n_ctx=4096,
    callback_manager=callback_manager, 
    verbose=True
)


if use_natural_tts: # Use TTS only if cuda is available
    print("Loading TTS...")
    model_path = get_user_data_dir("tts/tts_models--multilingual--multi-dataset--xtts_v2")
    if not os.path.isdir(model_path):
        shutil.copytree("tts_model/XTTS-v2", model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)


threading.Thread(target=play_audio_worker).start()
threading.Thread(target=generate_audio_worker).start()
threading.Thread(target=generate_answer_worker).start()

print("Loading Discord...")
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


@tree.command(name="fast-ask", description="ask a question to AI")
@app_commands.describe(question="The question")
async def slash_command(interaction: discord.Interaction, question: str = None): 
    global answer_queu

    if question == None:
        await interaction.response.send_message("You must provide a question")
    else:
        waiting_count = answer_queu.qsize()
        end_of_sentence = " person on the waiting list"
        if waiting_count > 1:
            end_of_sentence = " people on the waiting list"

        await interaction.response.send_message("> " + question + "\nI am thinking to generate you the best answer...\nThere's currently " + str(answer_queu.qsize()) + end_of_sentence)
        original_message = await interaction.original_response()

        answer_queu.put([llm_fast, question, original_message])

@tree.command(name="smart-ask", description="ask a question to AI, with smart answer")
@app_commands.describe(question="The question")
async def slash_command(interaction: discord.Interaction, question: str = None):
    global answer_queu

    if question == None:
        await interaction.response.send_message("You must provide a question")
    else: 
        waiting_count = answer_queu.qsize()
        end_of_sentence = " person on the waiting list"
        if waiting_count > 1:
            end_of_sentence = " people on the waiting list"

        await interaction.response.send_message("> " + question + "\nI am thinking to generate you the best answer...\nThere's currently " + str(answer_queu.qsize()) + end_of_sentence)
        original_message = await interaction.original_response()

        answer_queu.put([llm_large, question, original_message])

@tree.command(name="join", description="join your current voice channel")
async def slash_command(interaction: discord.Interaction):    
    global connected_voice_channels

    await interaction.response.send_message("Connecting to your channel...")
    original_message = await interaction.original_response()
    
    await client.wait_until_ready()
    
    try:
        channel = interaction.user.voice.channel
        connected_voice_channels.append(await channel.connect())
        await edit_message(original_message, "Connect with success to your channel")
    except:
        await edit_message(original_message, "Can't connect to your channel")


@tree.command(name="leave", description="leave your current voice channel")
async def slash_command(interaction: discord.Interaction):   
    await interaction.response.send_message("Leaving your channel...")
    original_message = await interaction.original_response()

    await client.wait_until_ready()

    for voice_channel in connected_voice_channels:
        if voice_channel.guild == interaction.guild:
            await edit_message(original_message, "Leave with success your channel")
            await voice_channel.disconnect()
            return
    await edit_message(original_message, "Can't leave your channel")

@client.event
async def on_ready():
    await tree.sync()
    await client.change_presence(activity=discord.Game('Github : https://github.com/konect-V/VirtualFrontier'))
    print(f'Logged in as `{client.user}`')

client.run(os.environ["DISCORD_TOKEN"])
