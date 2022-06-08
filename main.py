# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:40:33 2021

@author: Zhenzi Weng
"""

import os
import wave
import threading
import tkinter as tk
import numpy as np
from pyaudio import PyAudio, paInt16 
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showwarning
from tkinter import ttk
from deepscst import DeepSCST_model

chunk = 1024
sample_rate = 16000
level = 300
count_num = 20
save_length = 8
time_count = 60

audio_input_dir = "./audio_input/"
audio_record_dir = "./audio_record/"
if not os.path.exists(audio_record_dir):
    os.makedirs(audio_record_dir)
sys_output_dir = "./system_output/"
if not os.path.exists(sys_output_dir):
    os.makedirs(sys_output_dir)

pa = PyAudio() 
stream = pa.open(format=paInt16, channels=1, rate=sample_rate,
                 frames_per_buffer=chunk, input=True, output=True)

def open_audio_input_dir():
    v_trans.set("")
    file_name = askopenfilename(initialdir=audio_input_dir)
    if file_name:
        v_trans.set(file_name)

def open_audio_synthesis_dir():
    askopenfilename(initialdir=os.path.split(v_trans.get())[0])

def open_text_recognition_dir():
    askopenfilename(initialdir=os.path.split(v_trans.get())[0])

def start_record():
    global voice_string
    global save_buffer
    global tag 
    global wf
    global filename
    
    tag = True
    voice_string = []  
    save_buffer = []
    
    filename = os.path.join(audio_record_dir, "test_0.wav")
    file_index = 0
    while os.path.exists(filename):
        file_index += 1
        filename = os.path.split(filename)[0]+"/"+os.path.split(filename)[-1].split("_")[0]+"_{}.wav".format(file_index)
    wf = wave.open(filename, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    while tag:
        string_audio_data = stream.read(chunk)
        audio_data = np.frombuffer(string_audio_data, dtype=np.short) 
        large_sample_count = np.sum(audio_data>level)
        if large_sample_count>count_num:
            save_buffer.append(string_audio_data)

def stop_record(): 
    global voice_string
    global save_buffer
    global tag
    global wf
    global filename
    
    tag = False
    
    if not "save_buffer" in globals().keys():
        showwarning(title="Warning", message="You need to record the voice first!")
    else:
        if len(save_buffer)>0:
            voice_string = save_buffer 
            wf.writeframes(np.array(voice_string).tostring()) 
            wf.close()
            print("Record the voice successfully!")
            v_trans.set(filename)
            
            del save_buffer
            del voice_string
        else:
            showwarning(title="Warning", message="Voice not recorded, try speak loudly!")

def clear_record():
    global tag
    
    if not v_trans.get():
        showwarning(title="Warning", message="You need to choose the input file or record one!")
        tag = False
    else:
        entry_file.delete(0, tk.END)

def play_audio():
    if not v_trans.get():
        showwarning(title="Warning", message="You need to choose the input file or record one!")   
    else:
        wav_read = wave.open(v_trans.get(), "rb")
        read_data = wav_read.readframes(chunk)
        while len(read_data)>0:
            stream.write(read_data)
            read_data = wav_read.readframes(chunk)

def play_synthesized_audio():
    if len(entry_speech_output.get("0.0", "end"))==1:
        showwarning(title="Warning", message="You need to synthesize audio first!")
    else:
        audio_synthesis_file = entry_speech_output.get("0.0", "end").split("\n")[0]
        wav_read = wave.open(audio_synthesis_file, "rb")
        read_data = wav_read.readframes(chunk)
        while len(read_data)>0:
            stream.write(read_data)
            read_data = wav_read.readframes(chunk)
      
def thread_it(func, *args):
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    t.start()

def Start_Simulation():
    # delete previous decoded_str and synthesized_speech_file
    entry_text_output.delete("1.0", "end")
    entry_speech_output.delete("1.0", "end")
    # get file input
    file_input = v_trans.get()
    if not file_input:
        showwarning(title="Warning", message="You need to choose the input file or record one!")
        print("Resrart simulation")
    else:
        print("You chose file: ", file_input) 
        # get channel input
        if v_fading.get()==100:
            showwarning(title="Warning", message="You need to choose the input channel!")
            print("Resrart simulation")
        else:
            channel_input = channels[v_fading.get()][0]
            print("You chose channel: ", channel_input)
            # get snr input
            if not entry_snr.get():
                showwarning(title="Warning", message="You need to choose the input SNR!")
                print("Resrart simulation")
            else: 
                snr_input = float(entry_snr.get())
                print("You chose snr: ", snr_input,"dB")
                print("------------------------   running DeepSC-ST   -----------------------")
                decoded_str, synthesized_speech_file = model(file_input, channel_input, snr_input)
                print(decoded_str)
                print(synthesized_speech_file)
                # show new decoded_str and synthesized_speech_file.
                entry_text_output.insert( "insert", decoded_str)
                entry_speech_output.insert( "insert", synthesized_speech_file )
                print("Simulation end")

def Clear_Setting():
    # clear input .wav file, input channel, and input snr
    entry_file.delete(0, tk.END)
    v_fading.set(100)
    entry_snr.delete(0, tk.END)
# load DeepSC-ST model
model = DeepSCST_model(name="DeepSC-ST model")
# initialize GUI window
myWindow = tk.Tk()
myWindow.configure(bg="white")

screen_width = myWindow.winfo_screenwidth()
screen_height = myWindow.winfo_screenheight()
# title setting
myWindow.title("DeepSC-ST system for speech recognition and speech synthesis")
window_width = 850
window_height = 640
myWindow.geometry("{}x{}+{}+{}".format(window_width, window_height,
                                      (screen_width-window_width)//2,
                                      (screen_height-window_height)//2-36))
###############  transmitter input  ###############
tk.Label(myWindow, width=15, height=1, background="#90EE90", padx=0, pady=0,
         relief="flat", text="Transmittter Input", font=("times new roman", 24, "bold"),
         justify="center", anchor="center").grid(padx=0, pady=10, row=0, column=1, columnspan=3)

tk.Button(myWindow, width=10, height=1, background="#d9d9d9", padx=0, pady=0,
          relief="raised", text="Choose file", font=("times new roman", 16),
          justify="center", anchor="center",
          command=open_audio_input_dir).grid(padx=0, pady=0, row=1, column=0)

v_trans = tk.StringVar()
entry_file = tk.Entry(myWindow, width=50, relief="solid",
                      font=(None, 16), textvariable=v_trans)
entry_file.grid(padx=0, pady=0, row=1, column=1, columnspan=4)
###############  record .wav file  ###############
tk.Button(myWindow, width=10, height=1, background="#d9d9d9", padx=0, pady=0,
          relief="raised", text="Start record", font=("times new roman", 16),
          justify="center", anchor="center",
          command=lambda: thread_it(start_record)).grid(padx=42, pady=10, row=2, column=0)

tk.Button(myWindow, width=10, height=1, background="#d9d9d9", padx=0, pady=0,
          relief="raised", text="Stop record", font=("times new roman", 16),
          justify="center", anchor="center",
          command=lambda: thread_it(stop_record)).grid(padx=42, pady=10, row=2, column=1)

tk.Button(myWindow, width=10, height=1, background="#d9d9d9", padx=0, pady=0,
          relief="raised", text="Clear record", font=("times new roman", 16),
          justify="center", anchor="center",
          command=lambda: thread_it(clear_record)).grid(padx=42, pady=10, row=2, column=2)

tk.Button(myWindow, width=10, height=1, background="#d9d9d9", padx=0, pady=0,
          relief="raised", text="Play audio", font=("times new roman", 16),
          justify="center", anchor="center",
          command=lambda: thread_it(play_audio)).grid(padx=42, pady=10, row=2, column=3)
###############  channel condition  ###############
tk.Label(myWindow, width=15, height=1, background="#90EE90", padx=0, pady=0,
         relief="flat", text="Channel Condition", font=("times new roman", 24, "bold"),
         justify="center", anchor="center").grid(padx=0, pady=10, row=3, column=1, columnspan=3)

tk.Label(myWindow, width=15, background="#d9d9d9", padx=0, pady=3,
         relief="flat", text="Channel fading", font=("times new roman", 16),
         justify="center", anchor="center").grid(padx=0, pady=0, row=4, column=0)
# fading
v_fading = tk.IntVar()
v_fading.set(100)
# for loop to create each selection
channels = [("AWGN Channel", 0), ("Rayleigh Channel", 1), ("Rician Channel", 2)]
for chan, num in channels:
    choose_channel = tk.Radiobutton(myWindow, width=12, background="#FFFFFF",
                                    padx=0, pady=0, text=chan,
                                    font=("times new roman", 16), value=num,
                                    variable=v_fading)
    choose_channel.grid(padx=0, pady=0, row=4, column=num+1)
# snr
tk.Label(myWindow, width=15, background="#d9d9d9", padx=0, pady=3,
         relief="flat", text='Signal-to-noise ratio', font=("times new roman", 16),
         justify="center", anchor="center").grid(padx=0, pady=10, row=5, column=0)

entry_snr = tk.Entry(myWindow, width=17, relief="solid", font=("times new roman", 16))
entry_snr.grid(padx=0, pady=0, row=5, column=2)

tk.Label(myWindow, width=17, background="#FFFFFF", padx=0, pady=0,
         text="dB", font=("times new roman", 16),
         justify="left", anchor="w").grid(padx=0, pady=0, row=5, column=3)
###############  split line  ###############
line_style = ttk.Style()
line_style.configure("Line.TSeparator", background="#000000")
split_line = ttk.Separator(myWindow, orien = tk.HORIZONTAL, style="Line.TSeparator")
split_line.grid(padx=0, pady=0, row=6, column=0, columnspan=4, sticky="we")
###############  start simulation  ###############
tk.Button(myWindow, width=6, height=1, background="#a2c4c9", padx=0, pady=0,
          relief="raised", text="Clear", font=("times new roman", 20, "bold"),
          justify="center", anchor="center",
          command=lambda: thread_it(Clear_Setting)).grid(padx=0, pady=15, row=7, column=0, columnspan=2)

tk.Button(myWindow, width=6, height=1, background="#a2c4c9", padx=0, pady=0, 
          relief="raised", text="Run", font=("times new roman", 20, "bold"),
          justify="center", anchor="center",
          command=lambda: thread_it(Start_Simulation)).grid(padx=0, pady=15, row=7, column=1, columnspan=2)

tk.Button(myWindow, width=6, height=1, background="#a2c4c9", padx=0, pady=0,
          relief="raised", text="Quit", font=("times new roman", 20, "bold"),
          justify="center", anchor="center",
          command=lambda: thread_it(myWindow.quit)).grid(padx=0, pady=15, row=7, column=2, columnspan=2)
###############  split line  ###############
line_style = ttk.Style() 
line_style.configure("Line.TSeparator", background="#000000")
split_line = ttk.Separator(myWindow, orien = tk.HORIZONTAL, style="Line.TSeparator")
split_line.grid(padx=0, pady=0, row=8, column=0, columnspan=5, sticky="we")
###############  receiver output  ###############
tk.Label(myWindow, width=15, height=1, background="#90EE90", padx=0, pady=0,
         relief="flat", text="Receiver Output", font=("times new roman", 24, "bold"),
         justify="center", anchor="center").grid(padx=0, pady=20, row=9, column=1, columnspan=3)
# text output
tk.Button(myWindow, width=11, height=1, background="#d9d9d9", padx=0, pady=0,
          relief="raised", text="Text output", font=("times new roman", 16),
          justify="center", anchor="center",
          command=open_text_recognition_dir).grid(padx=0, pady=0, row=10, column=0)

entry_text_output = tk.Text(myWindow, height=3, width=51, relief="solid",
                            padx=0, pady=0, font=("times new roman", 16),
                            wrap=tk.WORD)
entry_text_output.grid(padx=0, pady=0, row=10, column=1, columnspan=3)
# speech output
tk.Button(myWindow, width=11, height=1, background="#d9d9d9", padx=0, pady=0,
          relief="raised", text="Speech output", font=("times new roman", 16),
          justify="center", anchor="center",
          command=open_audio_synthesis_dir).grid(padx=0, pady=10, row=11, column=0)

entry_speech_output = tk.Text(myWindow, height=2, width=51, relief="solid",
                              padx=0, pady=0, font=("times new roman", 16),
                              wrap=tk.CHAR)
entry_speech_output.grid(padx=0, pady=0, row=11, column=1, columnspan=3)

tk.Button(myWindow, width=12, height=1, background="#d9d9d9", padx=0, pady=0,
          relief="raised", text="Play audio", font=("times new roman", 16),
          justify="center", anchor="center",
          command=lambda: thread_it(play_synthesized_audio)).grid(padx=0, pady=0, row=12, column=1, columnspan=3)
###############  enter loop  ###############
myWindow.mainloop()

