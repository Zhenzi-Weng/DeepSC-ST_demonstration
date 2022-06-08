# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 18:28:36 2022

@author: Zhenzi Weng
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import argparse
import numpy as np
import wave
import audioop
import struct
import codecs
import soundfile
from asr_model import ASR_model
from decoder import DeepSpeechDecoder
from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import TFAutoModel

print("tensorflow version: ", tf.__version__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

###############    define global parameters    ###############
def parse_args():
    parser = argparse.ArgumentParser(description="DeepSC-ST system for speech recognition and speech synthesis.")
    # parameters of mel spectrogram
    parser.add_argument("--sample_rate", type=int, default=16000, help="sample rate for wav file.")
    parser.add_argument("--frame_length", type=int, default=20, help="the time duration in ms of a frame.")
    parser.add_argument("--frame_stride", type=int, default=10, help="the time duration in ms of frame stride.")
    parser.add_argument("--feature_norm", dest="feature_norm", action="store_true",
                        default=True, help="whether normalize the audio mfcc feature.") 
    # parameters of file path direction
    parser.add_argument("--vocabulary_path", type=str, default="./vocabulary.txt", help="the txt file path of vocabulary.")
    # parameters of model
    parser.add_argument("--num_rnn_layers", type=int, default=7, help="the number of cascaded rnn layers.")
    parser.add_argument("--rnn_hidden_size", type=int, default=800, help="the hidden size of each rnn layer.")
    parser.add_argument("--rnn_type", type=str, default="gru", help="the type of RNN cell.")
    parser.add_argument("--is_bidirectional", dest="is_bidirectional", action="store_true",
                        default=True, help="whether rnn unit is bidirectional.")
    parser.add_argument("--use_bias", dest="use_bias", action="store_true",
                        default=True, help="whether use bias in the last fully-connected layer.")
    parser.add_argument("--num_channel_units", type=int, default=40,
                        help="the unmber of units in each dense layer of channel encoding and decoding.")
    
    args = parser.parse_args()
    
    return args

args = parse_args()
print("Called with args:", args)

def compute_spectrogram_feature(audio_samples, sample_rate, frame_length=20,
                                frame_stride=10, max_freq=None, eps=1e-14):
    if max_freq is None:
        max_freq = sample_rate/2
    if max_freq>sample_rate/2: 
        raise ValueError("max_freq must not be greater than half of sample rate.")
    if frame_stride>frame_length:
        raise ValueError("stride size must not be greater than frame size.")
  
    stride_size = int(0.001*sample_rate*frame_stride)
    frame_size = int(0.001*sample_rate*frame_length)
    # extract strided windows
    truncate_size = (len(audio_samples)-frame_size)%stride_size
    audio_samples = audio_samples[:len(audio_samples)-truncate_size]
    nshape = (frame_size, (len(audio_samples)-frame_size)//stride_size+1)
    nstrides = (audio_samples.strides[0], audio_samples.strides[0]*stride_size)
    frames = np.lib.stride_tricks.as_strided(audio_samples, shape=nshape, strides=nstrides)
    assert np.all(frames[:, 1] == audio_samples[stride_size:(stride_size+frame_size)])
    # window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(frame_size)[:, None]
    fft = np.fft.rfft(frames*weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2
    scale = np.sum(weighting**2)*sample_rate
    fft[1:-1, :] *= (2.0/scale)
    fft[(0, -1), :] /= scale
    # prepare fft frequency list
    freqs = float(sample_rate)/frame_size*np.arange(fft.shape[0])
    # compute spectrogram feature
    ind = np.where(freqs<=max_freq)[0][-1]+1
    specgram = np.log(fft[:ind, :]+eps)
    specgram = np.transpose(specgram, (1, 0))
    
    return specgram

def normalize_audio_feature(audio_features):
    mean = np.mean(audio_features, axis=0)
    var = np.var(audio_features, axis=0)
    normalized_features = (audio_features-mean)/(np.sqrt(var)+1e-6)
    
    return normalized_features

def preprocess_audio(audio_file, args):
    # Load the audio file and compute spectrogram feature
    sample_rate = args.sample_rate
    frame_length = args.frame_length
    frame_stride = args.frame_stride
    feature_norm = args.feature_norm
    
    audio_samples, _ = soundfile.read(audio_file)
    audio_features = compute_spectrogram_feature(audio_samples, sample_rate, frame_length, frame_stride)
    # feature normalization
    if feature_norm:
        audio_features = normalize_audio_feature(audio_features)
    # adding Channel dimension for conv2D input.
    audio_features = np.expand_dims(audio_features, axis=2)
    
    return audio_features

def stereo_to_mono_downsampled(file_input):
    wave_read = wave.open(file_input, "rb")
    nchannels, samplewidth, framerate, nframes, ctype, compress = wave_read.getparams()
    str_data = wave_read.readframes(nframes)
    wave_read.close()
    if framerate!=16000:
        # downsample
        str_data = audioop.ratecv(str_data, samplewidth, nchannels, framerate, 16000, None)[0]
    if nchannels==2:
        wave_data = np.frombuffer(str_data, dtype=np.short)
        wave_data.shape = (-1, 2)
        wave_data = wave_data.T
        mono_wave = (wave_data[0]+wave_data[1])/2
        mono_audio_file = os.path.join(os.path.split(file_input)[0], 
                                       os.path.split(file_input)[-1].split(".")[0]+"_edited.wav") 
        mono = wave.open(mono_audio_file, "wb")
        mono.setparams((1, samplewidth, 16000, 0, "NONE", "Uncompressed"))
        for i in mono_wave:
            data = struct.pack("<h", int(i))
            mono.writeframesraw( data )
        mono.close()
        
        return mono_audio_file
    else:
        mono_audio_file = os.path.join(os.path.split(file_input)[0], 
                                       os.path.split(file_input)[-1].split(".")[0]+"_edited.wav") 
        mono = wave.open(mono_audio_file, "wb")
        mono.setparams((1, samplewidth, 16000, 0, "NONE", "Uncompressed"))
        mono.writeframes(str_data)
        mono.close()
        
        return mono_audio_file

def recognition_model(args, num_classes):
    features = tf.keras.layers.Input(name="features", shape=(None, 161, 1), dtype=tf.float32)
    AWGN_flag = tf.keras.layers.Input(name="AWGN_flag", shape=(), dtype=tf.float32)
    Rayleigh_flag = tf.keras.layers.Input(name="Rayleigh_flag", shape=(), dtype=tf.float32)
    Rician_flag = tf.keras.layers.Input(name="Rician_flag", shape=(), dtype=tf.float32)
    std = tf.keras.layers.Input(name="std", shape=(), dtype=tf.float32)
    
    model = ASR_model(args, num_classes)
    logits = model(features, AWGN_flag, Rayleigh_flag, Rician_flag, std)
    model = tf.keras.models.Model(inputs=[features, AWGN_flag, Rayleigh_flag, Rician_flag, std],
                                  outputs=logits)

    return model

class TextParams(object):
    def __init__(self, vocab_path):
        lines = []
        with codecs.open(vocab_path, "r", "utf-8") as fin:
            lines.extend(fin.readlines())
        self.character_to_index = {}
        self.index_to_character = {}
        self.speech_labels = ""
        index = 0
        for line in lines:
            if line.startswith("#"):
                continue   
            while line[-1] == "\n" or line[-1]=="\r":
                line = line[:-1]
            self.character_to_index[line] = index
            self.index_to_character[index] = line
            self.speech_labels += line
            index += 1

class DeepSCST_model(object): 
    def __init__(self, name):
        self.name = name
        ###############    initialize decoder    ###############
        text_params = TextParams(args.vocabulary_path)
        num_classes = len(text_params.speech_labels)
        self.greedy_decoder = DeepSpeechDecoder(text_params.speech_labels)
        ###############    define ASR model    ###############
        features = tf.keras.layers.Input(name="features", shape=(None, 161, 1), dtype=tf.float32)
        AWGN_flag = tf.keras.layers.Input(name="AWGN_flag", shape=(), dtype=tf.float32)
        Rayleigh_flag = tf.keras.layers.Input(name="Rayleigh_flag", shape=(), dtype=tf.float32)
        Rician_flag = tf.keras.layers.Input(name="Rician_flag", shape=(), dtype=tf.float32)
        std = tf.keras.layers.Input(name="std", shape=(), dtype=tf.float32)
        asr_model = ASR_model(args, num_classes)
        logits = asr_model(features, AWGN_flag, Rayleigh_flag, Rician_flag, std)
        self.asr_model = tf.keras.models.Model(inputs=[features, AWGN_flag, Rayleigh_flag, Rician_flag, std], outputs=logits)
        ###################    load ASR model    ###################
        trained_network = "./trained_model/"
        ckpt_path = os.path.join(trained_network, "saved_weights_{}".format(60)+".ckpt")
        self.asr_model.load_weights(ckpt_path)
        ###################    load TTS model    ###################
        self.processor = AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
        self.tacotron2 = TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
        self.mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")
    
    def __call__(self, file_input, channel_input, snr_input):
        # edit the input file to mono 16000Hz .wav file
        edited_file = stereo_to_mono_downsampled(file_input)
        features = preprocess_audio(edited_file, args)
        input_length = features.shape[0]
        # delete the edited wav file
        os.remove(edited_file)
        # append batch_size dim
        features = np.expand_dims(features, axis=0)
        
        if channel_input == "AWGN":
            AWGN_flag = np.array([1.0], dtype=np.float32)
            Rayleigh_flag = np.array([0.0], dtype=np.float32)
            Rician_flag = np.array([0.0], dtype=np.float32)
        elif channel_input == "Rayleigh":
            AWGN_flag = np.array([0.0], dtype=np.float32)
            Rayleigh_flag = np.array([1.0], dtype=np.float32)
            Rician_flag = np.array([0.0], dtype=np.float32)
        else:
            AWGN_flag = np.array([0.0], dtype=np.float32)
            Rayleigh_flag = np.array([0.5], dtype=np.float32)
            Rician_flag = np.array([0.5], dtype=np.float32)
        # ASR process
        snr = pow(10, (snr_input / 10))
        std = np.sqrt(1 / (2*snr))
        std = np.array([std], dtype=np.float32)
        logits = self.asr_model(inputs=[features, AWGN_flag, Rayleigh_flag, Rician_flag, std])
        decoded_str = self.greedy_decoder.decode(logits[0][0:int((int((input_length-1)/2+1)-1)/2+1)])    
        transcripts_file = os.path.join(os.path.split(os.path.split(file_input)[0])[0],
                                        "system_output",
                                        os.path.split(file_input)[-1].split(".")[0]+".txt")
        write_transcripts_file = open(transcripts_file, "w")
        write_transcripts_file.write(decoded_str)
        write_transcripts_file.close()
        # TTS process
        input_ids = self.processor.text_to_sequence(decoded_str)
        decoder_output, mel_outputs, stop_token_prediction, alignment_history = self.tacotron2.inference(
            input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            input_lengths=tf.convert_to_tensor([len(input_ids)], tf.int32),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32))
        # melgan inference (mel-to-wav)
        audio = self.mb_melgan.inference(mel_outputs)[0, :, 0]
        # synthesized .wav file
        synthesized_speech_file = os.path.split(os.path.split(file_input)[0])[0]+"/system_output/"+os.path.split(file_input)[-1].split(".")[0]+"_synthesized.wav"
        # .wav file must be 22050 Hz
        soundfile.write(synthesized_speech_file, audio, 22050, "PCM_16")
        # downsample 22050Hz to 16000Hz
        f = wave.open(synthesized_speech_file, "rb")
        params = f.getparams()   
        nchannels, sampwidth, framerate, nframes, ctype, compress = params
        read_data = f.readframes(nframes)
        f.close()
        os.remove(synthesized_speech_file)
        converted = audioop.ratecv(read_data, sampwidth, nchannels, framerate, 16000, None)
        s_write = wave.open(synthesized_speech_file, "w")
        s_write.setparams((nchannels, sampwidth, 16000, 0, "NONE", "Uncompressed"))
        s_write.writeframes(converted[0])
        s_write.close()
        
        return decoded_str, synthesized_speech_file

