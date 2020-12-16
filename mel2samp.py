# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************\
import os
import random
import argparse
import json
import torch
import torch.utils.data
import sys
from scipy.io.wavfile import read

# We're using the audio processing from TacoTron2 to make sure it matches
sys.path.insert(0, 'tacotron2')
from tacotron2.layers import TacotronSTFT

MAX_WAV_VALUE = 32768.0

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


class stylistic_dataset(torch.utils.data.Dataset):
    def __init__(self, dataBase, style=('happy', 'sad', 'angry')):
        """
        :param dataBase: a dict stored what datasets we use. Keys are the name, and values are the path
        :param style:

        The final data is stored in self.final_data. The structure is that:
        {
            'text': ['my name is blablab', ...], # a list of text
            'happy': [(text_id, audio_path), ...] # each sample is a tuple (text_id, audio_path)
            'sad': [(text_id, audio_path), ...]
              .
              .
              .
        }
        The example on how to use the dataset is shown at the bottom
        """
        self.dataBase = dataBase
        self.style = style
        self.final_data = {'text': []}
        for s in self.style:
            self.final_data[s] = []

        if 'ravdess' in self.dataBase.keys():
            self.process_ravdess()

        if 'cremad' in self.dataBase.keys():
            self.process_cremad()

    def process_cremad(self):
        style2id = {'angry': 'ANG', 'disgust': 'DIS', 'fearful': 'FEA', 'happy': 'HAP',
                    'neutral': 'NEU', 'sad': 'SAD'}
        id2style = {}
        for style_name in style2id.keys():
            id2style[style2id[style_name]] = style_name
        style_id_we_need = []
        for style in self.style:
            style_id_we_need.append(style2id[style])
        texts = ['it\'s eleven o\'clock', 'That is exactly what happened', 'I\'m on my way to the meeting',
                 'I wonder what this is about', 'The airplane is almost full',
                 'Maybe tomorrow it will be cold', 'I would like a new alarm clock',
                 'I think I have a doctor\'s appointment', 'Don\'t forget a jacket (DFA)',
                 'I think I\'ve seen this before', 'The surface is slick',
                 'We\'ll stop in a couple of minutes']
        text_short = {'IEO':0, 'TIE':1, 'IOM':2, 'IWW':3, 'TAI':4, 'MTI':5,
                      'IWL':6, 'ITH':7, 'DFA':8, 'ITS':9, 'TSI':10, 'WSI':11}
        text_id_base = len(self.final_data['text'])
        self.final_data['text']+=texts
        for audio_name in os.listdir(self.dataBase['cremad']):
            if len(audio_name) < 5:
                continue
            splits = audio_name.split('_')
            audio_text_id = text_id_base + text_short[splits[1]]
            if splits[2] in style_id_we_need:
                self.final_data[id2style[splits[2]]].append((audio_text_id,
                                                             os.path.join(self.dataBase['cremad'], audio_name)))


    def process_ravdess(self):
        style2id = {'neutral': '01', 'calm': '02', 'happy': '03', 'sad': '04',
                    'angry': '05', 'fearful': '06', 'disgust': '07', 'surprised': '08'}
        id2style = {}
        for style_name in style2id.keys():
            id2style[style2id[style_name]] = style_name
        style_id_we_need = []
        for style in self.style:
            style_id_we_need.append(style2id[style])
        text_id1 = len(self.final_data['text'])
        self.final_data['text'].append('kids are talking by the door')
        text_id2 = text_id1+1
        self.final_data['text'].append('dogs are sitting by the door')
        for actor in os.listdir(self.dataBase['ravdess']):
            audio_path = os.path.join(self.dataBase['ravdess'], actor)
            for audio_name in os.listdir(audio_path):
                if len(audio_name) < 5:
                    continue
                splits = audio_name.split('-')
                if splits[2] in style_id_we_need:
                    if splits[4] == '01':
                        current_text_id = text_id1
                    else:
                        current_text_id = text_id2
                    self.final_data[id2style[splits[2]]].append((current_text_id, os.path.join(audio_path, audio_name)))

    def pick_one_random_sample(self, style):
        """
        :param style: string
        :return:
        """
        if style not in self.final_data.keys():
            return None
        else:
            return random.choice(self.final_data[style])

    def __getitem__(self, index):
        pass



class Mel2Samp(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, training_files, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
        self.audio_files = files_to_list(training_files)
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate

    def get_mel(self, audio):
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

        mel = self.get_mel(audio)
        audio = audio / MAX_WAV_VALUE

        return (mel, audio)

    def __len__(self):
        return len(self.audio_files)

# ===================================================================
# Takes directory of clean audio and makes directory of spectrograms
# Useful for making test sets
# ===================================================================
if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Output directory')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    data_config = json.loads(data)["data_config"]
    mel2samp = Mel2Samp(**data_config)

    filepaths = files_to_list(args.filelist_path)

    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    for filepath in filepaths:
        audio, sr = load_wav_to_torch(filepath)
        melspectrogram = mel2samp.get_mel(audio)
        filename = os.path.basename(filepath)
        new_filepath = args.output_dir + '/' + filename + '.pt'
        print(new_filepath)
        torch.save(melspectrogram, new_filepath)
