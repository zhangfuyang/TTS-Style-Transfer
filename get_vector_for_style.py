import os
import numpy as np
import json
import random
import torch
from mel2samp import files_to_list, MAX_WAV_VALUE
from denoiser import Denoiser
from mel2samp import load_wav_to_torch
from scipy.io.wavfile import write
import resampy
from tacotron2.layers import TacotronSTFT
from glow import WaveGlow, WaveGlowLoss

class Get_mel():
    def __init__(self, filter_length, hop_length, win_length,
                 sampling_rate, mel_fmin, mel_fmax):
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
    def get_mel(self, audio):
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec


class voice_dataset:
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


def main(style, waveglow_path, sigma, output_dir, sampling_rate, is_fp16,
         denoiser_strength, args):
    #mel_files = files_to_list(mel_files)
    #print(mel_files)
    dataset = voice_dataset(dataBase={'ravdess': './our_data/ravdess', 'cremad': './our_data/cremad'},  style=('happy', 'sad', 'angry'))
    #print(len(dataset.final_data['happy']))

    #sample = dataset.pick_one_random_sample('happy')
    files = dataset.final_data[style]
    #files = ['/local-scratch/fuyang/cmpt726/waveglow/data/LJSpeech-1.1/LJ001-0001.wav']
    with open('config.json') as f:
        data = f.read()
    config = json.loads(data)
    waveglow_config = config["waveglow_config"]
    model = WaveGlow(**waveglow_config)
    checkpoint_dict = torch.load('waveglow_256channels_universal_v5.pt', map_location='cpu')
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    model.cuda()
    waveglow = model
    if is_fp16:
        from apex import amp
        waveglow, _ = amp.initialize(waveglow, [], opt_level="O1")

    if denoiser_strength > 0:
        denoiser = Denoiser(waveglow).cuda()

    mel_extractor = Get_mel(1024, 256, 1024, args.sampling_rate, 0.0, 8000.0)
    avg_z = np.zeros(8)
    _count = 0
    for i, (_, file_path) in enumerate(files):
        if i > 50:
            break
        try:
            audio, rate = load_wav_to_torch(file_path)
            if rate != sampling_rate:
                audio = resampy.resample(audio.numpy(), rate, sampling_rate)
                audio = torch.from_numpy(audio).float()
            #if audio.size(0) >= args.segment_length:
            #    max_audio_start = audio.size(0) - args.segment_length
            #    audio_start = random.randint(0, max_audio_start)
            #    audio = audio[audio_start:audio_start+args.segment_length]
            #else:
            #    audio = torch.nn.functional.pad(audio, (0, args.segment_length-audio.size(0)), 'constant').data
            mel = mel_extractor.get_mel(audio)
            audio = audio / MAX_WAV_VALUE

            mel = torch.autograd.Variable(mel.cuda().unsqueeze(0))
            audio = torch.autograd.Variable(audio.cuda().unsqueeze(0))
            audio = audio.half() if is_fp16 else audio
            mel = mel.half() if is_fp16 else mel
            outputs = waveglow((mel, audio))
            avg_z += outputs[0].squeeze(0).mean(1).detach().cpu().numpy()
            _count += 1
            z = outputs[0][:,4:]

            #print(outputs)
            mel_up = waveglow.upsample(mel)
            time_cutoff = waveglow.upsample.kernel_size[0]-waveglow.upsample.stride[0]
            mel_up = mel_up[:,:,:-time_cutoff]
            #mel_up = mel_up[:,:,:-(time_cutoff+128)]

            mel_up = mel_up.unfold(2, waveglow.n_group, waveglow.n_group).permute(0,2,1,3)
            mel_up = mel_up.contiguous().view(mel_up.size(0), mel_up.size(1), -1).permute(0, 2, 1)
            audio = z
            mel_up = mel_up[:,:,:audio.size(2)]

            sigma = 0.7
            z_i = 0
            for k in reversed(range(waveglow.n_flows)):
                n_half = int(audio.size(1)/2)
                audio_0 = audio[:,:n_half, :]
                audio_1 = audio[:, n_half:, :]

                output = waveglow.WN[k]((audio_0, mel_up))

                s = output[:,n_half:, :]
                b = output[:, :n_half, :]
                audio_1 = (audio_1-b)/torch.exp(s)
                audio = torch.cat([audio_0, audio_1],1)

                audio = waveglow.convinv[k](audio, reverse=True)

                if k % waveglow.n_early_every == 0 and k > 0:
                    z = outputs[0][:, 2-z_i:4-z_i]
                    #if mel_up.type() == 'torch.cuda.HalfTensor':
                    #    z = torch.cuda.HalfTensor(mel_up.size(0), waveglow.n_early_size, mel_up.size(2)).normal_()
                    #else:
                    #    z = torch.cuda.FloatTensor(mel_up.size(0), waveglow.n_early_size, mel_up.size(2)).normal_()
                    audio = torch.cat((sigma*z, audio),1)
            audio = audio.permute(0,2,1).contiguous().view(audio.size(0), -1).data
            audio = audio * MAX_WAV_VALUE
            audio = audio.squeeze()
            audio = audio.cpu().numpy()
            audio = audio.astype('int16')
            audio_path = os.path.join(
                output_dir, "{}_synthesis.wav".format(file_path[:-4]))
            if os.path.exists(os.path.join(*audio_path.split('/')[:-1])) is False:
                os.makedirs(os.path.join(*audio_path.split('/')[:-1]), exist_ok=True)
            write(audio_path, sampling_rate, audio)
            print(audio_path)
        except:
            continue

    avg_z = avg_z / _count
    np.save(style, avg_z)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-S', "--style", required=True)
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow decoder checkpoint with model')
    parser.add_argument('-o', "--output_dir", required=True)
    parser.add_argument("-s", "--sigma", default=1.0, type=float)
    parser.add_argument("--sampling_rate", default=22050, type=int)
    parser.add_argument("--segment_length", default=16000, type=int)
    parser.add_argument("--is_fp16", action="store_true")
    parser.add_argument("-d", "--denoiser_strength", default=0.0, type=float,
                        help='Removes model bias. Start with 0.1 and adjust')

    args = parser.parse_args()

    main(args.style, args.waveglow_path, args.sigma, args.output_dir,
         args.sampling_rate, args.is_fp16, args.denoiser_strength, args)
