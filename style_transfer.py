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


def main(style, wav_path, sigma, output_dir, sampling_rate, is_fp16,
         denoiser_strength, args):
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
    avg_style = np.load(style+'.npy')
    avg_style = torch.from_numpy(avg_style).float().cuda().unsqueeze(0).unsqueeze(-1)
    avg_style = avg_style.half() if is_fp16 else avg_style

    audio, rate = load_wav_to_torch(wav_path)
    if rate != sampling_rate:
        audio = resampy.resample(audio.numpy(), rate, sampling_rate)
        audio = torch.from_numpy(audio).float()
    mel = mel_extractor.get_mel(audio)
    audio = audio / MAX_WAV_VALUE

    mel = torch.autograd.Variable(mel.cuda().unsqueeze(0))
    audio = torch.autograd.Variable(audio.cuda().unsqueeze(0))
    audio = audio.half() if is_fp16 else audio
    mel = mel.half() if is_fp16 else mel
    outputs = waveglow((mel, audio))
    for ss in range(5):
        out_z = outputs[0] * (1 - ss*0.2) + avg_style * ss*0.2
        z = out_z[:,4:]
        mel_up = waveglow.upsample(mel)
        time_cutoff = waveglow.upsample.kernel_size[0]-waveglow.upsample.stride[0]
        mel_up = mel_up[:,:,:-time_cutoff]

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
                z = out_z[:, 2-z_i:4-z_i]
                audio = torch.cat((sigma*z, audio),1)
        audio = audio.permute(0,2,1).contiguous().view(audio.size(0), -1).data
        audio = audio * MAX_WAV_VALUE
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
        audio_path = os.path.join(
            output_dir, "{}_synthesis_{}.wav".format('transfer', ss))
        write(audio_path, sampling_rate, audio)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-S', "--style", required=True)
    parser.add_argument('-w', '--wavepath',
                        help='Path to waveglow decoder checkpoint with model')
    parser.add_argument('-o', "--output_dir", required=True)
    parser.add_argument("-s", "--sigma", default=1.0, type=float)
    parser.add_argument("--sampling_rate", default=22050, type=int)
    parser.add_argument("--segment_length", default=16000, type=int)
    parser.add_argument("--is_fp16", action="store_true")
    parser.add_argument("-d", "--denoiser_strength", default=0.0, type=float,
                        help='Removes model bias. Start with 0.1 and adjust')

    args = parser.parse_args()

    main(args.style, args.wavepath, args.sigma, args.output_dir,
         args.sampling_rate, args.is_fp16, args.denoiser_strength, args)
