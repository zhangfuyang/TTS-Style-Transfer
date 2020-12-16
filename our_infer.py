import os
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


def main(files, waveglow_path, sigma, output_dir, sampling_rate, is_fp16,
         denoiser_strength, args):
    #mel_files = files_to_list(mel_files)
    #print(mel_files)
    files = ['/local-scratch/fuyang/cmpt726/final_project/cremad/1091_WSI_SAD_XX.wav']
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
    #waveglow = torch.load(waveglow_path)['model']
    #waveglow = waveglow.remove_weightnorm(waveglow)
    #waveglow.cuda()
    waveglow = model
    if is_fp16:
        from apex import amp
        waveglow, _ = amp.initialize(waveglow, [], opt_level="O1")

    if denoiser_strength > 0:
        denoiser = Denoiser(waveglow).cuda()

    mel_extractor = Get_mel(1024, 256, 1024, args.sampling_rate, 0.0, 8000.0)

    for i, file_path in enumerate(files):
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
        z = outputs[0][:,4:]
        print(outputs)
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
            output_dir, "{}_synthesis.wav".format('fuyangz'))
        write(audio_path, sampling_rate, audio)
        print(audio_path)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
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

    main(args.filelist_path, args.waveglow_path, args.sigma, args.output_dir,
         args.sampling_rate, args.is_fp16, args.denoiser_strength, args)
