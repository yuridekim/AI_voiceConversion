import os
import time
import librosa
from speech_tools import *

#dataset = 'vcc2018'

speaker_list = ['VCC2SF1','VCC2SF2','VCC2SM1','VCC2SM2'] # VCC2

start_time = time.time()

sampling_rate = 22050
num_mcep = 36
frame_period = 5.0
n_frames = 128

dtype = "test"
data_root = os.path.join('corpus', dtype)
exp_root = os.path.join('data', dtype)

for speaker in speaker_list:
    train_dir = os.path.join(data_root, speaker)
    exp_dir = os.path.join(exp_root, speaker)

    cmd="find "+train_dir+" -iname '*.wav' > flist.txt"
    os.system(cmd)
    
    os.makedirs(exp_dir, exist_ok=True)
    print('Loading {} Wavs...'.format(speaker))
    with open("flist.txt", 'r') as f:
        for path in f:
            path = path[:-1]
            info = path.split(".")[-2]
            utt_id = info.split("/")[-1]
            print("Processing",utt_id)
            wav, _ = librosa.load(path, sr = sampling_rate, mono = True)
            f0, timeaxis, sp, ap, coded_sp = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period, num_mcep=num_mcep)
            frame_num = 4 * (len(f0) // 4)
            coded_sp = coded_sp[:frame_num]
            f0 = f0[:frame_num]
            ap = ap[:frame_num]

            save_pickle(os.path.join(exp_dir, '{}.p'.format(utt_id)), (coded_sp.T, f0, ap))
    
    os.system("rm flist.txt")
end_time = time.time()
time_elapsed = end_time - start_time

print('Preprocessing Done.')

print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (
    time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
