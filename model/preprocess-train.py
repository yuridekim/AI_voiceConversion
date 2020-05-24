import os
import time

from speech_tools import *

speaker_list = ['VCC2SF1','VCC2SF2','VCC2SM1','VCC2SM2'] 

start_time = time.time()

sampling_rate = 22050
num_mcep = 36
frame_period = 5.0
n_frames = 128

dtype = 'train'
data_root = os.path.join('corpus', dtype)
exp_root = os.path.join('data', dtype)
for speaker in speaker_list:
    train_dir = os.path.join(data_root, speaker)
    exp_dir = os.path.join(exp_root, speaker)
    
    os.makedirs(exp_dir, exist_ok=True)
    print('Loading {} Wavs...'.format(speaker))
    wavs = load_wavs(wav_dir=train_dir, sr=sampling_rate)

    print('Extracting acoustic features...')
    f0s, timeaxes, sps, aps, coded_sps = world_encode_data(wavs=wavs, fs=sampling_rate,
                                                                    frame_period=frame_period, num_mcep=num_mcep)

    print('Calculating F0 statistics...')

    log_f0s_mean, log_f0s_std = logf0_statistics(f0s)

    print('Log Pitch {}'.format(speaker))
    print('Mean: %f, Std: %f' % (log_f0s_mean, log_f0s_std))
    
    print('Normalizing data...')
    coded_sps_transposed = transpose_in_list(lst=coded_sps)

    coded_sps_norm, coded_sps_mean, coded_sps_std = mcs_normalization_fit_transform(
        mcs=coded_sps_transposed)
    
    print('Saving {} data...'.format(speaker))
    save_pickle(os.path.join(exp_dir, 'cache{}.p'.format(num_mcep)),
                (coded_sps_norm, coded_sps_mean, coded_sps_std, log_f0s_mean, log_f0s_std))
    
end_time = time.time()
time_elapsed = end_time - start_time

print('Preprocessing Done.')

print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (
    time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
