import librosa
import numpy as np
import pyworld
import pickle
import librosa
import numpy as np
import os
import pyworld
import pysptk
import time
import torch


def world_decompose(wav, fs, frame_period = 5.0, num_mcep=36):

    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    alpha = pysptk.util.mcepalpha(fs)
    mc = pysptk.conversion.sp2mc(sp, order=num_mcep-1, alpha=alpha)

    return f0, timeaxis, sp, ap, mc

def world_decode_mc(mc, fs):

    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    #coded_sp = coded_sp.astype(np.float32)
    #coded_sp = np.ascontiguousarray(coded_sp)
    alpha = pysptk.util.mcepalpha(fs)
    sp = pysptk.conversion.mc2sp(mc, alpha, fftlen)
    # decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)

    return sp



def transpose_in_list(lst):

    transposed_lst = list()
    for array in lst:
        transposed_lst.append(array.T)
    return transposed_lst


def world_decode_data(coded_sps, fs):

    decoded_sps =  list()

    for coded_sp in coded_sps:
        decoded_sp = world_decode_spectral_envelop(coded_sp, fs)
        decoded_sps.append(decoded_sp)

    return decoded_sps




def mcs_normalization_fit_transform(mcs):

    mcs_concatenated = np.concatenate(mcs, axis = 1)
    mcs_mean = np.mean(mcs_concatenated, axis = 1, keepdims = True)
    mcs_std = np.std(mcs_concatenated, axis = 1, keepdims = True)

    mcs_normalized = list()
    for mc in mcs:
        mcs_normalized.append((mc - mcs_mean) / mcs_std)
    
    return mcs_normalized, mcs_mean, mcs_std

def coded_sps_normalization_transform(coded_sps, coded_sps_mean, coded_sps_std):

    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)
    
    return coded_sps_normalized

def coded_sps_normalization_inverse_transform(normalized_coded_sps, coded_sps_mean, coded_sps_std):

    coded_sps = list()
    for normalized_coded_sp in normalized_coded_sps:
        coded_sps.append(normalized_coded_sp * coded_sps_std + coded_sps_mean)

    return coded_sps

def coded_sp_padding(coded_sp, multiple = 4):

    num_features = coded_sp.shape[0]
    num_frames = coded_sp.shape[1]
    num_frames_padded = int(np.ceil(num_frames / multiple)) * multiple
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    coded_sp_padded = np.pad(coded_sp, ((0, 0), (num_pad_left, num_pad_right)), 'constant', constant_values = 0)

    return coded_sp_padded


def logf0_statistics(f0s):

    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()

    return log_f0s_mean, log_f0s_std

def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):

    # Logarithm Gaussian normalization for Pitch Conversions
    f0_converted = np.exp((np.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    return f0_converted

def wavs_to_specs(wavs, n_fft = 1024, hop_length = None):

    stfts = list()
    for wav in wavs:
        stft = librosa.stft(wav, n_fft = n_fft, hop_length = hop_length)
        stfts.append(stft)

    return stfts



def wavs_to_mfccs(wavs, sr, n_fft = 1024, hop_length = None, n_mels = 128, n_mfcc = 36):

    mfccs = list()
    for wav in wavs:
        mfcc = librosa.feature.mfcc(y = wav, sr = sr, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels, n_mfcc = n_mfcc)
        mfccs.append(mfcc)

    return mfccs


def mfccs_normalization(mfccs):

    mfccs_concatenated = np.concatenate(mfccs, axis = 1)
    mfccs_mean = np.mean(mfccs_concatenated, axis = 1, keepdims = True)
    mfccs_std = np.std(mfccs_concatenated, axis = 1, keepdims = True)

    mfccs_normalized = list()
    for mfcc in mfccs:
        mfccs_normalized.append((mfcc - mfccs_mean) / mfccs_std)
    
    return mfccs_normalized, mfccs_mean, mfccs_std


def load_wavs(wav_dir, sr):
    
    debug_num = 0
    wavs = list()
    for file in os.listdir(wav_dir):
        if(file.count("wav") == 0):
            continue        
        """
        debug_num += 1
        if (debug_num > 100):
            break
        """
        file_path = os.path.join(wav_dir, file)
        wav, _ = librosa.load(file_path, sr = sr, mono = True)
        wav = librosa.util.normalize(wav, norm=np.inf, axis=None)
        #wav = wav.astype(np.float64)
        wavs.append(wav)

    return wavs





"""
def world_decode_spectral_envelop(coded_sp, fs):
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    # coded_sp = coded_sp.astype(np.float32)
    # coded_sp = np.ascontiguousarray(coded_sp)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)

    return decoded_sp
"""

def world_encode_data(wavs, fs, frame_period = 5.0, num_mcep = 36):

    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    mcs = list()

    for wav in wavs:
        f0, timeaxis, sp, ap, mc = world_decompose(wav = wav, fs = fs, frame_period = frame_period, num_mcep=num_mcep)
        f0s.append(f0)
        timeaxes.append(timeaxis)
        sps.append(sp)
        aps.append(ap)
        mcs.append(mc)

    return f0s, timeaxes, sps, aps, mcs





def world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period):
    #decoded_sp = decoded_sp.astype(np.float64)
    wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
    # Librosa could not save wav if not doing so
    wav = wav.astype(np.float64)

    return wav


def world_synthesis_data(f0s, decoded_sps, aps, fs, frame_period):
    wavs = list()

    for f0, decoded_sp, ap in zip(f0s, decoded_sps, aps):
        wav = world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period)
        wavs.append(wav)

    return wavs

"""
def coded_sps_normalization_fit_transform(coded_sps):
    coded_sps_concatenated = np.concatenate(coded_sps, axis=1)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_std = np.std(coded_sps_concatenated, axis=1, keepdims=True)

    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)

    return coded_sps_normalized, coded_sps_mean, coded_sps_std

"""



"""
def wav_padding(wav, sr, frame_period, multiple=4):
    assert wav.ndim == 1
    num_frames = len(wav)
    num_frames_padded = int(
        (np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (
                sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values=0)

    return wav_padded
"""
def wav_padding(wav, sr, frame_period, multiple = 4):

    assert wav.ndim == 1 
    num_frames = len(wav)
    num_frames_padded = int((np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values = 0)

    nlen = len(wav_padded) + 80
    a = 2**5
    wav_padded = np.pad(wav_padded, (0, (a-(nlen//80)%a)*80 - (nlen%80)), 'constant', constant_values = 0)
    
    return wav_padded





def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def sample_train_data(sp_list, n_frames=128, shuffle=False):
    """
    Input: [(D, T1), (D, T2), ... ]
    Output: [(D, 128), (D, 128), ... ]
    """
    total_num = len(sp_list)
    feat_idxs = np.arange(total_num)
    if shuffle:
        np.random.shuffle(feat_idxs)

    sp_mat_list = []
    for idx in feat_idxs:
        cur_data = sp_list[idx]
        cur_data_frames = cur_data.shape[-1]
        
        assert cur_data_frames >= n_frames, "Too short SP"
        start_idx = np.random.randint(cur_data_frames - n_frames + 1)
        end_idx = start_idx + n_frames
        sp_mat_list.append(cur_data[:, start_idx:end_idx])

    result = np.array(sp_mat_list)
    return result


def feat_loader(sp_list, spk_vec, batch_size, n_frames=128, shuffle=False):
    sampled_sp = sample_train_data(sp_list, n_frames=n_frames, shuffle=shuffle) #[(36, 128)]
    spk_vec = np.expand_dims(spk_vec, axis=0)

    total_num = len(sampled_sp)
    
    for start_idx in range(0, total_num, batch_size):
        end_idx = start_idx + batch_size
        x = sampled_sp[start_idx:end_idx]
        x = np.expand_dims(x, axis=1)
        x = torch.Tensor(x).float().cuda()

        batch_len = x.size()[0]
        y = np.repeat(spk_vec, batch_len, axis=0)
        y = torch.Tensor(y).float().cuda()

        yield x, y

def rnn_loader(sp_list, spk_vec, batch_size, shuffle=False):
    
    """
    Input: [(D, T1), (D, T2), ... ]
    Output: [(D, 128), (D, 128), ... ]
    
    1. shuffle sp_list
    2. calc min seq
    3. 
    """
    spk_vec = np.expand_dims(spk_vec, axis=0)
    
    total_num = len(sp_list)
    if shuffle:
        np.random.shuffle(sp_list)
      
    for start_idx in range(0, total_num, batch_size):
        end_idx = start_idx + batch_size
        cur_sp = sp_list[start_idx:end_idx]
        T_min = min([len(sp[0]) for sp in cur_sp])
        x = sample_train_data(cur_sp, n_frames=T_min) 
        # [(D, T_min), (D, T_min), ... ]
        # x = torch.cat(x, dim=0).permute(2,0,1)
        # (T_min, batch, D)
        x = torch.Tensor(x).float().cuda().permute(2,0,1)

        batch_len = x.size()[1]
        y = np.repeat(spk_vec, batch_len, axis=0)
        y = torch.Tensor(y).float().cuda()

        yield x, y

def pretrain_loader(sp_dict, spk_dict, batch_size, n_frames=128, shuffle=False):
    """
    Input: {spk_id: sp_list}
    """
    feat_list = []
    for spk_id, sp_list in sp_dict.items():
        sampled_sp = sample_train_data(sp_list, n_frames=n_frames, shuffle=shuffle)
        spk_idx = spk_dict[spk_id]

        for cur_sp in sampled_sp:
            feat_list.append((cur_sp, spk_idx))


    total_num = len(feat_list)
    if shuffle:
        np.random.shuffle(feat_list)
    
    for start_idx in range(0, total_num, batch_size):
        x=[]; y=[]
        for cur_idx in range(start_idx, min(start_idx+batch_size, total_num)):
            cur_x, cur_y = feat_list[cur_idx]
            x.append(cur_x)
            y.append(cur_y)
        
        x = np.expand_dims(x, axis=1)

        x = torch.Tensor(x).float().cuda()
        y = torch.Tensor(y).long().cuda()

        yield x, y