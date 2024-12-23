import os
import time
import shutil
import scipy
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf

import matplotlib.pyplot as plt

def mmss_to_seconds(time_str):
    try:
        # Parse the time string into a time structure
        time_struct = time.strptime(time_str, "%M:%S")
        
        # Convert minutes and seconds into total seconds
        total_seconds = time_struct.tm_min * 60 + time_struct.tm_sec
        return total_seconds
    except ValueError:
        print("Invalid time format. Please use MM:SS.")
        return None

def fade_in_out(audio, sample_rate, duration):
    # Calculate the number of samples for the fade duration
    fade_samples = int(sample_rate * duration)
    
    # Create the Hamming window
    hamming_window = np.hamming(2 * fade_samples)
    
    # Split the window into fade-in and fade-out parts
    fade_in_window = hamming_window[:fade_samples]
    fade_out_window = hamming_window[fade_samples:]
    
    # Create a copy of the audio to apply the fade effect
    faded_audio = np.copy(audio)
    
    # Apply fade-in
    if len(audio.shape) == 1:  # Mono
        faded_audio[:fade_samples] *= fade_in_window
    else:  # Stereo
        faded_audio[:fade_samples, :] *= fade_in_window[:, np.newaxis]
    
    # Apply fade-out
    if len(audio.shape) == 1:  # Mono
        faded_audio[-fade_samples:] *= fade_out_window
    else:  # Stereo
        faded_audio[-fade_samples:, :] *= fade_out_window[:, np.newaxis]
    
    return faded_audio

def download_and_crop(url, start, end):
    yt_dlp = "./yt-dlp_macos "
    command = f'{yt_dlp} -q --audio-quality 0 --audio-format wav "{url}" -o tmp.wav'
    os.system(command)

    fs = 22050
    wav, _fs = librosa.load("./tmp.wav", mono=True)
    os.remove("./tmp.wav")
    if _fs != fs:
        import pdb; pdb.set_trace()
        assert np.abs(1-_fs/fs) < 0.05
        wav = librosa.resample(wav, _fs, fs)

    if start is None:
        start = 0
    else:
        start = int(mmss_to_seconds(start)*fs)
    if end is None:
        end = -1
    else:
        end = int(mmss_to_seconds(end) * fs)
    wav = wav[start:end, None]

    max_amp = np.percentile(np.abs(wav), 100)
    wav = np.clip(wav, -max_amp, max_amp)
    wav /= max_amp
    wav *= 0.9

    fade = 3
    wav = fade_in_out(wav, fs, fade)
    return wav, fs

if __name__ == '__main__':
    csv = "./reception_entry.csv"
    df = pd.read_csv(csv)
    # df = df.sample(frac=1.0)
    # df.to_csv('./reception_songs_shuffled.csv')

    pbar = tqdm(range(df.shape[0]))
    songs = []
    print('downloading')
    for idx in pbar:
        pbar.set_description(f'{idx+1:03d}/{df.shape[0]}')
        dr = df.iloc[idx]
        print(dr['url'])
        song, fs = download_and_crop(dr['url'], dr['start'], dr['end'])
        songs.append(song)
    print('saving')
    full = np.concatenate(songs, axis=0)
    sf.write(f"{csv.replace('csv', 'wav')}", full, samplerate=fs, subtype='PCM_24')