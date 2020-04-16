#!/usr/bin/env python
# coding= UTF-8
#
# Author: Suhas
# Date  : 2020-02-18
#

import code
import glob
import os
import numpy as np
import soundfile as sf
import sounddevice as sd
import queue
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import warnings
warnings.filterwarnings("ignore")

def extract_feature(file_name=None):
    if file_name: 
        print('Extracting', file_name)
        X, sample_rate = sf.read(file_name, dtype='float32')
    else:  
        device_info = sd.query_devices(None, 'input')
        sample_rate = int(device_info['default_samplerate'])
        q = queue.Queue()
        def callback(i,f,t,s): q.put(i.copy())
        data = []
        with sd.InputStream(samplerate=sample_rate, callback=callback):
            while True: 
                if len(data) < 100000: data.extend(q.get())
                else: break
        X = np.array(data)

    if X.ndim > 1: X = X[:,0]
    X = X.T
    mfbe = logfbank(X,sample_rate)
    return mfbe

    

def parse_audio_files(parent_dir,file_ext='*.wav'):
    sub_dirs = os.listdir(parent_dir)
    sub_dirs.sort()
    features, labels = np.empty((0,26)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        if os.path.isdir(os.path.join(parent_dir, sub_dir)):
            for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                try: mfbe= extract_feature(fn)
                except Exception as e:
                    print("[Error] extract feature error in %s. %s" % (fn,e))
                    continue
                ext_features = np.hstack([mfbe])
                features = np.vstack([features,ext_features])
                # labels = np.append(labels, fn.split('/')[1])
                labels = np.append(labels, label)
            print("extract %s features done" % (sub_dir))
    return np.array(features), np.array(labels, dtype = np.int)

def parse_predict_files(parent_dir,file_ext='*.wav'):
    features = np.empty((0,26))
    filenames = []
    for fn in glob.glob(os.path.join(parent_dir, file_ext)):
        mfbe = extract_feature(fn)
        ext_features = np.hstack([mfbe])
        features = np.vstack([features,ext_features])
        filenames.append(fn)
        print("extract %s features done" % fn)
    return np.array(features), np.array(filenames)

def main():
    # Get features and labels
    features, labels = parse_audio_files('../data_backup/')
    np.save('mfbe.npy', features)
    #np.save('label_mfbe.npy', labels)

if __name__ == '__main__': main()
