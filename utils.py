#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:47:43 2017

@author: leandro
"""

import librosa
import numpy as np
import os
from math import ceil
import time
import sys

import umap

from ProcessStatus import saveStatus


distanceMatrix = None

def findMusic(directory):
    supportedFileTypes = ["wav", "mp3"] #flac ?
    musicFiles = []

    for file in os.listdir(directory):
        if os.path.isdir(directory + file):
            musicFiles += findMusic(directory + file + "/")
        elif os.path.splitext(file)[1][1:].lower() in supportedFileTypes:
            musicFiles.append( directory + file )
        else:
            if not file.endswith(".asd"):
                pass
#                print("Skipped:", directory + file)

    return musicFiles

def getSTFT(data, win_length = 2048, hop_length = 512):
    D = librosa.stft(data, n_fft = win_length, hop_length = hop_length)
    D = np.abs(D)

    return D.reshape( D.shape[0] * D.shape[1] )    

def getZeroCrossingRate(data):
    zc = librosa.feature.zero_crossing_rate(data)
    return zc.reshape( zc.shape[0] * zc.shape[1] )

def getRMS(data, win_length = 2048, hop_length = 512):
    rms = librosa.feature.rms(data, frame_length=win_length, hop_length=hop_length)
    return rms.reshape( rms.shape[0] * rms.shape[1] )

def getMFCC(data, sample_rate = 22050, win_length = 2048 , hop_length = 512 ):
    mfcc = librosa.feature.mfcc(data, sr = sample_rate, n_fft = win_length, hop_length = hop_length)
    
    return mfcc.reshape( mfcc.shape[0] * mfcc.shape[1] )
    
def getSpectralCentroid(data, sample_rate = 22050, win_length = 2048, hop_length = 512):
    spectralCentroid = librosa.feature.spectral_centroid(data, sr=sample_rate, n_fft = win_length, hop_length = hop_length )
    return spectralCentroid.squeeze(0)

def getChroma(data, sample_rate = 22050, win_length = 2048, hop_length = 512):
    chroma = librosa.feature.chroma_stft(data, sr=sample_rate, n_fft=win_length , hop_length = hop_length )
    return chroma.reshape( chroma.shape[0] * chroma.shape[1] )


# one day,...
#def _threadGetAudioData( file, features, sizeAudioRaw, win_length, hop_length ):
#    pass

def getAudioData( audioFiles, features, audioMaxLength = 3, qtyFilesToProcess = None, win_length = 2048, hop_length = 512, sample_rate = 22050 ):
    count = 0
    countFail = 0
    COUNT_NOTICE = 200
#    COUNT_FAIL = 20

    maxProgress = 0.5

    listAudioData = []

    tic = time.clock()

    audioFilesDone = []

    sizeAudioRaw = ceil(sample_rate * audioMaxLength)

    if qtyFilesToProcess == None:
        qtyFilesToProcess = len(audioFiles)

    for i in range(0, qtyFilesToProcess):
        try:
            file = audioFiles[i]
            sys.stdout.write('.')
            sys.stdout.flush()

            tmpAudioData, tmpSampleRate = librosa.core.load(file, sr = sample_rate)

            tmpAudioData.resize(sizeAudioRaw)

            featuresData = np.array([])

            for f in features:
                if f == "stft":
                    featuresData = np.concatenate((featuresData,getSTFT(tmpAudioData, win_length, hop_length)))
                if f == "mfcc":
                    featuresData = np.concatenate((featuresData,getMFCC(tmpAudioData, tmpSampleRate, win_length, hop_length)))
                if f == "spectral-centroid":
                    featuresData = np.concatenate((featuresData,getSpectralCentroid(tmpAudioData, tmpSampleRate, win_length, hop_length)))
                if f == "chromagram":
                    featuresData = np.concatenate((featuresData,getChroma(tmpAudioData, tmpSampleRate, win_length, hop_length)))
                if f == "rms":
                    featuresData = np.concatenate((featuresData,getRMS(tmpAudioData, win_length, hop_length)))

            listAudioData.append( featuresData )
            audioFilesDone.append(file)

            count += 1
            saveStatus( "progress", (i/qtyFilesToProcess) * maxProgress )

            if count % COUNT_NOTICE == 0:
                sys.stdout.write('\n\r')
                print("[", count, "/", qtyFilesToProcess, "]")
                sys.stdout.flush()

        except Exception as ex:
            countFail += 1
            sys.stdout.write('\n\r')
            print(file, "[FAIL]", ex)
            sys.stdout.flush()

            saveStatus("failFiles", [file])

            # if countFail >= COUNT_FAIL:
            #     break

    matrixAudioData = np.array(listAudioData, dtype=np.float32)
    #matrixAudioData = matrixAudioData.squeeze(1)
    audioFiles.clear()
    audioFiles += audioFilesDone

    print("")
    print("Matriz final:", matrixAudioData.shape)

    toc = time.clock()
    print("time:", toc - tic)
    return matrixAudioData

def saveAudioData( matrixAudioData, filename ):
    np.save(filename, matrixAudioData)

def loadAudioData( filename ):
    return np.load(filename)

def doPCA( matrixAudioData, n_components = 0.98 ):
    from sklearn.decomposition import PCA

    tic = time.clock()

    pca = PCA(n_components=n_components, svd_solver = "full")
    pca.fit(matrixAudioData)
    print("Variance explained:", pca.explained_variance_ratio_.sum())
    matrixAudioDataTransformed = pca.transform(matrixAudioData)

    toc = time.clock()

    print("shape transformed:", matrixAudioDataTransformed.shape)
    print("time:", toc - tic)
    return matrixAudioDataTransformed

#https://umap-learn.readthedocs.io/en/latest/parameters.html
'''
Low values on n_neighbors will force UMAP to concentrate on very local structure
(potentially to the detriment of the big picture),
while large values will push UMAP to look at larger neighborhoods of each point
when estimating the manifold structure of the data,
losing fine detail structure for the sake of getting the broader of the data.
'''
def doUMAP( matrixAudioData, n_neighbors=15, min_dist=0.1, metric='cosine' ): #metric = cosine ??
    tic = time.clock()

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    embedding = reducer.fit_transform( matrixAudioData )

    toc = time.clock()

    print("time:",toc-tic)

    return embedding



def doTSNE( matrixAudioDataTransformed, n_components = 2, perplexity = 30, learning_rate = 200.0, n_iter = 1000, metric = 'cosine' ):
    from sklearn.manifold import TSNE
    from sklearn.metrics import pairwise_distances
    from scipy.spatial import distance as dist

    tic = time.clock()

    tsne = TSNE(n_components=n_components, metric=metric, perplexity = perplexity, learning_rate = learning_rate, n_iter = n_iter)
    positions = tsne.fit(matrixAudioDataTransformed).embedding_

    toc = time.clock()

    print("time:", toc - tic)

    return positions

def doDBScan( tsneResult ):
    from sklearn.cluster import DBSCAN

    db = DBSCAN( eps=2, min_samples=3, metric="euclidean" )
    dbFit = db.fit( tsneResult )

    return dbFit.labels_
