#!/usr/bin/env python
import argparse
import os
import utils
import numpy as np
import sys
import warnings

import ProcessStatus
from ProcessStatus import saveStatus

#these modules are imported here so that the one-folder bundle created by PyInstller can be also used by onset-detection
import librosa
import scipy.io.wavfile
import json

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("audioFilesPath", help="Path to audio files to parse")
parser.add_argument("--n-files", default = -1, help = "Quantity of files to process. Default: all files found")
parser.add_argument("--audio-max-length", type = float, default = 1.2, help = "Seconds. Longer audios will be chopped, shorter ones will be zero-padded. Only for processing, original audios will not be affected. Default: 1.2")
parser.add_argument("--sample-rate", type = int, default = 22050, help = "Number of samples taken per second. Default: 22050")

parser.add_argument("--tsne-perplexity", type = float, default = 30, help = "Sets the number of effective nearest neighbors. Larger dataset requires a larger perplexity. Default: 30")
parser.add_argument("--tsne-learning-rate", type = float, default = 200.0, help = "tSNE's learning rate. Usually in the range 10.0, 1000.0. Default: 200")
parser.add_argument("--tsne-n-iter", type = int, default = 1000, help = "tSNE's iteration count. Maximum number of iterations for the optimization. Should be at least 250. Default: 1000")

parser.add_argument("--save-pca-results", default = False, action = "store_true")
parser.add_argument("--force-full-process", default = False, action = "store_true")

parser.add_argument("--win-length", type = int, default = 2048, help = "Smaller values improve the temporal resolution at the expense of frequency resolution Default: 2048")
parser.add_argument("--hop-length", type = int, default = 512, help = "Number of audio samples between adjacent columns. Default: 512")

parser.add_argument("--stft", default = False, action = "store_true", help = "Short-Time Fourier Transform. (default)")
parser.add_argument("--mfcc", default = False, action = "store_true", help = "Mel-frequency cepstral coefficients.")
parser.add_argument("--spectral-centroid", default = False, action = "store_true", help = "Compute the spectral centroid.")
parser.add_argument("--chromagram", default = False, action = "store_true", help = "Constant-Q chromagram.")
parser.add_argument("--rms", default = False, action = "store_true", help = "Compute root-mean-square (RMS) value for each frame, either from the audio samples y or from a spectrogram S.")

parser.add_argument("--viz-method", default = "tsne", help = "tsne, umap or pca. Default: tsne.")

parser.add_argument("--umap-n-neighbors",type = int , default = 15, help = "The size of the local neighborhood UMAP will look at. Default: 15")
parser.add_argument("--umap-min-dist", type = float, default = 0.1, help = "The minimum distance apart that points are allowed to be. Default: 0.1")
parser.add_argument("--metric", default = "cosine", help = "Controls how distance is computed in the ambient space of the input data. Default: cosine.")

parser.add_argument("--process-status-path", default = "./processInfo.json", help = "Full path for processInfo file")

args = parser.parse_args()

features = []
if args.stft:
    features.append('stft')
if args.mfcc:
    features.append('mfcc')
if args.spectral_centroid:
    features.append('spectral-centroid')
if args.chromagram:
    features.append('chromagram')
if args.rms:
    features.append('rms')
if len(features) == 0:
    features.append('stft') #default

featuresNames = ""
for f in features:
    featuresNames = featuresNames + "_" + str(f)
featuresNames = featuresNames[1:]

ProcessStatus.setStatusFilename(args.process_status_path) 

basename = args.audioFilesPath if args.audioFilesPath[-1] != "/" else args.audioFilesPath[:-1]
audioFilesPathDirectoryName = os.path.basename(basename)
PCAfile = "{}.{}.{}.{}_{}.{}secs".format( audioFilesPathDirectoryName, featuresNames, args.sample_rate, args.win_length, args.hop_length, args.audio_max_length )
PCAfullPath = os.path.join(args.audioFilesPath, "../", PCAfile + ".npy")

if args.force_full_process:
    PCAfullPath = ''

if os.path.isdir(args.audioFilesPath):
    if args.audioFilesPath[-1] != "/":
        args.audioFilesPath += "/"
#    print("reading directory", args.audioFilesPath, "...")
else:
    print("audioFilesPath not found :(")
    saveStatus("currentProcess", "Path not found")
    saveStatus("progress", -1)
    sys.exit()

audioFilesPathDirectoryName = os.path.basename(args.audioFilesPath[0:-1])

audioFiles = utils.findMusic( args.audioFilesPath )

saveStatus("qtyFiles",len(audioFiles))
saveStatus("progress", 0)
print(len(audioFiles), "files found")

if len(audioFiles) == 0:
    saveStatus("currentProcess", "No wav or mp3 files found")
    saveStatus("progress", -1)
    sys.exit()

args.n_files = int(args.n_files)
if args.n_files == -1:
    args.n_files = len(audioFiles)

if not os.path.exists(PCAfullPath):
    print("now processing")
    saveStatus("currentProcess","Extracting features")
    audioData = utils.getAudioData( audioFiles, features = features, qtyFilesToProcess = args.n_files, audioMaxLength = args.audio_max_length, win_length = args.win_length, hop_length = args.hop_length, sample_rate = args.sample_rate)

    saveStatus("qtyFiles",len(audioFiles))
    if len(audioFiles) < 5:
        print("Less than 5 audiofiles were processed, aborting.")
        saveStatus("currentProcess", "Less than 5 audiofiles were processed")
        saveStatus("progress", -1)
        sys.exit()

    print("DONE!")

    print("")
    print("Now doing PCA")
    saveStatus("progress", 0.5)
    saveStatus("currentProcess","Running principal component analysis")
    audioDataTransformed = utils.doPCA(audioData)

    if args.save_pca_results:
        np.save(os.path.join(args.audioFilesPath, "../", PCAfile), audioDataTransformed)
else :
    audioDataTransformed = np.load(PCAfullPath)

    if audioDataTransformed.shape[0] != len(audioFiles):
        print("Audio folder has changed. Check \"Force full process\" for re processing all files.")
        saveStatus("currentProcess", "Audio folder has changed. Check \"Force full process\" for re processing all files.")
        saveStatus("progress", -1)
        sys.exit()

saveStatus("progress", 0.7)

if args.viz_method == "tsne":
    saveStatus("currentProcess","Running t-SNE")
    print("")

    print("Now doing t-SNE")
    embedding = utils.doTSNE( audioDataTransformed , 2, perplexity = args.tsne_perplexity, learning_rate = args.tsne_learning_rate, n_iter = args.tsne_n_iter, metric = args.metric )
elif args.viz_method == "umap":
    saveStatus("currentProcess","Running UMAP")
    print("")

    print("Now doing UMAP")
    embedding = utils.doUMAP( audioDataTransformed, n_neighbors = args.umap_n_neighbors, min_dist = args.umap_min_dist, metric = args.metric )
elif args.viz_method == "pca":
    saveStatus("currentProcess","Running principal component analysis")
    print("")

    embedding = utils.doPCA( audioDataTransformed, 2 )

print("DONE!")

print("")
print("Now saving session file")

import json

jsonSession = {
    "audioFilesPath" : "%s/" % audioFilesPathDirectoryName,
    "tsv" : "",
    "sounds" : {
        "useOriginalClusters" : False,
    }
}

audioFilesForExport = list( map( lambda x : x[len(args.audioFilesPath):], audioFiles ) )

# if audios are in different directories, use them as initial clusters
clusters = np.repeat(0, embedding.shape[0])
folderNames = []
for f in audioFilesForExport:
    if os.sep in f:
        folderName = f[0:f.rfind(os.sep)]
        if not folderName in folderNames:
            folderNames.append(folderName)
if len(folderNames) > 1:
    CLUSTER_NOISE = -2
    clusters = []
    for f in audioFilesForExport:
        if os.sep in f:
            folderName = f[0:f.rfind(os.sep)]
            if folderName in folderNames: #this is redundant but safe
                clusters.append( folderNames.index( folderName ) )
            else:
                clusters.append( CLUSTER_NOISE )
        else:
            clusters.append( CLUSTER_NOISE )
    
    clusters = np.array(clusters)
    
    jsonSession["sounds"]["useOriginalClusters"] = True
    jsonSession["sounds"]["folderNames"] = folderNames
            
# build output
# x , y , z , clusterNumber, filename
output = np.c_[ embedding, np.repeat(0, embedding.shape[0] ) , clusters, audioFilesForExport ]

tsv = ""
for row in output:
    for field in row:
        tsv += field + ","
    tsv = tsv[0:-1]
    tsv += "|"

jsonSession["tsv"] = tsv

filenameOk = False
currentFilenameNumber = 1
while ( not filenameOk ):
    newFilename = '%s.%s.%d.json' % ( audioFilesPathDirectoryName, featuresNames, currentFilenameNumber )
    newFilepath = os.path.join(args.audioFilesPath, "../", newFilename )
    newFilepath = os.path.realpath(newFilepath)

    if ( os.path.exists(newFilepath) ):
        currentFilenameNumber += 1
    else:
        filenameOk = True

with open( newFilepath, 'w') as fp:
    json.dump(jsonSession, fp)

saveStatus("currentProcess", "All done")
saveStatus("generatedJSONPath", newFilepath)
saveStatus("progress", 1)

print("All done, have a great day")
