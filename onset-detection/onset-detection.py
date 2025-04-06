#!/usr/bin/env python

import librosa
import numpy as np
import scipy.io.wavfile
import os
import json
import argparse

print("[DEBUG] Script starting...")

parser = argparse.ArgumentParser()
parser.add_argument("--process-status-path", default = "./onsets-interprocess.json", help = "Full path for onsets interprocess file")

args = parser.parse_args()
print(f"[DEBUG] Using process status file: {args.process_status_path}")

#__location__ = os.path.realpath(
#    os.path.join(os.getcwd(), os.path.dirname(__file__)))
#
#json_name = f"{__location__}/../data-analysis/onset-detection/onsets-interprocess.json"
#

"""
Usage examples:

Find onsets in a file:

{"state": "onsetCut",
 "audioFilePath": "/path/to/file.wav",
 "params": {"window_max": 0.03,
            "window_avg": 0.1,
            "delta": 0.07,
            "backtrack": true
            }
}

Slice all files in a directory into onsets:

{"state": "onsetCut",
 "audioFilePath": "/path/to/dir",
 "folder": "/path/to/save/dir",
 "params": {"window_max": 0.2,
            "window_avg": 0.5,
            "delta": 0.1,
            "backtrack": false
            },
 "save": {"fade": 1000,
          "normalize": true
         }
}

Slice in fixed time an audio file:

{"state": "timeCut",
 "audioFilePath": "/path/to/file.mp3",
 "folder": "/path/to/save/dir",
 "time": 500,
 "save": true
}
"""

def findOnsetsAndCut(window_max, window_avg, delta, backtrack):
    global onset_frames, onset_times, times
    print(f"[DEBUG] Starting findOnsetsAndCut with params: window_max={window_max}, window_avg={window_avg}, delta={delta}, backtrack={backtrack}")
    
    print("[DEBUG] Computing onset strength...")
    o_env = librosa.onset.onset_strength(y=segmento, sr=sr)
    print(f"[DEBUG] Onset envelope shape: {o_env.shape}")
    
    times = librosa.times_like(o_env, sr=sr)
    print(f"[DEBUG] Times array shape: {times.shape}")

    window_max_final = window_max * sr
    window_avg_final = window_avg * sr
    print(f"[DEBUG] Computed window sizes: max={window_max_final}, avg={window_avg_final}")

    print("[DEBUG] Detecting onsets...")
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr,
                                          backtrack=backtrack,
                                          delta=delta,
                                          pre_avg=window_avg_final, post_avg=window_avg_final + 1,
                                          pre_max=window_max_final, post_max=window_max_final + 1)

    onset_times = librosa.frames_to_time(frames=onset_frames, sr=sr)
    print(f"[DEBUG] Found {len(onset_frames)} onsets at times: {onset_times}")

def loadFile(filename):
    global x, sr, segmento
    print(f"[DEBUG] Attempting to load file: {filename}")

    target = filename
    if not os.path.isfile(target):
        print(f"[DEBUG] ERROR: File not found: {target}")
        return
    else:
        print(f"[DEBUG] File exists, loading with librosa...")
        x, sr = librosa.load(target, sr=None)
        print(f"[DEBUG] File loaded. Sample rate: {sr}, Length: {len(x)} samples")

    segmento = x
    print(f"[DEBUG] Segmento set with shape: {segmento.shape}")

def fadeNormalizeSave(folder, fadeSamples=1000, normalize=True):
    print(f"[DEBUG] Starting fadeNormalizeSave: folder={folder}, fadeSamples={fadeSamples}, normalize={normalize}")
    
    segmentoFade = np.copy(segmento)
    print(f"[DEBUG] Created copy of segmento with shape: {segmentoFade.shape}")

    print("[DEBUG] Applying fades...")
    for f in range(1, len(onset_frames)):
        sampleFinish = librosa.frames_to_samples(frames=onset_frames[f])
        print(f"[DEBUG] Processing fade at frame {f}, sample {sampleFinish}")

        j = fadeSamples
        for i in np.arange(sampleFinish-fadeSamples, sampleFinish+1):
            segmentoFade[i] *= j/fadeSamples
            j-=1

    print(f"[DEBUG] Creating output folder if needed: {folder}")
    if not os.path.exists(folder):
        os.mkdir(folder)

    print("[DEBUG] Saving sliced segments...")
    for i in range(len(onset_frames)-1):
        timeF = librosa.frames_to_samples(frames=onset_frames[i])
        timeT = librosa.frames_to_samples(frames=onset_frames[i+1])
        filename, ext = os.path.splitext(audioFile)
        archivoAGuardar = f"{folder}/{filename}_{i}.wav"
        print(f"[DEBUG] Saving segment {i}: {archivoAGuardar} (samples {timeF} to {timeT})")
        
        segmentoAGuardar = segmentoFade[timeF:timeT]
        if normalize:
            print(f"[DEBUG] Normalizing segment {i}")
            segmentoAGuardar = np.array((segmentoAGuardar / np.max(np.abs(segmentoAGuardar))))

        scipy.io.wavfile.write(archivoAGuardar, sr, segmentoAGuardar)
        print(f"[DEBUG] Saved segment {i}")

def fadeNormalizeSaveMarkers(markers, folder, fadeSamples = 1000, normalize = True):
  
  segmentoFade = np.copy(segmento)
  
  markers.append(segmento.size - 1)

  json_file = {
      "state": "slicingPreprocessing",
      "sliced": 0,
      "countMarkers" : len(markers)
  }

  for f in range(1, len(markers)):
      sampleFinish = markers[f]

      j = fadeSamples
      for i in np.arange(sampleFinish-fadeSamples, sampleFinish+1):
          segmentoFade[i] *= j/fadeSamples
          j-=1
      
      json_file["sliced"] += 1
      with open(json_name, 'w') as file:
        json.dump(json_file, file)

  if not os.path.exists(folder):
      os.mkdir(folder)

  json_file = {
      "state": "slicingProcessing",
      "sliced": 0,
      "countMarkers" : len(markers)
  }

  for i in range(len(markers) - 1):
      timeF = markers[i]
      timeT = markers[i+1] 

      filename, ext = os.path.splitext(audioFile)
      archivoAGuardar = f"{folder}/{filename}_{i}.wav"
      segmentoAGuardar = segmentoFade[timeF:timeT]

      if normalize:
          segmentoAGuardar = np.array((segmentoAGuardar / np.max(np.abs(segmentoAGuardar))))
      
      scipy.io.wavfile.write( archivoAGuardar ,sr, segmentoAGuardar)

      json_file["sliced"] += 1
      with open(json_name, 'w') as file:
        json.dump(json_file, file)

def timeSlice(filePath, time):
    global y, sr, cutSeconds
    y, sr = librosa.load(filePath, sr = 44100)

    duration = librosa.get_duration(y=y, sr=sr)*1000
    timeLine = np.arange(0, duration, 1).tolist()
    timeLine.append(duration)

    cutTimes = timeLine[::time]
    cutSeconds = []
    for e in cutTimes:
        e = e/1000
        cutSeconds.append(e)

def saveTime(folder):
    filePath = os.path.dirname(audioFilePath)
    if not os.path.exists(folder):
        os.mkdir(folder)

    for i in range(len(cutSeconds)-1):
        timeF = librosa.time_to_samples( cutSeconds[i] )
        timeT = librosa.time_to_samples( cutSeconds[i+1] )
        filename, ext = os.path.splitext(audioFile)
        archivoAGuardar = f"{folder}{filename}_{i}.wav"
        scipy.io.wavfile.write( archivoAGuardar ,sr, y[timeF:timeT])

def segment(audioToProcess):
    global state
    print(f"[DEBUG] Starting segment processing for: {audioToProcess}")

    if state == "onsetCut":
        print("[DEBUG] Processing onsetCut")
        loadFile(audioToProcess)
        if segmento is None or segmento.size == 0:
            print("[DEBUG] ERROR: Failed to load audio or empty segmento")
            return
            
        print("[DEBUG] Calling findOnsetsAndCut...")
        findOnsetsAndCut(window_max, window_avg, delta, backtrack)
        
        if save:
            print("[DEBUG] Save flag is True, calling fadeNormalizeSave...")
            fadeNormalizeSave(folder, fade, normalize)
            
    elif state == "timeCut":
        print("[DEBUG] Processing timeCut")
        timeSlice(audioToProcess, fixedTime)
        if save:
            saveTime(folder)

json_file = {
    "state": "waiting..."
}

json_name = args.process_status_path

if not os.path.exists(json_name):
    with open( json_name, 'w') as fp:
        json.dump(json_file, fp)
            
save = False
supportedFileTypes = ["wav", "mp3"]

print("[DEBUG] Starting main loop...")
while True:
    try:
        print(f"[DEBUG] Reading JSON file: {json_name}")
        with open(json_name) as jsonFile:
            data = json.load(jsonFile)
            state = data["state"]
            print(f"[DEBUG] Current state: {state}")
            
            if state == "onsetCut" or state == "timeCut":
                audioFilePath = data["audioFilePath"]
                print(f"[DEBUG] Processing audio path: {audioFilePath}")
                
                if "params" in data:
                    window_max = data["params"]["window_max"]
                    window_avg = data["params"]["window_avg"]
                    delta = data["params"]["delta"]
                    backtrack = data["params"]["backtrack"]
                    print(f"[DEBUG] Loaded params: window_max={window_max}, window_avg={window_avg}, delta={delta}, backtrack={backtrack}")
                
                if "time" in data:
                    fixedTime = data["time"]
                    print(f"[DEBUG] Fixed time set to: {fixedTime}")

                if "save" in data:
                    save = True
                    folder = data["folder"]
                    print(f"[DEBUG] Save enabled. Output folder: {folder}")
                    if state == "onsetCut":
                        fade = data["save"]["fade"]
                        normalize = data["save"]["normalize"]
                        print(f"[DEBUG] Save params: fade={fade}, normalize={normalize}")

                resultado = []
                if os.path.isdir(audioFilePath):
                    print(f"[DEBUG] Processing directory: {audioFilePath}")
                    if audioFilePath[-1] != "/":
                        audioFilePath += "/"
                    for audioFile in os.listdir(audioFilePath):
                        if not os.path.isdir(audioFile):
                            if os.path.splitext(audioFile)[1][1:].lower() in supportedFileTypes:
                                finalPath = f"{audioFilePath}{audioFile}"
                                print(f"[DEBUG] Processing file: {finalPath}")
                                segment(finalPath)
                else:
                    print(f"[DEBUG] Processing single file: {audioFilePath}")
                    audioFile = os.path.basename(audioFilePath)
                    segment(audioFilePath)

                if segmento is None or segmento.size == 0:
                    print("[DEBUG] Error: segmento is empty or None")
                    state = "errorLoadingFile"
                else:
                    if save:
                        state = "doneAndSave"
                    else:
                        state = "done"
                
                print(f"[DEBUG] Final state: {state}")
                json_file = {
                    "state": state,
                    "results": resultado
                }
                
                with open(json_name, 'w') as file:
                    json.dump(json_file, file)
                    print("[DEBUG] Updated JSON file with results")
                    
            elif state == "saveSlices":
                audioFilePath = data["audioFilePath"]
                folder = data["folder"]
                fade = (data["save"]["fade"])
                normalize = data["save"]["normalize"]
                markers = data["results"]
                fadeNormalizeSaveMarkers(markers, folder, fade, normalize)
                
                json_file = {
                    "state": "doneAndSave",
                }
                with open(json_name, 'w') as file:
                    json.dump(json_file, file)
                
            elif state == "loading":
                json_file = {
                    "state": "waiting...",
                }
                with open(json_name, 'w') as file:
                    json.dump(json_file, file)
                    
            elif state == "errorLoadingFile":
                json_file = {
                    "state": "errorLoadingFile",
                }
                with open(json_name, 'w') as file:
                    json.dump(json_file, file)
                
            elif state == "end":
                print("[DEBUG] Received end state, exiting...")
                break

    except Exception as e:
        print(f"[DEBUG] Error in main loop: {str(e)}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")

print("[DEBUG] Script finished")
