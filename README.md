# Split_Y

An audio file splitter that uses onset detection to automatically split audio files at detected onsets. Perfect for splitting long recordings into individual segments.

## Features

- Onset-based splitting: Automatically detects and splits at musical events
- Time-based splitting: Split audio files into fixed-length segments
- Cross-fading between segments
- Normalization of output segments
- Supports WAV and MP3 files

## Usage

The script uses a JSON configuration file to control the splitting process. Here are some example configurations:

### Onset-based Splitting

```json
{
    "state": "onsetCut",
    "audioFilePath": "/path/to/file.wav",
    "folder": "/path/to/output",
    "params": {
        "window_max": 0.2,
        "window_avg": 0.3,
        "delta": 0.1,
        "backtrack": true
    },
    "save": {
        "fade": 1000,
        "normalize": true
    }
}
```

### Time-based Splitting

```json
{
    "state": "timeCut",
    "audioFilePath": "/path/to/file.mp3",
    "folder": "/path/to/save/dir",
    "time": 500,
    "save": true
}
```

## Parameters

- `window_max`: Window size for maximum detection (in seconds)
- `window_avg`: Window size for average detection (in seconds)
- `delta`: Threshold for onset detection
- `backtrack`: Whether to backtrack to find the exact onset point
- `fade`: Number of samples to use for crossfading
- `normalize`: Whether to normalize the output segments

## Requirements

- Python 3.x
- librosa
- numpy
- scipy

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install librosa numpy scipy
```

## Running

```bash
python onset-detection/onset-detection.py --process-status-path path/to/config.json
```

How to compile:

1. Install anaconda
2. conda create -n python36 python=3.6
3. conda activate python36
4. pip install -r requirements.txt
5. test that enviroment can run "python doProcess.py" correctly (it should)

5w. On windows 10 run this additional steps:
5w.1 pip install pypiwin32 (solves "win32com module not found")
5w.2 pip install --upgrade "setuptools<45.0.0" (solves No module named 'pkg_resources.py2_warn')

6. sh doPyInstaller.sh

7. cd onset-detection
8. sh doPyInstaller.sh
