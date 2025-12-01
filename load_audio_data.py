# load_audio_data.py
import os

def collect_audiomnist_paths(root_dir):
    speakers = []
    for sp in sorted(os.listdir(root_dir)):
        sdir = os.path.join(root_dir, sp)
        if not os.path.isdir(sdir):
            continue
        files = []
        for f in sorted(os.listdir(sdir)):
            if f.lower().endswith(".wav"):
                files.append(os.path.join(sdir, f))
        if files:
            speakers.append({"speaker": sp, "files": files})
    return speakers
