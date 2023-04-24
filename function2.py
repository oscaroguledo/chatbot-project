import vlc
import re
from playsound import playsound
import os, subprocess, glob

def get_media(media):
    folder = r"C:\Users\HP"
    files = []
    files.extend(glob.glob(folder+'/**', recursive=True))
    #rember to generrate database
    for i in files:

        if i.endswith(".mp3") or i.endswith(".wav") or i.endswith(".aiff") or i.endswith(".flac") or i.endswith(".ogg"):
            if media in i :
                get_audio(media,i)##play audio
        elif i.endswith(".mp4") or i.endswith(".mkv") or i.endswith(".avi"):
            if media in i:
                print(i, "video===========")
                get_video(media, i)##play video
def get_audio(audio, dir):
    os.chdir("C:\Program Files\VideoLAN\VLC")
    meta_data = subprocess.check_output(f'vlc {dir}')  ##playing the media
    #os.chdir(r"C:\Program Files\WindowsApps")
    #meta_data = subprocess.check_output(
     #   f"Microsoft.ZuneMusic_3.6.25021.0_x64__8wekyb3d8bbwe!Microsoft.ZuneMusic {aud}")

def get_video(video, dir):
    print("yes", video, dir)
    os.chdir("C:\Program Files\VideoLAN\VLC")
    meta_data = subprocess.check_output(f'vlc {dir}')##playing the media
    print("yes", video, dir)





def search(name):
    text = " MY apologies, but this functionality has not yet been added"
    return
