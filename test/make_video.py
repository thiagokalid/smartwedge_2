import matplotlib.pyplot as plt
import os
import subprocess

os.chdir("/home/tekalid/repos/smartwedge_2/figures/lens_frames_surf")
subprocess.call([
    'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
    'video_name.mp4'
])

