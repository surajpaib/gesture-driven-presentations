import os
from moviepy.editor import VideoFileClip, vfx


def change_videos(effect, video_directory_str):
    video_directory = os.fsencode(video_directory_str)

    for file in os.listdir(video_directory):
        filename = os.fsdecode(file)
        # os.rename(video_directory_str + "/" + filename, video_directory_str + "/" + filename.replace(" ", ""))
        if filename.endswith(".mp4"):
            video = VideoFileClip(video_directory_str + "/" + filename)
            if effect == "mirror_x":
                out = video.fx(vfx.mirror_x)
                new_filename = filename.split(".")[0] + '_mirror_x.mp4'
            else:
                out = video.fx(vfx.time_mirror)
                new_filename = filename.split(".")[0] + '_time_mirror.mp4'

            out.write_videofile("../videos_output/" + new_filename)
            print("Video save done", new_filename)
            print()
            video.close()


change_videos("mirror_x", '../videos')
