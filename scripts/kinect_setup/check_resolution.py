import freenect

def get_video_format():
    frame, _ = freenect.sync_get_video()
    print("Video frame resolution:", frame.shape)

def get_depth_format():
    frame, _ = freenect.sync_get_depth()
    print("Depth frame resolution:", frame.shape)

get_video_format()
get_depth_format()
