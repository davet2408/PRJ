"""
This module provides ways to access video streams using threading through the 
VideoCapture class, a proxy for OpenCVs VideoCapture class. 

If run as main then FPS and other information can be preinted.

Author: David Temple
Date: 27/03/2020
"""
import queue
import threading
import numpy as np
import cv2
from imutils.video import FPS
import time


class VideoCapture:
    """Class that offers a bufferless frame reader using threading.
    
    Class based on code from:
        https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-
        from-capture-device-camera-in-opencv-python
        (accessed 22/03/20)
    """

    def __init__(self, src="gstreamer"):
        self.cap = self._video_source(src)
        # Buffer to hold most recent frame
        self.buffer = queue.Queue()
        # Stopping condition
        self.done = False
        thread = threading.Thread(target=self._frame_reader)
        # Causes thread to die when calling thread does
        thread.daemon = True
        thread.start()

    def _frame_reader(self):
        """Only keep the most recent frame, read constantly."""
        while True:
            # Check if still going
            if self.done:
                self.cap.release()
                return

            (ret, frame) = self.cap.read()
            if not self.buffer.empty():
                # Try to discard the old frame and return val.
                try:
                    self.buffer.get(False)
                except queue.Empty:
                    pass
            self.buffer.put((ret, frame))

    def _gstreamer(self):
        """Initialize GStreamer capture"""
        port = 7000
        pipeline = (
            f'udpsrc port={port} caps = "application/x-rtp, '
            "media=(string)video, clock-rate=(int)90000, "
            'encoding-name=(string)H264, payload=(int)96" '
            "! rtph264depay ! decodebin ! videoconvert "
            "! appsink sync=false drop=true"
        )
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    def _video_source(self, src):
        """Create a video capture for either GStreamer or other 
        provided source"""

        if src == "gstreamer":
            return self._gstreamer()
        else:
            return cv2.VideoCapture(src)

    def read(self):
        """Get the most recent frame"""
        return self.buffer.get()

    def stop(self):
        """Tell the video capture to stop"""
        self.done = True


def receive():
    """Test the normal stream and get FPS"""
    port = 7000
    pipeline = (
        f'udpsrc port={port} caps = "application/x-rtp, '
        "media=(string)video, clock-rate=(int)90000, "
        'encoding-name=(string)H264, payload=(int)96" '
        "! rtph264depay ! decodebin ! videoconvert "
        "! appsink sync=false drop=true"
    )

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    while True:
        fps = FPS().start()
        ret, frame = cap.read()
        fps.update()

        cv2.imshow("receive", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    fps.stop()
    cap.release()
    print(fps.fps())


def receive_buffer(src=None):
    """Test the buffered stream and get FPS"""

    if src is not None:
        cap = VideoCapture(src)
    else:
        cap = VideoCapture()

    while True:
        fps = FPS().start()
        ret, frame = cap.read()
        fps.update()

        cv2.imshow("receive", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.stop()
    fps.stop()
    print(fps.fps())


if __name__ == "__main__":

    # receive() # Test stream
    # receive_buffer(0) # Test webcam
    receive_buffer()  # Test buffer
