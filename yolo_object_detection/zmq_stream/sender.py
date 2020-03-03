from imutils.video import VideoStream
import imagezmq
import argparse
import socket
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-s", "--target-ip", required=True, help="ip address of the reciever of the stream"
)
args = vars(ap.parse_args())
# initialize the ImageSender object with the target ip address
sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(args["server_ip"]))

# get the host name
rpiName = socket.gethostname()
vs = VideoStream(usePiCamera=True).start()
# allow camera to warm up
time.sleep(2.0)

while True:
    # read the frame from the camera and send it to the server
    frame = vs.read()
    sender.send_image(rpiName, frame)

