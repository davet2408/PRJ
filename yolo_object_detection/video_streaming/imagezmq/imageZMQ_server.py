import cv2
import imagezmq
import time
from imutils.video import FPS


image_hub = imagezmq.ImageHub()
fps = FPS().start()

count = 0
start_time = time.time()

while True:  # show streamed images until Ctrl-C
    try:
        rpi_name, image = image_hub.recv_image()
        cv2.imshow(rpi_name, image)  # 1 window for each RPi
        fps.update()
        count += 1
        cv2.waitKey(1)
        image_hub.send_reply(b"OK")
    except (KeyboardInterrupt, SystemExit):
        fps.stop()
        print(fps.fps())
        print(time.time() - start_time)
        print(count)

        quit()
