import argparse
from threading import Thread
import multiprocessing
import time
from imutils.video import FPS
from imutils.video import VideoStream
import cv2
import imutils
import detection


def detect(queue_in, queue_out):
    # loop indefinitely -- this function will be called as a daemon
    # process so we don't need to worry about joining it
    while True:
        # attempt to grab the next frame from the input queue
        frame = queue_in.get()
        if frame is None:
            continue

        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        detections = detector.detect(frame)
        boxes = [centroid for (conf, box, centroid) in detections]

        queue_out.put(boxes)


def detect_handle(queue_in):
    # loop indefinitely -- this function will be called as a daemon
    # process so we don't need to worry about joining it
    while True:
        # attempt to grab the next frame from the input queue
        boxes = queue_in.get()
        if frame is None:
            continue

        print(boxes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-md", "--model-dir", required=True,
                    help="path to pre-trained model dir")
    ap.add_argument("-i", "--input", type=str, default="",
                    help="path to (optional) input video file")
    ap.add_argument("-o", "--output", type=str, default="",
                    help="path to (optional) output video file")
    ap.add_argument("-g", "--use-gpu", type=int, default=0,
                    help="set 1 to indicate should use CUDA; otherwise use CPU instead")
    ap.add_argument("-s", "--skip-frames", type=int, default=30,
                    help="# of skip frames between detections")
    args = vars(ap.parse_args())

    # load our serialized model from disk
    print("[INFO] loading model...")
    model_cfg = args["model_dir"] + "/yolov3.cfg"
    model_weights = args["model_dir"] + "/yolov3.weights"
    net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
    if args.get("use_gpu", 0) == 1:
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    lay = net.getLayerNames()
    lay = [lay[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # initialize person detector
    detector = detection.Detector(net=net, layer=lay)

    # initialize queue
    # in: receive detection demand
    # out: send detection result
    detect_Q_in = multiprocessing.Queue()
    detect_Q_out = multiprocessing.Queue()

    # spawn a daemon process for object detection
    detect_thd = Thread(target=detect, name="Detector",
                        args=(detect_Q_in, detect_Q_out))
    detect_thd.daemon = True
    detect_thd.start()

    # spawn a daemon process for object detection
    detect_handle_thd = Thread(target=detect_handle, name="Detection Handler",
                               args=(detect_Q_out,))
    detect_handle_thd.daemon = True
    detect_handle_thd.start()

    src = args["input"] if args["input"] else 0

    # if a video path was supplied, grab a reference to the video file
    if args.get("input", False):
        print("[INFO] opening video file...")
        vs = cv2.VideoCapture(src)

    # otherwise, grab a reference to the webcam
    else:
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)

    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    total_frames = 0

    fps = FPS().start()

    # loop over frames from the video stream
    while True:
        start = time.time()

        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        frame = vs.read()

        frame = frame[1] if args.get("input", False) else frame

        # if we are viewing a video and we did not grab a frame then we have
        # reached the end of the video
        if args["input"] is not None and frame is None:
            break

        # resize the frame to have a maximum width of 500 pixels (the less data
        # we have, the faster we can process it)
        frame = imutils.resize(frame, width=500)

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if total_frames % args["skip_frames"] == 0:
            # set the status and initialize our new set of object trackers
            status = "Detecting"

            detect_Q_in.put(frame)

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            pass

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        total_frames += 1
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # if we are not using a video file, stop the camera video stream
    if not args.get("input", False):
        vs.stop()

    # otherwise, release the video file pointer
    else:
        vs.release()

    # close any open windows
    cv2.destroyAllWindows()
