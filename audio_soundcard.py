import soundcard as sc
import keyboard as kb
import numpy as np
import os
import threading
import crepe
import queue
import warnings
import time

from statistics import mode

# Import needed modules from osc4py3
from osc4py3.as_eventloop import *
from osc4py3 import oscbuildparse

# Suppress the SoundcardRuntimeWarning
warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

# get a list of all speakers:
speakers = sc.all_speakers()
# get the current default speaker on your system:
default_speaker = sc.default_speaker()
# get a list of all microphones:
mics = sc.all_microphones()
# get the current default microphone on your system:
default_mic = sc.default_microphone()

# create a queue to communicate with threads
q = queue.Queue()

# print(speakers)
# print(default_speaker)
# print(mics)
# print(default_mic)

def read_data(mic_input, numframes=1024):
    return mic_input.record(numframes)

def send_msg(name, value):
    osc_process()
    osc_send(oscbuildparse.OSCMessage("/keypoint/" + name, None, value), "localhost")
    time.sleep(0.475)


def detection_context():
    with default_mic.recorder(samplerate=96000) as mic:
        data = read_data(mic)
        dprev = np.array([])

        while data.any():
            time, frequency, confidence, activation = crepe.predict(data, 96000, step_size=16, model_capacity="large", viterbi=True, center=True, verbose=False)

            if q.qsize() > 0:
                q.task_done()
                q.join()

            # print("time", time)
            # print("freq", frequency)
            # print("confidence", confidence)
            # print("activation", activation)

            d = np.array(frequency * confidence)
            
            if confidence[0] < 0.20:
                d = dprev
            else:
                dprev = d
            
            t = threading.Thread(target=send_msg, name="osc-worker", args=["data", d.tolist()], daemon=True)
            q.put(t)
            t.start()        

            try:
                data = read_data(mic)

                if kb.is_pressed('q'):
                    print("Quitting")
                    return
            except:
                continue


if __name__ == '__main__':
    osc_startup()
    osc_udp_client("127.0.0.1", 9001, "localhost")
    detection_context()
    osc_terminate()