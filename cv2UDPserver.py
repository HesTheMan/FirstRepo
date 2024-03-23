#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      The Man
#
# Created:     02/03/2024
# Copyright:   (c) The Man 2024
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import socket
import numpy as np
import cv2
import signal
import time

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
MESSAGE_SIZE = 16 * 1024 * 8  # 16x1024 numpy array of 8-byte numbers

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

spike_gain=10
sine_gain=0
noise_gain=0.2

mySpike_uV = np.array([-6.8, -15.2, -15.6, -8.9, -10.0, -32.3, -70.9, -100.0, -95.5, -49.7,
            9.2, 51.2, 58.4, 41.4, 23.2, 18.0, 21.8, 23.9, 18.9, 11.7, 9.5,
            13.0, 16.8, 15.8, 10.8, 6.2, 4.4, 3.3, 0.6, -5.4, -3.3, 0.0])
uVperCount=0.030517578125000001
my_spike_counts=mySpike_uV/uVperCount
my_spike_counts=spike_gain*my_spike_counts



#parameters to create a smooth sine wave
freq = 1
fs = 30000
uVperCount = 0.030517578125000001
amp_uV = 100
amp = sine_gain*amp_uV / uVperCount
delta_phase = 0
my_sine=np.zeros([16,1024])


def signal_handler(sig, frame):
    print("Exiting the program...")
    server_socket.close()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)

spike_loc=300
while True:
    #Random component
    data = noise_gain*np.random.normal(0, 1, (16, 1024)) * 2**14
    #Spike Component smooth transition at block boundaries
    if spike_loc + len(my_spike_counts) > 1024:
        part1 = 1024 - spike_loc
        part2 = len(my_spike_counts) - part1
        data[:, spike_loc:spike_loc + part1] = data[:, spike_loc:spike_loc + part1] + my_spike_counts[:part1]
        data[:, :part2] = data[:, :part2] + my_spike_counts[part1:]
    else:
        data[:, spike_loc:spike_loc + len(my_spike_counts)] = data[:, spike_loc:spike_loc + len(my_spike_counts)] + my_spike_counts
    #sine wave component smooth transition at block boundaries
    for i in range(1024):
        delta_phase = delta_phase + 2 * np.pi * freq / fs
        delta_phase = delta_phase % (2 * np.pi)
        my_sine[:,i]=amp * np.sin(delta_phase)

        #my_sine[:,i]=np.clip(amp * np.sin(delta_phase), -32768, 32767).astype(np.int16)

    data=data+my_sine
    data = np.clip(data,-32768, 32767)
    data = data.astype(np.int16).tobytes()
    for i in range(0, len(data), MESSAGE_SIZE):
        server_socket.sendto(data[i:i + MESSAGE_SIZE], (UDP_IP, UDP_PORT))

    time.sleep(0.034)  # Introduce a 0.034-second delay
    spike_loc = (spike_loc + 1) % 1024
    #spike_loc = np.random.randint(0, 1024)

