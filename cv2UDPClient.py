#-------------------------------------------------------------------------------
# Name:        Cv2 Client
# Purpose
#
# Author:      Dinal Andreasen
#
# Created:     02/03/2024
# Copyright:   (c) The Man 2024
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import socket
import numpy as np
import cv2
import time
import scipy.signal as signal


UDP_IP = "127.0.0.1"
UDP_PORT = 5005
MESSAGE_SIZE = 16 * 1024 * 8  # 16x1024 numpy array of 8-byte numbers

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.bind((UDP_IP, UDP_PORT))

numSamples=1024
x = np.linspace(100, 1100, numSamples)
xS= np.linspace(10,330,33)
font = cv2.FONT_HERSHEY_SIMPLEX
time_last=0
data_rate=0
uVperCount = 0.030517578125000001

# Define the bandpass filter
lowcut = 10
highcut = 6000
fs = 30000
# Create a Butterworth bandpass filter
b, a = signal.butter(5, [lowcut, highcut], btype='band', fs=fs)
scope_gain=10.0
threshold_counts = -15000
threshold=threshold_counts*scope_gain/32000
spike_index=range(32)
while True:
    time_start=time.time()
    data, _ = client_socket.recvfrom(MESSAGE_SIZE)
    data = np.frombuffer(data, dtype=np.int16).reshape(16, 1024)
    data_uV=data*uVperCount

    image = np.zeros((900, 1200,3), dtype=np.uint8)
    imagef = np.zeros((900, 1200, 3), dtype=np.uint8)
    imageC = np.zeros((400, 400, 3), np.uint8)  #The Cluster image
    imageS = np.zeros((900, 350, 3), np.uint8)  #The Spike image

    for i in range(16):
        y_int16=data[i,:]
        y=y_int16*scope_gain/32000

        #y = np.random.randint(10,30 , numSamples)
        points = np.array([x, 100+y+40*i], dtype=np.int32).T
        cv2.polylines(image, [points], isClosed=False, color=(16*(17-i), 16*i+1, 0), thickness=1)

        yf = signal.lfilter(b, a, y)
        pointsf = np.array([x, 100+yf+40*i], dtype=np.int32).T
        cv2.polylines(imagef, [pointsf], isClosed=False, color=(16*(17-i), 16*i+1, 0), thickness=1)

        #exceeded_points = np.where(np.diff(yf > threshold, axis=0))
        exceeded_points = np.where((np.roll(yf, 1) > threshold) & (yf <= threshold))
        for point in exceeded_points[0]:
            scaled_point = int((point / numSamples) * 1000)  # Scale the x-coordinate to fit within 1000 pixels
            cv2.line(imagef, (scaled_point+100, 90+40*i), (scaled_point+100, 110+40*i), (0, 0, 255), 1)  #draw a line on the filtered data
            cv2.line(imagef, (scaled_point+100-10, int(threshold)+100+40*i), (scaled_point+100+10, int(threshold)+100+40*i), (0, 0, 255), 1)  #draw a line on the filtered data
            if point < 16 or point > len(yf) - 17:
                continue
            C_window = yf[point - 16: point + 17]
            xC=np.max(C_window)
            yC=-np.min(C_window)

            xC_int=int((xC / numSamples) * 20000)
            yC_int=400-int((yC / numSamples) * 20000)
            #xC_int=np.random.randint(0, 401)
            #yC_int=np.random.randint(0, 401)
            cv2.circle(imageC,(xC_int, yC_int), 3, (0, 255, 255), -1)

            points_S = np.array([xS, 100+yf[point - 16: point + 17]+40*i], dtype=np.int32).T
            cv2.polylines(imageS,[points_S], isClosed=False, color=(16*(17-i), 16*i+1, 0), thickness=1)
    # persistence
    imageC = np.where(imageC > 0, imageC/1.0005, 0)
    #imageC = np.where(imageC > 0, imageC/1.5, 0)

    timeNow = '{:.3f}'.format(time.time() % 60)
    delta_time=time.time()-time_last
    proc_time=time.time()-time_start
    delta_time_str='Delta Time = {:.3f}'.format(delta_time)
    proc_time_str='P Time = {:.3f}'.format(proc_time)
    data_rate=1024*0.001/(delta_time+.000001)
    data_rate_str='Channel Data Rate = {:>6.3} kSPS'.format(data_rate)
    scope_gain_str='Scope Gain = {}'.format(scope_gain)
        #cv2.putText(image, timeNow, (1000, 50), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, timeNow, (100, 50), font, 0.5, (255, 255, 255), 1, cv2.LINE_8)
    cv2.putText(image, delta_time_str, (200, 50), font, 0.5, (255, 255, 255), 1, cv2.LINE_8)
    cv2.putText(image, data_rate_str, (400, 50), font, 0.5, (255, 255, 255), 1, cv2.LINE_8)
    cv2.putText(image, proc_time_str, (700, 50), font, 0.5, (255, 255, 255), 1, cv2.LINE_8)

    cv2.putText(image, scope_gain_str,(900, 50), font, 0.5, (255, 255, 255), 1, cv2.LINE_8)

    time_last=time.time()


    cv2.imshow("Scope Traces", image)
    key = cv2.waitKey(1)  # Wait for a key event
    if key & 0xFF == ord('s'):  # Check if 's' key is pressed
        break
    elif key == 43:  # Check if + key is pressed
        scope_gain *= 2  # Double the scope gain
    elif key == 45:  # Check if - key is pressed
        scope_gain = scope_gain/2  # Double the scope gain

    cv2.imshow('Filtered Image', imagef)
    key = cv2.waitKey(1)  # Wait for a key event
    if key & 0xFF == ord('s'):  # Check if 's' key is pressed
        break
    cv2.imshow('Cluster', imageC)
    key = cv2.waitKey(1)  # Wait for a key event
    if key & 0xFF == ord('s'):  # Check if 's' key is pressed
        break
    cv2.imshow('Spike Window', imageS)
    key = cv2.waitKey(1)  # Wait for a key event
    if key & 0xFF == ord('s'):  # Check if 's' key is pressed
        break




cv2.destroyAllWindows()
