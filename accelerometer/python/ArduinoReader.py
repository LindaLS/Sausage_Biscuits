# This class is used to read values dumped by the MPU6050 accelerometer
# on to the Arduino serial port.
import numpy as np
import serial 
import time
import struct
import threading
import converters

class ArduinoReader:

    # Constructor
    def __init__(self, port, baud_rate):
        # Serial port configuration
        self.port = port
        self.baud_rate = baud_rate
        
        # Raw measurements
        self.quaternian = np.matrix('0.0; 0.0; 0.0; 0.0')
        self.acceleration = np.matrix('0.0; 0.0; 0.0')
        
        # Accelerometer related values
        self.gravity = np.matrix('0.0; 0.0; 9.81')
        self.velocity = np.matrix('0.0; 0.0; 0.0')
        self.displacement = np.matrix('0.0; 0.0; 0.0')
        self.calibration = np.matrix('1.0; 0.0; 1.0; 0.0; 1.0; 0.0')

        # Threads
        self.done_reading = False

        # Open port used by Arduino
        self.arduino_port = serial.Serial(self.port, self.baud_rate)
        self.arduino_port.flush()
        #self.arduino_port.BytesAvailableFcnCount = 22;
        #self.arduino_port.BytesAvailableFcnMode = 'byte';

        # Start reading from Arduino port
        #self.arduino_port.open()
        time.sleep(2)
        self.arduino_port.write('1')

        self.callback_thread = threading.Thread(target=readCallback, args=(self,))
        self.callback_thread.start()

    def close(self):
        self.done_reading = True
        self.callback_thread.join()
        self.arduino_port.close()

    def raw_read(self):
        q = self.quaternian;
        a = np.divide(self.acceleration, 1000.0);
        return q, a

    def read(self):
        q, a = self.raw_read()
        R = converters.quaternion_to_r(q)
        g = np.transpose(R) * self.gravity
        a_real = np.zeros((3, 1));
        a_real[0] = a[0]*self.calibration[0] + self.calibration[1] - g[0]
        a_real[1] = a[1]*self.calibration[2] + self.calibration[2] - g[1]
        a_real[2] = a[2]*self.calibration[4] + self.calibration[5] - g[2]
        return R, a_real

    # Calibration function
    # Calibrate accelerometer readings
    # TODO:Calculate sample rate
    def calibrate(self):
        N = 100 # Number of sample points
        delta = 0.5 # accepted delta from real gravity. Small delta gives high accuracy but may not converge.

        sample_num = 0
        M = np.zeros((3*N, 6))
        y = np.zeros((3*N, 1))
        lamda = 0.1
        while sample_num < N:
            q, a = self.raw_read()
            magnitude = np.sqrt(np.square(a).sum())
            
            if (magnitude > (self.gravity[2]-delta) and magnitude < (self.gravity[2]+delta)):
                R = converters.quaternion_to_r(q)
                g = np.transpose(R) * self.gravity
                M[3*sample_num, 0] = a[0]
                M[3*sample_num, 1] = 1
                M[3*sample_num+1, 2] = a[1]
                M[3*sample_num+1, 3] = 1
                M[3*sample_num+2, 4] = a[2]
                M[3*sample_num+2, 5] = 1
                y[3*sample_num,0] = g[0]
                y[3*sample_num+1,0] = g[1]
                y[3*sample_num+2,0] = g[2]
                sample_num = sample_num + 1
            
            time.sleep(0.1)

        # Calibrate using least sum of squares with penalization
        self.calibration = np.linalg.pinv(np.transpose(M).dot(M) + lamda * np.eye(6)).dot(np.transpose(M)).dot(y)
        
# This function will keep on reading from the serial port until close() is called.
# Call using a dedicated thread.
def readCallback(ar):
    while(not ar.done_reading):
        if(ar.arduino_port.in_waiting > 22):
            ar.quaternian[0] = struct.unpack('<f', ar.arduino_port.read(4))
            ar.quaternian[1] = struct.unpack('<f', ar.arduino_port.read(4))
            ar.quaternian[2] = struct.unpack('<f', ar.arduino_port.read(4))
            ar.quaternian[3] = struct.unpack('<f', ar.arduino_port.read(4))
            ar.acceleration[0] = struct.unpack('<h', ar.arduino_port.read(2))
            ar.acceleration[1] = struct.unpack('<h', ar.arduino_port.read(2))
            ar.acceleration[2] = struct.unpack('<h', ar.arduino_port.read(2))

