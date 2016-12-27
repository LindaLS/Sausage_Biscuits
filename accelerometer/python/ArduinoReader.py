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
        self.sensitivity = 1
        self.gravity_compensation = np.matrix('9.81 -9.81; 9.81 -9.81; 9.81 -9.81')
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
        #self.arduino_port.write('1')

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
        # Read raw values coming from accelerometer
        q, a_raw = self.raw_read()

        # Convert from quaternion to rotation matrix
        R = converters.quaternion_to_r(q)

        # Calculate how much to compensate gravity by
        gravity_compensation = np.zeros((3, 1))
        for axis in range(0, 3):
            # Each axis corresponds to x,y,z
            if R[2][axis] > 0:
                # Gravity alignes with +ve axis
                gravity_compensation[axis] = R[2, axis] * self.gravity_compensation[axis, 0]
            else:
                # Gravity alignes with -ve axis
                gravity_compensation[axis] = -1 * R[2, axis] * self.gravity_compensation[axis, 1]
        
        # Calculate acceleration with gravity compensation    
        a_real = np.multiply(self.sensitivity, a_raw - gravity_compensation)
        
        return R, a_real

    # Calibration function
    # Calibrate accelerometer readings
    def calibrate(self, calibration_file):
        # Get offset constants from file
        self.sensitivity = 1
        x = 0
        y = 1
        z = 2
        positive = 0
        negative = 1

        self.gravity_compensation[x, positive] = 8.14
        self.gravity_compensation[x, negative] = -7.47
        self.gravity_compensation[y, positive] = 7.72
        self.gravity_compensation[y, negative] = -7.94
        self.gravity_compensation[z, positive] = 10.83
        self.gravity_compensation[z, negative] = -5.7
        
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

