# This is sample code used to test whether data can be properly read from the arduino.

# Import libraries
import time
import ArduinoReader
import numpy as np

def sample():
    ar = ArduinoReader.ArduinoReader('COM10', 115200)
    
    print('Waiting for readings to stabilize...')
    time.sleep(30)
    #print('Calibrating...');
    #ar.calibrate();
    print('Done Calibrating.');

    start = time.time()
        
    while (time.time() - start) < 15 :
        R, a = ar.read()
        print(np.transpose(a))
        
        time.sleep(0.1)

    ar.close()

# MAIN #
if __name__ == "__main__":
    sample()