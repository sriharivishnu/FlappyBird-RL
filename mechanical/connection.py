#script to send commands to Arduino

import serial
class Connection():
    def __init__(self, path, frequency):
        self.serial = serial.Serial(path, frequency, timeout=1)
        self.serial.flush()
    
    def sendTap(self):
        self.serial.write(b"1\n")