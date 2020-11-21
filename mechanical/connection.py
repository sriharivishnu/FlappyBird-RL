#script to send commands to Arduino

import serial
class Connection():
    def __init__(self, path, frequency):
        self.serial = serial.Serial(path, frequency, timeout=1)
        self.serial.flush()
    
    def sendTap(self):
        self.serial.write(b"1\n")
    
    def sendTapAndWait(self):
        self.serial.write(b"1\n")
        while True:
            line = self.serial.readline().decode('utf-8').rstrip().strip()
            if line == "1":
                break