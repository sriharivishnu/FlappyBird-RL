#include <Servo.h>
Servo motor;
int MOTOR_PIN = 9;
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  motor.attach(MOTOR_PIN);
  motor.write(0);
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
     if (data == "1") {
      motor.write(39);
      delay(70);
      motor.write(10);
      Serial.println("1");
    }
  }
}
