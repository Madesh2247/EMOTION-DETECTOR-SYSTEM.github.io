// Advanced Water Level Controller using Arduino

const int lowSensor = 2;
const int midSensor = 3;
const int highSensor = 4;

const int motorRelay = 8;
const int buzzer = 9;
const int manualSwitch = 10;

const int ledLow = 5;
const int ledMid = 6;
const int ledHigh = 7;

bool motorState = false;

void setup() {
  pinMode(lowSensor, INPUT);
  pinMode(midSensor, INPUT);
  pinMode(highSensor, INPUT);

  pinMode(motorRelay, OUTPUT);
  pinMode(buzzer, OUTPUT);
  pinMode(manualSwitch, INPUT_PULLUP);

  pinMode(ledLow, OUTPUT);
  pinMode(ledMid, OUTPUT);
  pinMode(ledHigh, OUTPUT);

  Serial.begin(9600);
  Serial.println("Smart Water Level Controller Initialized");
}

void loop() {
  bool low = digitalRead(lowSensor);
  bool mid = digitalRead(midSensor);
  bool high = digitalRead(highSensor);
  bool manual = digitalRead(manualSwitch) == LOW; // manual ON

  // Update LED indicators
  digitalWrite(ledLow, !low);
  digitalWrite(ledMid, !mid);
  digitalWrite(ledHigh, !high);

  // Manual mode
  if (manual) {
    digitalWrite(motorRelay, HIGH);
    Serial.println("Manual Mode: Pump ON");
    delay(500);
    return;
  }

  // Auto Mode Control
  if (!low && !motorState) {
    digitalWrite(motorRelay, HIGH);
    motorState = true;
    Serial.println("Tank Low - Pump ON");
  }
  else if (high && motorState) {
    digitalWrite(motorRelay, LOW);
    motorState = false;
    Serial.println("Tank Full - Pump OFF");
    tone(buzzer, 1000, 1000); // alert
  }

  // Normal condition
  if (mid && !high && motorState == false) {
    Serial.println("Water Level Normal");
  }

  delay(1000);
}
