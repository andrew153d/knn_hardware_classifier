#include <Adafruit_NeoPixel.h>
#ifdef __AVR__
#include <avr/power.h> // Required for 16 MHz Adafruit Trinket
#endif
#include <Servo.h>

#define PIN        5
#define NUMPIXELS  4
#define BRIGHTNESS 250
#define DELAYVAL   500

Adafruit_NeoPixel pixels(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);
Servo myservo;
char command;
long last_time = 2000;
bool on = false;
void setup() {
  Serial.begin(9600);

#if defined(__AVR_ATtiny85__) && (F_CPU == 16000000)
  clock_prescale_set(clock_div_1);
#endif

  pixels.begin();
  
 
  pixels.clear();
  pixels.show();
  myservo.attach(9);
  last_time = millis();
}

void loop() {
  if (Serial.available() > 0) {
    command = Serial.read();
  }
  
  if(millis()-last_time > 10000 && on){
    pixels.clear();
    pixels.show();
    on = false;
  }

  if(command == 'a'){
    last_time = millis();
    command = 'c';

      for (int i = 0; i < NUMPIXELS; i++) {
       pixels.setPixelColor(i, pixels.Color(BRIGHTNESS, BRIGHTNESS, BRIGHTNESS));
      }
      on = true;
    pixels.show();
    Serial.flush();
    int pos;
    myservo.write(180);
    delay(1500);
    myservo.write(0);
  
  }
}
