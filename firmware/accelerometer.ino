/**
 * accelerometer.ino
 *
 * Streams MMA8452Q accelerometer data over serial at ~20 Hz.
 * Output format: X_g<TAB>Y_g<TAB>Z_g<NEWLINE>
 *
 * Hardware:
 *   Arduino --------------- MMA8452Q Breakout
 *     3.3V  ---------------     3.3V
 *     GND   ---------------     GND
 *   SDA (A4) --\/330 Ohm\/--    SDA
 *   SCL (A5) --\/330 Ohm\/--    SCL
 */

#include <Wire.h>
#include "SparkFun_MMA8452Q.h"

MMA8452Q accel;

static const int SAMPLE_INTERVAL_MS = 50;

void setup() {
    Serial.begin(9600);
    Wire.begin();

    if (!accel.begin()) {
        Serial.println("ERROR: MMA8452Q not found. Check wiring.");
        while (true);
    }

    Serial.println("MMA8452Q ready");
}

void loop() {
    if (accel.available()) {
        Serial.print(accel.getCalculatedX(), 3);
        Serial.print("\t");
        Serial.print(accel.getCalculatedY(), 3);
        Serial.print("\t");
        Serial.print(accel.getCalculatedZ(), 3);
        Serial.println();
        delay(SAMPLE_INTERVAL_MS);
    }
}
