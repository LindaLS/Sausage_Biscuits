#include<ADC.h>
#define HWSERIAL Serial
long t,t0;
int test=200;
int test2=200;
int  maxChannel=1;
int sensorValue[21];
ADC *adc=new ADC();
byte buf[3];
int refreshCounter=0;



// the setup routine runs once when you press reset:
void setup() {
  // initialize serial communication at 9600 bits per second:
  HWSERIAL.begin(9600);
  //HWSERIAL.begin(460800);
  

}

// the loop routine runs over and over again forever:
void loop() {
  // read the input on analog pin 0:
  int i=0;
  t=micros();
  adc->setResolution(16);
  sensorValue[0] = adc->analogRead(A0);
  sensorValue[1] = adc->analogRead(A1);
  sensorValue[2] = adc->analogRead(A2);
  sensorValue[3] = adc->analogRead(A3);

  //sensorValue[4] = adc->analogRead(A4);
  //sensorValue[5] = adc->analogRead(A5);
  //sensorValue[6] = adc->analogRead(A6);
  //sensorValue[7] = adc->analogRead(A7);
  /*
  sensorValue[3] = adc->analogRead(A3);
  sensorValue[4] = adc->analogRead(A4);
  sensorValue[5] = adc->analogRead(A5);
  sensorValue[6] = adc->analogRead(A6);
  sensorValue[7] = adc->analogRead(A7);
  sensorValue[8] = adc->analogRead(A8);
  sensorValue[9] = adc->analogRead(A9);
  sensorValue[10] = adc->analogRead(A10);
  sensorValue[11] = adc->analogRead(A11);
  sensorValue[12] = adc->analogRead(A12);
  sensorValue[13] = adc->analogRead(A13);
  sensorValue[14] = adc->analogRead(A14);
  sensorValue[15] = adc->analogRead(A15);
  sensorValue[16] = adc->analogRead(A16);
  sensorValue[17] = adc->analogRead(A17);
  sensorValue[18] = adc->analogRead(A18);
  sensorValue[19] = adc->analogRead(A19);

  //Serial.println(Serial.available());
 
 /*
  //testing purpose
  for(i=0;i<(maxChannel);i++){
      Serial.print(i);
      Serial.print(",");
      Serial.println(sensorValue[i]);
  }
  */
 
  for(i=0;i<(maxChannel);i++){
    buf[0]=(sensorValue[i]&127);
    buf[1]=(sensorValue[i]>>7)&127;
    buf[2]=(128|(i<<2))|((sensorValue[i]>>14)&3);
    HWSERIAL.write(buf[2]);
    HWSERIAL.write(buf[1]);
    HWSERIAL.write(buf[0]);
  }
  
  t0=2000-(micros()-t);
  if(t0>0){
    delayMicroseconds(t0);
  }
  
  refreshCounter++;
  if(refreshCounter==60){
    //refresh the usb buffer every 120ms
    HWSERIAL.send_now();
    refreshCounter=0;
  }
  //t0=micros()-t;
  //HWSERIAL.print("Time: ");
  //HWSERIAL.println(t0);
  //delayMicroseconds(100);
  
}
