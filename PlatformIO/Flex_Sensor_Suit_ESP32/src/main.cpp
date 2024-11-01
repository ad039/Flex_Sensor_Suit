#include <Arduino.h>
#include <ArduinoBLE.h>
#include <Wire.h>
#include "SparkFun_Displacement_Sensor_Arduino_Library.h" // Click here to get the library: http://librarymanager/All#SparkFun_Displacement_Sensor
#include "Adafruit_NeoPixel.h"

//define what sensors to use
//#define ADS_sensor

// define debug mode
#define DEBUG


////////////////////////////////// ADS Sensors //////////////////////////////////
#ifdef ADS_sensor
// ADS Objects
ADS elbowFlex;
ADS shoulderFlex1;
ADS shoulderFlex2;
ADS shoulderFlex3;
ADS forearmFlex;
ADS handFlex1;
ADS handFlex2;

void ADS_init();
#endif

////////////////////////////////// RTOS //////////////////////////////////
TaskHandle_t bleThread; // Rtos task handles
TaskHandle_t sensorThread;
TaskHandle_t batteryMonitorThread;
TaskHandle_t MCUTempThread;

//Mutex stdio_mutex;
//Mutex ble_mutex;

// a struct for the sensor task to ble task queue
typedef struct {
  int16_t data[7];
} message_sensor;


void ble_Task(void *pvParameters);
void sensor_Task(void *pvParameters);
void batteryMonitor_Task(void *pvParameters);
void MCUTemp_Task(void *pvParameters);


TickType_t BLE_UPDATE_INTERVAL;

////////////////////////////////// Interrupt //////////////////////////////////

TimerHandle_t sensorTimer;
TimerHandle_t batteryMonitorTimer;
TimerHandle_t MCUTempTimer;

void ISRCallback_Sensor(TimerHandle_t xTimer);
void ISRCallback_BatteryMonitor(TimerHandle_t xTimer);
void ISRCallback_MCUTemp(TimerHandle_t xTimer);


////////////////////////////////// BLE //////////////////////////////////
// Flex Sensor Service and characteristics
BLEService flexSensor_Service("0000fff0-0000-1000-8000-00805f9b34fb");

BLECharacteristic flexSensor_Char("0000fff1-0000-1000-8000-00805f9b34fb", BLERead | BLENotify, 14);

// Battery Level Service
BLEService batteryLevel_Service("180F");

BLEUnsignedCharCharacteristic batteryLevel_Char("2A19", BLERead | BLENotify);

BLEDescriptor batteryLevel_Descriptor = BLEDescriptor("2901", "Battery Service");

// MCU Temp Service
BLEService MCUTemp_Service("2A6E");

BLEShortCharacteristic MCUTemp_Char("2A6E", BLERead | BLENotify);


// Advertising parameters should have a global scope. Do NOT define them in 'setup' or in 'loop'
const uint8_t manufactData[4] = {0x01, 0x02, 0x03, 0x04};
const uint8_t serviceData[3] = {0x00, 0x01, 0x02};

uint8_t notification_status = 0;

void sensor_characteristicRead(BLEDevice central, BLECharacteristic thisChar);

void sensor_characteristicSubscribed(BLEDevice central, BLECharacteristic thisChar);

void sensor_characteristicUnsubscribed(BLEDevice central, BLECharacteristic thisChar);

void batteryMonitor_characteristicRead(BLEDevice central, BLECharacteristic thisChar);

void batteryMonitor_characteristicSubscribed(BLEDevice central, BLECharacteristic thisChar);

void batteryMonitor_characteristicUnsubscribed(BLEDevice central, BLECharacteristic thisChar);

void MCUTemp_characteristicRead(BLEDevice central, BLECharacteristic thisChar);

void MCUTemp_characteristicSubscribed(BLEDevice central, BLECharacteristic thisChar);

void MCUTemp_characteristicUnsubscribed(BLEDevice central, BLECharacteristic thisChar);

void blePeripheralConnectHandler(BLEDevice central);

void blePeripheralDisconnectHandler(BLEDevice central);

void BLE_init(void);

/* Neopixel */
// How many internal neopixels do we have? some boards have more than one!
#define NUMPIXELS        1

Adafruit_NeoPixel pixels(NUMPIXELS, PIN_NEOPIXEL, NEO_GRB + NEO_KHZ800);

/* Other */
int batteryPin = A2;
int counter = 0;

////////////////////////////////// Setup //////////////////////////////////
void setup()
{
  /* Serial begin */
  Serial.begin(115200);

  //while (!Serial) {
  //  ;
  //}
  
  /* Wire begin */
  Wire.begin();
  Wire.setClock(1000000);
  delay(10);

  /* Initialise ADS sensors */
  #ifdef ADS_sensor
  ADS_init();
  #endif
  
// Neopixel stuff to indicate a connection
#if defined(NEOPIXEL_POWER)
  // If this board has a power control pin, we must set it to output and high
  // in order to enable the NeoPixels. We put this in an #if defined so it can
  // be reused for other boards without compilation errors
  pinMode(NEOPIXEL_POWER, OUTPUT);
  digitalWrite(NEOPIXEL_POWER, HIGH);
#endif

  pixels.begin(); // INITIALIZE NeoPixel strip object (REQUIRED)
  pixels.setBrightness(20); // not so bright

  // set color to red
  pixels.fill(0xFF0000);
  pixels.show();

  /* Initialise BLE */
  BLE_init();

  Serial.println("Starting...");

  /* Initialise and start threads to run infinitley */

  xTaskCreate(ble_Task, "BLE Task", 2048, NULL, 2, &bleThread);
  xTaskCreate(sensor_Task, "Sensor Task", 2048, NULL, 3, &sensorThread);
  xTaskCreate(batteryMonitor_Task, "Battery Monitor Task", 2048, NULL, 1, &batteryMonitorThread);
  xTaskCreate(MCUTemp_Task, "MCU Temp Task", 2048, NULL, 1, &MCUTempThread);

  /* Initialise Timer, but dont start yet*/
  sensorTimer = xTimerCreate("Sensor Timer", 10, pdTRUE, NULL, ISRCallback_Sensor);
  batteryMonitorTimer = xTimerCreate("Battery Monitor Timer", 10000, pdTRUE, NULL, ISRCallback_BatteryMonitor);
  MCUTempTimer = xTimerCreate("MCU Temp Timer", 10000, pdTRUE, NULL, ISRCallback_MCUTemp);
}

void loop() { 
  // nothing to do here
  //portSUPPRESS_TICKS_AND_SLEEP( 900 );
}


// a task to control the ble polling
void ble_Task(void *pvParameters)
{
  // set the update interval for the ble thread
  TickType_t xLastWakeTime;
  BLE_UPDATE_INTERVAL = 1000;

  // Initialise the xLastWakeTime variable with the current time.
  
  long currentMillis = millis();

  while (1) 
  {
    xLastWakeTime = xTaskGetTickCount();
    uint32_t currentMillis = millis();

    // poll for ble updates
    //ble_mutex.lock();
    BLE.poll();
    //ble_mutex.unlock();
   

#ifdef DEBUG
    //stdio_mutex.lock();
    Serial.print("B ");
    Serial.print(millis()-currentMillis);
    Serial.print(" ");
    Serial.println(uxTaskGetStackHighWaterMark(bleThread));
    //stdio_mutex.unlock();
#endif

    // sleep this thread for the update interval minus the millis spent in this thread
    vTaskDelayUntil( &xLastWakeTime, BLE_UPDATE_INTERVAL);
  }
  
}


/* Task to read the sensors */
void sensor_Task(void *pvParameters) 
{
  // set the update interval for the ble thread
  TickType_t xSensorLastWakeTime;
  const TickType_t xSensor_Frequency = 10;

  // Initialise the xLastWakeTime variable with the current time.
  xSensorLastWakeTime = xTaskGetTickCount();
  
  uint32_t ulInterruptStatus; // variable to recive the notification value

  message_sensor *sensorMessage = new message_sensor();
  sensorMessage->data[0] = 0x0001;

  while(1) {
    /* Block indefinitely (without a timeout, so no need to check the function's
           return value) to wait for a notification. NOTE! Real applications
           should not block indefinitely, but instead time out occasionally in order
           to handle error conditions that may prevent the interrupt from sending
           any more notifications. */
    xTaskNotifyWaitIndexed( 0,                  /* Wait for 0th Notificaition */
                            0x00,               /* Don't clear any bits on entry. */
                            ULONG_MAX,          /* Clear all bits on exit. */
                            &ulInterruptStatus, /* Receives the notification value. */
                            portMAX_DELAY );    /* Block indefinitely. */

    
    long currentMillis = millis();

    // for debugging ble
#ifndef ADS_sensor
    if (counter == 1){
        sensorMessage->data[0] = 0x0000;
        sensorMessage->data[1] = 0x0001;
        sensorMessage->data[2] = 0x0002;
        sensorMessage->data[3] = 0x0003;
        sensorMessage->data[4] = 0x0004;
        sensorMessage->data[5] = 0x0005;
        sensorMessage->data[6] = 0x0006;
    }
    if (counter == 0) {
        sensorMessage->data[0] = 0x0001;
        sensorMessage->data[1] = 0x0001;
        sensorMessage->data[2] = 0x0002;
        sensorMessage->data[3] = 0x0003;
        sensorMessage->data[4] = 0x0004;
        sensorMessage->data[5] = 0x0005;
        sensorMessage->data[6] = 0x0006;
    }

    if (counter == 1) {
        counter = 0;
    }
    else {
        counter = 1;
    }
    
    delay(2);
  
#endif

    /////////////////// Read ADSs ///////////////////
#ifdef ADS_sensor
    // read elbow sensor
    if (elbowFlex.available()) {
        sensorMessage->data[0] = (int16_t)(elbowFlex.getX()*100);
    }
    
    // read shoulder1 sensor
    if (shoulderFlex1.available()) {
        sensorMessage->data[1] = (int16_t)(shoulderFlex1.getX()*100);
    }

    // read shoulder2 sensor
    if (shoulderFlex2.available()) {
        sensorMessage->data[2] = (int16_t)(shoulderFlex2.getX()*100);
    }

    // read shoulder3 sensor
    if (shoulderFlex3.available()) {
        sensorMessage->data[3] = (int16_t)(shoulderFlex3.getX()*100);
    }
    
    // read forearm sensor
    if (forearmFlex.available()) {
        sensorMessage->data[4] = (int16_t)(forearmFlex.getX()*100);
    }
    
    // read hand1 sensor
    if (handFlex1.available()) {
        sensorMessage->data[5] = (int16_t)(handFlex1.getX()*100);
    }
    
    // read hand2 sensor
    if (handFlex2.available()) {
        sensorMessage->data[6] = (int16_t)(handFlex2.getX()*100);
    }

#endif
    flexSensor_Char.writeValue(sensorMessage->data, 14);

#ifdef DEBUG
    //stdio_mutex.lock();
    Serial.print("S ");
    Serial.print(millis()-currentMillis);
    Serial.print(" ");
    Serial.println(uxTaskGetStackHighWaterMark(sensorThread));
    //stdio_mutex.unlock();
#endif

    //vTaskDelayUntil( &xSensorLastWakeTime, xSensor_Frequency );
  }
}

/* task to monitor the battery level */
void batteryMonitor_Task(void *pvParameters) {
  // set the update interval for the ble thread
  TickType_t xbatteryMonitor_LastWakeTime;
  const TickType_t xbatteryMonitor_Frequency = 1000;

  // define a variable to store the battery pin value
  uint16_t batteryPin_Value;
  uint8_t battery_Percentage;
  float battery_Voltage;
  float battery_lower_limit = 3.0;
  float battery_upper_limit = 3.7;

  // Initialise the xLastWakeTime variable with the current time.
  xbatteryMonitor_LastWakeTime = xTaskGetTickCount();

  uint32_t ulInterruptStatus; // variable to recive the notification value

  while (1) {
    
    long currentMillis = millis();

    batteryPin_Value = analogRead(A2);
    battery_Voltage = ( batteryPin_Value / 4096.0) * 2 * 3.3;
    battery_Percentage = (battery_Voltage - battery_lower_limit)/(battery_upper_limit - battery_lower_limit) * 100;


    batteryLevel_Char.writeValue(battery_Percentage);

#ifdef DEBUG
    //stdio_mutex.lock();
    Serial.print("BAT ");
    Serial.print(batteryPin_Value);
    Serial.print(" ");
    Serial.print(battery_Percentage);
    Serial.print(" ");
    Serial.print(millis()-currentMillis);
    Serial.print(" ");
    Serial.println(uxTaskGetStackHighWaterMark(batteryMonitorThread));
    //stdio_mutex.unlock();
#endif

    xTaskNotifyWaitIndexed( 0, 0x00, ULONG_MAX, &ulInterruptStatus, portMAX_DELAY );
  }

}

/* task to monitor the MCU Temp */
void MCUTemp_Task(void *pvParameters) {

  // define a variable to store the MCU Temp in
  int16_t MCUTemp_Value;

  uint32_t ulInterruptStatus; // variable to recive the notification value

  // Flag
  uint8_t flag = 0b00000000;

  while (1) {
    
    long currentMillis = millis();

    MCUTemp_Value = (int16_t)round(temperatureRead()*100);
    
    MCUTemp_Char.writeValue(MCUTemp_Value);

#ifdef DEBUG
    //stdio_mutex.lock();
    Serial.print("T ");
    Serial.print(MCUTemp_Value);
    Serial.print(" ");
    Serial.print(millis()-currentMillis);
    Serial.print(" ");
    Serial.println(uxTaskGetStackHighWaterMark(MCUTempThread));
    //stdio_mutex.unlock();
#endif

    xTaskNotifyWaitIndexed( 0, 0x00, ULONG_MAX, &ulInterruptStatus, portMAX_DELAY ); // wait for next call
  }

}


/*
*   @brief interrupt callback to trigger event call
*/
void ISRCallback_Sensor(TimerHandle_t xTimer)
{

  /* Notify the sensor task to take a reading */
  xTaskNotify(sensorThread, 0, eNoAction);
}

void ISRCallback_BatteryMonitor(TimerHandle_t xTimer)
{
  /* Notify the batter monitor task to take a reading */
  xTaskNotify(batteryMonitorThread, 0, eNoAction);
}

void ISRCallback_MCUTemp(TimerHandle_t xTimer)
{
  /* Notify the batter monitor task to take a reading */
  xTaskNotify(MCUTempThread, 0, eNoAction);
}


/*
 * @brief initialise the BLE
*/
void BLE_init() 
{
  if (!BLE.begin()) {
  Serial.println("* Starting Bluetooth® Low Energy module failed!");
    while (1) {
      ;
    }
  }
  
  
  // set the local name peripheral advertises
  BLE.setLocalName("FlexSensorSuit");
  // set the UUID for the service this peripheral advertises:
  BLE.setAdvertisedService(flexSensor_Service);

  // set BLE event handlers
  BLE.setEventHandler( BLEConnected,  blePeripheralConnectHandler);
  BLE.setEventHandler( BLEDisconnected,  blePeripheralDisconnectHandler);


  // add the Flex Sensor Service
  flexSensor_Service.addCharacteristic(flexSensor_Char);

  int16_t flexSensorChar_init[] = {0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006};
  flexSensor_Char.writeValue(flexSensorChar_init, 14);

  flexSensor_Char.setEventHandler( BLESubscribed, sensor_characteristicSubscribed);
  flexSensor_Char.setEventHandler( BLEUnsubscribed, sensor_characteristicUnsubscribed);
  flexSensor_Char.setEventHandler( BLERead, sensor_characteristicRead);

  BLE.addService(flexSensor_Service);

  // Add the Battery Monitor Service
  batteryLevel_Service.addCharacteristic(batteryLevel_Char);

  batteryLevel_Char.addDescriptor(batteryLevel_Descriptor);
  
  uint8_t batteryMonitorChar_init = 0x00;
  batteryLevel_Char.writeValue(batteryMonitorChar_init);

  batteryLevel_Char.setEventHandler( BLESubscribed, batteryMonitor_characteristicSubscribed);
  batteryLevel_Char.setEventHandler( BLEUnsubscribed, batteryMonitor_characteristicUnsubscribed);
  batteryLevel_Char.setEventHandler( BLERead, batteryMonitor_characteristicRead);

  BLE.addService(batteryLevel_Service);

  // add the MCU Temp Service
  MCUTemp_Service.addCharacteristic(MCUTemp_Char);

  int16_t MCUTempChar_init = {0x0000};
  MCUTemp_Char.writeValue(MCUTempChar_init);

  MCUTemp_Char.setEventHandler( BLESubscribed, MCUTemp_characteristicSubscribed);
  MCUTemp_Char.setEventHandler( BLEUnsubscribed, MCUTemp_characteristicUnsubscribed);
  MCUTemp_Char.setEventHandler( BLERead, MCUTemp_characteristicRead);

  BLE.addService(MCUTemp_Service);

  // set the connection interval from between 7.5ms to 4000ms in units of 1.25ms
  BLE.setConnectionInterval(0x0008, 0x0008); // 10ms

  // start advertising
  BLE.advertise();

  Serial.println("Bluetooth® device active, waiting for connections...");
}



// BLE Callbacks

void sensor_characteristicRead(BLEDevice central, BLECharacteristic thisChar) {
  // Read if central asks, queue a new sensorTask event to be excecuted in the bleThread
  xTaskNotify(sensorThread, 0, eNoAction);

#ifdef DEBUG
  Serial.println("Characteristic Read");
#endif
  
}

void batteryMonitor_characteristicRead(BLEDevice central, BLECharacteristic thisChar) {
  // Read if central asks, queue a new sensorTask event to be excecuted in the bleThread
  xTaskNotify(batteryMonitorThread, 0, eNoAction);

#ifdef DEBUG
  Serial.println("Characteristic Read");
#endif
  
}

void sensor_characteristicSubscribed(BLEDevice central, BLECharacteristic thisChar) {
  // central wrote new value to characteristic, update LED
  Serial.print("Sensor Characteristic subscribed. UUID: ");
  Serial.println(thisChar.uuid());

  // change the BLE update interval to every 5ms
  BLE_UPDATE_INTERVAL = 5; //ms
  
  // start sensor timer to run every 10 ms and call the sensor task 
  xTimerStart( sensorTimer, 0 );

}

void sensor_characteristicUnsubscribed(BLEDevice central, BLECharacteristic thisChar) {
  // central wrote new value to characteristic, update LED
  Serial.print("Sensor Characteristic unsubscribed. UUID: ");
  Serial.println(thisChar.uuid());

  // change ble update interval back to every 1 second
  BLE_UPDATE_INTERVAL = 1000; //ms
  
  // stop sensor timer
  xTimerStop(sensorTimer, 0);

}

void batteryMonitor_characteristicSubscribed(BLEDevice central, BLECharacteristic thisChar) {
  // central wrote new value to characteristic, update LED
  Serial.print("Battery Monitor Characteristic subscribed. UUID: ");
  Serial.println(thisChar.uuid());

  // start sensor timer to run every 10000 ms (10 seconds) and call the battery Monitor task 
  xTimerStart( batteryMonitorTimer, 0 );
}

void batteryMonitor_characteristicUnsubscribed(BLEDevice central, BLECharacteristic thisChar) {
  // central wrote new value to characteristic, update LED
  Serial.print("Battery Monitor Characteristic unsubscribed. UUID: ");
  Serial.println(thisChar.uuid());

  xTimerStop( batteryMonitorTimer, 0 );
}


void MCUTemp_characteristicRead(BLEDevice central, BLECharacteristic thisChar) {
  // Read if central asks, queue a new sensorTask event to be excecuted in the bleThread
  xTaskNotify(MCUTempThread, 0, eNoAction);

#ifdef DEBUG
  Serial.println("Characteristic Read");
#endif
  
}

void MCUTemp_characteristicSubscribed(BLEDevice central, BLECharacteristic thisChar) {
  // central wrote new value to characteristic, update LED
  Serial.print("Battery Monitor Characteristic subscribed. UUID: ");
  Serial.println(thisChar.uuid());

  // start sensor timer to run every 10000 ms (10 seconds) and call the battery Monitor task 
  xTimerStart( MCUTempTimer, 0 );
}

void MCUTemp_characteristicUnsubscribed(BLEDevice central, BLECharacteristic thisChar) {
  // central wrote new value to characteristic, update LED
  Serial.print("Battery Monitor Characteristic unsubscribed. UUID: ");
  Serial.println(thisChar.uuid());

  xTimerStop( MCUTempTimer, 0 );
}

void blePeripheralConnectHandler(BLEDevice central) {
  // central connected event handler
  Serial.println("Connected event, central");

  // set color to green
  pixels.fill(0x00FF00);
  pixels.show();
  

}

void blePeripheralDisconnectHandler(BLEDevice central) {
  // central disconnected event handler
  Serial.println("Disconnected event, central");

  // set color to red
  pixels.fill(0xFF0000);
  pixels.show();

  

}


#ifdef ADS_sensor
/*
 * @brief initialise the ADS sensors, there are seven sensors with different addresses
 * 
 * Elbow - 0x10
 * Shoulder1 - 0xE
 * Shoulder2 - 0xC
 * Shoulder3 - 0xA
 * Forearm - 0x16
 * Hand1 - 0x18
 * Hand2 - 0x1A
 * 
 */
void ADS_init()
{
    //elbow ADS begin
  if (elbowFlex.begin(0x10) == false)
  {
    Serial.println(F("Elbow sensor not detected. Check wiring. Freezing..."));
    while (1) {
      ;
    }

  }
  else {
    Serial.println(F("Elbow sensor detected"));
    delay(10);
    // run the sensor at 500Hz
    elbowFlex.setSampleRate(ADS_500_HZ);
    delay(10);
    elbowFlex.run();
    delay(10);
  }
  
  // shoulder1 ADS begin
  if (shoulderFlex1.begin(0xE) == false)
  {
    Serial.println(F("Shoulder1 sensor not detected. Check wiring. Freezing..."));


  }
  else {
    Serial.println(F("Shoulder1 sensor detected"));
    delay(10);
    // run the sensor at 500Hz
    shoulderFlex1.setSampleRate(ADS_500_HZ);
    delay(10);
    shoulderFlex1.run();
    delay(10);
  }
  
  
  // shoulder2 ADS begin
  
  if (shoulderFlex2.begin(0xC) == false)
  {
    Serial.println(F("Shoulder2 sensor not detected. Check wiring. Freezing..."));
    while (1) {
      ;
    }
  }
  else {
    Serial.println(F("Shoulder2 sensor detected"));
    delay(10);
    // run the sensor at 500Hz
    shoulderFlex2.setSampleRate(ADS_500_HZ);
    delay(10);
    shoulderFlex2.run();
    delay(10);
  }
  
  // shoulder3 ADS begin
  if (shoulderFlex3.begin(0xA) == false)
  {
    Serial.println(F("Shoulder3 sensor not detected. Check wiring. Freezing..."));
    while (1) {
      ;
    }
  }
  else {
    Serial.println(F("Shoulder3 sensor detected"));
    delay(10);
    // run the sensor
    shoulderFlex3.setSampleRate(ADS_500_HZ);
    delay(10);
    shoulderFlex3.run();
    delay(10);
    
  }
  
  //forearm ADS begin
  if (forearmFlex.begin(0x16) == false)
  {
    Serial.println(F("Forearm sensor not detected. Check wiring. Freezing..."));
    while (1) {
      ;
    }
  }
  else {
    Serial.println(F("Forearm sensor detected"));
    delay(10);
    // run the sensor at 500Hz
    forearmFlex.setSampleRate(ADS_500_HZ);
    delay(10);
    forearmFlex.run();
    delay(10);
  }
  
  //hand1 ADS begin
  if (handFlex1.begin(0x18) == false)
  {
    Serial.println(F("Hand1 sensor not detected. Check wiring. Freezing..."));
    while (1) {
      ;
    }
  }
  else {
    Serial.println(F("Hand1 sensor detected"));
    delay(10);
    // run the sensor at 500Hz
    handFlex1.setSampleRate(ADS_500_HZ);
    delay(10);
    handFlex1.run();
    delay(10);
  }
  //hand2 ADS begin
  if (handFlex2.begin(0x1A) == false)
  {
    Serial.println(F("Hand2 sensor not detected. Check wiring. Freezing..."));
    while (1) {
      ;
    }
  }
  else {
    Serial.println(F("Hand2 sensor detected"));
    delay(10);
    // run the sensor at 500Hz
    handFlex2.setSampleRate(ADS_500_HZ);
    delay(10);
    handFlex2.run();
    delay(10);
  }
}

void TCA9548A(uint8_t bus) {
  Wire.beginTransmission(0x70);  // TCA9548A address
  Wire.write(1 << bus);          // send byte to select bus
  Wire.endTransmission();
  //Serial.print(bus);
}
#endif





