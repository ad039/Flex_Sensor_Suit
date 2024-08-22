#include <Arduino.h>
#include <ArduinoBLE.h>
#include <Wire.h>
#include "SparkFun_Displacement_Sensor_Arduino_Library.h" // Click here to get the library: http://librarymanager/All#SparkFun_Displacement_Sensor

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

//Mutex stdio_mutex;
//Mutex ble_mutex;

// a struct for the sensor task to ble task queue
typedef struct {
  int16_t data[25];
} message_sensor;

//Queue<message_sensor, 50> sensorQueue; 
//MemoryPool<message_sensor, 50> sensorMpool;

//EventQueue eventQueue;

void ble_Task(void *pvParameters);
void sensor_Task(void *pvParameters);


TickType_t BLE_UPDATE_INTERVAL;

////////////////////////////////// Interrupt //////////////////////////////////

TimerHandle_t sensorTimer;

void ISRCallback(TimerHandle_t xTimer);

//std::chrono::milliseconds Ts(10);      //sensor sampling time in ms 

////////////////////////////////// BLE //////////////////////////////////
BLEService flexSensorService("0000fff0-0000-1000-8000-00805f9b34fb");
BLECharacteristic flexSensorCharacteristic("0000fff1-0000-1000-8000-00805f9b34fb", BLERead | BLENotify, 14);


// Advertising parameters should have a global scope. Do NOT define them in 'setup' or in 'loop'
const uint8_t manufactData[4] = {0x01, 0x02, 0x03, 0x04};
const uint8_t serviceData[3] = {0x00, 0x01, 0x02};

uint8_t notification_status = 0;

void characteristicRead(BLEDevice central, BLECharacteristic thisChar);

void characteristicSubscribed(BLEDevice central, BLECharacteristic thisChar);

void characteristicUnsubscribed(BLEDevice central, BLECharacteristic thisChar);

void blePeripheralConnectHandler(BLEDevice central);

void blePeripheralDisconnectHandler(BLEDevice central);

void BLE_init(void);

int counter = 0;

////////////////////////////////// Setup //////////////////////////////////
void setup()
{
  /////////////////// Serial begin ///////////////////
  Serial.begin(115200);

  //while (!Serial) {
  //  ;
  //}
  
  /////////////////// Wire begin ///////////////////
  Wire.begin();
  Wire.setClock(1000000);
  delay(10);

  /////////////////// Initialise ADS sensors ///////////////////
  #ifdef ADS_sensor
  ADS_init();
  #endif
  
  // timer pin for measuering Ts on oscilliscope
  pinMode(LED_BUILTIN, OUTPUT);

  digitalWrite(LED_BUILTIN, LOW);


  /////////////////// Initialise BLE ///////////////////
  BLE_init();

  Serial.println("Starting...");
  // initialise threads to run infinitley

  xTaskCreate(ble_Task, "BLE Task", 2048, NULL, 1, &bleThread);
  xTaskCreate(sensor_Task, "Sensor Task", 2048, NULL, 2, &sensorThread);

  sensorTimer = xTimerCreate("Sensor Timer", 10, pdTRUE, NULL, ISRCallback);

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

    // dispatch any queued sensor read events in the eventQueue
    //eventQueue.dispatch_once();
    // read the data queue if it is available
    /*
    message_sensor *sensorMessage = NULL;
    if (sensorQueue.try_get(&sensorMessage)) {
      ble_mutex.lock();
      flexSensorCharacteristic.writeValue(sensorMessage->data, 14);
      ble_mutex.unlock();

      sensorMpool.free(sensorMessage);
    }
    */
   

#ifdef DEBUG
    //stdio_mutex.lock();
    Serial.print("B ");
    Serial.print(millis()-currentMillis);
    Serial.print(" ");
    Serial.print(temperatureRead());
    Serial.print(" ");
    Serial.println(uxTaskGetStackHighWaterMark(bleThread));
    //stdio_mutex.unlock();
#endif
    // sleep this thread for the update interval minus the millis spent in this thread
    vTaskDelayUntil( &xLastWakeTime, BLE_UPDATE_INTERVAL);
  }
  
}



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
    flexSensorCharacteristic.writeValue(sensorMessage->data, 14);

    //sensorQueue.try_put(sensorMessage);

#ifdef DEBUG
    //stdio_mutex.lock();
    Serial.print("S ");
    Serial.print(millis()-currentMillis);
    Serial.print(" ");
    Serial.println(uxTaskGetStackHighWaterMark(bleThread));
    //stdio_mutex.unlock();
#endif

    //vTaskDelayUntil( &xSensorLastWakeTime, xSensor_Frequency );
  }
}


/*
*   @brief interrupt callback to trigger event call
*/
void ISRCallback(TimerHandle_t xTimer)
{

   /* Notify the sensor task to take a reading */
   xTaskNotify(sensorThread, 0, eNoAction);

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
  BLE.setAdvertisedService(flexSensorService);

  flexSensorService.addCharacteristic(flexSensorCharacteristic);

  // add the service
  BLE.addService(flexSensorService);
  
  flexSensorCharacteristic.writeValue((byte)0x00, 14);

  // set BLE event handlers
  BLE.setEventHandler( BLEConnected,  blePeripheralConnectHandler);
  BLE.setEventHandler( BLEDisconnected,  blePeripheralDisconnectHandler);

  flexSensorCharacteristic.setEventHandler( BLESubscribed, characteristicSubscribed);
  flexSensorCharacteristic.setEventHandler( BLEUnsubscribed, characteristicUnsubscribed);
  flexSensorCharacteristic.setEventHandler( BLERead, characteristicRead);
  
  // set the connection interval from between 7.5ms to 4000ms in units of 1.25ms
  BLE.setConnectionInterval(0x0008, 0x0008); // 10ms

  // start advertising
  BLE.advertise();

  Serial.println("Bluetooth® device active, waiting for connections...");
}



// BLE Callbacks

void characteristicRead(BLEDevice central, BLECharacteristic thisChar) {
  // Read if central asks, queue a new sensorTask event to be excecuted in the bleThread
  xTaskNotify(sensorThread, 0, eNoAction);

#ifdef DEBUG
  Serial.println("Characteristic Read");
#endif
  
}

void characteristicSubscribed(BLEDevice central, BLECharacteristic thisChar) {
  // central wrote new value to characteristic, update LED
  Serial.print("Characteristic subscribed. UUID: ");
  Serial.println(thisChar.uuid());
  // start sensor timer to run every 10 ms and call the sensor task 
  xTimerStart( sensorTimer, 0 );
  //xTaskCreate(sensor_Task, "Sensor Task", 2048, NULL, 2, &sensorThread);

  // initialise event queue to run every Ts ms
  
  //notification_status = 1;
}

void characteristicUnsubscribed(BLEDevice central, BLECharacteristic thisChar) {
  // central wrote new value to characteristic, update LED
  Serial.print("Characteristic unsubscribed. UUID: ");
  Serial.println(thisChar.uuid());
  // stop sensor timer
  xTimerStop(sensorTimer, 0);
  //vTaskDelete(sensorThread);

  //notification_status = 0;
}

void blePeripheralConnectHandler(BLEDevice central) {
  // central connected event handler
  Serial.println("Connected event, central: ");
  //Serial.println(central.address());
  digitalWrite(LED_BUILTIN, HIGH);
  // change the BLE update interval to every 5ms
  BLE_UPDATE_INTERVAL = 5; //ms
  
  //connectionStatus = 1;
}

void blePeripheralDisconnectHandler(BLEDevice central) {
  // central disconnected event handler
  Serial.println("Disconnected event, central: ");
  //Serial.println(central.address());
  digitalWrite(LED_BUILTIN, LOW);
  // change ble update interval back to every 1 second
  BLE_UPDATE_INTERVAL = 1000; //ms
  //connectionStatus = 0;
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





