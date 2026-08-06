// Stream BNO086 board tilt to the host as compact binary frames.
//
// Why binary rather than CSV. The USB-serial link on this rig is not reliable
// above 115200: flash reads at 921600 and 460800 both stalled at exactly
// 0x6000, and one 512 KB chunk needed a retry even at 115200. So the baud is
// fixed at 115200 and the frame has to fit inside it. A CSV line of quaternion
// floats runs ~60 bytes; at 200 Hz that is 12 kB/s against roughly 11.5 kB/s of
// usable bandwidth -- over budget, and the failure mode is silent backpressure
// that shows up as jitter in exactly the latency numbers sysid is measuring.
// The 17-byte frame below is 3.4 kB/s at 200 Hz.
//
// Why GAME rotation vector rather than the fused one. Game rotation vector is
// gravity-referenced but ignores the magnetometer. Two STS3215 servos drawing
// current a few centimetres away disturb any magnetometer-fused heading, and
// tilt is the one thing here that must stay clean. Yaw drift does not matter:
// the board's alpha/beta come from gravity alone.
//
// Quaternions are sent as Q14 fixed point, which is the BNO's own native
// scaling -- the library hands out floats it converted from Q14, so this
// converts back rather than introducing a new quantisation.

#include <Wire.h>
#include "SparkFun_BNO08x_Arduino_Library.h"

// ---- wiring ---------------------------------------------------------------
// SparkFun Qwiic: red 3V3, black GND, blue SDA, yellow SCL.
//
// GPIO16/17 are the PSRAM pins on WROVER modules. The board here reports as
// ESP32-D0WD-V3 with no embedded PSRAM, so they should be free -- but external
// PSRAM would not show in the eFuse feature list, so this is verified at
// runtime rather than assumed. If I2C never comes up, try 21/22.
static const int PIN_SDA = 16;
static const int PIN_SCL = 17;
static const uint32_t I2C_HZ = 400000;

static const uint32_t SERIAL_BAUD = 115200;

// MILLISECONDS, not microseconds. enableGameRotationVector takes a uint16_t and
// multiplies it by 1000 internally (library .cpp line 840), so passing 5000
// here would ask for one report every five seconds rather than 200 Hz -- a
// mistake that produces a working stream at 1/1000th the intended rate, which
// is exactly the kind that survives a bring-up check and ruins the dynamics.
static const uint16_t REPORT_INTERVAL_MS = 5;  // 200 Hz

// ---- framing --------------------------------------------------------------
// [0xA5][0x5A][type][len][payload...][crc8]
// crc8 covers type, len, and payload -- not the sync word, which is what the
// host uses to resynchronise after a dropped byte.
static const uint8_t SYNC0 = 0xA5;
static const uint8_t SYNC1 = 0x5A;

static const uint8_t TYPE_SAMPLE = 0x01;  // 12 bytes: u32 micros, 4x i16 quat, u8 acc, u8 seq
static const uint8_t TYPE_PONG   = 0x02;  // 8 bytes:  u32 token, u32 esp micros
static const uint8_t TYPE_STATUS = 0x03;  // ASCII, for bring-up messages

BNO08x imu;
static bool imu_ok = false;

// Frame counter, incremented per transmitted sample and wrapping at 256.
//
// Note what this does and does not catch. It is generated here, not taken from
// the SH-2 report header, so it detects frames lost between this device and the
// host -- serial overrun, a dropped byte resyncing the parser -- which is the
// failure this link is actually prone to. It does not detect a report the BNO
// itself skipped. Those show up instead as a gap in the micros timestamp, which
// is why the timestamp is sent per sample rather than assumed from the rate.
static uint8_t seq = 0;

// Recovery pacing for a sensor that failed to start.
static const uint32_t RETRY_INTERVAL_MS = 2000;
static uint32_t last_retry_ms = 0;

// Dallas/Maxim CRC-8 (poly 0x31). Small enough to be worth having: a corrupt
// quaternion that still parses would look like a real board movement.
static uint8_t crc8(const uint8_t *data, size_t len) {
  uint8_t crc = 0x00;
  for (size_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (uint8_t b = 0; b < 8; b++) {
      crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x31) : (uint8_t)(crc << 1);
    }
  }
  return crc;
}

static void sendFrame(uint8_t type, const uint8_t *payload, uint8_t len) {
  uint8_t header[2] = {type, len};
  uint8_t crc = crc8(header, 2);
  // Continue the CRC across the payload without copying it into a buffer.
  for (uint8_t i = 0; i < len; i++) {
    uint8_t byte = payload[i];
    crc ^= byte;
    for (uint8_t b = 0; b < 8; b++) {
      crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x31) : (uint8_t)(crc << 1);
    }
  }
  Serial.write(SYNC0);
  Serial.write(SYNC1);
  Serial.write(type);
  Serial.write(len);
  Serial.write(payload, len);
  Serial.write(crc);
}

static void sendStatus(const char *msg) {
  size_t n = strlen(msg);
  if (n > 200) n = 200;
  sendFrame(TYPE_STATUS, (const uint8_t *)msg, (uint8_t)n);
}

static void put_u32(uint8_t *dst, uint32_t v) {
  dst[0] = (uint8_t)(v);          dst[1] = (uint8_t)(v >> 8);
  dst[2] = (uint8_t)(v >> 16);    dst[3] = (uint8_t)(v >> 24);
}

static void put_i16(uint8_t *dst, int16_t v) {
  dst[0] = (uint8_t)(v);          dst[1] = (uint8_t)(v >> 8);
}

// Float quaternion component -> Q14, saturating. A unit quaternion component
// is within [-1, 1], so 16384 is the natural scale and saturation should never
// fire; if it does, something upstream is wrong and clipping is better than
// wrapping to the opposite sign.
static int16_t to_q14(float v) {
  long scaled = lroundf(v * 16384.0f);
  if (scaled >  32767) scaled =  32767;
  if (scaled < -32768) scaled = -32768;
  return (int16_t)scaled;
}

static bool enableReports() {
  return imu.enableGameRotationVector(REPORT_INTERVAL_MS);
}

// Free a bus that a device is holding low.
//
// The BNO086 has its own supply and no reset line wired here, so resetting the
// ESP32 does not reset the sensor. If it is stuck partway through a transaction
// it keeps SDA low, and every subsequent begin() fails while the device still
// ACKs its address -- which is exactly what happened after the rig was
// unplugged and reconnected: an I2C scan found 0x4B present, yet begin() would
// not complete. Clocking SCL lets the device finish the byte it was in.
static void recoverBus() {
  pinMode(PIN_SDA, INPUT_PULLUP);
  pinMode(PIN_SCL, OUTPUT);
  for (int i = 0; i < 9 && digitalRead(PIN_SDA) == LOW; i++) {
    digitalWrite(PIN_SCL, LOW);  delayMicroseconds(5);
    digitalWrite(PIN_SCL, HIGH); delayMicroseconds(5);
  }
  digitalWrite(PIN_SCL, HIGH);
}

// Try to bring the sensor up. Safe to call repeatedly.
static bool tryBegin() {
  recoverBus();
  Wire.begin(PIN_SDA, PIN_SCL, I2C_HZ);
  delay(50);
  if (!(imu.begin(0x4B, Wire) || imu.begin(0x4A, Wire))) return false;
  return enableReports();
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(200);

  // Two addresses exist in the wild: 0x4B is the SparkFun default, 0x4A is the
  // ADR-jumper alternative. tryBegin attempts both after recovering the bus.
  imu_ok = tryBegin();
  sendStatus(imu_ok ? "BNO086 ready: game rotation vector @ 200 Hz"
                    : "BNO08x did not start; will keep retrying. If it never "
                      "comes up, check 3V3/GND/SDA=16/SCL=17 -- and note a "
                      "WROVER uses 16/17 for PSRAM, so try 21/22.");
}

void loop() {
  // Host commands. 'P' + 4-byte token -> PONG, so the host can measure link
  // round-trip and de-bias step_latency_s. Without this, every latency number
  // silently includes an unknown amount of USB-serial transport delay.
  if (Serial.available() >= 1) {
    int cmd = Serial.read();
    if (cmd == 'P') {
      uint8_t token[4];
      size_t got = Serial.readBytes(token, 4);
      if (got == 4) {
        uint8_t payload[8];
        memcpy(payload, token, 4);
        put_u32(payload + 4, micros());
        sendFrame(TYPE_PONG, payload, 8);
      }
    } else if (cmd == 'R') {
      // Re-enable reports without a power cycle: the BNO stops reporting after
      // certain resets, and this recovers it without disturbing the rig. The
      // board must not be bumped while the level zero is in force.
      if (imu_ok) sendStatus(enableReports() ? "reports re-enabled"
                                             : "re-enable failed");
    }
  }

  // Retry rather than giving up for the lifetime of the boot. The sensor can
  // be left latched by a replug, and since resetting the ESP32 does not reset
  // it, a one-shot begin() in setup() would strand the stream permanently --
  // which is what it did.
  if (!imu_ok) {
    if (millis() - last_retry_ms >= RETRY_INTERVAL_MS) {
      last_retry_ms = millis();
      imu_ok = tryBegin();
      sendStatus(imu_ok ? "BNO086 recovered"
                        : "BNO08x still unavailable, retrying");
    }
    delay(20);
    return;
  }

  if (imu.wasReset()) {
    // A reset silently drops the report subscription. Saying so matters: the
    // alternative is a host that sees the stream stop and cannot tell whether
    // the board is still or the sensor is gone.
    sendStatus("BNO08x reset detected, re-enabling reports");
    enableReports();
  }

  if (imu.getSensorEvent()) {
    if (imu.getSensorEventID() == SENSOR_REPORTID_GAME_ROTATION_VECTOR) {
      uint8_t payload[12];
      put_u32(payload + 0, micros());
      put_i16(payload + 4, to_q14(imu.getGameQuatI()));
      put_i16(payload + 6, to_q14(imu.getGameQuatJ()));
      put_i16(payload + 8, to_q14(imu.getGameQuatK()));
      put_i16(payload + 10, to_q14(imu.getGameQuatReal()));
      // Payload is 12 bytes of quaternion + timestamp; accuracy and sequence
      // ride in the two trailing bytes below.
      uint8_t full[14];
      memcpy(full, payload, 12);
      // getQuatAccuracy, not getGameQuatAccuracy -- the latter does not exist.
      // The library keeps one shared quatAccuracy field for both the fused and
      // game rotation vectors.
      full[12] = (uint8_t)imu.getQuatAccuracy();
      full[13] = seq++;
      sendFrame(TYPE_SAMPLE, full, 14);
    }
  }
}
