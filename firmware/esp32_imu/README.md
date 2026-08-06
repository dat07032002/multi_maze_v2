# ESP32 BNO086 tilt telemetry

Streams board tilt from a SparkFun BNO086 to the host, for actuator system
identification. The host side is [`tag_vision/hardware/imu.py`](../../tag_vision/hardware/imu.py);
the wire format is defined here and the two must change together.

## Hardware

| Qwiic | Signal | ESP32 |
| --- | --- | --- |
| Red | 3.3 V | 3V3 |
| Black | GND | GND |
| Blue | SDA | GPIO16 |
| Yellow | SCL | GPIO17 |

The board on this rig is an **ESP32-D0WD-V3 rev v3.1**, MAC `28:05:a5:33:26:78`,
on a CP2102 bridge at `/dev/ttyUSB1`. The FEETECH servo bus is the CH340 at
`/dev/ttyUSB0` — never point a flashing tool at it.

> **GPIO16/17 are the PSRAM pins on WROVER modules.** This part reports no
> embedded PSRAM, but external PSRAM would not appear in the eFuse feature list,
> so the firmware reports an I2C failure rather than assuming. If the sensor is
> never found, move to GPIO21/22 and change `PIN_SDA`/`PIN_SCL`.

## Build

Arduino IDE or `arduino-cli`, with:

- board: ESP32 Dev Module
- library: **SparkFun BNO08x Arduino Library**
- the original firmware on this chip was built against arduino-esp32 **3.3.10**
  (ESP-IDF v5.5.4); anything recent will do for this sketch

```bash
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/esp32_imu
arduino-cli upload  --fqbn esp32:esp32:esp32 -p /dev/ttyUSB1 firmware/esp32_imu
```

## Wire format

```
[0xA5][0x5A][type][len][payload...][crc8]
```

CRC-8 is Dallas/Maxim, polynomial `0x31`, over `type`, `len`, and payload — not
the sync word, which is what the host resynchronises on after a dropped byte.

| type | name | payload |
| --- | --- | --- |
| `0x01` | sample | `u32` micros, 4× `i16` quaternion (Q14, i/j/k/real), `u8` accuracy, `u8` seq |
| `0x02` | pong | `u32` echoed token, `u32` device micros |
| `0x03` | status | ASCII, bring-up messages |

Host commands: `'P'` + 4-byte token requests a pong; `'R'` re-enables reports.

### Why these choices

**Binary, not CSV.** This USB link is demonstrably unreliable above 115200 —
flash reads at 921600 and 460800 both stalled at exactly `0x6000`, and one
512 KB chunk needed a retry even at 115200. So the baud is fixed at 115200 and
the frame has to fit inside it. CSV quaternion floats run ~60 B, which at 200 Hz
is 12 kB/s against roughly 11.5 kB/s usable — over budget, and the failure mode
is silent backpressure appearing as jitter in exactly the latency numbers sysid
exists to measure. The 17-byte frame is 3.4 kB/s.

**Game rotation vector, not the fused one.** Game rotation vector is
gravity-referenced but ignores the magnetometer. Two STS3215 servos drawing
current a few centimetres away disturb any magnetometer-fused heading, and tilt
is the one thing that must stay clean. Yaw drift does not matter here.

**Q14 quaternions.** That is the BNO's native scaling, so this converts back to
it rather than adding a second quantisation. The resulting angle resolution is
about 0.007°, asserted in [`test/test_imu.py`](../../test/test_imu.py) and far
below the sensor's own noise.

**Sequence numbers are generated here, not taken from SH-2.** They therefore
detect frames lost between this device and the host — serial overrun, a dropped
byte resyncing the parser — which is what this link is actually prone to. A
report the BNO itself skipped shows up instead as a gap in the micros timestamp,
which is why the timestamp is per-sample rather than inferred from the rate.

## Restoring the original firmware

The firmware that shipped on this board is backed up and **restore-tested** —
written back and read out again with matching hashes:

```
artifacts/esp32_backup/20260806_140245/
  flash_full_4MB.bin    sha256 770a1f9faaba1c33...
  manifest.json         chip, flash, partitions, provenance
```

```bash
esptool --port /dev/ttyUSB1 --baud 115200 write-flash 0x0 flash_full_4MB.bin
```

Use 115200. Higher rates fail on this link.

That image was built elsewhere (`/mnt/ssd2/trungbao-home/…`) and contains no
BNO08x driver, so it is not a predecessor of this sketch — it is someone else's
work that happened to be on the chip.
