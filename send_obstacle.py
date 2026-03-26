import sys
from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega as mavlink2
import time

master = mavutil.mavlink_connection(
    'udp:0.0.0.0:14552',
    input=False,
    dialect='ardupilotmega'
)
master.mav.srcSystem = 255
print("Connected to ArduPilot!")

print("Sending obstacle 2m directly ahead...")
while True:
    distances = [65535] * 72
    distances[0] = 200

    msg = mavlink2.MAVLink_obstacle_distance_message(
        int(time.time() * 1e6),  # time_usec
        mavlink2.MAV_DISTANCE_SENSOR_UNKNOWN,  # sensor_type
        distances,  # distances array
        5,  # increment
        20,  # min_distance
        1500  # max_distance
    )
    master.mav.send(msg)
    print("Obstacle sent! 200cm ahead")
    time.sleep(0.1)