#!/usr/bin/env python3
"""Minimal MAVLink heartbeat sender, standing in for a ground control station.

PX4 refuses to arm while its "No connection to the GCS" health check fails,
which happens whenever nothing is talking MAVLink to it. In SITL there is no
real ground station, so this script sends the one message PX4 needs: a GCS
heartbeat, once per second.

MAVROS also satisfies this check, so once the MAVROS bridge is running this
script is redundant. It stays because it is useful on its own for testing PX4
without any ROS 2 involvement.

Usage:
  ./gcs_heartbeat.py                 # default PX4 SITL port 18570
  ./gcs_heartbeat.py --port 18570
"""

import argparse
import time

from pymavlink import mavutil

SYSTEM_ID = 255      # conventional ground station system id
COMPONENT_ID = 190   # MAV_COMP_ID_MISSIONPLANNER


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18570,
                    help="PX4's MAVLink 'Normal' mode UDP port (default 18570)")
    ap.add_argument("--rate", type=float, default=1.0,
                    help="heartbeats per second (default 1)")
    args = ap.parse_args()

    conn = mavutil.mavlink_connection(
        f"udpout:{args.host}:{args.port}",
        source_system=SYSTEM_ID,
        source_component=COMPONENT_ID,
    )

    print(f"Sending GCS heartbeats to {args.host}:{args.port} at {args.rate} Hz.")
    print("PX4 should report 'Ready for takeoff!' within a few seconds.")
    print("Ctrl+C to stop.")

    try:
        while True:
            conn.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0,
            )
            time.sleep(1.0 / args.rate)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
