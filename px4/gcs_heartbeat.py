#!/usr/bin/env python3
"""Minimal MAVLink ground station: sends heartbeats, and sets SITL parameters.

PX4 refuses to arm while its "No connection to the GCS" health check fails,
which happens whenever nothing is talking MAVLink to it. In SITL there is no
real ground station, so this script sends the one message PX4 needs: a GCS
heartbeat, once per second.

It also applies the parameters a SITL run needs. MAVROS was doing this before,
but it does not work with several drones: MAVROS downloads the autopilot's
whole parameter list (about 800 of them) before it will set any, and with
three instances running that download does not finish. It reports "310 params
still missing" and then answers every attempt with "Unknown parameter to set".
The parameter protocol is one request and one reply, so it is simpler to speak
it directly.

The parameter type is read from the autopilot first instead of assumed,
because PX4 reads the value field differently depending on it.

Usage:
  ./gcs_heartbeat.py                 # default PX4 SITL port 18570
  ./gcs_heartbeat.py --port 18570
  ./gcs_heartbeat.py --port 18570 --set CBRK_SUPPLY_CHK=894281 --set COM_DISARM_PRFLT=0
"""

import argparse
import struct
import time

from pymavlink import mavutil

SYSTEM_ID = 255      # conventional ground station system id
COMPONENT_ID = 190   # MAV_COMP_ID_MISSIONPLANNER

PARAM_REPLY_TIMEOUT_S = 5.0
PARAM_ATTEMPTS = 8

# MAVLink carries every parameter value in a float field. PX4 treats that field
# as a real number only for REAL32; for the integer types it reinterprets the
# four bytes, so an integer has to be packed bitwise rather than converted.
TYPE_FORMATS = {
    mavutil.mavlink.MAV_PARAM_TYPE_INT8: "<b",
    mavutil.mavlink.MAV_PARAM_TYPE_UINT8: "<B",
    mavutil.mavlink.MAV_PARAM_TYPE_INT16: "<h",
    mavutil.mavlink.MAV_PARAM_TYPE_UINT16: "<H",
    mavutil.mavlink.MAV_PARAM_TYPE_INT32: "<i",
    mavutil.mavlink.MAV_PARAM_TYPE_UINT32: "<I",
}


def encode_param_value(value, param_type):
    if param_type not in TYPE_FORMATS:
        return float(value)
    raw = struct.pack(TYPE_FORMATS[param_type], int(float(value)))
    return struct.unpack("<f", raw.ljust(4, b"\x00"))[0]


def decode_param_value(msg):
    if msg.param_type not in TYPE_FORMATS:
        return float(msg.param_value)
    fmt = TYPE_FORMATS[msg.param_type]
    raw = struct.pack("<f", msg.param_value)
    return struct.unpack(fmt, raw[:struct.calcsize(fmt)])[0]


def send_heartbeat(conn):
    conn.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0, 0, 0,
    )


def await_param(conn, name, timeout):
    """Wait for the autopilot's report of one named parameter."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = conn.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if msg is not None and msg.param_id.strip("\x00") == name:
            return msg
    return None


def apply_params(conn, settings):
    everything_applied = True

    for name, value in settings:
        applied = False
        for _ in range(PARAM_ATTEMPTS):
            # Keep the link alive: PX4 drops a ground station that goes quiet.
            send_heartbeat(conn)

            conn.mav.param_request_read_send(
                conn.target_system, conn.target_component, name.encode(), -1)
            current = await_param(conn, name, PARAM_REPLY_TIMEOUT_S)
            if current is None:
                continue

            conn.mav.param_set_send(
                conn.target_system, conn.target_component, name.encode(),
                encode_param_value(value, current.param_type),
                current.param_type)
            confirmed = await_param(conn, name, PARAM_REPLY_TIMEOUT_S)
            if confirmed is None:
                continue

            written = decode_param_value(confirmed)
            if abs(float(written) - float(value)) <= 1e-3:
                print(f"PARAM OK {name} = {written}", flush=True)
                applied = True
                break

        if not applied:
            print(f"PARAM FAIL {name}", flush=True)
            everything_applied = False

    print("PARAMS DONE" if everything_applied else "PARAMS INCOMPLETE",
          flush=True)
    return everything_applied


def parse_setting(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            f"expected NAME=VALUE, got '{text}'")
    name, value = text.split("=", 1)
    return name.strip(), value.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18570,
                    help="PX4's MAVLink 'Normal' mode UDP port (default 18570)")
    ap.add_argument("--rate", type=float, default=1.0,
                    help="heartbeats per second (default 1)")
    ap.add_argument("--set", type=parse_setting, action="append", default=[],
                    metavar="NAME=VALUE", dest="settings",
                    help="parameter to apply once the autopilot answers; "
                         "may be given several times")
    args = ap.parse_args()

    conn = mavutil.mavlink_connection(
        f"udpout:{args.host}:{args.port}",
        source_system=SYSTEM_ID,
        source_component=COMPONENT_ID,
    )

    print(f"Sending GCS heartbeats to {args.host}:{args.port} at {args.rate} Hz.",
          flush=True)
    print("PX4 should report 'Ready for takeoff!' within a few seconds.",
          flush=True)
    print("Ctrl+C to stop.", flush=True)

    try:
        if args.settings:
            # PX4 only starts talking once it has heard from us, and its
            # heartbeat is what tells us which system to address.
            deadline = time.time() + 120.0
            while conn.target_system == 0 and time.time() < deadline:
                send_heartbeat(conn)
                conn.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)

            if conn.target_system == 0:
                print("PARAMS INCOMPLETE (no heartbeat from the autopilot)",
                      flush=True)
            else:
                print(f"Autopilot is system {conn.target_system}, "
                      f"component {conn.target_component}.", flush=True)
                apply_params(conn, args.settings)

        while True:
            send_heartbeat(conn)
            time.sleep(1.0 / args.rate)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
