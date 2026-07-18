import subprocess
import time
from concurrent.futures import ThreadPoolExecutor


# Drone start positions from the SDF file


DRONE_STARTS = {
    "drone_1": (0.0, -3.0, 0.3),
    "drone_2": (0.0,  0.0, 0.3),
    "drone_3": (0.0,  3.0, 0.3),
}



# Intentional collision point


TARGET_X = 3.0
TARGET_Y = 0.0
TARGET_Z = 2

# Timing


MISSION_TIME = 4.0
DT = 0.15

# Velocity limits

MAX_VX = 1
MAX_VY = 0.6
MAX_VZ = 1


def clamp(value, low, high):
    return max(low, min(value, high))


def run_gz_command(args):
    subprocess.run(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def enable_drone(drone_name):
    run_gz_command([
        "gz", "topic",
        "-t", f"/{drone_name}/enable",
        "-m", "gz.msgs.Boolean",
        "-p", "data: true",
    ])


def send_twist(drone_name, vx, vy, vz, yaw_rate=0.0):
    topic = f"/{drone_name}/gazebo/command/twist"

    msg = (
        f"linear: {{x: {vx} y: {vy} z: {vz}}} "
        f"angular: {{z: {yaw_rate}}}"
    )

    run_gz_command([
        "gz", "topic",
        "-t", topic,
        "-m", "gz.msgs.Twist",
        "-p", msg,
    ])


def compute_velocity(start_position):
    sx, sy, sz = start_position

    vx = (TARGET_X - sx) / MISSION_TIME
    vy = (TARGET_Y - sy) / MISSION_TIME
    vz = (TARGET_Z - sz) / MISSION_TIME

    vx = clamp(vx, -MAX_VX, MAX_VX)
    vy = clamp(vy, -MAX_VY, MAX_VY)
    vz = clamp(vz, -MAX_VZ, MAX_VZ)

    return vx, vy, vz


def send_all_parallel(commands):
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        futures = []

        for drone_name, velocity in commands.items():
            vx, vy, vz = velocity
            futures.append(
                executor.submit(send_twist, drone_name, vx, vy, vz)
            )

        for future in futures:
            future.result()


def hover_all(duration=1.0):
    steps = int(duration / DT)

    zero_commands = {
        drone_name: (0.0, 0.0, 0.0)
        for drone_name in DRONE_STARTS
    }

    for _ in range(steps):
        send_all_parallel(zero_commands)
        time.sleep(DT)


try:
    print("SYNCHRONIZED INTENTIONAL COLLISION TEST")
    print(f"Target point: x={TARGET_X}, y={TARGET_Y}, z={TARGET_Z}")
    print(f"Mission time: {MISSION_TIME} seconds")
    print("Important: reset/restart Gazebo before every run.")

    print("Enabling drones...")
    for drone_name in DRONE_STARTS:
        enable_drone(drone_name)

    time.sleep(0.5)

    print("Sending zero velocity before start...")
    hover_all(duration=1.0)

    commands = {}

    print("Computed velocity commands:")
    for drone_name, start_position in DRONE_STARTS.items():
        vx, vy, vz = compute_velocity(start_position)
        commands[drone_name] = (vx, vy, vz)

        print(
            f"{drone_name}: "
            f"vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}"
        )

    print("Starting in 3...")
    time.sleep(1.0)
    print("2...")
    time.sleep(1.0)
    print("1...")
    time.sleep(1.0)
    print("GO")

    start_time = time.perf_counter()

    while time.perf_counter() - start_time < MISSION_TIME:
        send_all_parallel(commands)
        time.sleep(DT)

    print("Collision attempt finished. Hovering...")
    hover_all(duration=2.0)

    print("Done.")

except KeyboardInterrupt:
    print("\nEmergency hover.")
    hover_all(duration=1.0)