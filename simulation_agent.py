import time
import cv2
from machathon_judge import Simulator, Judge
import numpy as np
from solution.perception import get_center_lane
from solution.perception import get_center
from solution.control import PID
from solution.utils import translate

previous_twist = 0


class FPSCounter:
    def __init__(self):
        self.frames = []

    def step(self):
        self.frames.append(time.monotonic())

    def get_fps(self):
        n_seconds = 5

        count = 0
        cur_time = time.monotonic()
        for f in self.frames:
            if cur_time - f < n_seconds:  # Count frames in the past n_seconds
                count += 1

        return count / n_seconds


def self_drive_car(simulator: Simulator):
    """
    Parameters
    ----------
    simulator : Simulator
        The simulator object to control the car
        The only functions that should be used are:
        - get_image()
        - set_car_steering()
        - set_car_velocity()
        - get_state()
    """
    global fps_counter, steering_controller, velocity_controller, previous_twist

    fps_counter.step()

    # Get the image and show it
    img = simulator.get_image()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    fps = fps_counter.get_fps()

    # draw fps on image
    cv2.putText(
        img,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("image", img)
    cv2.waitKey(1)

    lane_image = get_center_lane(img)
    cxm, cxy = get_center(lane_image)

    cv2.imshow("processed image", lane_image)
    cv2.waitKey(1)

    if cxm != None:
        #  BEGIN CONTROL
        err = (320 - cxm)
        twist = steering_controller.step(err)
        twist = translate(twist, -40, 40, -np.pi*.1, np.pi*.08)

        if cxy != None:
            err += abs(240 / 2 - cxy) * 100

        speed = velocity_controller.step(err)
        speed = translate(abs(speed), 0, 100, 45, -7)

    else:
        twist = previous_twist * -1.1
        speed = -0.7

    previous_twist = twist

    simulator.set_car_steering(twist)
    simulator.set_car_velocity(speed)


if __name__ == "__main__":
    # Initialize any variables needed

    steering_controller = PID(0.2, 0, 0.2, 200, 200, 5)
    velocity_controller = PID(0.4, 0.0005, -0.3, 300, 200, 10)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    fps_counter = FPSCounter()

    # You should modify the value of the parameters to the judge constructor
    # according to your team's info
    judge = Judge(team_code="3e9XrwPGz", zip_file_path="classical_agent_2.zip")

    # Pass the function that contains your main solution to the judge
    judge.set_run_hook(self_drive_car)

    # Start the judge and simulation
    judge.run(send_score=False, verbose=True)
