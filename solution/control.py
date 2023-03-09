import numpy as np


class PID:

    def __init__(self, Kp, Ki, Kd, integral_limit=200, error_limit=200, threshold=10):

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integral_limit = integral_limit
        self.error_limit = error_limit
        self.threshold = threshold

        self.integral = 0.0
        self.last_error = 0.0

    def update(self, error: float) -> float:
        """
        Calculate the PID output value for given reference input and feedback

        Parameters
        ----------
        error : float
            Error between reference input and feedback
        """

        # clip error to avoid windup
        error = np.clip(error, -self.error_limit, self.error_limit)

        # calculate the derivative
        derivative = (error - self.last_error)

        # calculate the integral
        self.integral += error

        # clip the integral
        self.integral = np.clip(
            self.integral, -self.integral_limit, self.integral_limit)

        if abs(error) < self.threshold:
            self.integral -= self.integral / 2

        # save the error for the next time
        self.last_error = error

        # calculate the output
        return int(self.Kp * error + self.Ki * self.integral + self.Kd * derivative)
