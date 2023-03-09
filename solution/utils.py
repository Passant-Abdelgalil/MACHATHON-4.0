def translate(value: float, leftMin: float, leftMax: float, rightMin: float, rightMax: float) -> float:
    """
    Re-maps a number from one range to another.

    Parameters
    ----------
    value : float
        The incoming value to be converted
    leftMin : float
        The lower bound of the value's current range
    leftMax : float
        The upper bound of the value's current range
    rightMin : float
        The lower bound of the value's target range
    rightMax : float
        The upper bound of the value's target range
    """
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)
