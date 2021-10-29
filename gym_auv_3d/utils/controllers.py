
class PI():
    def __init__(self, Kp=2, Ki=1.5):
        self.Kp = Kp
        self.Ki = Ki
        self._u = 0
        self.accumulated_error = 0

    def u(self, error):
        if abs(self._u) >= 1: self.accumulated_error += 0 # anti-windup
        else: self.accumulated_error += error*0.1
        
        self._u = self.Kp*error + self.Ki*self.accumulated_error
        return self._u


class PID():
    #def __init__(self, Kp=0.15, Ki=0.003, Kd=0.025):
    #def __init__(self, Kp=0.2, Ki=0.004, Kd=0.035):
    def __init__(self, Kp=3.5, Ki=0.05, Kd=0.03):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self._u = 0
        self.accumulated_error = 0
        self.last_error = 0

    def u(self, error):
        if abs(self._u) >= 1: self.accumulated_error += 0 # anti-windup
        else: self.accumulated_error += error*0.1

        derivative_error = (error-self.last_error)/0.1
        self._u = self.Kp*error + self.Ki*self.accumulated_error + self.Kd*derivative_error
        return self._u
