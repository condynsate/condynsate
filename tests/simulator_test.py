import time
import numpy as np
from condynsate import Simulator


if __name__ == "__main__":
    sim = Simulator(gravity=(0.0, 0.0, -9.81), dt=0.01)
    fnc = sim._client.getMatrixFromQuaternion()
    fnc((0., 0., 0., 1.))
    sim.terminate()
