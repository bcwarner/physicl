import sys
import os
sys.path.append(os.getcwd() + "\\..")

import phys
import phys.newton
import phys.light
import numpy as np
import time

sim = phys.Simulation(params={"bounds": np.array([1000, 1000, 1000]), "cl_on": True, "exit": lambda cond: cond.t >= 0.010})

def rand_ray():
	ret = {}
	ret["s"] = np.array([0] * 3, dtype=np.double)
	ret["v"] = np.array([phys.light.c, 0, 0], dtype=np.double)
	ret["E"] = np.double(1)
	return ret

for i in range(0, 10000):
	sim.add_obj(phys.light.PhotonObject(rand_ray()))

sim.add_step(0, phys.UpdateTimeStep(lambda s: np.double(0.001)))
sim.add_step(1, phys.newton.NewtonianKinematicsStep())
sim.add_step(2, phys.light.ScatterDeleteStep(np.double(0.001), np.double(0.001)))
sim.add_step(3, phys.light.ScatterMeasureStep("data_.csv", True, [np.array([0, np.nan, np.nan], dtype=np.double)]))
sim.add_step(4, phys.light.ScatterSignMeasureStep("data_2.csv", True))

sim.start()

while sim.running:
        time.sleep(0.1)
        print(sim.get_state())
