import sys
import os

import phys
import phys.newton
import phys.light
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sim = phys.Simulation(params={"bounds": np.array([1000, 1000, 1000]), "cl_on": False, "exit": lambda cond: cond.t >= 0.010})


sim.add_objs(phys.light.generate_photons(1000, bins=100, dist="gauss", min=phys.light.E_from_wavelength(200e-9), max=phys.light.E_from_wavelength(700e-9)))

sim.add_step(0, phys.UpdateTimeStep(lambda s: np.double(0.001)))
sim.add_step(1, phys.newton.NewtonianKinematicsStep())
sim.add_step(2, phys.light.ScatterDeleteStep(np.double(0.001), np.double(0.001)))
m1 = phys.light.ScatterMeasureStep("data_.csv", True, [np.array([0, np.nan, np.nan], dtype=np.double)])
sim.add_step(3, m1)
m2 = phys.light.ScatterSignMeasureStep("data_2.csv", True)
sim.add_step(4, m2)

sim.start()

while sim.running:
        time.sleep(0.1)
        print(sim.get_state())

plt.plot(sim.ts, [x[1] for x in m1.data], label="n")
plt.ylabel("Photons")
plt.xlabel("Time (s)")
plt.title("Photon Count vs. Time (s)")

plt.show()