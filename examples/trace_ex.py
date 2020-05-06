import phys
import numpy as np
import phys.light
import phys.newton
import time

sim = phys.Simulation({"cl_on": True, "exit": lambda cond: cond.t >= 0.100})

sim.add_step(3, phys.UpdateTimeStep(lambda c: 0.01))
sim.add_step(1, phys.newton.NewtonianKinematicsStep())
sim.add_step(2, phys.light.ScatterSphericalStep(0.001, 0.001))
sim.add_step(0, phys.light.TracePathMeasureStep(None))

sim.add_objs(phys.light.generate_photons(1000, bins=100, dist="gauss", min=phys.light.E_from_wavelength(200e-9), max=phys.light.E_from_wavelength(700e-9)))

sim.start()

while sim.running:
    time.sleep(0.1)
    print(sim.get_state())

