import sys
import os

import phys
import phys.newton
import phys.light
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def comp(prep, verbose = False, action=""):
	# Set this so we avoid 
	os.environ["PYOPENCL_CTX"] = "0"
	x = np.floor(10 ** np.linspace(2, 5.5, 9))
	cl_off = []
	cl_on = []
	for i in x:
		print(i, "CL off")
		# CL off
		sim = phys.Simulation(params={"bounds": np.array([1000, 1000, 1000]), "cl_on": False, "exit": lambda cond: len(cond.objects) == 0})

		prep(sim, i)
		sim.start()

		if verbose:
			while sim.running:
				time.sleep(0.1)
				print(sim.get_state())
		else:
			sim.join()
		cl_off.append(sim.run_time)

		# CL on
		print(i, "CL on")
		sim = phys.Simulation(params={"bounds": np.array([1000, 1000, 1000]), "cl_on": True, "exit": lambda cond: len(cond.objects) == 0})

		prep(sim, i)
		sim.start()

		if verbose:
			while sim.running:
				time.sleep(0.1)
				print(sim.get_state())
		else:
			sim.join()
		cl_on.append(sim.run_time)


	plt.plot(x, cl_off, label=action + " (Python)")
	plt.plot(x, cl_on, label=action + " (OpenCL)")
	plt.xlabel("Photon Count")
	plt.ylabel("Runtime (s)")
	plt.legend()
	plt.title("Photon Count vs. Runtime (s)")

	plt.show()

	return (x, cl_off, cl_on)

# Deletion
def del_prep(sim, i):
	sim.add_step(0, phys.UpdateTimeStep(lambda s: np.double(0.001)))
	sim.add_step(1, phys.newton.NewtonianKinematicsStep())
	sim.add_step(2, phys.light.ScatterDeleteStep(np.double(0.001), np.double(0.001)))

	sim.add_objs(phys.light.generate_photons(i, bins=1, dist="constant", min=phys.light.E_from_wavelength(200e-9), max=phys.light.E_from_wavelength(700e-9)))

if input("Test deletion? (y/n)") == "y":
	comp(del_prep, action="Deletion")

# Spherical w/o Wavelength Dep Scattering
def sphere_prep(sim, i):
	sim.exit = lambda cond: cond.t >= 0.5
	sim.add_step(0, phys.UpdateTimeStep(lambda s: np.double(0.001)))
	sim.add_step(1, phys.newton.NewtonianKinematicsStep())
	sim.add_step(2, phys.light.ScatterSphericalStep(np.double(0.001), np.double(0.001)))

	sim.add_objs(phys.light.generate_photons(i, bins=1, dist="constant", min=phys.light.E_from_wavelength(200e-9), max=phys.light.E_from_wavelength(700e-9)))

if input("Test spherical? (y/n)") == "y":
	comp(sphere_prep, action="Spherical")

# Spherical w/ Wavelength Dep Scattering

def sphere_wav_prep(sim, i):
	sim.exit = lambda cond: cond.t >= 0.5
	sim.add_step(0, phys.UpdateTimeStep(lambda s: np.double(0.001)))
	sim.add_step(1, phys.newton.NewtonianKinematicsStep())
	sim.add_step(2, phys.light.ScatterSphericalStep(np.double(0.001), np.double(0.001), wavelength_dep_scattering = True))

	sim.add_objs(phys.light.generate_photons(i, bins=1, dist="constant", min=phys.light.E_from_wavelength(200e-9), max=phys.light.E_from_wavelength(700e-9)))

if input("Test spherical w/ wavelength scattering? (y/n)") == "y":
	comp(sphere_wav_prep, action="Spherical w/ Wavelength Scattering")