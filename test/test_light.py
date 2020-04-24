import pytest

import sys
import os

import phys
import phys.newton
import phys.light
import numpy as np
import time

def rand_ray():
	ret = {}
	ret["s"] = np.array([0] * 3, dtype=np.double)
	ret["v"] = np.array([phys.light.c, 0, 0], dtype=np.double)
	ret["E"] = np.double(1)
	return ret

def sim():	
	s = phys.Simulation(bounds=np.array([1000, 1000, 1000]), cl_on=True, exit=lambda cond: cond.t >= 0.100)
	for i in range(0, 10000):
		s.add_obj(phys.light.PhotonObject(**rand_ray()))

	return s


def test_scatter_spherical():
	"""
	Tests that spherical scattering has the property of having equal proportions of positive and negative coordinates along the x-axis.
	"""
	x = sim()
	x.add_step(0, phys.UpdateTimeStep(lambda s: np.double(0.001)))
	x.add_step(1, phys.newton.NewtonianKinematicsStep())
	x.add_step(2, phys.light.ScatterSphericalStep(np.double(0.001), np.double(0.001)))
	step = phys.light.ScatterSignMeasureStep(None, True)
	x.add_step(3, step)

	x.start()
	x.join()
	error = (np.double(step.data[0][1] * 0.5) - (sum([y[2] for y in step.data]) / len(step.data))) / np.double(step.data[0][1] * 0.5)
	res = np.isclose(error, 0, 0, 0.10)
	print("Scatter spherical test error: " + str(error))
	assert res

def test_scatter_delete():
	"""
	Tests that deletion scattering has the property of being proportional to e ^ -1 at (n * A) ^ -1
	"""
	x = sim()
	x.exit = lambda x: len(x.objects) == 0
	N_i = len(x.objects)
	x.add_step(0, phys.UpdateTimeStep(lambda s: np.double(0.001)))
	x.add_step(1, phys.newton.NewtonianKinematicsStep())
	n = 0.001
	A = 0.001
	x.add_step(2, phys.light.ScatterDeleteStep(np.double(n), np.double(A)))
	step = phys.light.ScatterMeasureStep(None, True, [[1 / (n * A), np.nan, np.nan]])
	x.add_step(3, step)
	x.start()
	x.join()
	
	N_x = sum(step.data[2])
	error = (np.e ** -1 - (N_x / N_i)) / (np.e ** -1)
	res = np.isclose(error, 0, 0, 0.10)
	print("Scatter deletion test error: " + str(error))
	assert res

