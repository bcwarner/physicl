import phys

class NewtonianKinematicsStep(phys.Step):
	def __init__(self):
		pass

	def run(self, sim):
		for obj in sim.objects:
			for i in range(0, 3):
				obj.dr[i] = obj.v[i] * sim.dt
				obj.r[i] += obj.dr[i]
