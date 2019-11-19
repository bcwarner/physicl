import phys

class NewtonianKinematicsStep(phys.Step):
	"""
	Moves all objects according to Newtonian kinematics.
	"""
	def __init__(self):
		pass

	def run(self, sim):
		"""
		Runs this step in native Python, updating each object's position (r) and change in position (dr)
		"""
		for obj in sim.objects:
			for i in range(0, 3):
				obj.dr[i] = obj.v[i] * sim.dt
				obj.r[i] += obj.dr[i]
