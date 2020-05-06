import physicl

class NewtonianKinematicsStep(physicl.Step):
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
			obj.dr = obj.v * sim.dt
			obj.r += obj.dr
