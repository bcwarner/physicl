import numpy as np
import pyopencl as cl
import time 
import threading

class Step:
	def __init__(self):
		"""
		Initializes the step of the simulation with specific parameters
		"""
		pass

	def __compile_cl__(self, sim):
		"""
		Compiles the OpenCL kernel(s) associated with this step.
		"""
		pass

	def __run_cl(self, sim):
		"""
		Runs this step with compiled OpenCL.
		"""
		pass

	def __run_py(self, sim):
		"""
		Runs this step with native Python code.
		"""
		pass

	def run(self, sim):
		pass

	def terminate(self, sim):
		pass

class UpdateTimeStep(Step):
	"""
	Basic class for updating the time step. Does this according to a function that is passed when this step is initialized.

	"""
	def __init__(self, fn):
		"""
		Updates the simulation according to a function that takes one argument, the simulation, and returns a float indicating the amount of time needed to increment. Usually something as simple as:
		lambda x: 0.001
		"""
		super().__init__()
		self.fn = fn # Must take one argument, which is the simulation.

	def run(self, sim):
		"""
		Called when the simulation is run to update it's current time and change in time.
		"""
		sim.dt = self.fn(sim)
		sim.t += sim.dt
		sim.ts.append(sim.t)

class MeasureStep(Step):
	def __init__(self, out_fn = None):
		"""
		Initializes this step with an empty list. Also establishes what file will be written to when we finish.

		"""
		self.out_fn = out_fn
		self.data = []

	def run(self, sim):
		"""
		Measures some aspect of the simulation at this step.
		"""
		pass

	def terminate(self, sim):
		"""
		If there is an output file, saves to that. Otherwise does nothing.
		"""
		# Potentially remove this and let the user take care of this.
		if self.out_fn == None:
			return
		while True:
			try:
				with open(self.out_fn, "w") as f:
					if type(self.data) == list:
						for x in self.data:
							f.write(", ".join([str(i) for i in list(x)]) + "\n")
					elif type(self.data) == dict:
						for k, v in self.data:
							f.write(", ".join([str(i) for i in list(v)]) + "\n")
				break
			except:
				input("Error saving to '" + self.out_fn + "'. Hit any key to try again.")


class Object:
	"""
	Represents a generic object in the simulation. Inheriting subclasses will add defining features.
	"""
	def __init__(self, attrs={}):
		"""
		Initializes an object with basic attributes, including position, change in last position, velocity, and acceleration.
		Also sets any additional attributes as defined with attrs.
		"""
		self.r = np.array([0] * 3, dtype=np.double)
		self.dr = np.array([0] * 3, dtype=np.double) # change since last step
		self.dv = np.array([0] * 3, dtype=np.double) # change since last step
		self.v = np.array([0] * 3, dtype=np.double)
		self.a = np.array([0] * 3, dtype=np.double)
		for attr, val in attrs.items():
			self.__setattr__(attr, val)

	# Question: what do we do about unknown values?

class Simulation (threading.Thread):
	"""
	Represents a simulation that runs on a separate thread. Contains a number of attributes, holds objects, and applies steps in the order they are given.
	Must be started using start().
	"""
	def __init__(self, params={}):
		"""
		Initializes the simulation with a default number of parameters, which may be overriden using the `params` parameter.
		Also sets up the OpenCL context.
		"""
		threading.Thread.__init__(self)

		self.bounds = np.zeros(3)
		self.cl_on = True
		self.exit = lambda x: len(x.objects) == 0 # Default exit state
		self.state_fn = lambda x: {"objects": len(x.objects), "t": x.t, "dt": x.dt, "run_time": time.time() - x.start_time}
		self.state_need_lock = False
		for attr, val in params.items():
			self.__setattr__(attr, val)
		self.dt = np.double(0)
		self.t = np.double(0)
		self.objects = []
		self.steps = {}
		self.__state_lock = threading.Lock()
		self.running = False
		self.start_time = 0

		if self.cl_on == True:
			self.cl_ctx = cl.create_some_context()
			self.cl_q = cl.CommandQueue(self.cl_ctx)
		else:
			self.cl_ctx = None
			self.cl_q = None

	def add_step(self, idx, step):
		"""
		Adds a new step to the simulation. Index must be unique.
		"""
		if not idx in self.steps:
			 self.steps[idx] = step
		else:
			raise IndexException("Cannot add a step to an existing index.")

	def add_obj(self, obj):
		"""
		Adds an object to the list of current objects in the simulation.
		"""
		self.objects.append(obj)

	def add_objs(self, objs):
		"""
		Adds a list of objects to the list of current objects in the simulation.
		"""
		self.objects.extend(objs)

	def remove_obj(self, obj):
		"""
		Removes an object from this simulation.
		"""
		self.objects.remove(obj)

	def remove_step(self, idx):
		"""
		Removes a step if the simulation is not running. Otherwise raises a RuntimeError.
		"""
		if self.running == False:
			self.steps.pop(idx)
		else:
			raise RuntimeError("Cannot remove a Step while the simulation is running.")

	def get_device_info():
		"""
		Generates a dictionary identifying each platform and all associated devices. Properties for devices are listed if the associated OpenCL feature is supported on that device. 
		"""
		# Iterate through each platform.
		res = {}
		for plat in cl.get_platforms():
			k = plat.get_info(cl.platform_info.NAME)
			v = {}
			
			# Get platform info.

			for x in filter(lambda x: x.isupper(), dir(cl.platform_info)):
				v[x] = plat.get_info(eval("cl.platform_info." + x))

			# Get device info
			for dev in plat.get_devices():
				dk = dev.get_info(cl.device_info.NAME)
				dv = {}
				for x in filter(lambda x: x.isupper(), dir(cl.device_info)):
					try:
						dv[x] = dev.get_info(eval("cl.device_info." + x))
					except:
						pass # Some of these are not available with certain devices, so we ignore them.

				v[dk] = dv

			res[k] = v

		return res

	def run(self):
		"""
		Runs the simulation using threading as defined by threading.Thread.
		Sets t, dt = 0, repeatedly iterates through each step of the simulation until the exit condition is met.
		Then calls each step's terminate function and stops.
		"""
		self.start_time = time.time()
		self.t = 0
		self.dt = 0
		self.ts = []
		self.running = True
		while not self.exit(self):
			self.__state_lock.acquire()
			for idx, step in self.steps.items():
				step.run(self)
			self.__state_lock.release()

		# call terminate on all
		self.__state_lock.acquire()
		for idx, step in self.steps.items():
			step.terminate(self)
		self.run_time = time.time() - self.start_time
		self.running = False
		self.__state_lock.release()

	def set_dev(id):
		"""
		Sets the environment variable PYOPENCL_CTX to the string id so that OpenCL will default to the device with that corresponding id.
		"""


	def get_state(self):
		"""
		Gets the current state of the simulation. Locks if the user states the simulation should lock before getting the state.
		"""
		if self.state_need_lock:
			self.__state_lock.acquire()
		r = self.state_fn(self)
		if self.state_need_lock:
			self.__state_lock.release()
		return r