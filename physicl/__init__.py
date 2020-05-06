import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import time 
import threading
import re
import numbers
import copy
import math

class MeasurementError(ArithmeticError):
	pass


# Question: Add decorator for unit validation?


class Measurement(np.ndarray):
	# Dictionary of tuples 
	
	# https://www.bipm.org/en/measurement-units/
	# https://www.bipm.org/utils/common/pdf/si-brochure/SI-Brochure-9-EN.pdf
	# How do SI units translate to code units? abbrev => code unit, scale (CASE SENSITIVE)

	code_scale = {"s": [1, ("T", 1)],
					"m": [1, ("L", 1)],
					"kg": [1, ("M", 1)],
					"A": [1, ("I", 1)],
					"K": [1, ("Th", 1)],
					"mol": [1, ("N", 1)],
					"cd": [1, ("J", 1)]
					}

	unit_scale = {	# Time Units
					"s": [1, ("s", 1)],

					# Length units
					"m": [1, ("m", 1)],

					# Mass units
					"kg": [1, ("kg", 1)],

					# Electrical units
					"A": [1, ("A", 1)],

					# Heat units
					"K": [1, ("K", 1)],

					# Substance units
					"mol": [1, ("mol", 1)],

					# Luminance units
					"cd": [1, ("cd", 1)],
					

					# Other derived units
					"N" : [1, ("kg", 1), ("m", 1), ("s", -2)],
					"Pa" : [1, ("kg", 1), ("m", -1), ("s", -2)],

					"J" : [1, ("N", 1), ("m", 1)],
					"W" : [1, ("kg", 1), ("m", 2), ("s", -3)],
					"C" : [1, ("A", 1), ("s", 1)],
					"V" : [1, ("W", 1), ("A", -1)],
					"F" : [1, ("C", 1), ("V", -1)],
					"Ohm" : [1, ("V", 1), ("A", 1)],
					"Wb" : [1, ("V", 1), ("s", 1)],
					"T" : [1, ("Wb", 1), ("m", -2)],
					"H" : [1, ("Wb", 1), ("A", -1)],
					# Require Kelvins in all cases
					"lm" : [1, ("cd", 1)], # Ignoring unitless measures like radians and steradians.
					"Bq" : [1, ("s", -1)],
					"Gy" : [1, ("m", 2), ("s", -2)],
					"Sv" : [1, ("m", 2), ("s", -2)],
					"kat" : [1, ("mol", 1), ("s", -1)],


					# "Non-SI units accepted for use with the SI units"
					"min" : [60, ("s", 1)],
					"h" : [3600, ("s", 1)],
					"d" : [86400, ("s", 1)],

					"au" : [149597870700, ("m", 1)],

					"ha" : [10 ** 4, ("m", 2)],

					"L" : [10 ** -3, ("m", 3)],

					"t" : [10 ** 3, ("kg", 1)],
					"Da" : [1.6605390666050e-27, ("kg", 1)],

					"eV" : [1.602176634e-19, ("J", 1)]

					}

	unit_match = re.compile("""((?P<u>[a-zA-Z]*)\s*(\*\*|\^)\s*(?P<p>-?\d*))""")

	# Return new unit, new scaling factor.
	# Question: fractional dimensions?
	def __intermediate_to_base(unit, power):
		# u ** p <=> (subunit ** power * ... * subunit ** power) ** power
		scale = Measurement.unit_scale[unit][0] ** power
		units = []
		for subunit in Measurement.unit_scale[unit][1:]:
			# Recurse downward until the most basic units are hit.
			if subunit[0] in Measurement.code_scale: # Is it a base unit?
				units.append((subunit[0], subunit[1] * power))
			else:
				units.extend(Measurement.__intermediate_to_base(subunit[0], subunit[1] * power)[1:])

		# Deal with duplicate units?

		return [scale] + units

	def __base_to_code(base):
				# u ** p <=> (subunit ** power * ... * subunit ** power) ** power
		scale = base[0]
		units = []
		for unit in base[1:]:
			scale *= Measurement.code_scale[unit[0]][0] ** unit[1]
			units.append((Measurement.code_scale[unit[0]][1][0], Measurement.code_scale[unit[0]][1][1] * unit[1]))

		return [scale] + units

	# Assumption: Code scale is set at the beginning.
	def set_code_scale(base_unit, new_scale):
		Measurement.code_scale[base_unit][0] = new_scale
		

	def reset_code_scale(base_unit):
		Measurement.set_code_scale(base_unit, 1)

	def __new__(cls, raw_value, units):
		x = None
		if isinstance(raw_value, list):
			x = np.asarray([x.__unscaled__() if isinstance(x, Measurement) else x for x in raw_value], dtype=np.double).view(cls)
		else:
			x = np.asarray(raw_value, dtype=np.double).view(cls)
		x.__scale__(units)
		return x

	def __scale__(self, units):
		self.scale = np.double(1)

		units_raw = Measurement.unit_match.findall(units)
		self.units = {}
		self.original_units = {}
		for x in units_raw:
			power = int(x[3])
			base = Measurement.__intermediate_to_base(x[1], power)
			code = Measurement.__base_to_code(base)
			self.scale *= code[0]

			if x[1] not in self.original_units:
				self.original_units[x[1]] = power
			else:
				self.original_units[x[1]] += power

			for conv_unit in code[1:]:
				if conv_unit[0] not in self.units:
					self.units[conv_unit[0]] = conv_unit[1]
				else:
					self.units[conv_unit[0]] += conv_unit[1]

		for i in range(0, self.size):
			self.flat[i] *= self.scale

	def __unscaled__(self):
		x = np.copy(self).view(np.ndarray)
		try:
			for i in range(0, x.size):
				x.flat[i] /= self.scale
		except:
			print("Error: " + str(x))
		return x

	def __array_finalize__(self, obj):
		# Convert raw units to code units. Calculate new scale.
		if obj is None: return
		if getattr(self, "units", None) != None:
			self.scale(self.units)

	def value(self):
		return self.__unscaled__()

	def unitstr(self):
		try:
			return " ".join([k + "**" + str(v) for k, v in self.original_units.items()])
		except:
			return "" # Phantom Measurement objects

	def fstr(self):
		return str(float(self))

	def valstr(self):
		return str(self.value())

	def __str__(self):
		return str(self.view(np.ndarray)).upper()

	def __format__(self, fmt):
		return super().__format__(fmt).upper()

	def __repr__(self):
		return str(self.value()) + " " + self.unitstr()

	def __deepcopy__(self, memo):
		res = np.copy(self).view(Measurement)
		res.units = copy.deepcopy(self.units, memo)
		res.original_units = copy.deepcopy(self.original_units, memo)
		res.scale = self.scale
		return res

	def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
		# Coerce to first units for addition.
		units_idx = 0 if isinstance(inputs[0], Measurement) else 1
		inputs_conv = [item if (isinstance(item, Measurement) and "units" in dir(item)) else Measurement(item, inputs[units_idx].unitstr()) for item in inputs]
		inputs_nd = [x.view(np.ndarray) for x in inputs_conv]
		if "out" in kwargs:
			kwargs["out"] = tuple(item.view(np.ndarray) for item in kwargs["out"])

		res = None
		if ufunc.__name__ in ["add", "subtract"]:
			#for x in inputs_conv[1:]:
			#	if x.units != inputs[0].units:
			#		raise MeasurementError("Unit mismatch for " + str(x) + ", " + str(inputs[0]))
			# Ignore this for now, as a lot of code uses plain numbers.
			res = np.asarray(super(Measurement, self).__array_ufunc__(ufunc, method, *inputs_nd, **kwargs)).view(Measurement)
			res.scale = inputs_conv[0].scale
			res.units = copy.deepcopy(inputs_conv[0].units)
			res.original_units = copy.deepcopy(inputs_conv[0].original_units)
		elif ufunc.__name__ in ["multiply", "divide", "true_divide", "floor_divide"]:
			new_units = {}
			new_original_units = {}
			for unit, power in inputs_conv[0].units.items():
				new_units[unit] = power

			for unit, power in inputs_conv[1].units.items():
				if unit not in new_units:
					new_units[unit] = power * (-1 if ufunc.__name__ in ["divide", "true_divide", "floor_divide"] else 1)
				else:
					new_units[unit] += power * (-1 if ufunc.__name__ in ["divide", "true_divide", "floor_divide"] else 1)

			for unit, power in inputs_conv[0].original_units.items():
				new_original_units[unit] = power

			for unit, power in inputs_conv[1].original_units.items():
				if unit not in new_units:
					new_original_units[unit] = power * (-1 if ufunc.__name__ in ["divide", "true_divide", "floor_divide"] else 1)
				else:
					new_original_units[unit] += power * (-1 if ufunc.__name__ in ["divide", "true_divide", "floor_divide"] else 1)

			# Calculate new scale
			new_scale = inputs_conv[0].scale * (inputs_conv[1].scale ** (-1 if ufunc.__name__ in ["divide", "true_divide", "floor_divide"] else 1))
			res = Measurement(np.asarray(super(Measurement, self).__array_ufunc__(ufunc, method, *inputs_nd, **kwargs)), "")
			res.scale = new_scale
			res.units = new_units
			res.original_units = new_original_units
		elif ufunc.__name__ in ["power", "square", "sqrt"]:
			if ufunc.__name__ == "power":
				power = inputs_nd[1]
			elif ufunc.__name__ == "square":
				power = 2
			elif ufunc.__name__ == "sqrt":
				power = 1/2
			res = np.asarray(super(Measurement, self).__array_ufunc__(ufunc, method, *inputs_nd, **kwargs)).view(Measurement)
			res.units = copy.deepcopy(inputs_conv[0].units) # Unitless if being exponentiated to?
			res.original_units = copy.deepcopy(inputs_conv[0].original_units)
			res.scale = (inputs_conv[0].scale ** power) if isinstance(inputs_conv[0], Measurement) else 1

			for unit, value in self.units.items():
				res.units[unit] *= power
			for unit, value in self.original_units.items():
				res.original_units[unit] *= power
		else:
			res = np.asarray(super(Measurement, self).__array_ufunc__(ufunc, method, *inputs_nd, **kwargs)).view(Measurement)
			res.units = copy.deepcopy(inputs_conv[0].units)
			res.original_units = copy.deepcopy(inputs_conv[0].original_units)
			res.scale = inputs_conv[0].scale

		if "out" in kwargs:
			tup = []
			for item in kwargs["out"]:
				tup.append(item.view(Measurement))
				tup[-1].units = res.units
				tup[-1].scale = res.scale
				tup[-1].original_units = res.original_units
		return res

	def rescale(self):
		# Question: Should we allow multiple simultaneous scales, or one singular scale that is present throughout the program.
		pass

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
		sim.ts.append(copy.deepcopy(sim.t))

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
	def __init__(self, **kwargs):
		"""
		Initializes an object with basic attributes, including position, change in last position, velocity, and acceleration.
		Also sets any additional attributes as defined with attrs.
		"""
		self.r = Measurement([0] * 3, "m**1")
		self.dr = Measurement([0] * 3, "m**1") # change since last step
		self.dv = Measurement([0] * 3, "m**1 s**-2") # change since last step
		self.v = Measurement([0] * 3, "m**1 s**-1")
		self.a = Measurement([0] * 3, "m**1 s**-2")
		for attr, val in kwargs.items():
			self.__setattr__(attr, val)

	# Question: what do we do about unknown values?

class Simulation (threading.Thread):
	"""
	Represents a simulation that runs on a separate thread. Contains a number of attributes, holds objects, and applies steps in the order they are given.
	Must be started using start().
	"""
	def __init__(self, **kwargs):
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
		for attr, val in kwargs.items():
			self.__setattr__(attr, val)
		self.dt = Measurement(np.double(0), "s**1")
		self.t = Measurement(np.double(0), "s**1")
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

class CLInput:
	types = ["obj", "obj_def", "obj_action", "const", "other"]
	def __init__(self, **kwargs):
		self.name = kwargs["name"]
		self.type = kwargs["type"]
		if kwargs["type"] == "obj":
			self.code = "self." + self.name + ".append(obj." + kwargs["obj_attr"] + ")"
			self.ctype = "double" if "ctype" not in kwargs else kwargs["ctype"]
		elif kwargs["type"] == "obj_def":
			self.code = "self." + self.name + ".append(" + kwargs["obj_def"] + ")"
			self.ctype = "double" if "ctype" not in kwargs else kwargs["ctype"]
		elif kwargs["type"] == "obj_track":
			self.code = "self." + self.name + ".append(" + kwargs["obj_track"] + ")"
		elif kwargs["type"] in ["obj_action", "other"]:
			self.code = kwargs["code"]
		elif kwargs["type"] == "const":
			self.const_value = kwargs["const_value"]
			self.ctype = "double" if "ctype" not in kwargs else kwargs["ctype"] 

class CLOutput:
	def __init__(self, **kwargs):
		self.name = kwargs["name"]
		self.ctype = kwargs["ctype"] if "ctype" in kwargs else "double"

class CLProgram:
	def __init__(self, sim, name, kernel_code):
		self.variables = {}
		self.sim = sim
		self.prog = None
		self.prog_name = name
		self.prep_metadata = []
		self.output_metadata = []
		self.kernel_code = kernel_code

	# Steps:
	# Copy data over into arrays
	# Copy arrays over into memory
	# Run
	# Retrieve the results and store them.

	def build_kernel(self):
		kernel_outer = "__kernel void " + self.prog_name + "("
		cat = []
		for item in filter(lambda x: x.type in ["obj", "obj_def", "const"], self.prep_metadata):
			if item.type == "obj" or item.type == "obj_def":
				cat.append("__global " + item.ctype  + " *" + item.name)
			elif item.type == "const":
				cat.append(item.ctype + " " + item.name)

		cat.extend(["__global " + item.ctype + " *" + item.name for item in self.output_metadata])
		kernel_outer += ", ".join(cat) + "){"
		# TODO: Process the kernel code.
		kernel = kernel_outer + self.kernel_code + "}"

		self.prog = cl.Program(self.sim.cl_ctx, kernel).build()
		#for 


	# Move the data into memory
	def run(self):
		# Import necessary libraries?

		initial = ""
		obj_collection = """for obj in self.sim.objects:"""
		other = ""
		np_initial = ""
		np_device = ""
		for item in self.prep_metadata:
			if item.type in ["obj", "obj_def"]:
				initial += "self." + item.name + " = []\n"
				np_initial += "self." + item.name + "_np = np.array(self." + item.name + ", dtype=np.double)\n"
				np_device += "self." + item.name + "_dev = cl_array.to_device(self.sim.cl_q, self." + item.name + "_np)\n"
			if item.type == "obj_track":
				initial += "self." + item.name + " = []\n"
				obj_collection += "\n\t" + item.code 
			if item.type in ["obj", "obj_def", "obj_action"]:
				obj_collection += "\n\t" + item.code
			elif item.type == ["other"]:
				other += "\t" + item.code

		exec("import physicl\nimport physicl.light")

		exec(initial)
		exec(obj_collection)
		exec(np_initial)
		exec(np_device)
		exec(other)

		# Copy data over into arrays

		# Copy arrays over into memory.

		# Prepare to call upon the input.

		# Compile if we haven't already.

		# Find the first instance of an "obj" __CLInput instance, this will define the global size.
		self.global_shape = "None" # PLACEHOLDER
		for item in self.prep_metadata:
			if item.type == "obj":
				self.global_shape = "self." + item.name + "_np.shape"
				break

		self.local_shape = "None" # Probably unnecesary.

		self.call_params = ["self.sim.cl_q", self.global_shape, self.local_shape] + ["self." + item.name + "_dev.data" if item.type != "const" else "np.double(" + item.const_value + ")" for item in filter(lambda x: x.type in ["obj", "obj_def", "const"], self.prep_metadata)] + ["self.res_" + item.name + ".data" for item in self.output_metadata]

		# Prepare the output.
		for var in self.output_metadata:
			#print("res_" + var.name + " = cl_array.empty(self.sim.cl_q, " + global_shape + ", dtype=" + var.type +  ")")
			exec("self.res_" + var.name + " = cl_array.empty(self.sim.cl_q, " + self.global_shape + ", dtype=np." + var.ctype + ")")

		# Run (this could be very costly for performance, how are we going to mitigate this?)
		exec("self.prog." + self.prog_name + "(" + ", ".join(self.call_params) + ")")
		
		# Retrieve the output.
		out = {}
		for var in self.output_metadata:
			#print(""""out[var.name] = exec("self.res_" + var.name + ".get()")""")
			out[var.name] = eval("self.res_" + var.name + ".get()")

		return out