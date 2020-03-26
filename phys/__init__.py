import numpy as np
import pyopencl as cl
import time 
import threading
import re
import numbers
import copy
import math

class MeasurementError(ArithmeticError):
	pass


# Question: Add decorator for unit validation?


class Measurement(numbers.Real):
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

	unit_match = re.compile("""((?P<u>[a-zA-Z]*)(\*\*|\^)(?P<p>-?\d*))""")

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

	def __init__(self, raw_value, units):
		# Convert raw units to code units. Calculate new scale.
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

		self.value = np.double(raw_value) * self.scale

	def __hid_init__(value, scale, units, original_units):
		x = Measurement(1, "")
		x.value = value
		x.scale = scale
		x.units = copy.deepcopy(units)
		x.original_units = copy.deepcopy(original_units)
		return x

	def __float__(self):
		# Unscale according to each unit.
		return np.double(self.value / self.scale)

	def __str__(self):
		return " ".join([str(float(self))] + [k + "**" + str(v) for k, v in self.original_units.items()])

	def fstr(self):
		return str(float(self))

	def __repr__(self):
		return self.__str__()

	def __getattr__(self, name):
		if name == "real":
			return self.__float__()
		elif name == "imag":
			return 0
		else:
			raise AttributeError()

	def __add__(self, other):
		if self.units != other.units:
			raise MeasurementError("Unit dimensions of " + str(self) + " and " + str(other) + " do not match.")
		
		return Measurement.__hid_init__(self.value + other.value, self.scale, self.units, self.original_units)

	def __radd__(self, other):
		return Measurement.__add__(other, self)

	def __sub__(self, other):
		temp = Measurement.__hid_init__(-other.value, other.scale, other.units, other.original_units)
		return Measurement.__add__(self, temp)

	def __mul__(self, other):
		# Combine units
		if not isinstance(other, Measurement):
			return Measurement.__hid_init__(self.value * other, self.scale, self.units, self.original_units)
		else:
			new_units = {}
			new_original_units = {}
			for unit, power in list(self.units.items()) + list(other.units.items()):
				if unit not in new_units:
					new_units[unit] = power
				else:
					new_units[unit] += power

			for unit, power in list(self.original_units.items()) + list(other.original_units.items()):
				if unit not in new_original_units:
					new_original_units[unit] = power
				else:
					new_original_units[unit] += power

			# Calculate new scale
			new_scale = self.scale * other.scale
			new_value = (float(self) * float(other)) * new_scale
			
			return Measurement.__hid_init__(new_value, new_scale, new_units, new_original_units)

	def __rmul__(self, other):
		return Measurement.__hid_init__(other * self.value, self.scale, self.units, self.original_units)

	def __rpow__(self, other, modulo):
		raise MeasurementException("Cannot raise a scalar to the power of a Measurement")

	def __pow__(self, other, modulo=None):
		res = Measurement.__hid_init__(self.value, self.scale, self.units, self.original_units)

		for unit, value in self.units.items():
			res.units[unit] *= other 
		for unit, value in self.original_units.items():
			res.original_units[unit] *= other

		res.value = self.value ** other
		res.scale = self.scale ** other
		if modulo != None:
			res.value %= modulo
			res.scale %= modulo
		return res


	for name, op in [("__truediv__", "/"), ("__floordiv__", "//"), ("__mod__", "%")]:
		exec("""def """ + name + """(self, other): 
	if not isinstance(other, Measurement):
		return Measurement.__hid_init__(self.value """ + op + """ other, self.scale, self.units, self.original_units)
	else:
		new_units = {}
		new_original_units = {}
		for unit, power in self.units.items():
			new_units[unit] = power

		for unit, power in other.units.items():
			if unit not in new_units:
				new_units[unit] = -power
			else:
				new_units[unit] -= power

		for unit, power in self.original_units.items():
			new_original_units[unit] = power

		for unit, power in other.original_units.items():
			if unit not in new_original_units:
				new_original_units[unit] = -power
			else:
				new_original_units[unit] -= power

		# Calculate new scale
		new_scale = self.scale * other.scale
		new_value = (self.__float__() """ + op + """ other.__float__ ()) * new_scale
		
		return Measurement.__hid_init__(new_value, new_scale, new_units, new_original_units)
			""")

	"""
	def __truediv__(self, other):
			def __floordiv__(self, other):
		if not isinstance(other, Measurement):
			return Measurement.__hid_init__(self.value // other, self.scale, self.units, self.original_units)
		else:
			new_units = {}
			new_original_units = {}
			for unit, power in self.units.items():
				new_units[unit] = power

			for unit, power in other.units.items():
				if unit not in new_units:
					new_units[unit] = -power
				else:
					new_units[unit] -= power

			for unit, power in self.original_units.items():
				new_original_units[unit] = power

			for unit, power in other.original_units.items():
				if unit not in new_original_units:
					new_original_units[unit] = -power
				else:
					new_original_units[unit] -= power

			# Calculate new scale
			new_scale = self.scale * other.scale
			new_value = (self.__float__() // other.__float__ ()) * new_scale
			
			return Measurement.__hid_init__(new_value, new_scale, new_units, new_original_units)
	"""

	for name, op in [("__rtruediv__", "/"), ("__rfloordiv__", "/"), ("__rmod__", "%")]:
		exec("""def """ + name + """(self, other):
	new_units = {}
	new_original_units = {}
	for unit, value in self.units.items():
		new_units[unit] = -value
	for unit, value in self.original_units.items():
		new_original_units[unit] = -value
	return Measurement.__hid_init__(other """ + op + """ self.value, 1 """ + op + """ self.scale, new_units, new_original_units)
	""")


	def __abs__(self):
		return Measurement.__hid_init__(abs(self.value), self.scale, self.units, self.original_units)

	def __ceil__(self):
		return Measurement.__hid_init__(math.ceil(self.value), self.scale, self.units, self.original_units)

	def __eq__(self, other):
		if isinstance(other, Measurement):
			return self.value == other.value and self.scale == other.scale and self.units == other.units
		else:
			return self.value == other

	def __floor__(self):
		return Measurement.__hid_init__(math.floor(self.value), self.scale, self.units, self.original_units)

	def __le__(self, other):
		if isinstance(other, Measurement):
			return self.__float__() <= other.__float__()
		else:
			return self.__float__() <= other

	def __lt__(self, other):
		if isinstance(other, Measurement):
			return self.__float__() < other.__float__()
		else:
			return self.__float__() < other

	def __neg__(self):
		return Measurement.__hid_init__(-self.value, self.scale, self.units, self.original_units)

	def __pos__(self):
		return self

	def __trunc__(self):
		return Measurement.__hid_init__(math.trunc(self.value), self.scale, self.units, self.original_units)

	def __round__(self):
		return Measurement.__hid_init__(math.round(self.value), self.scale, self.units, self.original_units)


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

class __CLProgram:
	def __init__(self, sim):
		self.variables = {}
		self.sim = sim
		self.prog = None
		self.prog_name = None
		self.input_metadata = {}
		self.output_metadata = {}
		pass

	# Steps:
	# Copy data over into arrays
	# Copy arrays over into memory
	# Run
	# Retrieve the results and store them.

	# Move the data into memory
	def prepare(self):
		# Copy data over into arrays

		# Copy arrays over into memory.
		pass

	def run(self):
		# Prepare to call upon the input.
		data_shape = "None" # PLACEHOLDER
		device_data_names = [] # PLACEHOLDER
		call_params = ["self.sim.cl_q", data_shape]
		for dev in device_data_names:
			call_params.extend(dev)

		# Prepare the output.
		for var, typ in self.output_metadata.items():
			eval("result_" + var + " = cl_array.empty(self.sim.cl_q, " + data_shape + ", dtype=" + str() + ")")

		# Run
		eval("self.prog." + self.prog_name + "(" + ", ".join() + ")")

		# Retrieve the output.
		out = {}
		for var, typ in self.output_metadata.items():
			out[var] = eval("result_" + var + ".get()")

	def define(self):
		pass