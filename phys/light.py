import phys
import pyopencl as cl
import pyopencl.array as cl_array
import numpy.linalg as np_lin
import numpy as np
import scipy.stats as st
import copy


"""
SI definition for speed of light
"""
c = np.double(299792458) # Defined here: https://www.bipm.org/en/CGPM/db/17/1/
h = np.double(6.62607015e-34) # Defined here: https://www.bipm.org/utils/common/pdf/CGPM-2018/26th-CGPM-Resolutions.pdf

class PhotonObject(phys.Object):
	"""
	Represents a simple photon.
	Constrained to require an energy (E) and a velocity whose Euclidean norm is the speed of light.
	"""


	# Make it so we can have light that's slower than this?
	def __init__(self, attrs={}):
		"""
		Initializes the photon object and checks for constraints.

		"""
		super().__init__(attrs)
		if np_lin.norm(self.v) != np_lin.norm(c):
			raise Exception("Not a valid speed.") # May not work in non-vacuum mediums.
		if "E" not in attrs:
			raise Exception("Needs a valid energy.") # Handle wavelengths as an alternative?



def E_from_wavelength(wavelength):
	"""
	Converts a wavelength in meters to E in joules.
	"""
	return (h * c) / wavelength


def generate_photons(n, bins=100, dist="gauss", min=0, max=1, dir=[c, 0, 0]):
	"""
	Generates a collection of rays according to a distribution. Works as follows:
	1. Generates a distribution of E according to dist.
	2. Multiplies the quantity of rays desired per bin by the resulting distribution, then generates that number of rays per bin. This number is floored.
	3. Returns a list of the generated rays.

	Acceptable distributions: 
	- const: Generates n photons for each bin of photons.
	- gauss: Generates up to n photons for each bin of photons according to the Gaussian distribution centered around the average of the two.
	- loglin: Generates a distribution of photons such that the log of n is directly proportional to the log of the wavelength.
	"""
	# log N = log lambda
	# lambda = hc/E
	# log N = log (hc/E)^c
	# N = (hc/E)^c

	# fn: x is position in dist => dist count less-than-equal-to-1
	# min_fn x is miniumum => min in distribution generator
	# max_fn analagous to min_fn 
	dist_tab = {"gauss" : {"fn": lambda x: st.norm.pdf(x), "min_fn": lambda x: st.norm.ppf(0.001), "max_fn": lambda x: st.norm.ppf(0.999)},
				"constant": {"fn": lambda x: np.array([1] * len(x)), "min_fn": lambda x: min, "max_fn": lambda x:  max},
				"loglin": {"fn": lambda x: ((h * c) / x) ** -0.1, "min_fn": lambda x: min, "max_fn": lambda x: max}} # Change this constant of proportionality.

	dx = np.linspace(dist_tab[dist]["min_fn"](min), dist_tab[dist]["max_fn"](max), bins) # The distribution space.
	Ex = np.linspace(min, max, bins) # Linear energy distribution
	dcount = dist_tab[dist]["fn"](dx) * n # Distribution count for each bin.
	out = []
	for idx, x in enumerate(dcount):
		for i in range(int(np.floor(dcount[idx]))): # Generate dcount[idx] photons with Ex[idx] energy each.
			out.append(PhotonObject({"v": dir, "E": Ex[idx]})) 

	return out


class ScatterDeleteStep(phys.Step):
	"""
	Step that scatters photon objects from a simulation by removing them.
	Assumes that the simulation composes the entirety of the medium and has a constant number density and cross-sectional area.
	"""
	def __init__(self, n, A):
		"""
		Initializes this step with a constant number density (n) and cross-sectional area (A).
		"""
		self.n = n
		self.A = A
		self.cl_prog = None
		self.compiled = False

	def __compile_cl__(self, sim):
		kernel_del = """
			__kernel void light_scatter_step_del(__global double *dx, __global double *dy, __global double *dz, __global double *rand, double n, double A, __global int *result){
				int gid = get_global_id(0);
				double norm = sqrt(pow(dx[gid], 2) + pow(dy[gid], 2) + pow(dz[gid], 2));
				double pcoll = A * n * norm;
				if (pcoll >= rand[gid]){
					// Mark for removal.
					result[gid] = 1;
				} else {
					result[gid] = 0;
				}
			}
		"""

		self.cl_prog = cl.Program(sim.cl_ctx, kernel_del).build()
		self.compiled = True


	def __run_cl(self, sim):
		if self.compiled == False:
			self.__compile_cl__(sim)

		dx = []
		dy = []
		dz = []
		rand = []
		pht = []

		# Unwind
		for obj in sim.objects:
			if PhotonObject != type(obj):
				continue
			dx.append(obj.dr[0])
			dy.append(obj.dr[1])
			dz.append(obj.dr[2])
			rand.append(np.random.random())
			pht.append(obj)

		# Copy the memory over.

		dx_np = np.array(dx, dtype=np.double)
		dy_np = np.array(dy, dtype=np.double)
		dz_np = np.array(dz, dtype=np.double)
		rand_np = np.array(rand, dtype=np.double)

		dx_np_dev = cl_array.to_device(sim.cl_q, dx_np)
		dy_np_dev = cl_array.to_device(sim.cl_q, dy_np)
		dz_np_dev = cl_array.to_device(sim.cl_q, dz_np)
		rand_np_dev = cl_array.to_device(sim.cl_q, rand_np)


		# Run the kernel and place the results in res.
		res = cl_array.empty(sim.cl_q, dx_np.shape, dtype=np.int)
		self.cl_prog.light_scatter_step_del(sim.cl_q, res.shape, None, dx_np_dev.data, dy_np_dev.data, dz_np_dev.data, rand_np_dev.data, np.double(self.n), np.double(self.A), res.data)

		# Apply the results of the kernel to the simulation.
		out = res.get()
		for idx, x in enumerate(out):
			if x == 1:
				sim.remove_obj(pht[idx])

	def run(self, sim):
		"""
		Runs this step, and will either run it in native Python or OpenCL.
		"""
		if sim.cl_on == False:
			self.__run_py(sim)
		else:
			self.__run_cl(sim)

	def __run_py(self, sim):
		for obj in sim.objects:
			if PhotonObject != type(obj):
				continue
			p_coll = self.n * self.A * np_lin.norm(obj.dr)
			p_next = np.random.random()
			if p_coll >= p_next:
				sim.remove_obj(obj)

class ScatterSphericalStep(phys.Step):
	"""
	Step that scatters photons spherically, according to a defined number density (n) and a defined cross-sectional area (A).
	
	If the optional param wavelength_dep_scattering is set to True, then the probability of scattering will then be proportional to ((h * c) / E_wave)^-4 (i.e. divide by the wavelength to the power of 4)
	"""
	# Internal construction order:
	# 1. Default params.
	# 2. Wavelength dependent scattering.

	def __init__(self, n, A, wavelength_dep_scattering = False):
		self.n = n 
		self.A = A
		self.cl_prog = None
		self.compiled = False
		self.wavelength_dep_scattering = wavelength_dep_scattering


	def __compile_cl__(self, sim):

		params = ["__global double *dx", "__global double *dy", "__global double *dz", 
											"__global double *rand", "double n", "double A", 
											"__global double *rtheta", "__global double *rphi",
											"__global double *ovx", "__global double *ovy", "__global double *ovz",
											"__global double *nvx", "__global double *nvy", "__global double *nvz"]
		if self.wavelength_dep_scattering == True:
			params.extend(["__global double *E"])

		pcoll_vars = ["A", "n", "norm"]
		if self.wavelength_dep_scattering == True:
 			pcoll_vars.append("pow((" + str(h).upper() + " * " + str(c) + ") / E[gid], -4)")
		kernel_sph = """
			__kernel void light_scatter_step_sphere(""" + ", ".join(params) + """){

				int gid = get_global_id(0);
				double norm = sqrt(pow(dx[gid], 2) + pow(dy[gid], 2) + pow(dz[gid], 2));
				double pcoll = """ + " * ".join(pcoll_vars) + """;
				if (pcoll >= rand[gid]){
					// Change the velocity.
					nvx[gid] = """ + str(c) + """ * sin(rtheta[gid]) * cos(rphi[gid]);
					nvy[gid] = """ + str(c) + """ * sin(rtheta[gid]) * sin(rphi[gid]);
					nvz[gid] = """ + str(c) + """ * cos(rtheta[gid]);
				} else {
					nvx[gid] = NAN; // Mark it as unaffected
				}
			} 

		"""

		self.cl_prog = cl.Program(sim.cl_ctx, kernel_sph).build()
		self.compiled = True

	def __run_cl(self, sim):
		if self.compiled == False:
			self.__compile_cl__(sim)

		# Load the necessary data into memory.

		dx = []
		dy = []
		dz = []
		rtheta = []
		tau = 2 * np.pi
		rphi = []
		ovx = []
		ovy = []
		ovz = []
		rand = []
		pht = []
		Es = [] # 
		for obj in sim.objects:
			if PhotonObject != type(obj):
				continue
			dx.append(obj.dr[0])
			dy.append(obj.dr[1])
			dz.append(obj.dr[2])
			rtheta.append(np.random.random() * tau)
			rphi.append(np.random.random() * np.pi)
			ovx.append(obj.v[0])
			ovy.append(obj.v[1])
			ovz.append(obj.v[2])
			rand.append(np.random.random())
			pht.append(obj)
			if self.wavelength_dep_scattering:
				Es.append(obj.E)

		dx_np = np.array(dx, dtype=np.double)
		dy_np = np.array(dy, dtype=np.double)
		dz_np = np.array(dz, dtype=np.double)
		rtheta_np = np.array(rtheta, dtype=np.double)
		rphi_np = np.array(rphi, dtype=np.double)
		rand_np = np.array(rand, dtype=np.double)
		ovx_np = np.array(ovx, dtype=np.double)
		ovy_np = np.array(ovy, dtype=np.double)
		ovz_np = np.array(ovz, dtype=np.double)
		if self.wavelength_dep_scattering:
			e_np = np.array(Es, dtype=np.double)

		dx_np_dev = cl_array.to_device(sim.cl_q, dx_np)
		dy_np_dev = cl_array.to_device(sim.cl_q, dy_np)
		dz_np_dev = cl_array.to_device(sim.cl_q, dz_np)
		rand_np_dev = cl_array.to_device(sim.cl_q, rand_np)
		rtheta_np_dev = cl_array.to_device(sim.cl_q, rtheta_np)
		rphi_np_dev = cl_array.to_device(sim.cl_q, rphi_np)
		ovx_np_dev = cl_array.to_device(sim.cl_q, ovx_np)
		ovy_np_dev = cl_array.to_device(sim.cl_q, ovy_np)
		ovz_np_dev = cl_array.to_device(sim.cl_q, ovz_np)
		if self.wavelength_dep_scattering:
			e_np_dev = cl_array.to_device(sim.cl_q, e_np)

		nvx = cl_array.empty(sim.cl_q, ovx_np.shape, dtype=np.double)
		nvy = cl_array.empty(sim.cl_q, ovx_np.shape, dtype=np.double)
		nvz = cl_array.empty(sim.cl_q, ovx_np.shape, dtype=np.double)

		call_params = ["sim.cl_q", "nvx.shape", "None", "dx_np_dev.data", "dy_np_dev.data", "dz_np_dev.data", "rand_np_dev.data", "np.double(self.n)", "np.double(self.A)",
												"rtheta_np_dev.data", "rphi_np_dev.data", "ovx_np_dev.data", "ovy_np_dev.data", "ovz_np_dev.data", 
												"nvx.data", "nvy.data", "nvz.data"]

		if self.wavelength_dep_scattering:
			call_params.append("e_np_dev.data")

		eval("self.cl_prog.light_scatter_step_sphere(" + ", ".join(call_params) + ")")

		outx = nvx.get()
		outy = nvy.get()
		outz = nvz.get()

		# Set the velocity for each photon.
		for idx in range(0, len(outx)):
			if not np.isnan(outx[idx]):
				pht[idx].v = np.array([outx[idx], outy[idx], outz[idx]], dtype=np.double)

	def __run_py(self, sim):
		for obj in sim.objects:
			if PhotonObject != type(obj):
				continue
			p_coll = self.n * self.A * np_lin.norm(obj.dr)
			if self.wavelength_dep_scattering:
				p_coll *= ((h * c) / obj.E) ** -4 # Since we already have an A_targ term in there, we merely need to adjust it.
			p_next = np.random.random()
			if p_coll >= p_next:
				phi = np.random.random() * np.pi
				theta = np.random.random() * np.pi * 2
				obj.v = np.array([c * np.sin(theta) * np.cos(phi), c * np.sin(theta) * np.sin(phi), c * np.cos(theta)], dtype=np.double)


	def run(self, sim):
		"""
		Runs the simulation either in native Python or in OpenCL
		"""
		if sim.cl_on == False:
			self.__run_py(sim)
		else:
			self.__run_cl(sim)

class ScatterMeasureStep(phys.MeasureStep):
	"""
	Measures the number of photons that cross through a series of planes. 
	Each plane should be defined with a 3D numpy double array, with the coordinates not defining the location of the plane set to numpy.nan.
	"""
	def __init__(self, out_fn, measure_n=True, measure_locs=[]):
		super().__init__(out_fn)
		self.measure_locs = measure_locs
		self.measure_n = measure_n

	"""
	Runs this step and records how many photons passed through this plane based upon the current location and previous location.
	""" 
	def run(self, sim):
		out = [sim.t]
		if self.measure_n:
			out.append(len(sim.objects))
		for loc in self.measure_locs:
			nl = 0
			for obj in sim.objects:
				# Figure out how to do this efficiently.
				if not np.isnan(loc[0]):
					if (obj.r[0] - obj.dr[0] <= loc[0] and loc[0] <= obj.r[0]) or (obj.r[0] - obj.dr[0] >= loc[0] and loc[0] >= obj.dr[0]):
						nl += 1
				elif not np.isnan(loc[1]):
					if (obj.r[1] - obj.dr[1] <= loc[1] and loc[1] <= obj.r[1]) or (obj.r[1] - obj.dr[1] >= loc[1] and loc[1] >= obj.dr[1]):
						nl += 1
				else:
					if (obj.r[2] - obj.dr[2] <= loc[2] and loc[2] <= obj.r[2]) or (obj.r[2] - obj.dr[2] >= loc[2] and loc[2] >= obj.dr[2]):
						nl += 1
			out.append(nl)


		self.data.append(np.array(out))

class ScatterSignMeasureStep(phys.MeasureStep):
	"""
	Measures the number of objects with a strictly positive coordinate at each step with regards to x, y, and z coordinates.
	"""
	def __init__(self, out_fn, measure_n = True):
		super().__init__(out_fn)
		self.measure_n = measure_n

	def run(self, sim):
		# Do we count 0 as positive? No
		xp = 0
		yp = 0
		zp = 0

		out = [sim.t]
		if self.measure_n:
			out.append(len(sim.objects))
		for obj in sim.objects:
			xp += int(obj.v[0] > 0)
			yp += int(obj.v[1] > 0)
			zp += int(obj.v[2] > 0)
		out.append(xp)
		out.append(yp)
		out.append(zp)

		self.data.append(np.array(out))

class TracePathMeasureStep(phys.MeasureStep):
	"""
	Traces the path of object. If the object does not exist at any point, then the value 'nan;nan;nan' is traced. Otherwise the current coordinates are printed as a string.
	"""

	def __init__(self, out_fn, trace_type = phys.Object, id_info_fn = lambda x: str(type(x))):
		super().__init__(out_fn)
		self.trace_type = trace_type
		self.id_info_fn = id_info_fn
		self.id_counter = 0
		self.id_dict = {}
		self.pos_dict = {}

	def run(self, sim):
		for obj in sim.objects:
			#if type(obj) != self.trace_type:
			#	continue
			if "__trace_path_id" not in dir(obj):
				obj.__setattr__("__trace_path_id", self.id_counter)
				self.id_dict[self.id_counter] = self.id_info_fn(obj)
				self.pos_dict[self.id_counter] = {"start": sim.t, "pos": []}
				self.id_counter += 1
			self.pos_dict[obj.__getattribute__("__trace_path_id")]["pos"].append(np.array([obj.r[0], obj.r[1], obj.r[2]], dtype=np.double))
			# Append each object to outpu

	def terminate(self, sim):
		"""
		Turns the data collected into a 2D matrix of coordinate vectors. The first row will be the recorded times (useful if we write the data to a CSV).
		"""
		# Prettified data 
		# Cols: t, Rows: items
		rows = len(self.id_dict.keys())
		cols = len(sim.ts)
		dat_clean = []
		dat_clean.append(["t"] + copy.deepcopy(sim.ts))
		for i in range(0, rows):
			n = [self.id_dict[i]]
			b = sim.ts.index(self.pos_dict[i]["start"])
			a = cols - len(self.pos_dict[i]["pos"])
			n.extend([np.nan, np.nan, np.nan] * b) # Fill with nans before the data. Test this.
			n.extend([x for x in self.pos_dict[i]["pos"]]) # Add the main data
			n.extend([np.nan, np.nan, np.nan] * a)
			dat_clean.append(n)
		self.data = dat_clean

