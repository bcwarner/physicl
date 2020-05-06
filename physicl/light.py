import physicl
import pyopencl as cl
import pyopencl.array as cl_array
import numpy.linalg as np_lin
import numpy as np
import scipy.stats as st
import scipy.integrate
import copy


"""
SI definition for speed of light
"""
c = physicl.Measurement(np.double(299792458), "m**1 s**-1") # Defined here: https://www.bipm.org/en/CGPM/db/17/1/
h = physicl.Measurement(np.double(6.62607015e-34), "J**1 s**1") # Defined here: https://www.bipm.org/utils/common/pdf/CGPM-2018/26th-CGPM-Resolutions.pdf
kB = physicl.Measurement(np.double(1.380649e-23), "J**1 K**-1") # Boltzmann constant, defined here: https://www.bipm.org/utils/common/pdf/si-brochure/SI-Brochure-9.pdf

class PhotonObject(physicl.Object):
	"""
	Represents a simple photon.
	Constrained to require an energy (E) and a velocity whose Euclidean norm is the speed of light.
	"""


	# Make it so we can have light that's slower than this?
	def __init__(self, **kwargs):
		"""
		Initializes the photon object and checks for constraints.

		"""
		super().__init__(**kwargs)
		if np_lin.norm(self.v) != np_lin.norm(c):
			raise Exception("Not a valid speed.") # May not work in non-vacuum mediums.
		if "E" not in kwargs:
			raise Exception("Needs a valid energy.") # Handle wavelengths as an alternative?



def E_from_wavelength(wavelength):
	"""
	Converts a wavelength in meters to E in joules.
	"""
	return (h * c) / wavelength

def wavelength_from_E(E):
	"""
	Converts a photon energy in joules to a wavelength in meters.
	"""
	return (h * c) / E


# Optimize
def planck_distribution(E, T):
	E_conv = E.__unscaled__() if isinstance(E, physicl.Measurement) else E
	T_conv = T.__unscaled__() if isinstance(T, physicl.Measurement) else T
	kB_conv = kB.__unscaled__()
	coef1 = 15 / (np.pi ** 4 * kB_conv * T_conv) # J ** -1
	coef2 = E_conv / (kB_conv * T_conv) # unitless
	coef3 = 1 / (np.e ** (E_conv / (kB_conv * T_conv))) # unitless
	return physicl.Measurement(coef1 * (coef2 ** 3) * coef3, "J**-1")

# Do we care about error?
def planck_probability(E_min, E_max, T, integrator=lambda fn, a, b: scipy.integrate.quad(fn, a, b)):
	return integrator(lambda x: planck_distribution(x, T), E_min, E_max)

# Use dynamic programming here.
last_planck_params = None
last_planck_gamma_norm = None
last_planck_cdf = None

# Tie number of bins to number of photons. This will give us more statistically significant results.

def planck_phot_distribution(E_min, E_max, T, bins=1000):
	# Generate the bins.
	global last_planck_params
	global last_planck_gamma_norm
	global last_planck_cdf
	params = [x.__unscaled__() if isinstance(x, physicl.Measurement) else x for x in [E_min, E_max, T, bins]]
	E_min, E_max, T, bins = tuple(params)
	gamma_norm = []
	gamma_cdf = []
	E = np.linspace(E_min, E_max, bins)
	if last_planck_params != params:
		gamma = []
		for x in range(len(E) - 1):
			gamma.append(planck_probability(E[x], E[x + 1], T)[0])
		# Sum and normalize each bin
		tot_area = sum(gamma)
		gamma_norm = [x / tot_area for x in gamma]
		# Pick a random position along the curve.
		gamma_cdf = [gamma_norm[0]]
		for x in range(1, len(gamma_norm)):
			gamma_cdf.append(gamma_cdf[-1] + gamma_norm[x])
		last_planck_params = params
		last_planck_gamma_norm = gamma_norm
		last_planck_cdf = gamma_cdf
	else:
		gamma_norm = last_planck_gamma_norm
		gamma_cdf = last_planck_cdf

	rand = np.random.rand()
	for x in range(1, len(gamma_cdf)):
		if gamma_cdf[x] >= rand and rand >= gamma_cdf[x - 1]:
			return physicl.Measurement(E[x], "J**1")
	#return planck_phot_distribution(E_min, E_max, T, bins) # unsuccessful attempt, try again
	# Return the normalized proportion (generate_photons requires a proportion)


def generate_photons_from_E(E):
	return [PhotonObject(E=x, v=c * [1, 0, 0]) for x in E]

def generate_photons(n, fn=lambda: np.random.power(3), min=0, max=0, bins=-1):
	"""
	Generates a collection of rays according to a distribution. Works by taking n samples of fn and then using min + (max - min) * fn() to generate the photon. By default this is numpy's power distribution.
	"""
	# log N = log lambda
	# lambda = hc/E
	# log N = log (hc/E)^c
	# N = (hc/E)^c

	# fn: x is position in dist => dist count less-than-equal-to-1
	# min_fn x is miniumum => min in distribution generator
	# max_fn analagous to min_fn 
	out = []
	for i in range(int(n)):
		Eo = min + (max - min) * fn()
		out.append(PhotonObject(E=Eo, v=physicl.Measurement([c, 0, 0], "m**1 s**-1")))
	return out


class ScatterDeleteStepReference(physicl.Step):
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

class ScatterDeleteStep(physicl.Step):
    def __init__(self, n, A):
        self.n = n
        self.A = A
        self.built = False
        
    def run(self, sim):
        if self.built != True:
            skip = physicl.CLInput(name="photon_check", type="obj_action", code="if type(obj) != physicl.light.PhotonObject:\n \t\t continue")
            d0, d1, d2 = tuple([physicl.CLInput(name="d" + str(x), type="obj", obj_attr="dr[" + str(x) + "]") for x in range(0, 3)])
            rand = physicl.CLInput(name="rand", type="obj_def", obj_def="np.random.random()")
            A_, n_ = physicl.CLInput(name="A", type="const", const_value=str(self.n)), physicl.CLInput(name="n", type="const", const_value=str(self.A))
            pht = physicl.CLInput(name="pht", type="obj_track", obj_track="obj")
            res = physicl.CLOutput(name="res", ctype="int")
            kernel = """
                int gid = get_global_id(0);
                    double norm = sqrt(pow(d0[gid], 2) + pow(d1[gid], 2) + pow(d2[gid], 2));
                    double pcoll = A * n * norm;
                    if (pcoll >= rand[gid]){
                        // Mark for removal.
                        res[gid] = 1;
                    } else {
                        res[gid] = 0;
                    }
                """
            
            self.prog = physicl.CLProgram(sim, "test", kernel)
            self.prog.prep_metadata = [skip, d0, d1, d2, rand, pht, A_, n_]
            self.prog.output_metadata = [res]
            self.prog.build_kernel()
            self.built = True
        
        out = self.prog.run()
        for idx, x in enumerate(out["res"]):
            if x == 1:
                sim.remove_obj(self.prog.pht[idx])

class ScatterIsotropicStep(physicl.Step):
	"""
	Step that scatters photons isotropically, according to a defined number density (n) and a defined cross-sectional area (A).
	
	If the optional param wavelength_dep_scattering is set to True, then the probability of scattering will then be proportional to ((h * c) / E_wave)^-4 (i.e. divide by the wavelength to the power of 4)
	"""
	# Internal construction order:
	# 1. Default params.
	# 2. Wavelength dependent scattering.

	def __init__(self, **kwargs):
		self.n = kwargs.get("n", 1) 
		self.A = kwargs.get("A", 1)
		self.wavelength_dep_scattering = kwargs.get("wavelength_dep_scattering", False)
		self.variable_n = kwargs.get("variable_n", False)
		self.variable_n_fn = kwargs.get("variable_n_fn", None)
		self.prog = None
		self.built = False

	def __run_cl(self, sim):
		if self.built != True:
			skip = physicl.CLInput(name="photon_check", type="obj_action", code="if type(obj) != physicl.light.PhotonObject:\n \t\t continue")
			d0, d1, d2 = tuple([physicl.CLInput(name="d" + str(x), type="obj", obj_attr="dr[" + str(x) + "]") for x in range(0, 3)])
			rtheta, rphi, rand = tuple([physicl.CLInput(name=x, type="obj_def", obj_def="np.random.random()" + mul) for x, mul in [("rtheta", "* 2 * np.pi"), ("rphi", "* np.pi"), ("rand", "")]])
			ov0, ov1, ov2 = tuple([physicl.CLInput(name="ov" + str(x), type="obj", obj_attr="v[" + str(x) + "]") for x in range(0, 3)])
			A_, n_ = physicl.CLInput(name="A", type="const", const_value=str(self.n)), physicl.CLInput(name="n", type="const", const_value=str(self.A))
			pht = physicl.CLInput(name="pht", type="obj_track", obj_track="obj")
			res0, res1, res2 = [physicl.CLOutput(name="res" + str(x), ctype="double") for x in range(0, 3)]
			
			prep_metadata = [skip, d0, d1, d2, rtheta, rphi, rand, pht, A_, n_]
			if self.wavelength_dep_scattering:
				e = physicl.CLInput(name="E", type="obj", obj_attr="E")
				prep_metadata += [e]
			if self.variable_n:
				r0, r1, r2 = tuple([physicl.CLInput(name="r" + str(x), type="obj", obj_attr="r[" + str(x) + "]") for x in range(0, 3)])
				prep_metadata += [r0, r1, r2]

			pcoll_vars = ["A", "n" if self.variable_n == False else "(" + self.variable_n_fn + ")", "norm"]
			if self.wavelength_dep_scattering == True:
				pcoll_vars.append("pow((" + str(h).upper() + " * " + str(c) + ") / E[gid], -4)")

			kernel = """
					int gid = get_global_id(0);
					double norm = sqrt(pow(d0[gid], 2) + pow(d1[gid], 2) + pow(d2[gid], 2));
					double pcoll = """ + " * ".join(pcoll_vars) + """;
					if (pcoll >= rand[gid]){
						// Change the velocity.
						res0[gid] = """ + str(c) + """ * sin(rtheta[gid]) * cos(rphi[gid]);
						res1[gid] = """ + str(c) + """ * sin(rtheta[gid]) * sin(rphi[gid]);
						res2[gid] = """ + str(c) + """ * cos(rtheta[gid]);
					} else {
						res0[gid] = NAN; // Mark it as unaffected
					}
			"""

			self.prog = physicl.CLProgram(sim, "light_scatter_step_sphere", kernel)
			self.prog.prep_metadata = prep_metadata
			self.prog.output_metadata = [res0, res1, res2]
			self.prog.build_kernel()
			self.built = True

		out = self.prog.run()

		for idx in range(0, len(out["res0"])):
			if not np.isnan(out["res0"][idx]):
				vold = self.prog.pht[idx].v
				self.prog.pht[idx].v = np.array([out["res0"][idx], out["res1"][idx], out["res2"][idx]], dtype=np.double)
				self.prog.pht[idx].dv = self.prog.pht[idx].v - vold
			else:
				self.prog.pht[idx].dv = np.array([0, 0, 0], dtype=np.double)


	# Note: this does not support variable n scattering.
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
				vold = obj.v
				obj.v = np.array([c * np.sin(theta) * np.cos(phi), c * np.sin(theta) * np.sin(phi), c * np.cos(theta)], dtype=np.double)
				obj.dv = vold
			else:
				obj.dv = np.array([0,0,0])

	def run(self, sim):
		"""
		Runs the simulation either in native Python or in OpenCL
		"""
		if sim.cl_on == False:
			self.__run_py(sim)
		else:
			self.__run_cl(sim)

class ScatterMeasureStep(physicl.MeasureStep):
	"""
	Measures the number of photons that cross through a series of planes. 
	Each plane should be defined with a 3D numpy double array, with the coordinates not defining the location of the plane set to numpy.nan.
	"""
	def __init__(self, out_fn, measure_n=True, measure_locs=[], measure_E = False):
		super().__init__(out_fn)
		self.measure_locs = measure_locs
		self.measure_n = measure_n
		self.measure_E = measure_E
	"""
	Runs this step and records how many photons passed through this plane based upon the current location and previous location.
	""" 
	def run(self, sim):
		out = [sim.t]
		if self.measure_n:
			out.append(len(sim.objects))
		for loc in self.measure_locs:
			nl = 0
			if self.measure_E:
				Es = []
			for obj in sim.objects:
				# Figure out how to do this efficiently.
				# Could we do this by seeing the plane equation swap signs?
				if not np.isnan(loc[0]):
					if (obj.r[0] - obj.dr[0] <= loc[0] and loc[0] <= obj.r[0]) or (obj.r[0] - obj.dr[0] >= loc[0] and loc[0] >= obj.r[0]):
						nl += 1
						if self.measure_E:
							Es.append(obj.E)
				elif not np.isnan(loc[1]):
					if (obj.r[1] - obj.dr[1] <= loc[1] and loc[1] <= obj.r[1]) or (obj.r[1] - obj.dr[1] >= loc[1] and loc[1] >= obj.r[1]):
						nl += 1
						if self.measure_E:
							Es.append(obj.E)
				else:
					if (obj.r[2] - obj.dr[2] <= loc[2] and loc[2] <= obj.r[2]) or (obj.r[2] - obj.dr[2] >= loc[2] and loc[2] >= obj.r[2]):
						nl += 1
						if self.measure_E:
							Es.append(obj.E)
			out.append(nl)
			if self.measure_E:
				out.append(Es)

		self.data.append(np.array(out))

class ScatterSignMeasureStep(physicl.MeasureStep):
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

class TracePathMeasureStep(physicl.MeasureStep):
	"""
	Traces the path of object. If the object does not exist at any point, then the value 'nan;nan;nan' is traced. Otherwise the current coordinates are printed as a string.
	"""

	def __init__(self, out_fn, trace_type = physicl.Object, id_info_fn = lambda x: str(type(x)), trace_dv=False):
		super().__init__(out_fn)
		self.trace_type = trace_type
		self.id_info_fn = id_info_fn
		self.id_counter = 0
		self.id_dict = {}
		self.pos_dict = {}
		self.trace_dv = trace_dv

	def run(self, sim):
		for obj in sim.objects:
			#if type(obj) != self.trace_type:
			#	continue
			if "__trace_path_id" not in dir(obj):
				obj.__setattr__("__trace_path_id", self.id_counter)
				self.id_dict[self.id_counter] = self.id_info_fn(obj)
				self.pos_dict[self.id_counter] = {"start": copy.deepcopy(sim.t), "pos": []}
				if self.trace_dv:
					self.pos_dict[self.id_counter]["freq"] = 0
				self.id_counter += 1
			self.pos_dict[obj.__getattribute__("__trace_path_id")]["pos"].append(copy.deepcopy(obj.r))
			if self.trace_dv and not np.array_equal(obj.dv, np.array([0,0,0])):
				self.pos_dict[obj.__getattribute__("__trace_path_id")]["freq"] += 1
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
			if self.trace_dv:
				n.append(self.pos_dict[i]["freq"])
			b = sim.ts.index(self.pos_dict[i]["start"])
			a = cols - len(self.pos_dict[i]["pos"])
			n.extend([np.nan, np.nan, np.nan] * b) # Fill with nans before the data. Test this.
			n.extend([x for x in self.pos_dict[i]["pos"]]) # Add the main data
			n.extend([np.nan, np.nan, np.nan] * a)
			dat_clean.append(n)
		self.data = dat_clean

