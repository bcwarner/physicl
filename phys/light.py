import phys
import pyopencl as cl
import pyopencl.array as cl_array
import numpy.linalg as np_lin
import numpy as np

c = np.double(299792458) # Defined here: https://www.bipm.org/en/CGPM/db/17/1/

class PhotonObject(phys.Object):
	def __init__(self, attrs={}):
		super().__init__(attrs)
		if np_lin.norm(self.v) != np_lin.norm(c):
			raise Exception("Not a valid speed.") # May not work in non-vacuum mediums.
		if "E" not in attrs:
			raise Exception("Needs a valid energy.") # Handle wavelengths as an alternative?

class ScatterDeleteStep(phys.Step):
	def __init__(self, n, A):
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
		for obj in sim.objects:
			if PhotonObject != type(obj):
				continue
			dx.append(obj.dr[0])
			dy.append(obj.dr[1])
			dz.append(obj.dr[2])
			rand.append(np.random.random())
			pht.append(obj)

		dx_np = np.array(dx, dtype=np.double)
		dy_np = np.array(dy, dtype=np.double)
		dz_np = np.array(dz, dtype=np.double)
		rand_np = np.array(rand, dtype=np.double)

		dx_np_dev = cl_array.to_device(sim.cl_q, dx_np)
		dy_np_dev = cl_array.to_device(sim.cl_q, dy_np)
		dz_np_dev = cl_array.to_device(sim.cl_q, dz_np)
		rand_np_dev = cl_array.to_device(sim.cl_q, rand_np)

		res = cl_array.empty(sim.cl_q, dx_np.shape, dtype=np.int)
		self.cl_prog.light_scatter_step_del(sim.cl_q, res.shape, None, dx_np_dev.data, dy_np_dev.data, dz_np_dev.data, rand_np_dev.data, np.double(self.n), np.double(self.A), res.data)

		out = res.get()
		for idx, x in enumerate(out):
			if x == 1:
				sim.remove_obj(pht[idx])

	def run(self, sim):
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
	def __init__(self, n, A):
		self.n = n
		self.A = A
		self.cl_prog = None
		self.compiled = False

	def __compile_cl__(self, sim):
		kernel_sph = """
			__kernel void light_scatter_step_sphere(__global double *dx, __global double *dy, __global double *dz, 
											__global double *rand, double n, double A, 
											__global double *rtheta, __global double *rphi,
											__global double *ovx, __global double *ovy, __global double *ovz,
											__global double *nvx, __global double *nvy, __global double *nvz){

				int gid = get_global_id(0);
				double norm = sqrt(pow(dx[gid], 2) + pow(dy[gid], 2) + pow(dz[gid], 2));
				double pcoll = A * n * norm;
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

		dx_np = np.array(dx, dtype=np.double)
		dy_np = np.array(dy, dtype=np.double)
		dz_np = np.array(dz, dtype=np.double)
		rtheta_np = np.array(rtheta, dtype=np.double)
		rphi_np = np.array(rphi, dtype=np.double)
		rand_np = np.array(rand, dtype=np.double)
		ovx_np = np.array(ovx, dtype=np.double)
		ovy_np = np.array(ovy, dtype=np.double)
		ovz_np = np.array(ovz, dtype=np.double)

		dx_np_dev = cl_array.to_device(sim.cl_q, dx_np)
		dy_np_dev = cl_array.to_device(sim.cl_q, dy_np)
		dz_np_dev = cl_array.to_device(sim.cl_q, dz_np)
		rand_np_dev = cl_array.to_device(sim.cl_q, rand_np)
		rtheta_np_dev = cl_array.to_device(sim.cl_q, rtheta_np)
		rphi_np_dev = cl_array.to_device(sim.cl_q, rphi_np)
		ovx_np_dev = cl_array.to_device(sim.cl_q, ovx_np)
		ovy_np_dev = cl_array.to_device(sim.cl_q, ovy_np)
		ovz_np_dev = cl_array.to_device(sim.cl_q, ovz_np)

		nvx = cl_array.empty(sim.cl_q, ovx_np.shape, dtype=np.double)
		nvy = cl_array.empty(sim.cl_q, ovx_np.shape, dtype=np.double)
		nvz = cl_array.empty(sim.cl_q, ovx_np.shape, dtype=np.double)
		self.cl_prog.light_scatter_step_sphere(sim.cl_q, nvx.shape, None, dx_np_dev.data, dy_np_dev.data, dz_np_dev.data, rand_np_dev.data, np.double(self.n), np.double(self.A),
												rtheta_np_dev.data, rphi_np_dev.data, ovx_np_dev.data, ovy_np_dev.data, ovz_np_dev.data, 
												nvx.data, nvy.data, nvz.data)

		outx = nvx.get()
		outy = nvy.get()
		outz = nvz.get()
		for idx in range(0, len(outx)):
			if not np.isnan(outx[idx]):
				pht[idx].v = np.array([outx[idx], outy[idx], outz[idx]], dtype=np.double)

	def run(self, sim):
		if sim.cl_on == False:
			self.__run_py(sim)
		else:
			self.__run_cl(sim)

class ScatterMeasureStep(phys.MeasureStep):
	def __init__(self, out_fn, measure_n=True, measure_locs=[], measure_pos = True):
		super().__init__(out_fn)
		self.measure_locs = measure_locs
		self.measure_n = measure_n
		self.measure_pos = measure_pos

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
