cdef class Particle:
	cdef:
		double mass, position, velocity
	
	def __init__(self, mass, position, velocity):
		self.mass = mass
		self.position = position
		self.velocity = velocity
	
	def get_momentum(self):
		return self.mass * self.velocity
	
	def get_mass(self):
		return self.mass
	
	def get_position(self):
		return self.position
	
	def get_velocity(self):
		return self.velocity