cdef class Particle:
	cdef:
		double mass, position, velocity
	
	def __init__(self, m, p, v):
		self.mass = m
		self.position = p
		self.velocity = v
	
	cpdef double get_momentum(self):
		return self.mass * self.velocity

def add_momentums(particles):
	total_mom = 0.0
	
	for particle in particles:
		total_mom += particle.get_momentum()
	
	return total_mom