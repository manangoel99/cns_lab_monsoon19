import numpy as np
import matplotlib.pyplot as plt
import copy

class Simulator():

	def __init__(self):
		self.AllPositions = []
		self.AllVelocities = []
		self.AllEnergies = []
		self.AllMomenta = []
		self.AllMomentaNorm = []
		self.AllForces = []

		p_file = open("momentum.txt", "w")
		p_file.close()

		x_file = open("position.txt", "w")
		x_file.close()

		e_file = open("energy.txt", "w")
		e_file.close()

	def setupSimulation(self, N, epsilon, sigma, volume, steps, step_size):
		'''
		Setup the simulations 
		'''
		self.prev_positions = []
		self.prev_velocities = []
		self.N = N
		self.volume = volume
		self.epsilon = epsilon
		self.sigma = sigma
		self.dt = step_size
		self.mass = 6.6335209 * 1e-23
		self.steps = steps
		self.forces = [0 for i in range(N)]


	def initializePostions(self):
		'''
		Initalize the positions. 
		'''
		self.positions = (np.random.random((self.N, 3)) * 20) - 10

	def initializeVelocities(self):
		'''
		Initalize the velocities. 
		'''
		self.velocities = np.random.random((self.N, 3))
	
	def ForceField(self, particle1, particle2):
		vec = self.positions[particle1] - self.positions[particle2]
		r = np.linalg.norm(vec)

		force = 4 * self.epsilon * ((-12 * (self.sigma ** 12) * (r**(-13))) - (-6 * (self.sigma ** 6) * (r**(-7)))) * (vec / r)
		return force

	def doMultipleSteps(self):
		'''
		do the steps
		'''
		for i in range(self.steps):
			# self.eulerIntegrate()
			self.velocityVerletIntegrate()

			self.AllEnergies.append(self.totalEnergy())
			print(i, self.totalEnergy(), self.totalMomentum())
			self.AllMomenta.append(self.particleMomentum()[0])
			self.AllMomentaNorm.append(self.totalMomentum())
	
	def eulerIntegrate(self):
		'''
		apply euler scheme
		'''
		self.AllPositions.append(self.positions)

		for i in range(self.N):
			self.forces[i] = 0
			for j in range(self.N):
				if i != j:
					self.forces[i] += self.ForceField(i, j)
			self.velocities[i] = self.velocities[i] + (self.forces[i] / self.mass) * self.dt
			self.positions[i] = self.positions[i] + self.velocities[i] * self.dt
			

	def velocityVerletIntegrate(self):
		'''
		apply velocity verlet integrator
		'''
		self.AllPositions.append(self.positions)

		for i in range(self.N):
			prev_forces = copy.copy(self.forces[i])
			self.forces[i] = 0
			for j in range(self.N):
				if i != j:
					self.forces[i] += self.ForceField(i, j)
			
			self.positions[i] = self.positions[i] + (self.velocities[i] * self.dt) + (1 / (2 * self.mass)) * self.forces[i] * (self.dt**2)
			self.velocities[i] = self.velocities[i] + 1 / (2 * self.mass) * (prev_forces + self.forces[i]) * self.dt

	def totalEnergy(self):
		'''
		calculate total energy
		'''
		return self.potentialEnergy() + self.kineticEnergy()

	def particleMomentum(self):
		pres_momentum = self.velocities * self.mass
		return pres_momentum, np.linalg.norm(pres_momentum, axis=1)

	def totalMomentum(self):
		return np.sum(self.particleMomentum()[1])

	def potentialEnergy(self):
		'''
		calculate potential energy
		'''
		total_energy = 0
		for i in range(self.N - 1):
			for j in range(i + 1, self.N):
				r = np.linalg.norm(self.positions[i] - self.positions[j])
				total_energy += 4 * self.epsilon * (((self.sigma / r) ** 12) - (self.sigma / r) ** 6)
		return total_energy		

	def kineticEnergy(self):
		'''
		calculate kinetic energy
		'''
		v = np.expand_dims(np.linalg.norm(self.velocities, axis=1), axis=1)
		energies = (1 / 2) * self.mass * (v**2)
		return np.sum(energies)

	def Plot(self):
		self.AllEnergies = np.array(self.AllEnergies)

		plt.figure(1)
		plt.scatter(np.arange(self.steps), np.squeeze(self.AllEnergies))

		plt.title("Energy")
		plt.figure(2)
		plt.scatter(np.arange(self.steps), self.AllMomentaNorm)
		plt.title("Momentum")
	
	def Save(self):
		self.AllPositions = np.array(self.AllPositions)
		self.AllMomenta = np.array(self.AllMomenta)

		with open("position.txt", 'a') as f:
			for i in range(self.steps):
				print("Step : ", i + 1, file=f)
				for j in range(self.N):
					print("{},{},{}".format(self.AllPositions[i][j][0], self.AllPositions[i][j][1], self.AllPositions[i][j][2]), file=f)
		
		with open("energy.txt", 'a') as f:
			for i in range(self.steps):
				print("Step : ", i + 1, file=f)
				print("{}".format(self.AllEnergies[i]), file=f)

		with open("momentum.txt", "a") as f:
			for i in range(self.steps):
				print("Step : ", i + 1, file=f)
				for j in range(self.N):
					# print(self.AllMomenta[i].shape)
					print("{}, {}, {}".format(self.AllMomenta[i][j][0], self.AllMomenta[i][j][1], self.AllMomenta[i][j][2]), file=f)
#Initialize the simulator class
System=Simulator()
System.setupSimulation(100, 1, 1, 1, 50, 1e-10)
System.initializePostions()
System.initializeVelocities()
System.doMultipleSteps()
System.Plot()
System.Save()

plt.show()
#Call the methods from the simulator class appropriately 