import numpy as np
import random


class PSOPortfolioOptimizationCardinalityConstraints:
    def __init__(
            self,
            objective_func,
            objective_func_arguments,
            portfolio_size,
            cardinality_min,
            cardinality_max,
            num_particles=100,
            num_iterations=1000,
            c1=2,
            c2=1,
            w=0.8,
            verbose=True,
    ):
        self.objective_func = objective_func
        self.objective_func_arguments = objective_func_arguments
        self.n = portfolio_size
        self.cardinality_min = int(cardinality_min)
        self.cardinality_max = int(cardinality_max)
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.hyperparameters = {
            "c1": c1,
            "c2": c2,
            "w": w,
        }
        self.verbose = verbose
        
        # initialize swarm
        self.x = np.zeros((self.num_particles, self.n))
        for particle_index in range(self.num_particles):
            k = np.random.randint(cardinality_min, cardinality_max)
            for _ in range(k):
                self.x[particle_index, np.random.randint(0, self.n - 1)] += 1
                
        print(f"Initial x = {self.x}")
        
        # initialize velocities of particles
        self.v = np.random.rand(self.num_particles, self.n) * 0.1
        
        # initialize particle and global best
        self.pbest = self.x
        self.pbest_obj = np.array([objective_func(row, **self.objective_func_arguments) for row in self.x])
        self.gbest = self.pbest[self.pbest_obj.argmin(), :]
        print(f"Initial g_best = {self.gbest}")
        self.gbest_obj = self.pbest_obj.min()
        
        # training history
        self.training_history = []
    
    def optimize(self):
        c1, c2, w = self.hyperparameters["c1"], self.hyperparameters["c2"], self.hyperparameters["w"]
        
        for iter_num in range(self.num_iterations):
            r1, r2 = np.random.rand(2)
            
            # update v
            self.v = w * self.v + c1 * r1 * (self.pbest - self.x) + c2 * r2 * (self.gbest - self.x)
            
            # print(f"self.x before update x = {self.x} >>>>>>>>>>>>.")
            
            # update x
            self.x = self.x + self.v
            # make sure constraints are satisfied
            # print(f"self.x after update raw = {self.x}...........")
            
            self.x = np.round(self.x)
            self.x[self.x < 0] = 0

            # print(f"self.x after round and remove negatives = {self.x} >>>>>>>>>>>>>><<<<<<<<<<<")
            # print("............")
            
            row_sums = np.sum(self.x, axis=1)
            # print(f"row_sums min = {row_sums}")
            incorrect_particles_min = np.where(row_sums < self.cardinality_min)[0]
            # print(f"Incorrect particles min = {incorrect_particles_min}")
            
            for particle_index in incorrect_particles_min:
                # print("here")
                while np.sum(self.x[particle_index, :]) < self.cardinality_min:
                    # print(f"here 1, particle index = {[particle_index]}")
                    self.x[particle_index, np.random.randint(0, self.n - 1)] += 1

            row_sums = np.sum(self.x, axis=1)
            # print(f"row_sums max = {row_sums}")
            incorrect_particles_max = np.where(row_sums > self.cardinality_max)[0]
            # print(f"Incorrect particles max = {incorrect_particles_max}")
            
            for particle_index in incorrect_particles_max:
                while np.sum(self.x[particle_index, :]) > self.cardinality_max:
                    positive_indices = np.where(self.x[particle_index, :] >= 1)
                    self.x[particle_index, random.choice(positive_indices)] -= 1
            
            # update pbest, pbest_obj and gbest, gbest_obj
            p_obj = np.array([self.objective_func(row, **self.objective_func_arguments) for row in self.x])
            self.pbest[p_obj < self.pbest_obj, :] = self.x[p_obj < self.pbest_obj, :]
            
            self.pbest_obj = np.array([self.pbest_obj, p_obj]).min(axis=0)
            self.gbest = self.pbest[self.pbest_obj.argmin(), :]
            
            self.gbest_obj = self.pbest_obj.min()
            
            # append to training history
            self.training_history.append(self.gbest_obj)
            
            print(f"Iteration {iter_num + 1}")
            print(f"gbest obj = {self.gbest_obj}")
            if self.verbose:
                print(f"gbest = {self.gbest}")
            print(">>>>>>>>>>>>>>>>>>>>>>>>")
        
        print(f"Optimization complete. Resultant gbest = {self.gbest}")
        print(f"gbest_obj = {self.gbest_obj}")


