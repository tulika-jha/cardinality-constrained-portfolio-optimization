import numpy as np


class PSOPortfolioOptimization:
    def __init__(
            self,
            objective_func,
            objective_func_arguments,
            portfolio_size,
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
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.hyperparameters = {
            "c1": c1,
            "c2": c2,
            "w": w,
        }
        self.verbose = verbose
        
        # initialize swarm
        self.x = np.random.rand(self.num_particles, self.n)
        self.x = self.x / self.x.sum(axis=1)[:, np.newaxis]
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
            
            # update x
            self.x = self.x + self.v
            # normalize x and make sure constraints are satisfied
            self.x[self.x < 0] = 0
            self.x = self.x / self.x.sum(axis=1)[:, np.newaxis]
            
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
        
    

