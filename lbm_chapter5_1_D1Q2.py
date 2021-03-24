import taichi as ti
import matplotlib.pyplot as plt
import numpy as np
ti.init(arch=ti.cpu)
@ti.data_oriented
class lbm_solver:
    def __init__(self): # total steps to run
        self.m = 101
        self.dx = 1.
        self.rho = ti.field(dtype=ti.f32,shape=self.m)
        self.f1 = ti.field(dtype=ti.f32,shape=self.m)
        self.f2 = ti.field(dtype=ti.f32,shape=self.m)
        self.flux = ti.field(dtype=ti.f32,shape=self.m)
        self.x = ti.field(dtype=ti.f32,shape=self.m)
        self.alpha = 0.25
        self.omega = 1/(self.alpha+0.5)
        self.twall = 1.0
        self.nstep = 20000
        self.x[0] = 0.
        for i in range(1,self.m):
            self.x[i] = self.x[i-1]+self.dx
    @ti.kernel
    def init(self):
        for i in range(self.m):
            self.rho[i] = 0.
            self.f1[i] = 0.
            self.f2[i] = 0.
            self.flux[i] = 0.
    @ti.kernel
    def collision(self):
        for i in range(self.m):
            feq = 0.5 * self.rho[i]
            self.f1[i] = (1-self.omega)*self.f1[i]+self.omega*feq
            self.f2[i] = (1-self.omega)*self.f2[i]+self.omega*feq
    @ti.pyfunc
    def stream(self):
        # Streaming
        for i in range(self.m-1):
            self.f1[self.m-i-1]=self.f1[self.m-i-2]
            self.f2[i] = self.f2[i+1]
    @ti.pyfunc
    def boundary(self):
        # Boundary conditions
        self.f1[0] = self.twall - self.f2[0]
        self.f1[self.m-1] = self.f1[self.m-2]
        self.f2[self.m-1] = self.f2[self.m-2]
    @ti.kernel
    def update(self):
        for i in range(self.m):
            self.rho[i] = self.f1[i] + self.f2[i]
    @ti.kernel
    def post(self):
        for i in range(self.m):
            self.flux[i] = self.omega*(self.f1[i]-self.f2[i])
    def plot(self):
        plt.subplot(1,2,1)
        plt.plot(self.x.to_numpy(),self.rho.to_numpy())
        plt.title("Temperature")
        plt.xlabel("X")
        plt.ylabel("T")
        plt.subplot(1,2,2)
        plt.plot(self.x.to_numpy(),self.flux.to_numpy(),'o')
        plt.title("Flux")
        plt.xlabel("X")
        plt.ylabel("T")
        plt.show()
    @ti.pyfunc
    def gui_plot(self):
        pos = np.stack((self.x.to_numpy()/200,self.rho.to_numpy()),axis=-1)
        self.gui.circles(pos, color = 0x000000, radius = 5)
        pos = np.stack((self.x.to_numpy()/200+0.5,self.flux.to_numpy()/0.1),axis=-1)
        self.gui.circles(pos, color = 0x000000, radius = 5)
        self.gui.show()
        
    def solve(self):
        self.init()
        self.gui = ti.GUI('Diffusion equation D1Q2',res = (1000,500),background_color = 0xFFFFFF)
        for k in range(self.nstep):
            self.collision()
            self.stream()
            self.boundary()
            self.update()
            self.post()
            self.gui_plot()
        self.plot()
if __name__ == '__main__':
    lbm = lbm_solver()
    lbm.solve()
    # print(lbm.rho)
    # print(lbm.flux)

