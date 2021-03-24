import taichi as ti
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
ti.init(arch=ti.cpu)


@ti.data_oriented
class lbm_solver:
    def __init__(self):  # total steps to run
        self.m = 101
        self.w = ti.field(dtype=ti.f32, shape=3)
        self.c2 = 1./3.
        self.dx = 1.
        self.rho = ti.field(dtype=ti.f32, shape=self.m)
        self.f = ti.Vector.field(3, dtype=ti.f32, shape=self.m)
        self.flux = ti.field(dtype=ti.f32, shape=self.m)
        self.fluxq = ti.field(dtype=ti.f32, shape=self.m)
        self.x = ti.field(dtype=ti.f32, shape=self.m)
        self.tk = ti.field(dtype=ti.f32, shape=self.m)
        self.dtkdx = ti.field(dtype=ti.f32, shape=self.m)
        self.cpr = ti.field(dtype=ti.f32, shape=self.m)
        self.alpha = ti.field(dtype=ti.f32, shape=self.m)
        self.omega = ti.field(dtype=ti.f32, shape=self.m)
        self.twall = 1.0
        self.nstep = 1500
        for i in range(1, self.m):
            self.x[i] = self.x[i-1]+self.dx
        arr = np.array([4./6., 1./6., 1./6.], dtype=np.float32)
        self.w.from_numpy(arr)
    @ti.kernel
    def init(self):
        for i in range(self.m):
            self.tk[i] = 20.+30./(2.*self.x[i]+1.)
            self.alpha[i]=self.tk[i]/100
            self.omega[i] = 1./(3.*self.alpha[i]+0.5)

    @ti.kernel
    def collision(self):
        for i in range(self.m):
            feq0 = self.w[0] * self.rho[i]
            feq = self.w[1] * self.rho[i]
            self.f[i][0] = (1-self.omega[i])*self.f[i][0]+self.omega[i]*feq0
            self.f[i][1] = (1-self.omega[i])*self.f[i][1]+self.omega[i]*feq
            self.f[i][2] = (1-self.omega[i])*self.f[i][2]+self.omega[i]*feq

    @ti.pyfunc
    def stream(self):
        # Streaming
        for i in range(self.m-1):
            self.f[self.m-i-1][1] = self.f[self.m-i-2][1]
            self.f[i][2] = self.f[i+1][2]

    @ti.pyfunc
    def boundary(self):
        # Boundary conditions
        self.f[0][1] = self.twall - self.f[0][0] - self.f[0][2]
        self.f[self.m-1][0] = self.f[self.m-2][0]
        self.f[self.m-1][1] = self.f[self.m-2][1]
        self.f[self.m-1][2] = self.f[self.m-2][2]

    @ti.kernel
    def update(self):
        for i in range(self.m):
            self.rho[i] = self.f[i][0] + self.f[i][1] + self.f[i][2]
            self.flux[i] = self.tk[i]*self.omega[i]*(self.f[i][1]-self.f[i][2])/self.c2

    @ti.pyfunc
    def post(self):
        for i in range(self.m-1):
            self.fluxq[i] = self.tk[i]*(self.rho[i] - self.rho[i+1])
        self.fluxq[self.m-1] = self.fluxq[self.m-2]

    def plot(self):
        plt.subplot(1, 2, 1)
        plt.plot(self.x.to_numpy(), self.rho.to_numpy())
        plt.title("Temperature")
        plt.xlabel("X")
        plt.ylabel("T")
        plt.subplot(1, 2, 2)
        plt.plot(self.x.to_numpy(), self.flux.to_numpy(), 'x')
        plt.plot(self.x.to_numpy(), self.fluxq.to_numpy(), 'o')
        plt.title("fluxq")
        plt.xlabel("X")
        plt.ylabel("T")
        plt.show()

    @ti.pyfunc
    def gui_plot(self):
        pos = np.stack((self.x.to_numpy()/200, (self.rho.to_numpy())/1), axis=-1)
        self.gui.circles(pos, color=0x000000, radius=5)
        pos = np.stack((self.x.to_numpy()/200+0.5,
                       self.flux.to_numpy()/1), axis=-1)
        self.gui.circles(pos, color=0x00FFFF, radius=5)
        pos = np.stack((self.x.to_numpy()/200+0.5,
                       self.fluxq.to_numpy()/1), axis=-1)
        self.gui.circles(pos, color=0x000000, radius=5)
        self.gui.show()

    def solve(self):
        self.init()
        self.gui = ti.GUI('Diffusion equation D1Q2', res=(
            1000, 500), background_color=0xFFFFFF)
        for k in tqdm(range(self.nstep)):
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
    print(lbm.rho)
    print(lbm.flux)
