# import taichi as ti
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.cm as cm
from numba import jit,prange
# ti.init(arch=ti.cpu)


class lbm_solver:

    def __init__(self):  # total steps to run
        self.m = 201
        self.n = 201
        self.xl = 1.
        self.yl = 1.
        self.dx = self.xl/(self.m-1.)
        self.dy = self.yl/(self.n-1.)
        self.w = np.array([4./9., 1./9., 1./9., 1./9., 1./9.
            , 1./36., 1./36., 1./36., 1./36.],dtype=np.float32)
        self.e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1],
                        [-1, 1], [-1, -1], [1, -1]],dtype=np.int32)
        self.c2 = 1./3.
        self.rho = np.zeros([self.m,self.n],dtype=np.float32)
        self.f = np.zeros([self.m,self.n,9], dtype=np.float32)
        self.f_old = np.zeros([self.m,self.n,9], dtype=np.float32)
        self.flux = np.zeros(self.m,dtype=np.float32)
        self.fluxq = np.zeros(self.m,dtype=np.float32)
        self.Tm = np.zeros(self.m,dtype=np.float32)
        self.x = np.zeros(self.m,dtype=np.float32)
        self.y = np.zeros(self.n,dtype=np.float32)
        self.Z = np.zeros([self.m,self.n],dtype=np.float32)
        self.alpha = 0.25
        self.omega = 1./(3.*self.alpha+0.5)
        self.twall = 1.0
        self.nstep = 10000
        for i in range(1, self.m):
            self.x[i] = self.x[i-1]+self.dx

    def init(self):
        pass

    def feq(self,i,j,k):
        return self.w[k] * self.rho[i,j]
        
    @jit(parallel=True)
    def collision(self):
        for i in prange(self.m):
            for j in prange(self.n):
                for k in prange(9):
                    self.f_old[i,j,k] = (1-self.omega)*self.f[i,j,k]+self.omega*self.feq(i,j,k)
    @jit(parallel=True)
    def stream(self):
        # Streaming
        for i in prange(self.m):
            for j in prange(self.n):
                for k in prange(9):
                    ip = i - self.e[k, 0]
                    jp = j - self.e[k, 1]
                    if 0<=ip<self.m and 0<=jp<self.n:
                        self.f[i,j,k] = self.f_old[ip,jp,k]
        # for i, j in ti.ndrange((0, self.m-1), (0, self.n)):
        #         self.f[self.m-i-1,j,1] = self.f_old[self.m-i-2,j,1]
        #         self.f[i,j,2] = self.f_old[i+1,j,2]
        # for i, j in ti.ndrange((0, self.m), (0, self.n-1)):
        #         self.f[i,self.n-j-1,3] = self.f_old[i,self.n-j-2,3]
        #         self.f[i,j,4] = self.f_old[i,j+1,4]
    @jit(parallel=True)
    def boundary(self):
        # Boundary conditions
        # left boundary, twall=1.0
        for j in prange(self.n):
            self.f[0,j,1] = self.w[1]*self.twall + self.w[3]*self.twall - self.f[0,j,3]
            self.f[0,j,5] = self.w[5]*self.twall + self.w[7]*self.twall - self.f[0,j,7]
            self.f[0,j,8] = self.w[8]*self.twall + self.w[6]*self.twall - self.f[0,j,6]
        # bottom boundary, adiabatic, bounce back
        for i in prange(self.m):
            self.f[i,0,2] = self.f[i,1,2]
            self.f[i,0,5] = self.f[i,1,5]
            self.f[i,0,6] = self.f[i,1,6]
        # top boundary, T = 0.0
        for i in prange(self.m):
            self.f[i,self.n-1,7] = -self.f[i,self.n-1,5]
            self.f[i,self.n-1,4] = -self.f[i,self.n-1,2]
            self.f[i,self.n-1,8] = -self.f[i,self.n-1,6]
        # right hand boundary
        for j in prange(self.n):
            self.f[self.m-1,j,3] = -self.f[self.m-1,j,1]
            self.f[self.m-1,j,7] = -self.f[self.m-1,j,5]
            self.f[self.m-1,j,6] = -self.f[self.m-1,j,8]
    @jit(parallel=True)
    def update(self):
        for i in prange(self.m):
            for j in prange(self.n):
                self.rho[i,j] = self.f[i,j,0] + self.f[i,j,1] + self.f[i,j,2]+ self.f[i,j,3]+ self.f[i,j,4]+ \
                    self.f[i,j,5] + self.f[i,j,6] + self.f[i,j,7]+ self.f[i,j,8]
            # self.flux[i] = self.tk[i]*self.omega[i]*(self.f[i,1]-self.f[i,2])/self.c2
        for i in prange(self.m):
            self.Tm[i] = self.rho[i,(self.n-1)//2]
    @jit(parallel=True)
    def post(self):
        for i in prange(self.m):
            for j in prange(self.n):
                self.Z[j,i] = self.rho[i,j]
        # for i in range(self.m-1):
        #     self.fluxq[i] = self.tk[i]*(self.rho[i] - self.rho[i+1])
        # self.fluxq[self.m-1] = self.fluxq[self.m-2]

    def plot(self):
        plt.subplot(1, 2, 1)
        plt.plot(self.x.to_numpy(), self.Tm.to_numpy())
        plt.title("Temperature")
        plt.xlabel("X")
        plt.ylabel("T")
        plt.subplot(1, 2, 2)
        X, TM = np.meshgrid(self.x.to_numpy(), self.Tm.to_numpy())
        a = plt.contourf(X, TM, self.Z.to_numpy(), 10, cmap=plt.cm.Spectral)
        b = plt.contour(X, TM, self.Z.to_numpy(), 10, colors='black', linewidths=1, linestyles='solid')
        plt.colorbar(a, ticks=[i for i in np.linspace(0,1,10)])
        plt.clabel(b, inline=True, fontsize=10)
        # plt.plot(self.x.to_numpy(), self.flux.to_numpy(), 'x')
        # plt.plot(self.x.to_numpy(), self.fluxq.to_numpy(), 'o')
        # plt.title("fluxq")
        # plt.xlabel("X")
        # plt.ylabel("T")
        plt.show()


    def gui_plot(self):
        # rho = self.rho.to_numpy()
        rho_img = cm.plasma(self.rho)
        self.gui.set_image(rho_img)
        pos = np.stack((self.x, (self.Tm)), axis=-1)
        self.gui.circles(pos, color=0x000000, radius=3)
        # pos = np.stack((self.x.to_numpy()/200+0.5,
        #                self.flux.to_numpy()/1), axis=-1)
        # self.gui.circles(pos, color=0x00FFFF, radius=5)
        # pos = np.stack((self.x.to_numpy()/200+0.5,
        #                self.fluxq.to_numpy()/1), axis=-1)
        # self.gui.circles(pos, color=0x000000, radius=5)
        self.gui.show()

    def solve(self):
        self.init()
        # self.gui = ti.GUI('Diffusion equation D1Q2', res=(
        #     201, 201), background_color=0xFFFFFF)
        # self.gui.fps_limit = 500
        for k in tqdm(range(self.nstep)):
            self.collision()
            self.stream()
            self.boundary()
            self.update()
            self.post()
            # self.gui_plot()
        self.plot()


if __name__ == '__main__':
    lbm = lbm_solver()
    lbm.solve()
    print(lbm.Tm)
    print(lbm.rho)
