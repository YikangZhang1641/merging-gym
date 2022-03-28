# from cvxopt  import solvers, matrix
import numpy as np
from qpsolvers import solve_qp
from numpy import array, dot
# import sympy as sym
import math
import matplotlib.pyplot as plt
import sys

class spline5:
    def __init__(self, Xseq, Yseq, yaw=None, Tseq=None, vx=None, vy=None):
        self.Xseq = Xseq
        self.Yseq = Yseq
        self.vx = vx
        self.vy = vy
        self.Tseq = Tseq if Tseq is not None else range(len(Xseq)) # Tseq应该定义为真实时刻序列， 0.1s, 0.2s, ...

        self.T = max(Tseq)
        order = 3
        self.order = order

        if len(self.Tseq) == 1:
            return

        p = np.zeros([len(self.Tseq) * 2, (order + 1) * 2])
        q = np.zeros([len(self.Tseq) * 2, 1])
        for i in range(len(self.Tseq)):
            t = self.Tseq[i] / self.T
            for j in range(order + 1):
                ttt = np.power(t, j)
                p[i][j] = ttt
                p[i + len(self.Tseq)][j + order + 1] = ttt
            q[i] = Xseq[i]
            q[i + len(self.Tseq)] = Yseq[i]

        soft_cons_len = 10
        Soft_3rd = np.zeros([soft_cons_len * 2, (order + 1) * 2])

        # bound = 2
        # ###### -b <= d2y/d2x <= b
        # g = np.zeros([len(self.Tseq) * 2, (order + 1) * 2])
        # h = np.zeros([len(self.Tseq) * 2])
        # for i in range(len(self.Tseq)):
        #     t = self.Tseq[i] / (self.T - 1)
        #
        #     for j in range(2, order + 1):
        #         ttt = np.power(t, j - 2)
        #         g[i][j] = -bound * j * (j - 1) * ttt
        #         g[i][j + order + 1] = j * (j - 1) * ttt
        #
        #         g[i + len(self.Tseq)][j] = -bound * j * (j - 1) * ttt
        #         g[i + len(self.Tseq)][j + order + 1] = -j * (j - 1) * ttt
        #
        # ####### -b <= d2x/d2y <= b
        # g2 = np.zeros([len(self.Tseq) * 2, (order + 1) * 2])
        # h2 = np.zeros([len(self.Tseq) * 2])
        # for i in range(len(self.Tseq)):
        #     t = self.Tseq[i] / (self.T - 1)
        #
        #     for j in range(2, order + 1):
        #         ttt = np.power(t, j - 2)
        #         g2[i][j] = j * (j - 1) * ttt
        #         g2[i][j + order + 1] = -bound * j * (j - 1) * ttt
        #
        #         g2[i + len(self.Tseq)][j] = -j * (j - 1) * ttt
        #         g2[i + len(self.Tseq)][j + order + 1] = -bound * j * (j - 1) * ttt

        #######

        #         np.set_printoptions(precision=2)
        #         print('p', p)
        #         P = np.dot(p.T, p) + np.dot(g.T, g) * 0.001 + np.eye((order + 1) * 2) * 0.0001
        P = np.dot(p.T, p) + np.eye((order + 1) * 2) * 0.01
        Q = -np.dot(q.T, p)

        Q = Q.reshape(((order + 1) * 2,))

        A = np.zeros([5, (order + 1) * 2])
        B = np.zeros([A.shape[0], 1])

        A[0][0] = 1
        A[1][order + 1] = 1
        B[0] = Xseq[0]
        B[1] = Yseq[0]

        if vx is not None and vy is not None:
            A[2][1] = 1
            B[2] = vx
            A[3][1 + order + 1] = 1
            B[3] = vy

        if yaw is not None:
            A[-1][1] = math.tan(yaw)
            A[-1][1 + order + 1] = -1

            B[-1] = 0

        B = B.reshape((A.shape[0],))

        # G = np.concatenate([g, g2])
        # H = np.concatenate([h, h2])

        #         self.sol = solve_qp(P=P, q=Q)
        self.sol = solve_qp(P=P, q=Q, A=A, b=B)

    #         self.sol = solve_qp(P=P, q=Q, A=A, b=B, G=G, h=H)

    def cal(self, t):
        if len(self.Tseq) == 1:
            return self.Xseq[0], self.Yseq[0], 0, 0

        x = 0
        y = 0
        order = self.order
        dx = 0
        dy = 0

        if t >= 0 and t <= 1:
            for i in range(order + 1):
                ttt = np.power(t, i)
                x += self.sol[i] * ttt
                y += self.sol[i + order + 1] * ttt

            for i in range(1, order + 1):
                ttt = np.power(t, i - 1)
                dx += self.sol[i] * i * ttt
                dy += self.sol[i + order + 1] * i * ttt

        return x, y, dx, dy

    def plot_scatter(self):
        X, Y = [], []

        T = [i / 100 * self.T for i in range(100)]
        for t in T:
            x, y, dx, dy = self.cal(t)
            X.append(x)
            Y.append(y)

        plt.scatter(T, X)
        plt.scatter(self.Tseq, self.Xseq, color='red')
        plt.show()
        #
        plt.scatter(T, Y)
        plt.scatter(self.Tseq, self.Yseq, color='red')
        plt.show()

        plt.scatter(X, Y)
        plt.scatter(self.Xseq, self.Yseq, color='red')
        plt.show()

class mpc_1d:
    def __init__(self, x0, v0, xt, vt, t):
        self.x0 = x0
        self.v0 = v0
        self.xt = xt
        self.vt = vt
        self.t = t
        self.T_len = 10
        self.dt = self.t / self.T_len

        a = np.array([[1, self.dt], [0, 1]])
        b = np.array([[0], [self.dt]])
        A = np.zeros([2, self.T_len])

        ### [A9B, A8B, ...AB, B]
        tmp = np.eye(2)
        for i in range(self.T_len)[::-1]:
            A[:, i] = np.matmul(tmp, b).T
            tmp = np.matmul(a, tmp)

        B = (np.array([[xt], [vt]]) - np.matmul(tmp, np.array([[x0], [v0]])))
        B = B[0].reshape(1,)

        p = np.zeros([self.T_len - 1, self.T_len])
        for i in range(self.T_len - 1):
            p[i][i] = 1
            p[i][i + 1] = -1
        P = np.matmul(p.T, p) + np.eye(self.T_len) * 0.01
        q = np.zeros([self.T_len, 1]).reshape(self.T_len,)

        self.u = solve_qp(P=P, q=q, A=A[0,:], b=B)
        # print("A", A)
        # print("B", B)
        # print("P", P)
        # print("q", q)
        # print("u", u)

    def action(self):
        assert self.u is not None
        return self.u[0]

    def plot(self):
        plt.figure()
        plt.scatter(np.array([0, self.T_len]), np.array([self.x0, self.xt]), color='red')

        X = [self.x0]
        V = [self.v0]
        for u in self.u:
            X.append(X[-1] + self.dt * V[-1])
            V.append(V[-1] + self.dt * u)
        plt.plot(range(self.T_len + 1), X)
        plt.show()

if __name__ == '__main__':
    # Xseq = [0, 20]
    # Yseq = [0, 0]
    # Tseq = [0, 1]
    # vx = 20
    # vy = 0
    # sp = spline5(Xseq=Xseq, Yseq=Yseq, Tseq=Tseq, vx=vx, vy=vy)
    # print(sp.cal(0))
    # print(sp.cal(1))
    # sp.plot_scatter()

    res = mpc_1d(0, 0, 1, 1, 1)
    res.plot()