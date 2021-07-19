
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import math
import numpy as np
import sys

if __name__ == "__main__":
    # c = 1.5
    # m = 6
    # x = np.ones((m, m))
    # for i in range(1, 4):
    #     x[i, :] = np.array([(i+1)**(m-1-idx) for idx in range(m)])
    # x[4, :] = np.array([2*(2**(m-1-idx)) - 1 - c*(3**(m-1-idx)) for idx in range(m)])
    # x[5, :] = np.array([2*(3**(m-1-idx)) - 2**(m-1-idx) - c*(4**(m-1-idx)) for idx in range(m)])
    # # x[6, :] = 0
    # # x[6, -1] = 1
    # # x[:-1, -1] -= np.random.rand(m-1)*.01
    # # print(x)
    # print(np.linalg.det(x))
    # x_inv = np.linalg.inv(x)
    # print("inv", x_inv)
    # print(np.dot(x_inv, x))
    # a_vec = np.dot(x_inv, np.ones(m))
    # print(a_vec)
    #
    # print("eye?", np.dot(x_inv, x))

    # print("chebyshev center")
    # print(pypoman.polyhedron.compute_chebyshev_center(-x, -np.ones(m)))
    # print(x)
    #
    # lp_soln = linprog(np.ones(m), A_ub=-1*x, b_ub=-0.001*np.ones(m), bounds=None)
    # print(lp_soln)
    # a_vec_1 = lp_soln.x
    # print(a_vec_1)
    # print(">= 1?", np.dot(x, a_vec_1))

    # a_vec = np.linalg.lstsq(x, np.ones(m))[0]
    # print(x)
    # print(a_vec)

    # print("1?", np.dot(x, a_vec))

    # for i in range(1, 5):
    #     x_vals = np.array([i**(m-1-idx) for idx in range(m)])
    #     print(x_vals)
    #     print(a_vec_1)
    #     print(i, np.dot(x_vals, a_vec_1))
    #
    # for i in [1, 2]:
    #     g = []
    #     for j in [i, i+1, i+2]:
    #         g.append(np.dot(np.array([i**(m-1-idx) for idx in range(m)]), a_vec_1))
    #     print(2*g[1] - g[0] - c*g[2])

    n = 118

    # delta = 0.9


    c = 1.26

    # for a in np.arange(-200, -190):
    #     for b in np.arange(-20, -10):
    # sin((x + 62) * 0.1) / (x + 62)

    x = np.arange(1, n + 1, 1)

    # y = [2*math.sin(0.01*(_x + 629))/(_x+629) for _x in x]
    # y = [(math.sin((_x - 500 * math.pi)/1000)+3) for _x in x]

    # y = np.array([math.sin(_x*0.5*math.pi/(1*118) + 4*math.pi)*(((_x**1e-10)*0.5*math.pi/(1*118) + 4*math.pi)) for _x in x])
    # y = np.array([1/(1+math.exp(-_x*(1/1e15)))-.5 for _x in x])
    # y = np.array([_x/n-float(1-(_x/n))**(1/(n^2)) for _x in x])
    # y = np.array([math.log(float(_x)) for _x in x])
    # y = np.array([1 - (n/(_x+n)) for _x in x])
    y = np.array([float(_x)**n for _x in x])
    print(y)
    # y -= np.min(y)
    y /= np.max(y)
    # y = [math.sin((_x+4500)*.001)/((_x+4500)**c) + 1.002 for _x in x]

    # y = [np.exp(_x) for _x in x]
# k = 1 + 1e-4/c
#     y = [1, 2]
#     for i in x[2:]:
#         y.append((2*y[-1]-y[-2])/c - .3*y[-1]**2)
        # y.append(((2*y[-1]-y[-2]))*delta/(c*y[-1]))
        # y.append(y[-1]/(2/c - k))
# y.append((delta*c**2 + 2 * c * delta + 2*y[-1])/(4-c))

    # if y[8] >= 0:
    # print(a, b)
    plt.plot(x, y)
    # plt.plot(x, [-10*math.cos(_x)/_x for _x in x])
    # plt.savefig("g_%d_%d.png" % (a, b))
    plt.savefig("g.png")
    plt.clf()
    #
    print(y)
    for i in range(len(y) - 2):
        print(2*y[i+1] - y[i] - c*y[i+2])

    # n = 4
    # x_vals = np.arange(1, n + .1, 1)
    # c = float(sys.argv[1])
    #
    # x = np.array([[7-9*c, 3-3*c, 1-c], [14-16*c, 4-4*c, 1-c], [1, 1, 1]])
    # x_inv = np.linalg.inv(x)
    #
    # def find_eps():
    #     search_range = np.arange(0.01, 1, 0.01)
    #     for eps_1 in search_range:
    #         for eps_2 in search_range:
    #             for eps_3 in search_range:
    #                 # Solution based on using matrix inverse
    #                 a = np.dot(x_inv, np.array([eps_1, eps_2, eps_3]))
    #
    #                 # My manual solution, I don't think it is correct but it's probably just algebraic error.
    #                 # gamma = eps_2 - (1-c)*eps_3
    #                 #
    #                 # eps_1_prime = -1*(eps_1 - eps_2 + gamma/2)/c
    #                 # eps_2_prime = gamma/(2*(1-c)) - ((7-8*c)/(1-c))*eps_1_prime
    #                 # eps_3_prime = eps_3 - 4*eps_1_prime - 2*eps_2_prime
    #                 #
    #                 y = np.array([a[0]*(x**2) + a[1]*x + a[2] for x in x_vals])
    #
    #                 if y[-1] > 0 and np.isclose(np.max(y), y[-1]):
    #                     a_2 = a[0]
    #                     a_1 = a[1]
    #                     a_0 = a[2]
    #                     eps = [eps_1, eps_2, eps_3]
    #                     return a_0, a_1, a_2, eps
    #
    # a_0, a_1, a_2, eps = find_eps()
    # print(a_2)
    # print(a_1)
    # print(a_0)
    # print("eps", eps)
    #
    # g = []
    # if a_2 or a_1 or a_0:
    #     for x in x_vals:
    #         y = a_2*(x**2)+a_1*x + a_0
    #         print(y)
    #         g.append(y)
    #
    # for i in range(len(g) - 2):
    #     print(2*g[i+1] - g[i] - c*g[i+2])


                # plt.plot(x_vals, y)
                # plt.savefig("parabola_%0.2f_%0.2f_%0.2f.png" % (eps_1, eps_2, eps_3))
                # plt.clf()
