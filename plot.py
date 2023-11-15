import numpy as np
from numpy import exp, sqrt, cos, pi, e, sin
import matplotlib.pyplot as plt
import itertools
import os
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 16})
plt.rcParams.update(
    {'font.family': 'serif', "font.serif": "Times New Roman", "text.usetex": True, "font.weight": "bold"})

def Himmelblau(x, y):
    return (x**2+y-11)**2 + (x+y**2-7)**2

def Rosenbrock(x,y):
    return (1-x)**2 + 100*(y-x**2)**2

def Ackley(x, y):
    return -20.0*exp(-0.2*sqrt(0.5*(x**2+y**2)))-exp(0.5*(cos(2*pi*x)+cos(2*pi*y)))+e+20

def Goldstein(x, y):
    return (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))  

def DropWave(x, y):
    return -(1+np.cos(12*np.sqrt(x**2+y**2))) / (0.5*(x**2+y**2)+2)

def EggHolder(x, y):
    a=sqrt(abs(y+x/2+47))
    b=sqrt(abs(x-(y+47)))
    c=-(y+47)*sin(a)-x*sin(b)
    return c

def Michalewicz(x, y):
    return -1*((sin(x)*np.sin((1*x**2)/np.pi)**20)+(np.sin(y)*np.sin((2*y**2)/np.pi)**20))

def Booth(x, y):
    return (x+2*y-7)**2 + (2*x+y-5)**2

def Levy(x, y):
    return sin(3*pi*x)**2 + (x-1)**2*(1+sin(3*pi*y)*sin(3*pi*y))+ (y-1)*(y-1)*(1+sin(2*pi*y)*sin(2*pi*y))

func = Levy
rang = 5
step = 0.01

x_range = np.arange(-rang, rang, step)
y_range = np.arange(-rang, rang, step)

z = np.zeros((len(x_range), len(y_range)))
for i, x in enumerate(x_range):
    for j, y in enumerate(y_range):
        z[i][j] = func(x, y)

fig = plt.figure(figsize = (8,8))
plt.pcolormesh(z.T, edgecolors='k', linewidth=0.0, cmap = "Spectral_r")
plt.contour(np.log(z.T), levels = 80, linewidths = 0.7, colors = 'k')
plt.xticks([])
plt.yticks([])
plt.savefig("back_ackley.png", dpi = 90, bbox_inches = 'tight')

counter = 1
pairs = list(itertools.combinations([0,1,2], 2))
for j in range(5):
    data = np.load(f"./data/traj_{j}.npy")
    iterations, points, coords = np.shape(data)
    for i in range(iterations):
        fig = plt.figure(figsize = (8,8))
        plt.plot([data[i][0,0],data[i][1,0]], [data[i][0,1],data[i][1,1]], color = 'k', lw = 3, ls = ':')
        plt.plot([data[i][1,0],data[i][2,0]], [data[i][1,1],data[i][2,1]], color = 'k', lw = 3, ls = ':')
        plt.plot([data[i][2,0],data[i][0,0]], [data[i][2,1],data[i][0,1]], color = 'k', lw = 3, ls = ':')
        plt.scatter(data[i][:,0],data[i][:,1], c = ["r", "g", "b"], edgecolor = 'k', s = 200, zorder = 10)
        plt.xlim(np.min(x_range), np.max(x_range))
        plt.ylim(np.min(y_range), np.max(y_range))
        plt.text(0,-4.7,f'Iteration: {i:5d}', fontsize = 42, weight='bold')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"./animation/{counter:05d}.png", dpi = 90, bbox_inches = 'tight', transparent = True)
        plt.close()
        counter += 1

images = os.popen("ls ./animation/*.png").read().split()

for i,image in enumerate(images):
    os.system(f"convert -composite -gravity center back_ackley.png {image} ./final_images/{i:05d}.png")



