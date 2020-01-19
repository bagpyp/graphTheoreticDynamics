# graphTheoreticDynamics
![screenshot](screenshot.png)
"""
Created on Tue May 9 15:14:52 2016
@author: Robbie Cunningham
"""


This short program models the trajectory of a variable number of autonomous mobile agents based off your choice of the following: number of agents, starting positions, network topology (graph-theoretic), power threshold (e.g. voltage limit), and two conflicting gains (how strongly agents want to swarm together vs. how strongly they want to minimize some cost function (which in this case amounts to "heading toward the origin").

"""preliminaries"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
from mpl_toolkits.mplot3d import Axes3D




"""parameters"""

#number of agents
N=10
#which graph?
graph = 'path'
#final time
T=10
#time incriment
j=.1
#optimality gain
a=1
#consensus gain
b=1
#saturation bound (Delta)
D=20
#dimension, either 2 or 3
m=2
