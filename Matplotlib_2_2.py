
# # Matplotlib
# Read the tutorials: https://matplotlib.org/stable/users/explain/quick_start.html and https://matplotlib.org/stable/tutorials/pyplot.html before solving the exercises below. The "Pyplot Tutorial" you do not read in detail but it is good to know about since the fact that there are two approaches to plotting can be confusing if you are not aware of both of the approaches.


import numpy as np
import matplotlib.pyplot as plt


# ### Plotting in Matplotlib can be done in either of two ways, which ones? Which way is the recommended approach?


## PlyPlot : where you use plyplot functions for plotting and where these function are called globally. 
## used for quick and simple plots
## Objected Oriented : where we create the Figure and Axes and then call methods on them.
## the OO-way is more felxible and explicit which allows us to have a better control over the figure's layout and its elements.
## PlyPlot is recommended when we want to do a quick and simple plot while, 
## the OO-way is much better for more detailed and complex plots where we are allowed flexibility and control.


# ### Explain shortly what a figure, axes, axis and an artist is in Matplotlib.


## -- Figure = is the "canvas" which keeps track of the child Axes, 'special' Artists and even nested subfigures.
## -- Artists = everything which is visable on the Figure is an Artist, ALL texts, line2D, collections, patch objects including  
## the Figure, Axes, and Axis objects are Artists. 
## -- Axes = is the primary interface and is a method for configuring most parts of the plot.
## It includes two (or three for 3D) Axis object which provide the tick and and ticklabels to provide scales for the data in the Axes. 
## The Axes is a Artist which is attached to the figure. 
## All Axes have a title, x-label and a y-label which is set with set_title(), set_xlabel() and set_ylabel().
## -- Axis = Objects which sets the scales and limits and generates the ticks snd ticklabels.
## Locator determines the locationa of the ticks and the ticklable strings are formatted by a Formatter.


# ### When plotting in Matplotlib, what is the expected input data type?


## the expected inputs are numpy.array or numpy.ma.masked.array or objects which can be passed to numpy.asarray.
## if there are objects which which is not a NumPy array - ex, tuples, pandas and list - then Matplotlib will with numpy.asarray()
## convert these into NumPy arrays and then the data can be plotted. 


# ### Create a plot of the function y = x^2 [from -4 to 4, hint use the np.linspace function] both in the object-oriented approach and the pyplot approach. Your plot should have a title and axis-labels.


## Object Oriented Plot
x = np.linspace(-4, 4, 200)
y = x**2

fig, ax = plt.subplots()

ax.plot(x, y)
ax.set_title('Object Oriented Plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


## PlyPlot
x = np.linspace(-4, 4, 200)
y = x**2

plt.plot(x, y)

plt.title('PlyPlot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# ### Create a figure containing 2  subplots where the first is a scatter plot and the second is a bar plot. You have the data below. 


## Data for scatter plot
np.random.seed(15)
random_data_x = np.random.randn(1000)
random_data_y = np.random.randn(1000)
x = np.linspace(-2, 2, 100)
y = x**2

# Data for scatter plot
np.random.seed(15)
random_data_x = np.random.randn(1000)
random_data_y = np.random.randn(1000)
x = np.linspace(-2, 2, 100)
y = x**2

fig, ax = plt.subplots(figsize=[8, 8])

ax.scatter(random_data_x, random_data_y, color='red', alpha=0.5, label='Data')
ax.plot(x, y, color='orange', label='y = x**2')
ax.set_title('Data for Scatter Plot')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')

plt.show()

## Data for bar plot
fruit_data = {'grapes': 22, 'apple': 8, 'orange': 15, 'lemon': 20, 'lime': 25}
names = list(fruit_data.keys())
values = list(fruit_data.values())

fig, ax = plt.subplots(figsize=[10, 10])
ax.bar(names, values, color='pink')
ax.set_title('Data for Bar Plot')
ax.set_xlabel('Fruits')
ax.set_ylabel('Count')

plt.show()



