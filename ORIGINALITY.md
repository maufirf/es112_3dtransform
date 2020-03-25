In this file, I'm only going to define what is not original from my code instead of what is original as my original code significantly outweights the code i copied from other places.

## Axes3D imports

```python
#https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
```

This part of code was copied from the [Matplotlib's `Axes3D` tutorial](https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html) because this is the boilerplate i need to use their `Axes3D` to plot in 3D. In fact, I modified the last two lines to suit my needs:

- Make the figure symmetrical so the axes' scale are the same to each other.
- Make the figure size bigger than the default settings
- Create different figures for each rotation operation

```python
f_or = plt.figure(figsize=(8,8))
f_or_ax = f_or.add_subplot(111, projection='3d')
```

## Axes3D interactive support on jupyter notebooks

```python
#https://stackoverflow.com/questions/47311632/jupyter-how-to-rotate-3d-graph
%matplotlib notebook
```

This enables me to interact the figures i made in jupyter notebooks, so i can rotate and see my results in different angles.

## Specific HTML parsing support on jupyter notebooks

```
#https://gist.github.com/christopherlovell/e3e70880c0b0ad666e7b5fe311320a62
from IPython.display import HTML
```

This enables me to embed youtube video on my notebook.

## Giving colors to points in `scatter` plot points

```python
#https://stackoverflow.com/questions/33287156/specify-color-of-each-point-in-scatter-plot-matplotlib
cseq = np.array([115,169,255,64])/255
c = list(cseq for _ in range(len(cube_orig)))
```

-------

The rest is original.