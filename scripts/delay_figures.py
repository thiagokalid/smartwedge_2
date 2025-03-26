import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
font = {
    'weight' : 'normal',
    'size'   : 9
}
linewidth = 6.3091141732 # LaTeX linewidth

# Set the default font to DejaVu Serif
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]
matplotlib.rc('font', **font)

t = np.arange(0, 1, 1e-3)

def pulse(t, tau, s=.5e-3):
  return np.exp(-((t-tau)**2)/s)


# Figure 1:
s=.5e-3
x = pulse(t, .2, s)
x += .5*pulse(t, .8, s)

fig, ax = plt.subplots(figsize=(linewidth*.5, 1.5))
t = np.arange(.7, 1, 1e-3)
x = pulse(t, .2, s)
x += .5*pulse(t, .8, s)

plt.plot(t, x, '--', color='C0')
plt.xticks([])
plt.yticks([])
t_ = np.arange(0, 1, 1e-3)
x_ = pulse(t_, .2, s) + .3*pulse(t_, .6, s)
plt.plot(t_, x_, '-', color='C0')
plt.xlabel('Time')
plt.ylabel('Amplitude')

ax.annotate("Front wall", xy=(.27, .325), xytext=(.355, .84),
            arrowprops=dict(arrowstyle="-|>", color='k', alpha=1, linewidth=1),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )
ax.annotate("", xy=(.770, .325), xytext=(0.695, .645),
            arrowprops=dict(arrowstyle="-|>", color='k', alpha=1, linewidth=1),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )
ax.annotate("Back wall (spurious)", xy=(.770, .325), xytext=(0.78, .68),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )
ax.annotate("Pit", xy=(.56, .2), xytext=(0.485, .5),
            arrowprops=dict(arrowstyle="-|>", color='k', alpha=1, linewidth=1),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )

plt.tight_layout()
plt.savefig('../figures/delays_tight.pdf')
plt.show()


# Figure 2:
s=.5e-3
x = pulse(t, .2, s)
x += .5*pulse(t, .8, s)

fig, ax = plt.subplots(figsize=(linewidth*.5,1.5))
t = np.arange(.7, 1, 1e-3)
x = pulse(t, .2, s)
x += .5*pulse(t, .8, s)
plt.xticks([])
plt.yticks([])
t_ = np.arange(0, 1, 1e-3)
x_ = pulse(t_, .2, s) + .5*pulse(t_, .8, s)
plt.plot(t_, x_, '-', color='C0')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.ylim([None, 1.5])

ax.annotate("", xy=(.2, 1.2), xytext=(.8, 1.2),
            arrowprops=dict(arrowstyle="|-|", color='k', alpha=1, linewidth=1),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )
ax.annotate(r"$\tau_w$", xy=(.5, 1.25), xytext=(.5, 1.25),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )


ax.annotate("Front wall", xy=(.27, .325), xytext=(.355, .645),
            color="k",
            arrowprops=dict(arrowstyle="-|>", color='k', alpha=1, linewidth=1),
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )
ax.annotate("Back wall", xy=(.770, .325), xytext=(0.695, .645),
            color="k",
            ha="center",  # Center text horizontally
            va="bottom"  # Position text below arrow
            )

arrowStop =(.770, .325)
arrowStart = (0.695, .645)
ax.annotate("",arrowStop,xytext=arrowStart,arrowprops=dict(arrowstyle="-",shrinkA=0,shrinkB=1,edgecolor='k',facecolor="none",linestyle="dashed"))
ax.annotate("",arrowStop,xytext=arrowStart,arrowprops=dict(linewidth=0,arrowstyle="-|>",shrinkA=0,shrinkB=0,edgecolor="none",facecolor='k',linestyle="solid"))

plt.tight_layout()
plt.savefig('../figures/delays_utm.pdf')

plt.show()