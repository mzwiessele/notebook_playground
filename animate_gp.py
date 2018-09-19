#===============================================================================
# Copyright (c) 2018, Max Zwiessele
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of animate_gp nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#===============================================================================
import numpy as np
import GPy
from matplotlib import pyplot as plt
from matplotlib import animation

def exp_map_sphere(mu, E):
    theta = np.sqrt((E ** 2).sum(0))[None, :]
    M = mu * np.sin(theta)
    M = M + (E * (np.cos(theta)) / theta)
    M[:, np.abs(theta[0]) <= 1e-7] = mu
    return M

def exp_map(mu, E):
    theta = np.sqrt((E ** 2).sum(0))[None]
    M = mu * np.sin(theta)
    M = M + (E * (np.cos(theta)) / theta)
    M[:, np.abs(theta[0]) <= 1e-7] = mu
    return M

def animation_matrix(N, n):
    u = np.random.normal(0, 1, size=(N, 1))
    r = np.sqrt((u ** 2).sum())
    u /= r
    t = np.random.normal(0, 1, size=(N, 1))
    t = t - (t.T.dot(u)).dot(u.T).T
    t /= np.sqrt((t ** 2).sum())
    # start = np.random.uniform(0,2*np.pi)
    # T = np.linspace(start, start+2*np.pi, n)[None, :] * t
    return r * exp_map_sphere(u, np.linspace(0.001, 2 * np.pi, n)[None] * t)


def get_percs(X, mu, K):
    s = np.random.multivariate_normal(mu.squeeze(), K, size=(50000)).T
    return np.percentile(s, np.linspace(0, 100, 75), overwrite_input=True, axis=1)
    
    return 

def create_empty_ax():
    fig, ax = plt.subplots(figsize=(4.2 * (16 / 9), 4.20))
    
    ax.set_frame_on(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    return fig, ax

def plot_data(ax, X, Y):
    return ax.scatter(X, Y, marker='x', color='k')

def fill_grad(ax, X, mu, K):
    from GPy.plotting.matplot_dep.plot_definitions import MatplotlibPlots
    mat_plot = MatplotlibPlots()
    mat_plot.fill_gradient(ax, X[:, 0], get_percs(X, mu, K), color='#687C8E', linewidth=0, alpha=1.)

def animate_kernel(fig, ax, X, mu, K, out, frames=200):
    colors = ['#f7fbff',
              '#deebf7',
              '#c6dbef',
              "#9ecae1",
              "#6baed6",
              "#4292c6",
              '#2171b5',
              '#08519c',
              '#08306b',
              ]
    
    L = GPy.util.linalg.pdinv(K + np.eye(K.shape[0]) * 1e-8)[1]
    lines = [ax.plot([], [], lw=.8, color=c)[0] for c in colors]
    
    Rs = [animation_matrix(X.shape[0], frames) for _ in lines]
    
    def init():
        for line in lines:
            line.set_data([], [])
        return lines
    def animate(i):
        for animatrix, line in zip(Rs, lines):
            # print y[:,i].shape, x.shape
            line.set_data(X[:, 0], mu + L.dot(animatrix[:, [i]]))
        return lines
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames, interval=20, blit=False,
                               repeat=True, save_count=frames,
                              )
    
    writer = animation.FFMpegFileWriter(
        fps=30,
        codec='libx264',
        extra_args=[
            '-pix_fmt', 'yuva420p',
        ],
    )
    anim.save(
        out,
        writer=writer,
        dpi=150,
        savefig_kwargs={'transparent': False, 'facecolor': 'white'},
    )
    return anim
