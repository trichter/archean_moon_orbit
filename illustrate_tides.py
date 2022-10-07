# Copyright 2022 Tom Eulenfeld, MIT license

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.image import imread
import numpy as np
from numpy.fft import rfft, rfftfreq
import os.path


TIDES = 'data/tides.npz'
OUT = 'figs/'


def run_pygtide():
    """Calculate tidal gravity using ETERNA predict and pygtide"""
    if os.path.exists(TIDES):
        with np.load(TIDES) as npz:
            return npz['t'], npz['d1'], npz['d2'], npz['d3']
    import pygtide
    d1 = pygtide.predict_series(0, 0, 0, '2000-01-01', 24*365, 60)
    d2 = pygtide.predict_series(30, 0, 0, '2000-01-01', 24*365, 60)
    d3 = pygtide.predict_series(60, 0, 0, '2000-01-01', 24*365, 60)
    t = np.arange(len(d1)) / 60 / 24
    np.savez(TIDES, t=t, d1=d1, d2=d2, d3=d3)
    return t, d1, d2, d3


def plot_tides(t, d1, d2, d3):
    fig = plt.figure(figsize=(10, 10))
    m = fig.subplot_mosaic('0000000;1111111;2222233')
    ax0 = m['0']
    ax1 = m['1']
    ax2 = m['2']
    ax3 = m['3']
    plot_tidal_spectrum(t, d1, d2, d3, ax=ax0)
    ind = t < 100
    ax1.plot(t[ind], d1[ind])
    ax1.plot(t[ind], d2[ind])
    ax1.plot(t[ind], d3[ind])

    kw = dict(text='', xycoords='data', textcoords='data')
    kw2 = dict(arrowstyle='|-|,widthA=0.5,widthB=0.5', lw=2)
    ax1.annotate(xy=(13, -1900), xytext=(13 + 29.53, -1900), arrowprops=dict(color='C1', **kw2), **kw)
    ax1.annotate(xy=(13, -1600), xytext=(13 + 27.32, -1600), arrowprops=dict(color='C2', **kw2), **kw)
    ax1.annotate(r'synodic month ($T_{\rm l}$)', xy=(30, -1850), ha='center')
    ax1.annotate('sidereal month ($T$)', xy=(30, -1550), ha='center')
    ax1.set_xlim(0, 100)
    ax1.set_yticks([])
    ax1.set_ylim(-2050, None)
    ax1.xaxis.set_minor_locator(MultipleLocator(5))
    ind = t < 2
    ax2.plot(t[ind], d1[ind], label='latitude 0°  $\it{semidiurnal}$ tide, two neap-spring-neap cycles per $\it{synodic}$ month')
    ax2.plot(t[ind], d2[ind], label='latitude 30° mixed tide with dominant $\it{semidiurnal}$ component, two neap-spring-neap cycles per $\it{synodic}$ month')
    ax2.plot(t[ind], d3[ind], label='latitude 60° mixed tide with dominant $\it{diurnal}$ component, two neap-spring-neap cycles per $\it{sidereal}$ month')
    ax2.set_xlim(0, 2)
    ax2.set_xticks([0, 1, 2])
    #ax2.set_xticks([0.25, 0.5,0.75, 1.25,  1.5, 1.75], minor=True)
    ax2.xaxis.set_minor_locator(MultipleLocator(0.25))
    kw = dict(facecolor='0.7')
    ax2.axvspan(0.11, 0.73, **kw)
    ax2.axvspan(1.11, 1.73, **kw)
    kw = dict(xycoords='data')
    ax2.annotate('strong current\ndeposit of sand', (0.75, -1300), **kw)
    ax2.annotate('weak current\ndeposit of mud', (1.13, -1300), **kw)
    ax2.set_ylim(-1400, None)
    ax2.set_yticks([])

    ax2.set_xlabel('days since 2000-01-01')
    ax1.set_ylabel('tidal gravity', labelpad=10)
    ax2.set_ylabel('tidal gravity', labelpad=10)

    imdata = imread(OUT + 'cutout_with_scale.png')
    ax3.imshow(imdata)
    ax3.set_xticks([])
    ax3.set_yticks([])

    akw = dict(bbox=dict(boxstyle='circle', facecolor='w', edgecolor='k', alpha=0.5))
    for l, x in zip('1212', (0.08, 0.39, 0.58, 0.89)):
        ax2.annotate(l, (x, 0.23), xycoords='axes fraction', **akw)
    for l, x in zip('121212', (150, 230, 310, 390, 480, 580)):
        ax3.annotate(l, (x, 620), **akw)
    for ax, label in zip((ax0, ax1, ax2, ax3), 'abcd'):
        ax.annotate(label + ')', (0, 1), (8, -6), 'axes fraction', 'offset points', va='top', size='large')

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    fig.legend(loc='lower center', bbox_to_anchor=(0.1, 0., 0.8, 0.1))
    fig.savefig(OUT + 'tidal_gravity.pdf', bbox_inches='tight')


def find_nearest(a, v):
    idx = (np.abs(a - v)).argmin()
    return idx


def plot_tidal_spectrum(t, d1, d2, d3, ax=None):
    N = len(d1)
    freq = rfftfreq(N, t[1] - t[0])
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
    specs = []
    for d in (d1, d2, d3):
        spec = rfft(d - np.mean(d)) * 2 / N
        specs.append(spec)
    for spec in specs:
        ax.plot(freq, np.abs(spec))
    # for c, spec in zip('210', specs[::-1]):
    #     ax.plot(freq[freq<1.5], np.abs(spec)[freq<1.5], color='C' + c)
    max_spec = np.max(np.abs(specs), axis=0)
    ax.set_xlim(0, 2.5)
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlabel('frequency (cycles per day)')
    ax.set_ylabel('tidal gravity', labelpad=10)
    ax.set_yticks([])
    constituents = {'M2': 1.9322736, 'S2': 2, 'O1': 0.9295357, 'K1': 1.0027379}
    for c, f in constituents.items():
        idx = find_nearest(freq, f)
        ax.annotate(f'${c[0]}_{c[1]}$', (freq[idx], max_spec[idx]), (8, -2), 'data', 'offset points', ha='center')
    kw = dict(text='', xycoords='data', textcoords='data')
    kw2 = dict(arrowstyle='<->,head_length=0.2,head_width=0.1')
    ypos = 380
    ax.annotate(xy=(constituents['O1'], ypos), xytext=(constituents['K1'], ypos), arrowprops=kw2, **kw)
    ax.annotate(xy=(constituents['M2'], ypos), xytext=(constituents['S2'], ypos), arrowprops=kw2, **kw)
    ax.annotate('2/$T$', xy=(0.965, ypos+30), ha='center')
    ax.annotate(r'2/$T_{\rm l}$', xy=(1.978, ypos+30), ha='center')

    if fig is not None:
        ax.legend()
        fig.savefig(OUT + 'tidal_gravity_spectrum.pdf')


if __name__ == '__main__':
    args = run_pygtide()
    plot_tides(*args)
    plot_tidal_spectrum(*args)
    plt.show()
