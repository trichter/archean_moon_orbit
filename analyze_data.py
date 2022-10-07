# Copyright 2022 Tom Eulenfeld, MIT license

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft, rfftfreq
from matplotlib.ticker import MultipleLocator
from scipy.stats import median_abs_deviation as mad
import scipy.signal


DATA = 'data/thickness.txt'
OUT = 'figs/'


def _sub(x1, x2):
    dx = (x1-x2) / 10
    return list(np.linspace(x1-dx, x2+dx, 9))


DMLOC = MultipleLocator(10)
PTICKS = (50, 30, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2)
PMTICKS =  _sub(30, 20) + _sub(20, 10) + sum((_sub(i, i-1) for i in range(10, 0, -1)), [])
SMLOC = MultipleLocator(0.01)


def load_data():
    with open(DATA) as f:
        data = f.read()
    dataset = {}
    for line in data.splitlines()[1:]:
        label, *data = line.split()
        dataset[label] = np.array(list(map(int, data)))
    return dataset


def _inverse(f):
    """frequency to period, period to frequency"""
    return 1 / f


def calc_specs(data, dt, N2, shift, N=None):
    """Multiple digital Fourier transform of different parts of the data"""
    if N2 > len(data):
        N2 = len(data)
    if N is None:
        N = N2
    freq = rfftfreq(N, dt)
    specs = []
    i = 0
    while i + N2 <= len(data):
        data2 = data[i:i+N2]
        spec = rfft(data2 - np.mean(data2), N) * 2 / N2
        specs.append(spec)
        i += shift
    return freq, specs


def calc_spec(data, dt, N=None):
    """Digital Fourier transform"""
    if N is None:
        N = len(data)
    freq = rfftfreq(N, dt)
    spec = rfft(data - np.mean(data), N) * 2 / len(data)
    return freq, spec


def get_max(f, spec, f1, f2):
    """
    Frequency and spectum at maximal spectral amplitude for f1<f<f2
    """
    i0 = np.count_nonzero(f<=f1)
    i = np.argmax(np.abs(spec[(f>f1) * (f<f2)])) + i0
    return f[i], spec[i]


def print_max(*args, freqres=None, header=False):
    f, spec = get_max(*args)
    if header:
        print('freq  period amplitude  freqres  periodres')
    if freqres is None:
        print(f'{f:.3f}  {1/f:.3f}  {abs(spec):.1f}')
    else:
        print(f'{f:.3f}  {1/f:.3f}  {abs(spec):.1f}  {freqres:.2e}  {freqres/f**2:.2e}')


def plot_overview(dataset):
    fig1 = plt.figure(figsize=(16,10))
    fig2 = plt.figure(figsize=(16,10))
    fig3 = plt.figure(figsize=(16,10))
    ax1 = None
    ax2 = None
    ax4 = None
    saxes = []
    for i, (label, data) in enumerate(dataset.items()):
        label = label.replace('_', ' ')
        ax1 = fig1.add_subplot(7, 1, 1+i, sharex=ax1)
        ax2 = fig2.add_subplot(7, 1, 1+i, sharex=ax2)
        ax4 = fig3.add_subplot(7, 1, 1+i, sharex=ax4)
        saxes.append(ax2)

        ax1.bar(1+np.arange(len(data)), data, color='k')

        f, spec = calc_spec(data, 1)
        f2, spec2 = calc_spec(data, 1, 10000)
        f4, specs4 = calc_specs(data, 1, 50, 25, 10000)
        ax2.plot(f, np.abs(spec), '.k')
        ax2.plot(f2, np.abs(spec2), 'k')
        # ax4.plot(f3, np.abs(spec3), '.k')


        print(label)
        print(f'length {len(data)}, sum {sum(data)/10:.1f}cm')
        print_max(f2, spec2, 0.05, 0.097, header=True, freqres=f[1]-f[0])
        print_max(f2, spec2, 0.097, 0.15, freqres=f[1]-f[0])
        print()

        for s in specs4:
            ax4.plot(f4, np.abs(s), 'k', alpha=0.3)
        ax4.plot(f4, np.average(np.abs(specs4), axis=0), 'k')
        ax1.annotate(label, (0.95, 0.9), xycoords='axes fraction', va='top', ha='right')
        ax2.annotate(label, (0.95, 0.9), xycoords='axes fraction', va='top', ha='right')
        ax4.annotate(label, (0.95, 0.9), xycoords='axes fraction', va='top', ha='right')
        ax3 = ax2.secondary_xaxis('top', functions=(_inverse, _inverse))
        ax5 = ax4.secondary_xaxis('top', functions=(_inverse, _inverse))

        ax1.xaxis.set_minor_locator(DMLOC)
        ax2.xaxis.set_minor_locator(SMLOC)
        ax3.set_xticks(PTICKS)
        ax3.set_xticks(PMTICKS, minor=True)
        ax4.xaxis.set_minor_locator(SMLOC)
        ax5.set_xticks(PTICKS)
        ax5.set_xticks(PMTICKS, minor=True)

        if i == 0:
            # ax3.set_xlabel('period (days)')
            ax3.set_xlabel('period (layers per cycle)')
            ax5.set_xlabel('period (layers per cycle)')
        if i > 0:
            plt.setp(ax3.get_xticklabels(), visible=False)
            plt.setp(ax5.get_xticklabels(), visible=False)
        if i < 6:
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax2.get_xticklabels(), visible=False)
            plt.setp(ax4.get_xticklabels(), visible=False)
        if i == 3:
            label = 'thickness amplitude spectrum (mm)\n'
            ax4.set_ylabel(label)
            ax2.set_ylabel(label)
            ax1.set_ylabel('thickness (mm)')
    for ax, label in zip(saxes, 'abcdefg'):
        ax.annotate(label + ')', (0, 1), (8, -6), 'axes fraction', 'offset points', va='top', size='large')

    ax2.set_xlabel('frequency (cycles per layer)')
    ax4.set_xlabel('frequency (cycles per layer)')
    ax1.set_xlabel('layer')
    ax2.set_xlim((-0.01, 0.51))
    ax4.set_xlim((-0.01, 0.51))
    fig1.savefig(OUT + 'datasets_Heubeck2022_fig10.pdf')
    fig2.savefig(OUT + 'spectra_Heubeck2022_fig14.pdf')
    fig3.savefig(OUT + 'spectra_subsets.pdf')


def series(f, amp):
    """Time series corresponding to single Fourier coefficient"""
    return lambda t: amp * np.exp(2j*np.pi*f*t)


def get_confidence(yf, n=None, p=95):
    # https://stackoverflow.com/questions/67992691/how-to-calculate-95-confidence-level-of-fourier-transform-in-python
    from scipy.stats import gamma
    if n is None:
        n = len(yf)
    threshold = np.percentile(np.abs(yf)**2, 95)
    filtered = [x for x in np.abs(yf)**2 if x <= threshold]
    var = np.mean(filtered) #/n    # already divided by n in calc_spec
    level = gamma.isf(q=(1-p/100)/2, a=1, scale=var)
    return level ** 0.5   # amplitude


def plot_composite(data):
    f, spec = calc_spec(data, 1)
    f2, spec2 = calc_spec(data, 1, 10000)

    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax4 = fig.add_subplot(313)
    nr = 1+np.arange(len(data))
    ax1.bar(nr, data, color='k')
    ax2.plot(f, np.abs(spec), '.k', label='without zero-padding')
    ax2.plot(f2, np.abs(spec2), 'k', label='with zero-padding')
    conf = get_confidence(spec, p=95)
    conf2 = get_confidence(spec, p=99)
    ax2.axhline(conf, ls='--', color='gray', label='95%, 99% confidence level', zorder=-30)
    ax2.axhline(conf2, ls='--', color='gray', zorder=-30)

    print('some relative maxima in spectra  (freq, period, amplitude')
    print_max(f2, spec2, 0, 0.5, header=True, freqres=f[1]-f[0])
    fm1, sm1 = get_max(f2, spec2, 0, 0.5)
    func = series(fm1, sm1)
    ax1.plot(nr, func(nr)+np.median(data), label=f'period {1/fm1:.2f}')

    print_max(f2, spec2, 0.1, 0.12)
    fm2, sm2 = get_max(f2, spec2, 0.1, 0.12)
    func = series(fm2, sm2)
    # ax1.plot(nr, func(nr)+0.5*np.median(data), label=f'period {1/fm2:.2f}')

    print_max(f2, spec2, 0.3, 0.4)
    print_max(f2, spec2, 0.4, 0.45)

    f3, _ = calc_specs(data, 1, 50, 25)
    f4, specs4 = calc_specs(data, 1, 50, 25, N=10000)
    fsm, sm = zip(*[get_max(f4, s, 0.05, 0.09) for s in specs4])
    fsm = np.array(fsm)
    fsmed = np.median(fsm)
    ax4.axvline(fsmed, color='0.2')
    for i, s in enumerate(specs4):
        ax4.plot(f4, np.abs(s), '0.4', alpha=0.3)
        ax4.plot((fsm[i], fsmed), np.abs((sm[i], sm[i])), color='0.2')


    kw = dict(xytext=(30, -5), textcoords='offset points')
    err = mad(1 / fsm, scale='normal') / len(fsm)**0.5
    ax4.annotate(f'{1/fsmed:.2f} +- {err:.2f}', (fsmed, np.max(np.abs(sm))-0.5), arrowprops=dict(arrowstyle='->', color='C0'), **kw)



    ax3 = ax2.secondary_xaxis('top', functions=(_inverse, _inverse))
    ax5 = ax4.secondary_xaxis('top', functions=(_inverse, _inverse))


    ax2.annotate(round(1/fm1, 2), (fm1, abs(sm1)), arrowprops=dict(arrowstyle='->', color='C0'), **kw)
    # ax2.annotate(round(1/fm2, 2), (fm2, abs(sm2)), arrowprops=dict(arrowstyle='->', color='C1'), **kw)

    ax1.xaxis.set_minor_locator(DMLOC)
    ax2.xaxis.set_minor_locator(SMLOC)
    ax4.xaxis.set_minor_locator(SMLOC)
    ax3.set_xticks(PTICKS)
    ax3.set_xticks(PMTICKS, minor=True)
    ax5.set_xticks(PTICKS)
    ax5.set_xticks(PMTICKS, minor=True)
    label = 'thickness\namplitude spectrum (mm)'
    ax1.set_xlabel('layer')
    ax1.set_ylabel('thickness (mm)')
    ax2.set_xlabel('frequency (cycles per layer)')
    ax4.set_xlabel('frequency (cycles per layer)')
    ax2.set_ylabel(label)
    ax4.set_ylabel(label)
    ax1.set_xlim(-4, len(data)+4)
    ax2.set_xlim((-0.01, 0.51))
    ax4.set_xlim((-0.01, 0.51))
    ax3.set_xlabel('period (layers per cycle)')
    ax5.set_xlabel('period (layers per cycle)')
    for ax, label in zip((ax1, ax2, ax4), 'abc'):
        ax.annotate(label + ')', (0, 1), (8, -6), 'axes fraction', 'offset points', va='top', size='large')
    ax1.legend()
    ax2.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(OUT + 'composite_Heubeck2022_fig15.pdf')
    print('length of composite dataset', len(data))
    print('frequency resolution', np.round(np.diff(f)[0], 5))
    print('frequency sampling', np.round(np.diff(f2)[0], 6))
    print('frequency resolution for short data sets', np.round(np.diff(f3)[0], 5))
    print(f'median and standard error {1/fsmed:.2f} +- {err:.2f}')


def _cut_spec(f, t, s):
    i1 = np.nonzero(t>=80)[0][0]
    i2 = np.nonzero(t>115)[0][0]
    i3 = np.nonzero(f>=0.055)[0][0]
    i4 = np.nonzero(f>0.085)[0][0]
    return f[i3:i4], t[i1:i2], s[i3:i4, i1:i2]


def plot_spectrogram(data, N, ax, analyze=False):
    f, t, spec = scipy.signal.spectrogram(data-np.mean(data), 1,
                                          window='boxcar', nperseg=N,
                                          noverlap=N-1, nfft=1024,
                                          mode='complex')
    im = ax.pcolormesh(t, f, np.abs(spec), shading='auto', cmap='plasma', rasterized=True)
    if analyze:
         # cut out interesting range
         f2, _, s2 = _cut_spec(f, t, spec)
         # plot specs on the right side
         ax.plot(200+np.abs(s2), f2[:, np.newaxis], '0.5', alpha=0.2)

         fs = f2[np.argmax(np.abs(s2), axis=0)]  # median frequency with maximal amplitude
         tm = np.median(1/fs) # median period
         print(N, tm)
         err = mad(1/fs)#, scale='normal')
         print(f"{tm:.2f} +- {err:.2f}")
         # plot median frequency with maximal amplitude as horizontal bar
         ax.plot([80, 115], [1/tm]*2, 'k')
         # plot median and MAD on the rigth side
         yerr = [[1/tm-1/(tm+err)], [1/(tm-err)-1/tm]]
         xpos = np.mean(np.max(np.abs(s2), axis=0))
         ax.errorbar([200+xpos], [1/tm], yerr=yerr, zorder=20, color='k')
         ax.plot([200+xpos-5, 200+xpos+5], [1/tm, 1/tm], 'k')
    msg = f'window length {N}'
    if analyze:
        msg = msg + f'\nly/2={tm:.2f}$\\pm${err:.2f}'
    ax.annotate(msg, (1, 1), (-5, -5), 'axes fraction', 'offset points',
                ha='right', va='top')
    return f, t, spec, im


def spectrograms(data):
    fig = plt.figure(figsize=(16,10))
    ax1 = fig.add_subplot(511)
    ax2 = fig.add_subplot(512, sharex=ax1)
    ax3 = fig.add_subplot(513, sharex=ax1)
    ax4 = fig.add_subplot(514, sharex=ax1, sharey=ax3)
    ax5 = fig.add_subplot(515, sharex=ax1, sharey=ax3)

    ax1.bar(1+np.arange(len(data)), data, color='k')
    plot_spectrogram(data, 50, ax2)
    plot_spectrogram(data, 50, ax3, analyze=True)
    plot_spectrogram(data, 75, ax4, analyze=True)
    _, _, _, im = plot_spectrogram(data, 100, ax5, analyze=True)

    for ax in (ax3, ax4, ax5):
        ax_ = ax.secondary_yaxis('right', functions=(_inverse, _inverse))
        if ax == ax4:
            ax_.set_ylabel('period (layers per cycle)')
    ax1.set_xlim(-2, len(data)+3)
    y1, y2 = 0.055, 0.085
    ax3.set_ylim(y1, y2)
    # ax2.plot(*rect, 'C0', zorder=-20)
    # ax2.axhspan(y1, y2, color='0.5', zorder=-20)
    ax2.axhline(y1, color='w')
    ax2.axhline(y2, color='w')

    ax1.set_xlabel('layer')
    ax1.set_ylabel('thickness (mm)')
    ax4.set_ylabel('frequency (cycles per layer)')
    cax = fig.add_axes([0.15, 0.3, 0.005, 0.08])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks(cbar.ax.get_ylim())
    cbar.ax.set_yticklabels(['low', 'high'])
    cbar.set_label('spectral\namplitude')
    ax5.set_xlabel('layer (centrum of window for Short-time Fourier transform)')
    for ax, label in zip((ax1, ax2, ax3, ax4, ax5), 'abcde'):
        ax.annotate(label + ')', (0, 1), (8, -6), 'axes fraction', 'offset points', va='top', size='large')
    fig.savefig(OUT + 'spectrograms.pdf', dpi=200)


if __name__ == '__main__':
    dataset = load_data()
    plot_overview(dataset)  # Heubeck et al. 2022 figure 10 and 14
    plot_composite(dataset['DJvR+CH_composite'])  # figure 3 resp. Heubeck et al. 2022 figure 15
    spectrograms(dataset['DJvR+CH_composite'])  # figure 3
    plt.show()
