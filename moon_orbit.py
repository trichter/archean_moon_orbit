# Copyright 2022 Tom Eulenfeld, MIT license
"""
Relationship between number of days per month and Moon-Earth distance and related plots

Main function in this module is func:a2ly

a  ratio of lunar orbital radius to todays value
ly2 semidiuanal tides: lunar days per synodical month (Tl / tl)
ly1 diurnal tides: sidereal days per sideaial month (T / t)

diurnal: True -> diurnal tides, False -> semidiurnal tides
"""


from functools import partial
from math import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.legend_handler import HandlerLine2D
import numpy as np
from scipy.optimize import root_scalar


t0 = 86164  # s, todays sidereal day
ts0 = 86400   # s, todays solar day
T0 = 27.322 * ts0  # s, todays sidereal month (orbital period)
Y = 365.24 * ts0  # s, length of year (assumed constant)
beta0 = 0.211  # todays ratio of solar tide-raising torque and lunar tide-raising torque (msun/rsun**3/m*r0**3)**2
alpha0 = 0.203  # todays ratio of Earths angular momentum and lunar orbital angular momentum Le0/Lm0

# used for correction due to higher Earth's bulge
Ie = 8.034e37  # kg∗m**2, Earth's moment of ineartia, Lambeck, 1980
k2f = 0.93  # Earth's fluid Love number around rotation axis
rearth = 6.371e6  # m, radius of Earth
G = 6.6743e-11  # m**3/kg/s**2, gravitational constant
gamma = 8*pi**2*k2f*rearth**5 / (9*G*Ie) / t0 ** 2

# The following constants are only used in non-essential functions
# e.g. a2ly_Coughenour2013, plot_other_parameters
m = 7.342e22  # kg, mass of Moon
msun = 1.989e30  # kg, mass of Sun
r0 = 3.844e8  # m, todays mean lunar distance (semi-major axis)
rsun = 1.496e11  # m, todays Earth-Sun distance
Lm0 = 2*pi*m*r0**2/T0  # kg∗m**2/s, todays lunar orbital angular momentum
Le0 = 2 * pi * Ie / t0  # kg∗m**2/s, todays angular momentum of Earth's rotation
Ltot0 = Lm0 + Le0  # kg∗m**2/s, todays total angular momentum
mearth = 5.972e24  # kg, mass of Earth
beta0check = ((msun / m) * r0**3 / rsun**3)**2
alpha1 = 2*pi*mearth*rsun**2/Y/ Lm0  # todays ratio of Earths *orbital* angular momentum and lunar orbital angular momentum

print(f'beta0 is set to {beta0}, exact theoretical value is {beta0check:.5f}')
assert abs(1/ts0 - 1/t0 - 1/Y) < 1e-6
print(f'gamma={gamma:.6f}')

ALIM = (0.48, 1.02)

WILLIAMS2000 = {
    'a': np.array([51.9, 54.6, 54.7, #57.1,
                   58.34, 60.27]) / 60.27,
    'Tl/tl': [31.7, 31.1, 31.4, #30.3,
              29.5, 28.53],
    'Y/T': np.array([16.7, 15.5, 15.3, #14.5,
                     14.1, 13.37]),
    'ts': [17.1, 18.8, 18.9, #20.8,
           21.9, 24.00],
    'Y/t': np.array([515, 467, 465, #423,
                     401, 366.24]),
    'age': [2.45, 2.45, 0.90, 0.62, 0],
    'reference': [
        'Walker and Zahnle 1986, Trendall 1973, Western Australia',
        'Williams 1989c, 1999, Western Australia',
        'Sonett and Chan, 1998, Big Cottonwood Formation, Utah',
        'Sonett et al., 1996b, Big Cottonwood Formation, Utah',
        'Williams, 1989a, b, c, 1990, 1991, 1994, 1997, Elatina Formation and Reynella Siltstone, South Australia',  # a from Deubner
        'modern'
        ],
    'plot':
        []
    }
WILLIAMS2000['T/t'] = WILLIAMS2000['Y/t'] / WILLIAMS2000['Y/T']

ORBIT_TIME_DATA = [
     (2.45, 0.861, 'd', 1, 'Weeli Wolli, Walker & Zahnle 1986'),
     (2.45, 0.906, 'd', 0.5, 'Weeli Wolli, Williams 2000'),
     (0.90, 0.908, 's', 1, 'Cottonwood, Sonnet & Chan 1998'),
     (0.62, 0.968, '^', 1, 'Elatina, Reynella, Williams 2000'),
     (1.40, 0.886, 'o', 1, 'Xiamaling, Meyers & Malinverno 2018'),
     (2.465, 51.65*rearth/r0, '.', 1, 'Dales Gorge, OCR2019'),
     (0.259, 3.7611e8/r0, 'x', 1, 'Dales Gorge & others, Zhou et al. 2022'),
     (0.455, 3.7064e8/r0, 'x', 1, None),
     (0.655, 3.5731e8/r0, 'x', 1, None),
     (2.465, 3.2432e8/r0, 'x', 1, None),
     (2.460, 321.8e6/r0, 'D', 1, 'Joffre, Lantink et al. 2022')
    ]

PMODS = ('Webb 1982 curve d',
         'Daher et al. 2021 PD',
         'Daher et al. 2021 55Ma',
         'Daher et al. 2021 116Ma',
         'Daher et al. 2021 252Ma',
         'Tyler 2021 T=50',
         'Tyler 2021 T=40',
         'Tyler 2021 T=30',
         'Farhat et al. 2022')


def a2ly_Runcorn1979(a, diurnal=False, beta0=beta0):
    """
    Runcorn 1979, equation 12, solved for ly, assuming constant moment of inertia

    Equals a2ly_simple when applying the substitions.
    13.4 = Y/T0
    4.82 = 1 / alpha0
    27.3 = T0/t0
    """
    assert not diurnal
    bracket_term = 1 + (1 + beta0/13 - a**0.5 - beta0/13*a**6.5) * 4.82 #/ alpha0
    rhs = bracket_term * a** 1.5 * 27.3 #T0 / t0
    lhs = (1-a**1.5 /13.4)#* T0 / Y)
    return (rhs - 1) / lhs

def a2ly_Coughenour2013(a, diurnal=False):
    """
    Coughenour et al. 2013, equation 13, solved for ly, added Ie at appropriate place
    """
    rhs = Ltot0 + beta0*(Lm0-2*pi*m*a**2*r0**2/T(a))*a**6
    t1 = 2*pi/T(a)
    t2 = (rhs / t1  - m*a**2*r0**2) / Ie
    if diurnal:
        return t2
    else:
        return (t2 - 1) / T(a) * Tl(a)

def X(a, beta0=beta0, a0=1):
    return a0**0.5 + beta0/13*a0**6.5 - a**0.5 - beta0/13*a**6.5

def a2ly_simple(a, diurnal=False, alpha0=alpha0, beta0=beta0, mu=1):
    """Equation 16 and 17"""
    rhs = X(a, beta0=beta0)/alpha0 + 1
    if diurnal:
        return rhs/t0/mu * T(a)
    else:
        return (rhs/t0/mu - 1/T(a)) * Tl(a)

def cardanos_formula(p, q):
    t = (q**2/4 + p**3/27)**0.5
    return np.cbrt(-q/2 + t) + np.cbrt(-q/2 - t)

def a2ly(a, diurnal=False, alpha0=alpha0, beta0=beta0, mu=1, gamma=gamma,
         t01=t0, a0=1):
    """
    Equation 22 and 23

    For a0=1, t01=t0, gamma=0 this equals a2ly_simple.

    :param diurnal: True -> diurnal tides, False -> semidiurnal tides
    :param alpha0:
    """
    LedivLe0 = X(a, beta0=beta0, a0=a0) / alpha0 + t0/t01
    if gamma == 0:
        t0divt = LedivLe0 / mu
    else:
        p = (1 - gamma) / gamma
        q = -LedivLe0 / gamma / mu
        t0divt = cardanos_formula(p, q)
    if diurnal:
        return t0divt / t0 * T(a)
    else:
        return (t0divt/t0 - 1/T(a)) * Tl(a)

def a2ly_resonance(a, diurnal=False):
    ts_ = 21 * 3600
    t_ = 1 / (1/ts_+1/Y)
    if diurnal:
        ly = T(a) / t_
    else:
        ly = (1 / t_ - 1 / T(a)) / (1 / T(a) - 1 / Y)
    return ly

def T(a):
    """Sidereal month (Keppler's 3rd law, equation 5)"""
    return T0 * a ** 1.5

def Tl(a):
    """Synodic month"""
    return 1 / (1 / T(a) - 1 / Y)

def t(a, ly, diurnal=False):
    """Sidereal day (equations 13 and 14)"""
    if diurnal:
        return T(a) / ly  # K1, beat of O1 and K1
    else:
        return 1 / (1 / T(a) + ly / Tl(a))  # M2, beat of S2 and M2, ly = Tl / tl

def ts(a, ly, diurnal=False):
    """Solar day"""
    return 1 / (1 / t(a, ly, diurnal=diurnal) - 1 / Y)

def _a2ly_solve(a, ly, **kwargs):
    return ly - a2ly(a, **kwargs)

def solve(lys, bracket=None, **kwargs):
    func = partial(_a2ly_solve, **kwargs)
    aa = [root_scalar(func, (ly,), bracket=bracket).root for ly in lys]
    return np.array(aa)

def format_err(vs, d=1):
    vmin, v, vmax = sorted(vs)
    return f'{v:.{d}f} +{vmax-v:.{d}f} -{v-vmin:.{d}f}'

def format_err_ltx(vs, d=1):
    vmin, v, vmax = sorted(vs)
    return f'${v:.{d}f}^{{+{vmax-v:.{d}f}}}_{{-{v-vmin:.{d}f}}}$'

def format_array(vs, d=1):
    return ' '.join(f'{v:.{d}f}' for v in vs)

def print_moon_orbit(lys, abracket=(0.4, 0.8), fmt=format_array,
                     diurnal=False, **kwargs):
    try:
        a = solve(lys, bracket=abracket, diurnal=diurnal, **kwargs)
    except ValueError as ex:
        print(ex)
    else:
        ts_ = ts(a, lys, diurnal=diurnal)
        t_ = t(a, lys, diurnal=diurnal)
        tl_ = 1 / (1 / t_ - 1 / T(a))
        print('assuming ' + 'semi' * (not diurnal) + 'diurnal tides')
        print('    ly', fmt(lys))
        print('   ly1', fmt(T(a) / t_))
        print('   ly2', fmt(Tl(a) / tl_))
        print('     a', fmt(a, d=3))
        print('   Y/T', fmt(Y / T(a)))
        # print('   T/ts', fmt(T(a) / ts_))
        print('  Tl/ts', fmt(Tl(a) / ts_))
        print(' ts (h)', fmt(ts_ / 3600))
        print('   Y/ts', fmt(Y / ts_, d=0))
    print()

def print_moon_orbit_today(fmt=format_array):
    a = np.ones(1)
    ts_ = ts0
    t_ = t0
    tl_ = 1 / (1 / t_ - 1 / T0)
    print('today')
    print('   ly1', round(T0 / t_, 3))
    print('   ly2', fmt(Tl(a) / tl_))
    print('     a', fmt(a, d=2))
    print('   Y/T', fmt(Y / T(a), d=2))
    # print('   T/ts', fmt(T(a) / ts_, d=3))
    print('  Tl/ts', fmt(Tl(a) / ts_, d=3))
    print('   Y/ts', round(Y / ts_))
    print()

def print_moon_orbit_21_hours(lys, fmt=format_array, diurnal=False):
    lys = np.array(lys)
    ts_ = 21 * 3600
    t_ = 1 / (1 / Y + 1 / ts_)
    if diurnal:
        T_ = lys * t_
    else:
        T_ = (1 + lys) / (1 / t_ + lys / Y )
    tl_ = 1 / (1 / t_ - 1 / T_)
    a = (T_ / T0) ** (2 / 3)
    print('21-hour resonance assuming ' + 'semi' * (not diurnal) + 'diurnal tides')
    print('    ly', fmt(lys))
    print('   ly1', fmt(T_ / t_, 3))
    print('   ly2', fmt(Tl(a) / tl_))
    print('     a', fmt(a, d=3))
    print('   Y/T', fmt(Y / T_))
    print('   Tl/ts', fmt(Tl(a) / ts_))
    # print('   T/ts', fmt( T / ts_))
    print('   Y/ts', round(Y / ts_))
    print()


def plot_moon_orbit_different_params(alim=ALIM):
    a = np.linspace(*alim, 1001)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)

    bs = (0.24, beta0, 0.18, 0.15, 0.12)
    colors = ('0.8', 'k', '0.3', '0.5', '0.7')
    for ax, diurnal in [[axes[0, 0], False], [axes[1, 0], True]]:
        for b0, c in zip(bs, colors):
            lys = a2ly(a, diurnal=diurnal, beta0=b0)
            ax.plot(a, lys, color=c, label=r'$\beta_0$={}'.format(b0))
        ax.axhline(30, ls='--', zorder=-50)

    mus = (0.99, 1.00, 1.01, 1.02)
    colors = ('0.8', 'k', '0.5', '0.7')
    for ax, diurnal in [[axes[0, 1], False], [axes[1, 1], True]]:
        for mu, c in zip(mus, colors):
            lys = a2ly(a, diurnal=diurnal, mu=mu)
            ax.plot(a, lys, color=c, label=r'$\mu$={}'.format(mu))
        ax.axhline(30, ls='--', zorder=-50)

    ts_ = 21 * 3600
    t_ = 1 / (1 / Y + 1 / ts_)
    aa = (0.9, 0.91, 0.92, 0.93, 0.94)
    for ax, diurnal in [[axes[0, 2], False], [axes[1, 2], True]]:
        lys = a2ly(a, diurnal=diurnal)
        ax.plot(a, lys, color='k')
        lyr = a2ly_resonance(a, diurnal=diurnal)
        ind = np.logical_and(0.9 <= a, a <0.951)
        ax.plot(a[ind], lyr[ind], 'k--')#, color='gray')
        for a0 in aa:
            lyr2 = a2ly(a, a0=a0, t01=t_, diurnal=diurnal)
            ax.plot(a[a<a0], lyr2[a<a0], '--', color='0.5')
        ax.axhline(30, ls='--', zorder=-50)

    ax.set_xlim(alim)
    ax.set_ylim(25.8, 33.2)
    lkw = dict(bbox_to_anchor=(0.7, 0.01), loc='lower center', fontsize='small', frameon=False)
    axes[0, 0].legend(**lkw)
    axes[1, 0].legend(**lkw)
    axes[0, 1].legend(**lkw)
    axes[1, 1].legend(**lkw)
    axes[0, 0].xaxis.set_minor_locator(MultipleLocator(0.02))
    axes[1, 0].set_xlabel('orbital radius a')
    axes[1, 1].set_xlabel('orbital radius a')
    axes[1, 2].set_xlabel('orbital radius a')
    Tltl = r'$T_{\rm l}/t_{\rm l}$'
    axes[0, 0].set_ylabel(f'lunar days per synodic month\nly$_2$={Tltl}')
    axes[1, 0].set_ylabel('sidereal days per sidereal month\nly$_1$=$T/t$')
    ankw = dict(xy=(0, 1), xytext=(8, -8), xycoords='axes fraction',
                textcoords='offset points', va='top', size='large')
    axes[0, 0].annotate('a) ratio of torques\n    semidiurnal', **ankw)
    axes[1, 0].annotate('b) ratio of torques\n    diurnal', **ankw)
    axes[0, 1].annotate('c) change in moment of inertia\n    semidiurnal', **ankw)
    axes[1, 1].annotate('d) change in moment of inertia\n    diurnal', **ankw)
    axes[0, 2].annotate('e) atmospheric resonance\n    semidiurnal', **ankw)
    axes[1, 2].annotate('f) atmospheric resonance\n    diurnal', **ankw)
    axes[0, 2].annotate('21-hour resonance', (0.90, 26.6), rotation=71, fontsize='small')
    axes[1, 2].annotate('21-hour resonance', (0.902, 26.1), rotation=68, fontsize='small')
    fig.tight_layout()
    fig.savefig('figs/moon_orbit_params.pdf')


def plot_moon_orbit_different_params2(alim=ALIM):
    a = np.linspace(*alim, 1001)
    fig, axes = plt.subplots(2, 1, figsize=(5, 6), sharex=True, sharey=True)

    alphas = (0.201, alpha0, 0.205, 0.2075)
    colors = ('0.8', 'k', '0.5', '0.7')
    for ax, diurnal in [[axes[0], False], [axes[1], True]]:
        for a0, c in zip(alphas, colors):
            lys = a2ly(a, diurnal=diurnal, alpha0=a0)
            ax.plot(a, lys, color=c, label=r'$\alpha_0$={}'.format(a0) + ' (Runcorn 1979)' * (a0 == 0.2075))
        ax.axhline(30, ls='--', zorder=-50)

    ax.set_xlim(alim)
    ax.set_ylim(25.8, 33.2)
    lkw = dict(bbox_to_anchor=(0.7, 0.01), loc='lower center', fontsize='small', frameon=False)
    axes[0].legend(**lkw)
    axes[1].legend(**lkw)
    axes[0].xaxis.set_minor_locator(MultipleLocator(0.02))
    axes[1].set_xlabel('orbital radius a')
    Tltl = r'$T_{\rm l}/t_{\rm l}$'
    axes[0].set_ylabel(f'lunar days per synodic month\nly$_2$={Tltl}')
    axes[1].set_ylabel('sidereal days per sidereal month\nly$_1$=$T/t$')
    ankw = dict(xy=(0, 1), xytext=(8, -8), xycoords='axes fraction',
                textcoords='offset points', va='top', size='large')
    axes[0].annotate('a) ratio of angular momenta\n    semidiurnal', **ankw)
    axes[1].annotate('b) ratio of angular momenta\n    diurnal', **ankw)
    fig.tight_layout()
    fig.savefig('figs/moon_orbit_params2.pdf')


def plot_moon_orbit(lys=None, ly_range=None, ly_print_error=None,
                    ly_annotate=None, pos_annotate=0.3,
                    alim=ALIM, abracket=(0.4, 0.8),
                    abracket2=None,
                    orbit_data=WILLIAMS2000,
                    check_goldreich1966=False):
    a = np.linspace(*alim, 101)
    ar = np.linspace(alim[0], 0.951, 101)
    ly = a2ly(a, diurnal=False)
    ly2 = a2ly(a, diurnal=True)
    ly3 = a2ly(a, diurnal=False, gamma=0)
    ly4 = a2ly_Coughenour2013(a, diurnal=False)
    lyr = a2ly_resonance(ar, diurnal=False)

    Tltl = r'$T_{\rm l}/t_{\rm l}$'
    Yts = r'$Y/t_{\rm s}$'
    tsl = r'$t_{\rm s}$'

    fig, mosaic = plt.subplot_mosaic('1\n1\n1\n1\n2\n2\n3\n3', figsize=(10, 9), sharex=True)
    ax1 = mosaic['1']
    ax2 = mosaic['2']
    ax3 = mosaic['3']
    ax1.plot(a, ly, 'k', label=f'lunar days per synodic month {Tltl}, semidiurnal tide, equation (23)')
    ax1.plot(a, ly2, '0.5', zorder=0, label='sidereal days per sidereal month $T/t$, diurnal tide, equation (22)')
    ax1.plot(a, ly3, 'k--', dashes=(1, 2, 4, 2), label=f'{Tltl} without correction, $\gamma$=0, semidiurnal tide, equation (17)')
    ax1.plot(a, ly4, 'k:', dashes=(1, 5), label=f'{Tltl} Coughenour et al. 2013, semidiurnal tide')
    #ax1.plot(a, ly4, 'r:', label=f'{Tltl} Runcorn 1979, semidiurnal tide')
    ax1.plot(ar, lyr, 'k--', label='21-hour resonance, semidiurnal tide')
    # Tldivts = Tl(a) / ts(a, ly)
    # ax1.plot(a, Tldivts, 'C3', label='Tl/ts')
    ax2.plot(a, Y / T(a), 'k')
    ax3.plot(a, ts(a, ly) / 3600, 'k')
    ax3.plot(ar, np.ones(len(ar))*21, 'k--')
    if lys is not None:
        label = 'observed number of layers\nper two neap-spring-neap cycles   '
        ax1.axhline(lys[0], ls='--', zorder=-50, label=label)

    ax1.set_ylabel('number of "days"\nper two neap-spring-neap cycles ly')
    ax2.set_ylabel('sidereal months\nper year $Y/T$')
    ax3.set_ylabel('solar day ts (hours)')
    ax3.set_xlabel('orbital radius a')
    ax1.set_yticks(np.arange(10, 36, 2))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.yaxis.set_minor_locator(MultipleLocator(5))
    ax3.set_yticks(np.arange(8, 25, 4))
    ax3.yaxis.set_minor_locator(MultipleLocator(1))
    ax3.xaxis.set_minor_locator(MultipleLocator(0.02))
    ax3.set_xlim(alim[0]-0.01, alim[1]+0.01)

    def _inverse_factor(v):
        return Y / (v * 3600)

    ax5 = ax3.secondary_yaxis(-0.08, functions=(_inverse_factor, _inverse_factor))
    ax5.set_yticks([400, 500, 600, 700, 800, 1000])
    ax5.yaxis.set_minor_locator(MultipleLocator(50))
    ax5.set_ylabel(f'solar days per year {Yts}')

    if orbit_data is not None:
        adata = orbit_data['a']
        ax1.plot(adata, orbit_data['Tl/tl'], 'kx')
        ax1.plot(adata, orbit_data['T/t'], 'x', color='0.5')
        ax2.plot(adata, orbit_data['Y/T'], 'kx')
        ax3.plot(adata, orbit_data['ts'], 'kx')
    if lys is not None:
        a = solve(lys, diurnal=True, bracket=abracket)
        for ax in (ax1, ax2, ax3):
            for a_ in a:
                vline1 = ax.axvline(a_, color='C0')
        a = solve(lys, diurnal=False, bracket=abracket)
        for ax in (ax1, ax2, ax3):
            a_ = a[0]
            vline2 = ax.axvline(a_, color='C1', ls='--', alpha=0.3)
        if abracket2:
            a = solve(lys, diurnal=True, bracket=abracket2)
            for ax in (ax1, ax2, ax3):
                for a_ in a:
                    ax.axvline(a_, color='C0', alpha=0.6)
            a = solve(lys, diurnal=False, bracket=abracket2)
            for ax in (ax1, ax2, ax3):
                a_ = a[0]
                ax.axvline(a_, color='C1', ls='--', alpha=0.2)

    if ly_range is not None:
        a = sorted(solve(ly_range, diurnal=True, bracket=abracket))
        for ax in (ax1, ax2, ax3):
            spanplot = ax.axvspan(*a, color='C0', alpha=0.2, zorder=-30)
        if abracket2 is not None:
            a = sorted(solve(ly_range, diurnal=True, bracket=abracket2))
            for ax in (ax1, ax2, ax3):
                ax.axvspan(*a, color='C0', alpha=0.15, zorder=-30)
    if ly_annotate is not None:
        ly = ly_annotate
        a = solve([ly], diurnal=True, bracket=abracket)[0]
        ts_ = ts(a, ly, diurnal=True)
        ax1.annotate(f'$ly_1$ = {ly:.1f}\na = {a:.3f}', (pos_annotate, 0.75), xycoords='axes fraction', va='bottom')
        ax2.annotate(f'$Y/T$ = {Y / T(a):.1f}', (pos_annotate, 0.05), xycoords='axes fraction', va='bottom')
        label = f'{tsl} = {ts_/3600:.1f}h\n{Yts} = {Y/ts_:.0f}'
        ax3.annotate(label, (pos_annotate, 0.75), xycoords='axes fraction')
    for ax, label in zip((ax1, ax2, ax3), 'abc'):
        ax.annotate(label + ')', (0, 1), (8, -15), 'axes fraction', 'offset points', va='top', size='large')
    ax1.set_xlim(alim)
    ax1.set_ylim(22, 33)

    if check_goldreich1966:
        def ts2ly(a, ts_):
            t_ = 1 / (1/ts_ + 1/Y)
            ly1 = T(a) / t_
            tl = 1 / (1/t_ - 1/T(a))
            ly2 = Tl(a)/tl
            return ly1, ly2
        with open('data/goldreich1966.txt') as f:
            data = f.read()
        line1, line2 = data.splitlines()
        a = np.array(list(map(float, line1.split()[1:]))) / 60.2
        ts_ = np.array(list(map(float, line2.split()[1:]))) * 3600

        ly1, ly2 = ts2ly(a, ts_)
        ax1.plot(a, ly1, 'C1', alpha=0.3)
        ax1.plot(a, ly2, 'C1', label='Goldreich 1966')
        ax3.plot(a, ts_ / 3600, 'C1')

    def update_prop(handle, orig):
        handle.update_from(orig)
        x,y = handle.get_data()
        handle.set_data([np.mean(x)]*2, [-y[0], 2*y[0]])

    ax1.legend(loc='lower center', ncol=2, fontsize=8.8)
    ax2.legend([(spanplot, vline1), vline2],
                ['result and bounds for diurnal tides',
                'result for semidiurnal tides'],
                handler_map={plt.Line2D:HandlerLine2D(update_func=update_prop)},
                handleheight=2)
    fig.tight_layout()
    fig.savefig('figs/moon_orbit.pdf')


def plot_other_parameters(alim=ALIM):
    a = np.linspace(*alim, 101)
    ly = a2ly(a)
    IedivIe0 = 1 + gamma * (t0**2/t(a, ly)**2 - 1)
    ratio_LeLm = alpha0 * IedivIe0 * t0 / t(a, ly) /a ** 0.5
    ratio_Ltotal = (Le0*t0/t(a, ly)*IedivIe0 + Lm0*a**0.5) / (Le0 + Lm0)
    ratio_dLm_dLsun = 1 / (beta0 * a ** 6)

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(511)
    ax2 = fig.add_subplot(512, sharex=ax1)
    ax3 = fig.add_subplot(513, sharex=ax1)
    ax4 = fig.add_subplot(514, sharex=ax1)
    ax5 = fig.add_subplot(515, sharex=ax1)
    ax1.plot(a, ratio_LeLm, 'k')
    ax2.plot(a, ratio_Ltotal, 'k')
    ax3.semilogy(a, ratio_dLm_dLsun, 'k')
    ratioY = ((1/13 * beta0 * a ** 6.5 - 1 / 13 * beta0) / alpha1 + 1) ** 3
    ax4.plot(a, (ratioY-1), 'k')
    ax5.plot(a, IedivIe0, 'k')  # simplified solution without solving Cardanos formula
    from matplotlib.ticker import FormatStrFormatter
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    ankw = dict(xy=(0, 1), xytext=(1, -1), xycoords='axes fraction',
                textcoords='offset points', va='top', size='small')
    ax1.set_ylabel('Le / Lm')
    ax1.annotate('a) ratio of Earths angular momentum to lunar orbital angular momentum', **ankw)
    ax2.set_ylabel('(Le+Lm) / (Le0+Lm0)')
    ax2.annotate('b) ratio of total momentum in Earth-Moon system (including moments lost later) to todays value', **ankw)
    ax3.set_ylabel('dLm / dLsun')
    ax3.annotate('c) ratio of momenta transferred from rotation of Earth to orbit of Moon and orbit of Earth around sun', **ankw)
    ax4.set_ylabel('Y / Y0 - 1')
    ax4.annotate('d) change of length of the year due to larger Earth-Sun distance due to (c)', **ankw)
    ax5.set_ylabel('Ie / Ie0')
    ax5.annotate("e) change in Earth's moment of inertia, due to faster rotation and flattening", **ankw)
    ax5.set_xlabel('orbital radius a')
    fig.tight_layout()
    fig.savefig('figs/moon_orbit_params3.pdf')


def plot_moon_orbit_time(layers=None, abracket=(0.4, 0.8),
                         orbit_data=ORBIT_TIME_DATA,
                         abracket2=None,
                         plot_models=PMODS):
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    if plot_models is not None:
        models = np.load('data/orbit_models.npz')
        m1handle, = ax.plot(*models[PMODS[0]], '0.8', linestyle='-.', label=PMODS[0])
        for i in range(1, 5):
            m2handle, = ax.plot(*models[PMODS[i]], '0.8', linestyle='--', label='Daher et al. 2021')
        for i in range(5, 8):
            m3handle, = ax.plot(*models[PMODS[i]], '0.8', linestyle=':', label='Tyler 2021')
        m4handle, = ax.plot(*models[PMODS[8]], '0.8', label=PMODS[8])
        m5handle = ax.fill_between(*models[PMODS[8]+' 2sigma'], color='0.87')
    if orbit_data is not None:
        orbit_data_handles = [ax.plot(age, a_, marker, mec='k', mfc='None', label=label, alpha=alpha)[0]
                              for (age, a_, marker, alpha, label) in orbit_data]
    if layers is not None:
        lys = layers[:3]
        a = solve(lys, diurnal=True, bracket=abracket)
        amin, a_, amax = sorted(a)
        yerr = [[a_-amin], [amax-a_]]
        this_study_handle  = ax.errorbar(layers[-1], a_, yerr=yerr, fmt='x', zorder=10, color='C0', label='Moodies, this study')
        if abracket2 is not None:
            a = solve(lys, diurnal=True, bracket=abracket2)
            amin, a_, amax = sorted(a)
            yerr = [[a_-amin], [amax-a_]]
            ax.errorbar(layers[-1], a_, yerr=yerr, fmt='x', zorder=10, color='C0', alpha=0.7)
    ax.invert_xaxis()
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.set_ylim(0.465, None)
    ax.set_ylabel('orbital radius a')
    ax.set_xlabel('time in billion years before present')
    handles = [m1handle, m2handle, (m5handle, m4handle), m3handle]
    labels = [PMODS[0], 'Daher et al. 2021', PMODS[8], 'Tyler 2021']
    legend1 = ax.legend(handles, labels, loc='upper left', frameon=False)
    ax.add_artist(legend1)
    ax.legend(handles=[this_study_handle] + orbit_data_handles[::-1], loc='lower right', frameon=False, fontsize='small')
    fig.savefig('figs/moon_orbit_time.pdf')


_atest = np.linspace(0.2, 1.2, 101)
np.testing.assert_allclose(a2ly_simple(_atest), a2ly(_atest, gamma=0))


if __name__ == '__main__':
    lys = (30.0, 28.4, 30.6)
    lys2 = (30.0,)
    print_moon_orbit(lys, diurnal=True, abracket=(0.4, 0.8))
    print('gamma=0')
    print_moon_orbit(lys2, diurnal=True, abracket=(0.4, 0.8), gamma=0)
    print('1% higher moment of inertia')
    print_moon_orbit(lys2, diurnal=True, abracket=(0.4, 0.8), mu=1.01)
    print('high orbit')
    print_moon_orbit(lys2, diurnal=True, abracket=(0.85, 1.1))
    print_moon_orbit(lys2, diurnal=False, abracket=(0.4, 0.8))
    print_moon_orbit_21_hours((15.0,), diurnal=False)
    print_moon_orbit_today()
    print('Observation from Eriksson & Simpson is 18.6, here we use 20')
    print_moon_orbit([20], diurnal=False, abracket=(0.3, 0.8))
    plot_moon_orbit_time(layers=lys+ (3.20,))
    plot_moon_orbit(lys=(30.0,),
                    ly_range=(28.4, 30.6),
                    ly_annotate=30.0,
                    pos_annotate=0.15,
                    orbit_data=None
                    )
    plot_moon_orbit_different_params()
    plot_moon_orbit_different_params2()
    plot_other_parameters(alim=(0.2, 1.01))
    plt.show()
