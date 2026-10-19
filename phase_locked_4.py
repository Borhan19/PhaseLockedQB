#!/usr/bin/env python3
"""Quantum battery comparison: original F2 plus two coupling-resource landscapes.

Exactly THREE PDF figures are generated:
    F2_finite_capacity_optimized.pdf        (unchanged manuscript comparison)
    F3_option_A_fixed_pumps.pdf             (fixed pumps; growth may differ)
    F4_option_B_matched_growth.pdf          (pumps retuned to common growth)

F2 remains identical in its physics, plotting styles, and signed-input efficiency
convention to the uploaded manuscript-consistent script. The new landscape plots
compare DIFFERENT architectures, with x = Gamma_col/kappa0 and y = |J|/kappa0.
The color is log10(SNR_diss/SNR_coh,opt) at equal first-passage target occupation.
The coherent driving FREQUENCY, and no other drive parameter, is optimized for
fastest first passage; the dissipative drive remains on midpoint resonance.

Option A: coherent pump = 2.2*kappa0, dissipative pump = reference 1.30457*kappa0,
          both held fixed across the independent J/Gamma sweeps.
Option B: each pump is recalibrated to lambda_max=0.05*kappa0 at each point.

Both options retain |E|/kappa0=10, k_down=k_b=kappa0, theta=1.15 and phi=0.
Coupling and engineered damping are physically DISTINCT resources. Neither
landscape establishes equal wall-plug/source/reservoir thermodynamic costs.
For theta=pi/2 the battery decouples: SNR at nonzero target is undefined.

Commands:
  python phase_locked_resource_landscapes.py all
  python phase_locked_resource_landscapes.py heatmaps --map-nj 40 --map-ng 40
  python phase_locked_resource_landscapes.py linear
  python phase_locked_resource_landscapes.py validate
Run unchanged in Jupyter; change NOTEBOOK_COMMAND/NOTEBOOK_EXTRA_ARGS below.
Dependencies: numpy, scipy, pandas, matplotlib (no QuTiP).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.linalg import expm
from scipy.optimize import brentq, minimize_scalar
from scipy.special import roots_legendre

# ===========================================================================
# ONE-CLICK SETTINGS FOR JUPYTER
# ===========================================================================
NOTEBOOK_COMMAND = 'all'              # 'all': F2 + both maps; 'heatmaps': two maps
NOTEBOOK_TARGET_N_STAR = 100.0         # maximum target occupation on x axis
NOTEBOOK_EXTRA_ARGS = [
    '--t-max', '70',                  # maximum search time, in 1/kappa0
    '--nu-max', '6.5',                # range of coherent drive offsets
    '--nu-points', '161',             # initial global frequency search grid
    '--step', '0.025',                # time grid for stationary point detection
    '--out', 'Phase_Locked_Resource_Landscapes',  # output directory
    '--theta', '1.15',                   # collective-mode angle used in maps
    # Maps use the same target as F2 unless --map-target is supplied.
    '--map-nj', '53',                    # number of independent coherent couplings
    '--map-ng', '55',                    # number of collective rates
]

# Physical units: the original manuscripts use kappa0 = 0.01.
# The numerical solver rescales every rate by kappa0 and therefore uses
# kappa0 = 1, E = 10. In unscaled units BOTH schemes have E = 0.10.
PHYSICAL_KAPPA0 = 0.01
DIMENSIONLESS_DRIVE_E = 10.0

# Exact colors and line styles from manuscript Fig. 3.
FIG3_COHERENT_COLOR = '#54BC88'       # green, solid
FIG3_DISSIPATIVE_COLOR = '#B088E4'    # purple, solid
FIG3_MIDPOINT_LINEWIDTH = 1.15

# The frequency-optimized coherent coupling remains in the coherent color
# family, but uses a DARKER green and a DASHED line to avoid ambiguity.
FIG3_OPTIMIZED_COHERENT_COLOR = '#167B55'
FIG3_OPTIMIZED_COHERENT_LINESTYLE = (0, (5.0, 2.2))
FIG3_OPTIMIZED_COHERENT_LINEWIDTH = 1.55
MAIN_TEXT_FONT_SIZE = 10
SINGLE_COLUMN_WIDTH = 3.37
OMEGA_A = 1.0  # The paper's omega_a and omega_b are both 1 in physical units.
_GL_NODES, _GL_WEIGHTS = roots_legendre(128)


def is_notebook_launcher() -> bool:
    name = Path(sys.argv[0]).name.lower() if sys.argv else ''
    return ('ipykernel_launcher' in name or
            (len(sys.argv) >= 3 and sys.argv[1] in ('-f', '--f') and
             Path(sys.argv[2]).name.startswith('kernel-') and
             Path(sys.argv[2]).suffix == '.json'))


def setup_style():
    families = ('Times New Roman', 'Nimbus Roman', 'TeX Gyre Termes',
                'STIXGeneral', 'DejaVu Serif')
    names = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in families if f in names), 'DejaVu Serif')
    mpl.rcParams.update({
        'font.family':'serif', 'font.serif':[chosen], 'mathtext.fontset':'stix',
        'font.size':10, 'axes.labelsize':10, 'axes.titlesize':10,
        'xtick.labelsize':10, 'ytick.labelsize':10, 'legend.fontsize':10,
        'axes.linewidth':1., 'lines.linewidth':1.5, 'lines.markersize':4.,
        'xtick.direction':'in', 'ytick.direction':'in',
        'xtick.top':True, 'ytick.right':True,
        'pdf.fonttype':42, 'ps.fonttype':42,
        'figure.dpi':150, 'savefig.format':'pdf',
        'savefig.bbox':'tight', 'savefig.pad_inches':.035,
        'figure.max_open_warning':0,
    })
    return chosen


@dataclass(frozen=True)
class Model:
    kind: str
    k_down: float = 1.
    k_up: float = 2.2
    k_b: float = 1.
    J: float = 3.2
    gamma_col: float = 0.
    theta: float = 1.15
    phi: float = 0.
    bare_delta: float = 0.
    E: complex = 10. + 0.j

    def drift(self) -> np.ndarray:
        if self.kind == 'coherent':
            if self.gamma_col != 0:
                raise ValueError('Coherent reference must have no collective jump')
            ga = (self.k_down-self.k_up)/2.
            gb = self.k_b/2.
            return np.array([[-ga-1j*self.bare_delta, -1j*self.J],
                             [-1j*np.conj(self.J), -gb+1j*self.bare_delta]],complex)
        if self.kind == 'dissipative':
            if abs(self.J)>1e-14:
                raise ValueError('Dissipative reference must have J=0')
            c,s=np.cos(self.theta),np.sin(self.theta)
            g=.5*self.gamma_col*np.exp(1j*self.phi)*c*s
            ga=(self.k_down-self.k_up+self.gamma_col*c*c)/2.
            gb=(self.k_b+self.gamma_col*s*s)/2.
            return np.array([[-ga-1j*self.bare_delta,-g],
                             [-np.conj(g),-gb+1j*self.bare_delta]],complex)
        raise ValueError('Unknown kind: '+self.kind)

    def growth(self) -> float:
        return float(np.max(np.linalg.eigvals(self.drift()).real))

    def force(self) -> np.ndarray:
        return np.array([-1j*self.E,0.], complex)

    def noise(self) -> np.ndarray:
        # Normally ordered spontaneous injection for k_up * D[a^dagger]
        return np.diag([self.k_up, 0.]).astype(complex)


def target_models():
    theta=1.15
    gc=2.*3.2/(np.sin(theta)*np.cos(theta))
    coherent=Model('coherent', k_up=2.2, E=DIMENSIONLESS_DRIVE_E)
    diss=Model('dissipative', J=0., gamma_col=gc, theta=theta,
               k_up=1.30457, E=DIMENSIONLESS_DRIVE_E)
    # Refit dissipative pump so BOTH spectral growth rates are exactly .05.
    g=lambda kp: replace(diss,k_up=kp).growth()-.05
    diss=replace(diss,k_up=float(brentq(g,0.,4.,xtol=1.e-13)))
    assert abs(coherent.growth()-.05)<1.e-12
    assert abs(diss.growth()-.05)<1.e-12
    return coherent,diss


def int_exp(z,t):
    """Stable analytic integral I(z,t)=integral_0^t exp(z*u)du."""
    z,t=np.broadcast_arrays(np.asarray(z,dtype=complex),np.asarray(t,dtype=float))
    out=np.empty(z.shape,complex)
    small=np.abs(z*t)<1.e-5
    y=z[small]*t[small]
    out[small]=t[small]*(1+y/2+y*y/6+y*y*y/24+y**4/120)
    out[~small]=np.expm1(z[~small]*t[~small])/z[~small]
    return out


class Gaussian:
    """Exactly solved Gaussian amplitudes and NORMAL ordered covariance.

    dot(alpha) = (M0+i nu I)*alpha+f;
    dot(N)=M0*N+N*M0^dag+K, K=diag(k_up,0).
    Global source-frequency offset nu does NOT affect N(t).
    """
    def __init__(self,p:Model):
        self.p=p
        self.M=p.drift()
        self.lam,self.V=np.linalg.eig(self.M)
        if np.linalg.cond(self.V)>1.e8:
            raise ValueError('Drift nearly defective; need matrix exponential method')
        self.U=np.linalg.inv(self.V)
        self.src_modes=self.U @ p.force()
        self.noise_modes=self.U @ p.noise() @ self.U.conj().T
        self.denom=self.lam[:,None]+self.lam.conj()[None,:]
        self.source_proj=np.array([self.V[i,:]*self.src_modes for i in range(2)])
        self.noise_proj=np.array([
            np.outer(self.V[i,:],self.V[i,:].conj())*self.noise_modes
            for i in range(2)])

    def amplitude(self,t,nu=0.,mode=1):
        t=np.asarray(t,float)
        z=self.lam+1j*nu
        response=int_exp(z.reshape((2,)+(1,)*t.ndim),t)
        y=np.sum(self.source_proj[mode].reshape((2,)+(1,)*t.ndim)*response,axis=0)
        return complex(y) if y.ndim==0 else y

    def amplitude_prime(self,t,nu=0.,mode=1):
        t=np.asarray(t,float)
        z=self.lam+1j*nu
        resp=np.exp(z.reshape((2,)+(1,)*t.ndim)*t)
        y=np.sum(self.source_proj[mode].reshape((2,)+(1,)*t.ndim)*resp,axis=0)
        return complex(y) if y.ndim==0 else y

    def n_noise(self,t,mode=1):
        t=np.asarray(t,float)
        z=self.denom.reshape((2,2)+(1,)*t.ndim)
        v=self.noise_proj[mode].reshape((2,2)+(1,)*t.ndim)
        n=np.sum(v*int_exp(z,t),axis=(0,1)).real
        if np.any(n < -1.e-8):
            raise ArithmeticError('Unphysical negative noise population')
        n=np.maximum(n,0.)
        return float(n) if np.ndim(n)==0 else n

    def n_noise_prime(self,t,mode=1):
        t=np.asarray(t,float)
        z=self.denom.reshape((2,2)+(1,)*t.ndim)
        v=self.noise_proj[mode].reshape((2,2)+(1,)*t.ndim)
        n=np.sum(v*np.exp(z*t),axis=(0,1)).real
        return float(n) if np.ndim(n)==0 else n

    def active_energy_at(self,t,nu=0.):
        """Reproduce the paper's signed active-channel energetic convention.

        Integrate -2 omega_a Im(E* alpha_a(t)) plus
        omega_a k_up [n_a(t)+1] with a 128-point Gaussian quadrature.
        The coherent drive amplitude E is held fixed in BOTH models.
        This is NOT a gross incident-field or source-generation energy cost.
        """
        if not np.isfinite(t) or t < 0:
            return dict(charging_efficiency=np.nan,active_energy_net=np.nan,
                        drive_work_net=np.nan,gain_energy_net=np.nan)
        if t == 0:
            return dict(active_energy_net=0.0,drive_work_net=0.0,
                        gain_energy_net=0.0)
        sample_t = 0.5 * float(t) * (_GL_NODES + 1.0)
        amplitude_a = self.amplitude(sample_t,nu,mode=0)
        n_a_inc = self.n_noise(sample_t,mode=0)
        drive_power = -2.0 * OMEGA_A * np.imag(
            np.conj(self.p.E) * amplitude_a)
        gain_power = OMEGA_A * self.p.k_up * (
            np.abs(amplitude_a)**2 + n_a_inc + 1.0)
        factor=0.5*float(t)
        e_drive = float(factor*np.dot(_GL_WEIGHTS,drive_power))
        e_gain = float(factor*np.dot(_GL_WEIGHTS,gain_power))
        return dict(drive_work_net=e_drive,gain_energy_net=e_gain,
                    active_energy_net=e_drive+e_gain)

    def battery_occ(self,t,nu=0.):
        return abs(self.amplitude(t,nu))**2+self.n_noise(t)

    def battery_occ_prime(self,t,nu=0.):
        b=self.amplitude(t,nu)
        return 2*np.real(np.conj(b)*self.amplitude_prime(t,nu))+self.n_noise_prime(t)

    def metrics(self,t,nu=0.):
        if not np.isfinite(t):
            return dict(reached=False,tau=np.nan,drive_offset=np.nan)
        ninc=float(self.n_noise(t))
        w=float(abs(self.amplitude(t,nu))**2)
        n=w+ninc
        sd=np.sqrt(ninc*(ninc+1.)+(2*ninc+1.)*w)
        return dict(reached=True,tau=float(t),drive_offset=float(nu),
                    n_total=float(n),ergotropy=float(w),passive=float(ninc),
                    ergotropy_fraction=float(w/n) if n>0 else np.nan,
                    quadrature_snr=float(w/(ninc+.5)),
                    energy_fluctuation_snr=float(w/sd) if sd>0 else np.nan,
                    ergotropic_power=float(w/t) if t>0 else np.nan)


def stationary_partition(s:Gaussian,nu:float,tmax:float,step:float):
    """Find all monotone segments of total occupation by bracketing derivative roots."""
    grid=np.linspace(0.,tmax,int(np.ceil(tmax/step))+1)
    der=s.battery_occ_prime(grid,nu)
    idx=np.flatnonzero(der[:-1]*der[1:]<0.)
    roots=[0.]
    for i in idx:
        root=brentq(lambda u:s.battery_occ_prime(u,nu),
                    grid[i],grid[i+1],xtol=1.e-11)
        if root>roots[-1]+1.e-8:
            roots.append(float(root))
    roots.append(tmax)
    pts=np.asarray(roots,float)
    return pts,np.asarray(s.battery_occ(pts,nu),float)


def first_passage_from_partition(s:Gaussian,target:float,nu:float,part):
    pts,values=part
    if values[0]>=target:
        return 0.
    for k in range(1,len(pts)):
        if values[k]>=target:
            if values[k-1]>=target:
                raise ArithmeticError('Partition invariant failed')
            return float(brentq(lambda t:s.battery_occ(t,nu)-target,
                                pts[k-1],pts[k],xtol=5.e-11))
    return np.nan


def target_grid(max_target:float,n_points:int, reference:Gaussian,
                tmax:float, step:float):
    """Paper's dense near-uniform n_star grid with exact knee bracketing.

    The source paper used 701 uniform capacities from 0.2 to 100 together
    with 3 close points bracketing every new record-high coherent peak.
    """
    lower=min(0.2,max_target/100.0)
    pts,vals=stationary_partition(reference,0.0,tmax,step)
    record=0.0
    knees=[]
    for j in range(1,len(vals)-1):
        if vals[j] > vals[j-1] and vals[j] > vals[j+1]:
            if vals[j] > record + 1e-6:
                record=float(vals[j])
                if lower < record < max_target:
                    knees.append(record)
    bracket=[]
    for knee in knees:
        e=max(.003,1.e-4*knee)
        bracket.extend([knee-e,knee,knee+e])
    targets=np.unique(np.concatenate((np.linspace(lower,max_target,n_points),
                                     np.asarray(bracket,dtype=float))))
    targets=targets[(targets>=lower)&(targets<=max_target)]
    return targets,np.asarray(knees,dtype=float)


def first_passages_bulk(s:Gaussian, targets:np.ndarray, nu:float, part):
    """Exact first monotonic crossing, vectorized to resolve >700 capacities.

    A running-maximum of the values at every stationary point identifies which
    monotonic rising segment first reaches each threshold. A vector bisection
    then finds the earliest crossing without relying on fixed-time sampling.
    """
    pts,vals=part
    records=np.maximum.accumulate(vals)
    index=np.searchsorted(records,targets,side='left')
    answer=np.full_like(targets,np.nan,dtype=float)
    initially=(targets<=vals[0])
    answer[initially]=float(pts[0])
    good=(index>0)&(index<len(pts))
    if np.any(good):
        lo=pts[index[good]-1].astype(float).copy()
        hi=pts[index[good]].astype(float).copy()
        q=targets[good]
        for _ in range(35):
            m=(lo+hi)*0.5
            above=s.battery_occ(m,nu)>=q
            hi=np.where(above,m,hi)
            lo=np.where(above,lo,m)
        answer[good]=(lo+hi)*0.5
    return answer


def complete_metrics(s:Gaussian, t:float, nu:float, target:float):
    metrics=s.metrics(t,nu)
    metrics['n_target']=float(target)
    if not metrics['reached']:
        metrics.update(dict(charging_efficiency=np.nan,
            active_energy_net=np.nan,drive_work_net=np.nan,gain_energy_net=np.nan))
        return metrics
    cost=s.active_energy_at(t,nu)
    metrics.update(cost)
    net=cost['active_energy_net']
    metrics['charging_efficiency']=(float(metrics['ergotropy']/net)
                                    if net>1.e-15 else np.nan)
    return metrics


def midpoint_scan(s,targets,tmax,step):
    part=stationary_partition(s,0.,tmax,step)
    times=first_passages_bulk(s,targets,0.,part)
    rows=[complete_metrics(s,float(t),0.,float(target))
          for t,target in zip(times,targets)]
    return pd.DataFrame(rows)


def coherent_optimization(s,targets,nu_max,n_points,tmax,step):
    """Optimize FIRST-PASSAGE TIME only; reuse nu(n*) across ALL four panels.

    The frequency sweep is a global finite grid. Each frequency yields exact
    stationary-point intervals, with first crossings resolved simultaneously
    for all targets by bracketed vectorized bisection. Only the MAXIMUM target
    is polished continuously, to avoid introducing artificial roughness from
    a different objective for each Fig. 3 panel.
    """
    nus=np.linspace(-nu_max,nu_max,n_points)
    best_t=np.full(len(targets),np.inf,dtype=float)
    best_nu=np.full(len(targets),np.nan,dtype=float)
    best_j=np.zeros(len(targets),dtype=int)
    for j,nu in enumerate(nus):
        part=stationary_partition(s,float(nu),tmax,step)
        times=first_passages_bulk(s,targets,float(nu),part)
        improving=np.isfinite(times) & (times<best_t)
        best_t[improving]=times[improving]
        best_nu[improving]=float(nu)
        best_j[improving]=j
        if (j+1)%40==0 or j+1==n_points:
            print(f'  coherent frequency samples: {j+1}/{n_points}',flush=True)
    if np.any(~np.isfinite(best_t)):
        x=targets[~np.isfinite(best_t)][0]
        raise RuntimeError(f'Coherent scheme did not reach n*={x:g}. '
                           'Try increasing --t-max or --nu-max.')
    k=len(targets)-1
    j=int(best_j[k]); target=float(targets[k])
    left=float(nus[max(0,j-1)]);right=float(nus[min(n_points-1,j+1)])
    if right>left:
        def first_t(nu):
            part=stationary_partition(s,float(nu),tmax,step)
            return float(first_passage_from_partition(s,target,float(nu),part))
        result=minimize_scalar(first_t,bounds=(left,right),method='bounded',
                               options={'xatol':1.e-6,'maxiter':60})
        for nu in (left,float(result.x),right):
            t=first_t(nu)
            if np.isfinite(t) and t<best_t[k]:
                best_t[k]=t;best_nu[k]=nu
    rows=[complete_metrics(s,float(t),float(nu),float(target))
          for t,nu,target in zip(best_t,best_nu,targets)]
    for r in rows:
        r['first_passage_residual']=float(r['n_total']-r['n_target'])
        r['optimizes']='earliest_occupation_passage'
    return pd.DataFrame(rows)


def independent_covariance_check(s:Gaussian,nu:float,t:float):
    """Independent 4-real-quadrature Lindblad computation of battery added noise."""
    p=s.p
    M=s.M+1j*nu*np.eye(2)
    A=np.zeros((4,4))
    for i in range(2):
        for j in range(2):
            z=M[i,j]
            A[2*i:2*i+2,2*j:2*j+2]=((z.real,-z.imag),(z.imag,z.real))
    def ann(v):
        out=np.zeros(4,complex)
        for j,x in enumerate(v):
            out[2*j]=x/np.sqrt(2.)
            out[2*j+1]=1j*x/np.sqrt(2.)
        return out
    def cre(v):
        out=ann(v)
        out[1::2]*=-1
        return out
    jumps=[np.sqrt(p.k_down)*ann((1,0)),np.sqrt(p.k_up)*cre((1,0)),
           np.sqrt(p.k_b)*ann((0,1))]
    if p.kind=='dissipative':
        jumps.append(np.sqrt(p.gamma_col)*ann((np.cos(p.theta),
                np.exp(1j*p.phi)*np.sin(p.theta))))
    Y=sum((np.outer(x,x.conj()) for x in jumps),np.zeros((4,4),complex))
    om=np.kron(np.eye(2),np.array([[0.,1.],[-1.,0.]]))
    D=om @ Y.real @ om.T
    B=np.kron(np.eye(4),A)+np.kron(A,np.eye(4))
    G=np.zeros((17,17))
    G[:16,:16]=B
    G[:16,-1]=D.flatten(order='F')
    x0=np.r_[.5*np.eye(4).flatten(order='F'),1.]
    V=(expm(G*t)@x0)[:16].reshape((4,4),order='F')
    return float((V[2,2]+V[3,3]-1.)/2.)


def validate(verbose=True):
    c,d=target_models()
    assert abs(c.E)==abs(d.E)==DIMENSIONLESS_DRIVE_E
    assert np.isclose(d.gamma_col*np.cos(d.theta)*np.sin(d.theta)/2,abs(c.J))
    assert abs(c.growth()-.05)<1e-12 and abs(d.growth()-.05)<1e-12
    s_coh,s_dis=Gaussian(c),Gaussian(d)
    for s in (s_coh,s_dis):
        for nu in (-3.,0.,2.7):
            for t in (.1,1.,3.,7.):
                exact=independent_covariance_check(s,nu,t)
                assert np.isclose(s.n_noise(t),exact,rtol=1e-8,atol=1e-8),(
                    s.p.kind,nu,t,s.n_noise(t),exact)
                block=np.zeros((3,3),complex)
                block[:2,:2]=s.M+1j*nu*np.eye(2)
                block[:2,2]=s.p.force()
                ref=expm(block*t)[:2,2]
                for mode in (0,1):
                    assert np.isclose(s.amplitude(t,nu,mode),ref[mode],rtol=1e-8,atol=1e-8)
    for s,tc_ref,eta_ref in ((s_coh,12.772,.7009),(s_dis,2.731,.9951)):
        part=stationary_partition(s,0.,35.,.025)
        tc=first_passage_from_partition(s,100.,0.,part)
        eta=s.metrics(tc,0.)['ergotropy_fraction']
        assert abs(tc-tc_ref)<.025 and abs(eta-eta_ref)<.002
        if verbose:print(f'  {s.p.kind} paper reference: t={tc:.6f}, eta={eta:.6f}')
    # Independently reproduce original Fig. 3 charging-efficiency values.
    # These are the original signed active-energy convention, not W/E_b.
    for model,expected in ((s_coh,.0827127580135),(s_dis,.0704628)):
        tc=first_passage_from_partition(model,100.,0.,
                    stationary_partition(model,0.,35.,.025))
        cost=model.active_energy_at(tc,0.)['active_energy_net']
        eta=model.metrics(tc,0.)['ergotropy']/cost
        if not np.isclose(eta,expected,rtol=5e-5,atol=2e-6):
            raise AssertionError(f'Paper signed efficiency mismatch: {eta} vs {expected}')
        if verbose:print(f'  {model.p.kind} paper signed eta_ch(100)={eta:.7f}')
    # Cross-check the separate signed work and gain integrals against an
    # independent adaptive integration. This is important because their sum,
    # not W/E_b, is the denominator in the original manuscript panel (b).
    from scipy.integrate import quad
    for model,nu,t in ((s_coh,0.,12.7715483866),
                       (s_coh,-2.8982,1.6837),
                       (s_dis,0.,2.7311472)):
        ledger=model.active_energy_at(t,nu)
        ref_drive=quad(lambda u:-2.*OMEGA_A*np.imag(
            np.conj(model.p.E)*model.amplitude(u,nu,mode=0)),0.,t,
            epsabs=1e-10,epsrel=1e-10)[0]
        ref_gain=quad(lambda u:OMEGA_A*model.p.k_up*(
            abs(model.amplitude(u,nu,mode=0))**2+
            model.n_noise(u,mode=0)+1),0.,t,
            epsabs=1e-10,epsrel=1e-10)[0]
        assert np.isclose(ledger['drive_work_net'],ref_drive,atol=2e-8,rtol=3e-9)
        assert np.isclose(ledger['gain_energy_net'],ref_gain,atol=2e-8,rtol=3e-9)
    # For a resonant dissipative drift, e^{M t}_{ba} has constant phase,
    # so triangle inequality ensures nu=0 maximizes |b(t)| at each t.
    for t in (.3,1.,3.,8.):
        max0=abs(s_dis.amplitude(t,0.))**2
        for nu in (-3.,-1.,1.,3.):
            assert abs(s_dis.amplitude(t,nu))**2<=max0*(1+1e-10)
    if verbose:
        print('PASS: matched parameters, Lindblad covariance, exact means, '
              'reference first-passages, dissipative resonant optimization')


def format_axes(axes):
    for a in np.ravel(axes):
        a.minorticks_on()
        a.tick_params(which='both',direction='in',top=True,right=True)
        # Manuscript Fig. 3 style: no background grid.


def make_F2(coh0,coh_opt,diss,out_pdf,max_target,first_knee):
    """Exact manuscript Figure 3 layout, metrics, and knee marker, plus one curve."""
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2,2,figsize=(SINGLE_COLUMN_WIDTH,3.37),sharex=True)
    axes=np.asarray(axes)
    lines=[
        (coh0,'Coherent',dict(color=FIG3_COHERENT_COLOR,marker=None,
            linestyle='-',linewidth=FIG3_MIDPOINT_LINEWIDTH)),
        (coh_opt,'Coherent (opt.)',dict(color=FIG3_OPTIMIZED_COHERENT_COLOR,
            marker=None,linestyle=FIG3_OPTIMIZED_COHERENT_LINESTYLE,
            linewidth=FIG3_OPTIMIZED_COHERENT_LINEWIDTH)),
        (diss,'Dissipative',dict(color=FIG3_DISSIPATIVE_COLOR,
            marker=None,linestyle='-',linewidth=FIG3_MIDPOINT_LINEWIDTH)),
    ]
    panels=[('tau',r'$\kappa_0t_{\rm ch}$'),
            ('charging_efficiency',r'$\eta_{\rm ch,\star}$'),
            ('quadrature_snr',r'$\mathrm{SNR}_{b,\star}$'),
            ('ergotropic_power',r'$P_{\mathcal{W},\star}/(\omega_b\kappa_0)$')]
    for j,((metric,ylabel),ax) in enumerate(zip(panels,axes.flat)):
        for df,name,style in lines:
            good=df[df.reached & np.isfinite(df[metric])]
            if metric=='quadrature_snr':
                ax.semilogy(good.n_target,np.maximum(good[metric],1.e-16),
                            label=name,**style)
            else:
                ax.plot(good.n_target,good[metric],label=name,**style)
        if first_knee is not None:
            ax.axvline(first_knee,color='#303030',ls=':',lw=1.12,alpha=.9,zorder=5)
        ax.set_ylabel(ylabel)
        ax.text(.04,.95,f'({"abcd"[j]})',transform=ax.transAxes,
                ha='left',va='top',fontsize=MAIN_TEXT_FONT_SIZE,color='black',zorder=20,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.78,'pad':.6})
        ax.set_xlim(0,max_target)
        if metric=='charging_efficiency': ax.set_ylim(-.02,1.02)
        if metric in ('tau','ergotropic_power'): ax.set_ylim(bottom=0)
    axes[1,0].set_xlabel(r'$n_\star$')
    axes[1,1].set_xlabel(r'$n_\star$')
    format_axes(axes)
    handles=[Line2D([0],[0],color=style['color'],linestyle=style['linestyle'],
                    lw=style['linewidth'],label=label)
             for _,label,style in lines]
    fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,
               bbox_to_anchor=(.50,.995),fontsize=7.3,
               columnspacing=.65,handlelength=1.65,handletextpad=.4)
    # Same layout geometry as the original single-column manuscript figure.
    fig.subplots_adjust(left=.18,right=.98,bottom=.13,top=.88,
                        wspace=.4,hspace=.22)
    fig.savefig(out_pdf,bbox_inches='tight',pad_inches=.01)
    print('Figure saved:',out_pdf.resolve())
    if 'agg' not in mpl.get_backend().lower():
        plt.show()
    # Avoid plt.close() so the figure also appears in Jupyter.


def analyze(args):
    setup_style()
    validate(verbose=True)
    coherent,dissipative=target_models()
    sc,sd=Gaussian(coherent),Gaussian(dissipative)
    targets,knees=target_grid(float(args.target),int(args.target_points),
                              sc,args.t_max,args.step)
    first_knee=float(knees[0]) if len(knees) else None
    if first_knee is not None:
        print(f'  First coherent-capacity knee: n*={first_knee:.9f}')
    else:
        print('  No coherent knee occurs in the requested occupation interval.')
    out=Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    print('\nCONTROLLED BENCHMARK (normalized rates):')
    print(f'  fixed |E|/kappa0 = {coherent.E:g} for BOTH schemes '
          f'(physical E={coherent.E*PHYSICAL_KAPPA0:g}, kappa0={PHYSICAL_KAPPA0:g})')
    print(f'  |J|=|g|={abs(coherent.J):g}; max Re(lambda)={coherent.growth():.6g}')
    print(f'  incoherent pumps k_up: coherent={coherent.k_up:.6f}; '
          f'dissipative={dissipative.k_up:.6f}')
    print('  coherent frequency varied; dissipative nu=0 is frequency-optimal '
          'for resonant dissipative model')
    print(f'  target range: {targets[0]:.3g}..{targets[-1]:g}; '
          f't_max={args.t_max:g}; first-passage step={args.step:g}')

    baseline=midpoint_scan(sc,targets,args.t_max,args.step)
    dis=midpoint_scan(sd,targets,args.t_max,args.step)
    if not np.all(dis.reached):
        raise RuntimeError('Dissipative benchmark failed to reach every target. '
                           'Increase --t-max.')
    opt=coherent_optimization(sc,targets,args.nu_max,args.nu_points,args.t_max,args.step)
    if not np.all(opt.reached):
        raise RuntimeError('Optimized coherent protocol did not reach every target.')

    # All panels use the original paper's observables and signed energetic ledger;
    # optimized coherent curves use nu chosen ONLY by minimum charging time.
    assert np.allclose(opt.n_total.to_numpy(),targets,atol=3e-7,rtol=1e-8)
    assert np.allclose(dis.n_total.to_numpy(),targets,atol=3e-7,rtol=1e-8)
    mask=baseline.reached.to_numpy(bool)
    assert np.all(opt.tau.to_numpy()[mask]<=baseline.tau.to_numpy()[mask]+1e-8)

    # Save ONE combined CSV; no gross-cost or unrelated numerical tasks.
    rows=[]
    for name,df in [('coherent_midpoint',baseline),
                    ('coherent_fastest_frequency',opt),
                    ('dissipative_midpoint',dis)]:
        part=df.copy()
        part.insert(0,'protocol',name)
        rows.append(part)
    together=pd.concat(rows,ignore_index=True)
    together.to_csv(out/'F2_finite_capacity_optimized.csv',index=False)
    selected=[]
    for name,df in [('coherent_midpoint',baseline),
                    ('coherent_fastest_frequency',opt),
                    ('dissipative_midpoint',dis)]:
        r=df.iloc[-1]
        selected.append(dict(protocol=name,drive_offset=float(r.drive_offset),
               kappa0_t=float(r.tau),
               charging_efficiency=float(r.charging_efficiency),
               drive_work_net=float(r.drive_work_net),
               gain_energy_net=float(r.gain_energy_net),
               active_energy_net=float(r.active_energy_net),
               ergotropy_fraction=float(r.ergotropy_fraction),
              quadrature_snr=float(r.quadrature_snr),
              energy_fluctuation_snr=float(r.energy_fluctuation_snr),
              ergotropic_power=float(r.ergotropic_power)))
    settings=dict(
        interpretation='Matched fixed coherent-drive AMPLITUDE, not equal source-side energy consumption',
        optimization='For EACH target select fastest frequency from a global grid; '
                     'polish the maximum target locally. Use SAME frequency in all panels.',
        reference_kappa0=PHYSICAL_KAPPA0,
        physical_drive_E_both=PHYSICAL_KAPPA0*DIMENSIONLESS_DRIVE_E,
        normalized_drive_E_both=DIMENSIONLESS_DRIVE_E,
        coupling_magnitude_normalized=3.2,
        growth_max_normalized=.05,
        coherent_gain_pump_normalized=coherent.k_up,
        dissipative_gain_pump_normalized=dissipative.k_up,
        dissipative_gamma_col_normalized=dissipative.gamma_col,
        theta_dissipative=dissipative.theta,
        target_max=float(args.target),
        nu_scan_range_normalized=[-args.nu_max,args.nu_max],
        nu_scan_points=int(args.nu_points),
        figure_metrics={
            'a':'kappa0*t_first_passage',
            'b':'signed_net_charging_efficiency_W_over_integral_drive_plus_gain',
            'c':'manuscript_SNR_b_star = |<b>|^2/(n_inc+0.5)',
            'd':'ergotropic_power_W_over_t',
        },
        first_coherent_knee_occupation=first_knee,
        manuscript_signed_input_ledger='P_drive=-2 omega_a Im(E*<a>); P_gain=omega_a k_up(<n_a>+1)',
        note='The input-ledger efficiency is NOT a full source-side energetic efficiency.',
        figure_colors={
            'coherent_midpoint':FIG3_COHERENT_COLOR,
            'dissipative_midpoint':FIG3_DISSIPATIVE_COLOR,
            'coherent_frequency_optimized':FIG3_OPTIMIZED_COHERENT_COLOR,
        },
        first_passage_search_max=float(args.t_max),
        stationary_grid_step=float(args.step),
        selected_at_max_target=selected,
        model_limitations=[
            'Equal coherent-drive amplitude and growth rate are NOT equal total energetics.',
            'Incoherent gain rates differ and collective-reservoir implementation is not charged.',
            'Ideal linear gain is valid as a transient small-signal model; no saturation included.',
            'For nonzero bare detuning, dissipative frequency nu=0 need not be optimal.',
        ])
    (out/'F2_parameters_and_final_target.json').write_text(json.dumps(settings,indent=2))
    make_F2(baseline,opt,dis,out/'F2_finite_capacity_optimized.pdf',
            args.target,first_knee)
    print(f'\nAt target n*={args.target:g} (ALL metrics evaluated at FIRST passage):')
    for x in selected:
        print(f'  {x["protocol"]:<29s} nu={x["drive_offset"]:+7.4f}'
              f'  t={x["kappa0_t"]:9.4f}  eta_ch={x["charging_efficiency"]:.6f}'
              f'  SNR_b={x["quadrature_snr"]:8.4f}'
              f'  P={x["ergotropic_power"]:8.3f}')
    print('\nOriginal manuscript figure generated: F2_finite_capacity_optimized.pdf')
    print('Data and provenance saved as CSV and JSON in:',out)

# ==========================================================================
# RESOURCE-LANDSCAPE EXTENSION: TWO NEW FIGURES AND NO OTHER PLOTS
# ==========================================================================

def coherent_pump_for_growth(j, growth, kdown=1., kb=1.):
    """On resonance: analytic coherent gain rate giving max Re(lambda)=growth.

    For strong coherent hybridization the eigenvalues are complex conjugates
    and k_up = k_down+k_b+4*growth.
    For weak coherent mixing impose eigenvalue growth in the real-eigenvalue
    branch. The branches join at |J|=k_b/2+growth.
    """
    j=float(j); growth=float(growth)
    if j<=0:raise ValueError('J=0 disconnects the battery; SNR undefined')
    strong_p=kdown+kb+4.*growth
    j_crit=kb/2.+growth
    if j>=j_crit:
        return strong_p
    return kdown+2.*growth+2.*j*j/(kb/2.+growth)


def dissipative_pump_for_growth(gamma,theta,growth,kdown=1.,kb=1.):
    """Exact on-resonance pump matching a real leading eigenvalue to growth.

    gamma_a = (1-k_up+gamma*cos(theta)^2)/2,
    gamma_b = (1+gamma*sin(theta)^2)/2.
    Setting det(M-growth*I)=0 yields the expression below.
    """
    gamma=float(gamma); theta=float(theta);growth=float(growth)
    c2=np.cos(theta)**2; s2=np.sin(theta)**2
    z=kb+2.*growth
    return kdown+2.*growth+gamma*c2*z/(z+gamma*s2)


def ref_dissipative_model(base_d,theta):
    """Use the ORIGINAL fitted pump as the fixed-pump reference, even for scans."""
    if abs(theta-base_d.theta)<1e-12:
        return base_d
    # If changing theta, keep the original *numerical pump rate*, not growth.
    return replace(base_d,theta=theta)


def fastest_frequency_for_coupling(s,target,nu_max,num_nu,tmax,step):
    """First-passage optimum for CW coherent driving, using a global grid and
    locally refined exact first passages; it does NOT maximize SNR separately.

    Resonant and real-J symmetry makes |beta(t,nu)| even in nu, so nu>=0
    suffices. A dense vectorized time/frequency grid supplies candidate basins;
    stationary-point bracketing and Brent refine crossings in each basin.
    """
    grid_t=np.linspace(0.,tmax,int(np.ceil(tmax/step))+1)
    grid_nu=np.linspace(0.,nu_max,num_nu)
    tt=grid_t[:,None]
    bb=np.zeros((len(grid_t),len(grid_nu)),complex)
    for zz,cc in zip(s.lam,s.source_proj[1]):
        bb += cc*int_exp(zz+1j*grid_nu[None,:],tt)
    occ=np.abs(bb)**2+s.n_noise(grid_t)[:,None]
    crosses=occ>=target
    any_cross=np.any(crosses,axis=0)
    index=np.argmax(crosses,axis=0)
    left_index=np.maximum(index-1,0)
    low=occ[left_index,np.arange(len(grid_nu))]
    high=occ[index,np.arange(len(grid_nu))]
    frac=(target-low)/np.where(abs(high-low)>1e-15,high-low,1.)
    rough_time=np.where(any_cross,grid_t[left_index]+(grid_t[index]-grid_t[left_index])*
                        np.clip(frac,0.,1.),np.inf)
    candidates=[i for i in range(len(grid_nu))
                if np.isfinite(rough_time[i])
                and (i==0 or rough_time[i]<=rough_time[i-1])
                and (i==len(grid_nu)-1 or rough_time[i]<=rough_time[i+1])]
    candidates=sorted(candidates,key=lambda i:rough_time[i])[:6]
    if not candidates:
        return dict(reached=False,tau=np.nan,nu=np.nan,snr=np.nan,
                    passive=np.nan,ergotropy_fraction=np.nan,power=np.nan)
    best=(np.inf,np.nan)
    cache={}
    exact_step=min(.06,step*2.)
    def crossing(nu):
        key=round(float(nu),11)
        if key not in cache:
            part=stationary_partition(s,float(nu),tmax,exact_step)
            out=first_passage_from_partition(s,target,float(nu),part)
            cache[key]=float(out) if np.isfinite(out) else tmax*3.
        return cache[key]
    for i in candidates:
        v0=float(grid_nu[i]);lo=float(grid_nu[max(0,i-1)]);
        hi=float(grid_nu[min(len(grid_nu)-1,i+1)])
        trials=[v0]
        if hi>lo:
            opt=minimize_scalar(crossing,bounds=(lo,hi),method='bounded',
                                options={'xatol':2e-7,'maxiter':80})
            trials.extend([float(opt.x),lo,hi])
        for nu in trials:
            t=crossing(nu)
            if t<best[0]:best=(t,nu)
    if best[0]>=tmax*2:
        return dict(reached=False,tau=np.nan,nu=np.nan,snr=np.nan,
                    passive=np.nan,ergotropy_fraction=np.nan,power=np.nan)
    metr=s.metrics(best[0],best[1])
    if abs(metr['n_total']-target)>1e-5:
        raise ArithmeticError('Frequency optimization did not reach the target')
    return dict(reached=True,tau=best[0],nu=best[1],
                snr=metr['quadrature_snr'],passive=metr['passive'],
                ergotropy_fraction=metr['ergotropy_fraction'],
                power=metr['ergotropic_power'])


def dissipative_first_passage_at_target(s,target,tmax):
    """At phi=delta=0, the dissipative battery occupation is monotone.

    The battery transfer response has a fixed phase, and added noise is a
    positive integral. Consequently its first passage is the bracketed root.
    """
    if s.battery_occ(tmax)<target:
        return dict(reached=False,tau=np.nan,nu=0.,snr=np.nan,
                    passive=np.nan,ergotropy_fraction=np.nan,power=np.nan)
    t=float(brentq(lambda u:s.battery_occ(u)-target,0.,tmax,xtol=1e-11))
    metr=s.metrics(t,0.)
    if abs(metr['n_total']-target)>1e-6:
        raise ArithmeticError('Dissipative first passage did not reach target')
    return dict(reached=True,tau=t,nu=0.,snr=metr['quadrature_snr'],
                passive=metr['passive'],
                ergotropy_fraction=metr['ergotropy_fraction'],
                power=metr['ergotropic_power'])


def cell_edges(v,geometric=False):
    v=np.asarray(v,float)
    if geometric:
        x=np.log(v); middle=.5*(x[1:]+x[:-1]);ends=np.r_[x[0]-.5*(x[1]-x[0]),
             middle,x[-1]+.5*(x[-1]-x[-2])]
        return np.exp(ends)
    middle=.5*(v[1:]+v[:-1])
    return np.r_[v[0]-.5*(v[1]-v[0]),middle,v[-1]+.5*(v[-1]-v[-2])]


def plot_landscape(gammas,js,coh_rows,diss_rows,theta,mode,color_metric,
                   shared_vmax,reference_gamma,reference_j,out_pdf,target,growth):
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib import patheffects as pe
    sn_c=np.asarray([r['snr'] for r in coh_rows],float)
    sn_d=np.asarray([r['snr'] for r in diss_rows],float)
    with np.errstate(invalid='ignore',divide='ignore'):
        log_ratio=np.log10(sn_d[None,:]/sn_c[:,None])
    delta=sn_d[None,:]-sn_c[:,None]
    z=log_ratio if color_metric=='log_ratio' else delta
    if not np.any(np.isfinite(z)):
        raise RuntimeError('No valid first-passages in the heatmap domain')
    vmax=max(shared_vmax,0.001)
    norm=TwoSlopeNorm(vcenter=0.,vmin=-vmax,vmax=vmax)
    fig,ax=plt.subplots(figsize=(2.5,2.5))
    xedges=cell_edges(gammas)
    yedges=cell_edges(js,True)
    mesh=ax.pcolormesh(xedges,yedges,np.ma.masked_invalid(z),shading='flat',
                       cmap='RdBu_r',norm=norm,rasterized=True)
    # A checkerboard is used only for genuinely undefined/unreached points.
    if np.any(~np.isfinite(z)):
        ax.contourf(gammas,js,np.isnan(z).astype(float),levels=[.5,1.5],
                    colors=['#bcbcbc'],hatches=['///'],alpha=.6)
    G,J=np.meshgrid(gammas,js)
    has_equal_snr=float(np.nanmin(z))<0.<float(np.nanmax(z))
    if has_equal_snr:
        ax.contour(G,J,z,levels=[0.],colors=['#242424'],
                   linewidths=1.45,linestyles='solid',zorder=7)
    cs=np.sin(theta)*np.cos(theta)/2.
    matching=cs*gammas
    inside=(matching>=js[0])&(matching<=js[-1])
    ax.plot(gammas[inside],matching[inside],ls='--',lw=1.4,
            color='black',dashes=(3,2),zorder=5,
            path_effects=[pe.Stroke(linewidth=2.8,foreground='white'),pe.Normal()])
    if mode=='A':
        rates_c=np.asarray([r['growth'] for r in coh_rows])
        rates_d=np.asarray([r['growth'] for r in diss_rows])
        mismatch=rates_d[None,:]-rates_c[:,None]
        if np.min(mismatch)<0 and np.max(mismatch)>0:
            ax.contour(G,J,mismatch,levels=[0],colors=['cyan'],linewidths=1.45,
                       linestyles=':',zorder=6)
    ax.scatter([reference_gamma],[reference_j],marker='*',s=150,c='#FFD54F',
               linewidths=.95,edgecolors='#191919',zorder=9)
    ax.set_yscale('log')
    ax.set_xlim(gammas[0],gammas[-1])
    ax.set_ylim(js[0],js[-1])
    ax.set_xlabel(r'$\Gamma_{\rm col}/\kappa_0$')
    ax.set_ylabel(r'$|J|/\kappa_0$')
    ax.set_yticks([x for x in [.1,.2,.5,1.,2.,3.2,5.,8.] if js[0]<=x<=js[-1]])
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%g'))
    ax.grid(False)
    ax.tick_params(which='both',direction='in',top=True,right=True)
    cbar=fig.colorbar(mesh,ax=ax,fraction=.052,pad=.04,extend='neither')
    if color_metric=='log_ratio':
        cbar.set_label(r'$\log_{10}(\mathrm{SNR}_{\rm diss}/\mathrm{SNR}_{\rm coh,opt})$')
    else:
        cbar.set_label(r'$\Delta\mathrm{SNR}=\mathrm{SNR}_{\rm diss}-\mathrm{SNR}_{\rm coh,opt}$')
    #title=('A: Pump rates held fixed' if mode=='A' else
           #r'B: Matched maximum growth rate, $\lambda_{\max}/\kappa_0=0.05$')
    #ax.set_title(title,pad=17,fontsize=11)
    labels=[Line2D([],[],ls='--',color='black',label=r'$|J|=|g|$'),
            Line2D([],[],ls='',marker='*',ms=10,markerfacecolor='#FFD54F',
                   markeredgecolor='#191919',label='Manuscript point')]
    if mode=='A':
        labels.append(Line2D([],[],ls=':',color='cyan',lw=1.3,
                             label=r'$\lambda_{\max}^{\rm coh}=\lambda_{\max}^{\rm diss}$'))
    if has_equal_snr:
        labels.append(Line2D([],[],ls='-',color='#242424',lw=1.4,
                             label=r'Equal SNR'))
    fig.legend(handles=labels,loc='upper center',ncol=len(labels),frameon=False,
               bbox_to_anchor=(.49,1.025),fontsize=10,handlelength=1.8,
               columnspacing=1.05,handletextpad=.4)
    ax.text(.028,.035,fr'$n_\star={target:g}$, $\theta={theta:.3f}$',
            transform=ax.transAxes,fontsize=9.5,va='bottom',ha='left',
            bbox=dict(facecolor='white',alpha=.82,edgecolor='none',pad=2))
    fig.subplots_adjust(left=.14,right=.925,bottom=.14,top=.83)
    fig.savefig(out_pdf,bbox_inches='tight',pad_inches=.06)
    print('Figure saved:',out_pdf.resolve())
    if 'agg' not in mpl.get_backend().lower():plt.show()
    plt.close(fig)
    return log_ratio,delta


def run_heatmaps(args):
    """Sweep J and Gamma independently: O(N_J+N_G) exact Gaussian solves."""
    setup_style()
    base_c,base_d=target_models()
    theta=float(args.theta)
    if not 0.<theta<np.pi/2-1e-6:
        raise ValueError('Require 0 < theta < pi/2: at pi/2 the battery decouples '
                         'and cannot reach a nonzero target.')
    d_ref=ref_dissipative_model(base_d,theta)
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    gammas=np.linspace(float(args.gamma_min),float(args.gamma_max),int(args.map_ng))
    js=np.geomspace(float(args.j_min),float(args.j_max),int(args.map_nj))
    target=float(args.target if args.map_target is None else args.map_target)
    lam=float(args.growth_target)
    if args.map_nj<5 or args.map_ng<5:
        raise ValueError('Both map grids must have at least five points')
    if gammas[0]<=0 or js[0]<=0 or target<=0:
        raise ValueError('Positive Gamma, J, and target are required')
    if abs(lam-.05)>1e-12:
        print('INFO: Option B uses requested new growth target',lam)
    print('\nRESOURCE LANDSCAPES: same |E|/kappa0=10; independent J and Gamma')
    print('  Both schemes start from vacuum; dissipative theta =',theta)
    print('  Coherent CW drive frequency separately optimized at each J')
    print('  Dissipative remains at midpoint frequency, which is optimal here')
    print('  Option A holds separate reference pumps fixed; growth can differ')
    print('  Option B retunes BOTH pumps to max Re lambda =',lam)
    cohA=[];cohB=[];disA=[];disB=[]
    cache_coh={}
    for k,j in enumerate(js,1):
        for mode in ('A','B'):
            pump=(base_c.k_up if mode=='A' else coherent_pump_for_growth(j,lam))
            key=(round(float(j),13),round(float(pump),13))
            if key not in cache_coh:
                model=replace(base_c,J=float(j),k_up=float(pump))
                try:
                    s=Gaussian(model)
                except ValueError as e:
                    if 'nearly defective' not in str(e):raise
                    print('  WARNING: near coherent exceptional point at J=',j,
                          ': masked; use a finer grid without the exact EP')
                    result=dict(reached=False,tau=np.nan,nu=np.nan,snr=np.nan,
                                passive=np.nan,ergotropy_fraction=np.nan,power=np.nan)
                else:
                    result=fastest_frequency_for_coupling(
                        s,target,max(float(args.nu_max),float(j)+2.),
                        int(args.map_nu_points),float(args.map_tmax),
                        float(args.map_step))
                cache_coh[key]=result
            else:result=cache_coh[key]
            rec=dict(result,J=float(j),pump=float(pump),growth=float(
                replace(base_c,J=float(j),k_up=float(pump)).growth()))
            (cohA if mode=='A' else cohB).append(rec)
        if k%8==0 or k==len(js):
            print(f'  Coherent independent couplings: {k}/{len(js)}',flush=True)
    for k,gamma in enumerate(gammas,1):
        for mode in ('A','B'):
            pump=(d_ref.k_up if mode=='A' else
                  dissipative_pump_for_growth(gamma,theta,lam))
            model=replace(d_ref,gamma_col=float(gamma),k_up=float(pump))
            s=Gaussian(model)
            result=dissipative_first_passage_at_target(s,target,float(args.map_tmax))
            rec=dict(result,Gamma=float(gamma),pump=float(pump),growth=model.growth(),
                     offdiag=float(gamma*np.sin(theta)*np.cos(theta)/2.))
            (disA if mode=='A' else disB).append(rec)
        if k%12==0 or k==len(gammas):
            print(f'  Dissipative collective rates: {k}/{len(gammas)}',flush=True)
    for j,r in zip(js,cohB):
        if r['reached'] and abs(r['growth']-lam)>2e-9:
            raise AssertionError(f'Coherent matched growth failed at J={j}')
    for gamma,r in zip(gammas,disB):
        if abs(r['growth']-lam)>2e-9:
            raise AssertionError(f'Dissipative matched growth failed at Gamma={gamma}')
    # Independent 4-quadrature covariance test at non-reference parameter points.
    for model in (replace(base_c,J=.28,k_up=coherent_pump_for_growth(.28,lam)),
                  replace(d_ref,gamma_col=3.5,k_up=dissipative_pump_for_growth(3.5,theta,lam))):
        s=Gaussian(model)
        for t in (.4,1.1,3.):
            chk=independent_covariance_check(s,0.,t)
            if not np.isclose(s.n_noise(t),chk,rtol=1e-8,atol=1e-8):
                raise AssertionError('Independent covariance check failed')
    values={}
    for mode,crows,drows in [('A',cohA,disA),('B',cohB,disB)]:
        cvals=np.asarray([x['snr'] for x in crows]);dvals=np.asarray([x['snr'] for x in drows])
        with np.errstate(divide='ignore',invalid='ignore'):
            ratio=np.log10(dvals[None,:]/cvals[:,None])
        raw=dvals[None,:]-cvals[:,None]
        values[mode]=dict(log_ratio=ratio,raw_delta=raw)
    use='log_ratio' if args.color_metric=='log_ratio' else 'raw_delta'
    # One symmetric shared color scale makes Options A and B directly comparable.
    vlim=max(float(np.nanmax(np.abs(values[mode][use]))) for mode in ('A','B'))
    if not np.isfinite(vlim):raise RuntimeError('Landscape has no finite comparisons')
    ref_gamma=float(base_d.gamma_col)
    if abs(theta-base_d.theta)>1e-8:
        print('NOTE: nonreference theta. Reference star denotes the original '
              'J=3.2, Gamma=17.165 coordinate, not a matched-g point.')
    for mode,crows,drows,fname in [('A',cohA,disA,'F3_option_A_fixed_pumps.pdf'),
                                    ('B',cohB,disB,'F4_option_B_matched_growth.pdf')]:
        logr,delta=plot_landscape(gammas,js,crows,drows,theta,mode,args.color_metric,
                       vlim,ref_gamma,3.2,out/fname,target,lam)
        records=[]
        for iy,j in enumerate(js):
            cc=crows[iy]
            for ix,gamma in enumerate(gammas):
                dd=drows[ix]
                records.append(dict(option=mode,J_over_kappa0=float(j),
                       Gamma_col_over_kappa0=float(gamma),theta=theta,
                       g_over_kappa0=float(dd['offdiag']),
                       coherent_pump_over_kappa0=float(cc['pump']),
                       dissipative_pump_over_kappa0=float(dd['pump']),
                       lambda_coherent_over_kappa0=float(cc['growth']),
                       lambda_dissipative_over_kappa0=float(dd['growth']),
                       coherent_nu_opt_over_kappa0=float(cc['nu']),
                       coherent_t_ch_kappa0=float(cc['tau']),
                       dissipative_t_ch_kappa0=float(dd['tau']),
                       coherent_snr=float(cc['snr']),
                       dissipative_snr=float(dd['snr']),
                       log10_snr_ratio=float(logr[iy,ix]),
                       delta_snr=float(delta[iy,ix]),
                       coherent_passive_occupation=float(cc['passive']),
                       dissipative_passive_occupation=float(dd['passive']),
                       coherent_ergotropy_fraction=float(cc['ergotropy_fraction']),
                       dissipative_ergotropy_fraction=float(dd['ergotropy_fraction']),
                       both_reached=bool(cc['reached'] and dd['reached'])))
        csv_name=fname.replace('.pdf','.csv')
        pd.DataFrame(records).to_csv(out/csv_name,index=False)
        print(f'  {mode}: log10 SNR ratio range',float(np.nanmin(logr)),
              float(np.nanmax(logr)),'positive grid share',float(np.nanmean(logr>0)))
    # At reference coordinate, use EXACT reference models, not interpolation.
    ref_scoh=Gaussian(base_c);ref_sdis=Gaussian(base_d)
    opt=fastest_frequency_for_coupling(ref_scoh,target,
                   max(args.nu_max,5.5),max(101,args.map_nu_points),
                   args.map_tmax,args.map_step)
    ref_dis=dissipative_first_passage_at_target(ref_sdis,target,args.map_tmax)
    if abs(target-100)<1e-12:
        assert abs(opt['snr']-35.1499)<.03, opt
        assert abs(ref_dis['snr']-100.832)<.03,ref_dis
    reference=dict(J_over_kappa0=3.2,Gamma_col_over_kappa0=ref_gamma,
          coherent_opt=opt,dissipative_midpoint=ref_dis,
          log10_snr_ratio=float(np.log10(ref_dis['snr']/opt['snr'])),
          delta_snr=float(ref_dis['snr']-opt['snr']))
    metadata=dict(quantity='SNR_b,star = |<b>|^2/(n_b_inc + 0.5)',
      comparison='two separate physical devices; J and Gamma not same resource',
      options={'A':'Both gain rates fixed independently at manuscript values; no growth matching in the scan',
               'B':'Gain rate recalibrated separately in each architecture to fixed maximal growth'},
      same_target_first_passage=True,target=target,theta=theta,phi=0.,
      normalized_drive_amplitude_both=10.,coupling_axis='J (Hamiltonian)',
      dissipative_axis='Gamma_col (collective Lindblad rate)',
      colormap='RdBu_r; zero=equal SNR, red=dissipative advantage, blue=coherent advantage',
      coherent_drive='single monochromatic frequency optimized for earliest first passage at each J',
      dissipative_drive='midpoint, frequency optimal for delta=0, theta in (0,pi/2)',
      baseline_pumps=dict(coherent=base_c.k_up,dissipative=base_d.k_up),
      gamma_reference=ref_gamma,common_growth_B=lam,
      original_reference=reference,
      warnings=['NOT an equal energetic cost or equal microscopic hardware-resource comparison',
                'Changing Gamma changes BOTH off-diagonal g and local dissipative terms',
                'Noise covariances include exact Gaussian gain noise and passive reservoir influence',
                'Pointwise pump fitting in Option B is not an experimentally free operation',
                'First passages exceeding map time horizon are masked rather than assigned zero',
                'The coherent frequency maximum uses a global grid with local continuous refinement'],
      figures=['F2_finite_capacity_optimized.pdf',
               'F3_option_A_fixed_pumps.pdf','F4_option_B_matched_growth.pdf'])
    (out/'resource_landscape_parameters.json').write_text(
        json.dumps(metadata,indent=2,default=lambda x:float(x) if isinstance(x,np.generic) else x),
        encoding='utf-8')
    print('Reference J=3.2, Gamma=17.165 at n_star=',target,
          'coherent optimized SNR=',opt['snr'],
          'dissipative SNR=',ref_dis['snr'])
    print('Only two heatmap figures created (alongside original F2).')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',choices=['linear','heatmaps','all','validate'],default='all')
    parser.add_argument('--target',type=float,default=100.,help='F2 maximum target')
    parser.add_argument('--target-points',type=int,default=701)
    parser.add_argument('--t-max',type=float,default=70.)
    parser.add_argument('--step',type=float,default=.025)
    parser.add_argument('--nu-max',type=float,default=6.5)
    parser.add_argument('--nu-points',type=int,default=161)
    parser.add_argument('--out',default='Phase_Locked_Resource_Landscapes')
    parser.add_argument('--theta',type=float,default=1.15)
    parser.add_argument('--map-target',type=float,default=None,
                        help='Map target; defaults to the F2 --target setting')
    parser.add_argument('--j-min',type=float,default=.08)
    parser.add_argument('--j-max',type=float,default=8.)
    parser.add_argument('--gamma-min',type=float,default=1.)
    parser.add_argument('--gamma-max',type=float,default=45.)
    parser.add_argument('--map-nj',type=int,default=53)
    parser.add_argument('--map-ng',type=int,default=55)
    parser.add_argument('--map-tmax',type=float,default=18.)
    parser.add_argument('--map-step',type=float,default=.025)
    parser.add_argument('--map-nu-points',type=int,default=101)
    parser.add_argument('--growth-target',type=float,default=.05)
    parser.add_argument('--color-metric',choices=['log_ratio','difference'],default='log_ratio')
    args=parser.parse_args(argv)
    for name in ('target','t_max','step','nu_max','j_min',
                 'j_max','gamma_min','gamma_max','map_tmax','map_step','growth_target'):
        x=getattr(args,name)
        if not np.isfinite(x) or x<=0:parser.error(f'--{name.replace("_","-")} must be positive and finite')
    if args.map_target is not None and (not np.isfinite(args.map_target) or args.map_target<=0):
        parser.error('--map-target must be positive and finite')
    if args.j_max<=args.j_min or args.gamma_max<=args.gamma_min:
        parser.error('Coupling scan upper limits must exceed the lower limits')
    if args.nu_points<9 or args.target_points<3 or args.map_nu_points<9:
        parser.error('nu-points>=9, target-points>=3, map-nu-points>=9 required')
    if args.command=='validate':validate()
    if args.command in ('linear','all'):analyze(args)
    if args.command in ('heatmaps','all'):run_heatmaps(args)


if __name__=='__main__':
    if is_notebook_launcher():
        main([NOTEBOOK_COMMAND,'--target',str(NOTEBOOK_TARGET_N_STAR),
              *NOTEBOOK_EXTRA_ARGS])
    else:
        main()