"""DMFF ADMPDispPmeForce dispersion for an OpenMM System that carries an ADMPPmeForce.

DMFF: E = -(E_real + E_recip + E_self) with per-atom c = sqrt(C) and
E_real = sum_{pairs within rc} (mscale + g_p(kappa^2 r^2) - 1) c_i c_j / r^p.
The ADMPPmeForce dispersion channels already return -(E_recip + E_self) with kappa equal to
the multipole PME alpha, so this module only supplies sqrt(C) to the plugin and the real-space
part: -g_p c c / r^p for non-excluded pairs and +(1 - g_p) c c / r^p for zero-mscale pairs.
"""
import numpy as np
from openmm import CustomBondForce, CustomNonbondedForce

G6 = "exp(-x2)*(1+x2+0.5*x2^2)"
G8 = "exp(-x2)*(1+x2+0.5*x2^2+x2^3/6)"
G10 = "exp(-x2)*(1+x2+0.5*x2^2+x2^3/6+x2^4/24)"


def excluded_pairs(topology, max_bonds):
    adj = {i: set() for i in range(topology.getNumAtoms())}
    for a, b in topology.bonds():
        adj[a.index].add(b.index)
        adj[b.index].add(a.index)
    pairs = set()
    for i in adj:
        seen, front = {i}, {i}
        for _ in range(max_bonds):
            front = {j for u in front for j in adj[u] if j not in seen}
            seen |= front
        pairs |= {(min(i, j), max(i, j)) for j in seen if j != i}
    return sorted(pairs)


def add_dmff_dispersion(system, admp, topology, C, kappa, cutoff, max_bonds=5):
    """C: (N, 3) raw C6/C8/C10 in kJ/mol nm^p; kappa: ADMPPmeForce alpha (nm^-1); cutoff in nm.

    max_bonds=5 excludes 1-2 through 1-6 pairs (phyneo_ecl.xml sets mScale12..16 = 0).
    """
    c = np.sqrt(np.asarray(C, dtype=float))
    if admp.getNumDispersionParticles() == 0:
        for row in c:
            admp.addDispersionParticle(*map(float, row))
    else:
        for i, row in enumerate(c):
            admp.setDispersionParameters(i, *map(float, row))
    excl = excluded_pairs(topology, max_bonds)

    sr = CustomNonbondedForce(
        f"-(c61*c62*({G6})/r^6 + c81*c82*({G8})/r^8 + c101*c102*({G10})/r^10); x2=kappa2*r^2")
    sr.setName("DispersionPmeReal")
    for p in ("c6", "c8", "c10"):
        sr.addPerParticleParameter(p)
    sr.addGlobalParameter("kappa2", kappa * kappa)
    sr.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    sr.setCutoffDistance(cutoff)
    sr.setUseSwitchingFunction(False)
    sr.setUseLongRangeCorrection(False)
    for row in c:
        sr.addParticle(list(map(float, row)))
    for i, j in excl:
        sr.addExclusion(i, j)

    ex = CustomBondForce(
        f"step(rc-r)*((1-{G6})*c6/r^6 + (1-{G8})*c8/r^8 + (1-{G10})*c10/r^10); x2=kappa2*r^2")
    ex.setName("DispersionPmeExclusionCorrection")
    for p in ("c6", "c8", "c10"):
        ex.addPerBondParameter(p)
    ex.addGlobalParameter("kappa2", kappa * kappa)
    ex.addGlobalParameter("rc", cutoff)
    ex.setUsesPeriodicBoundaryConditions(True)
    for i, j in excl:
        ex.addBond(i, j, list(map(float, c[i] * c[j])))

    system.addForce(sr)
    system.addForce(ex)
    return sr, ex
