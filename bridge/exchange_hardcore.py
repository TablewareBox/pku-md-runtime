"""Extra Slater exchange hard core. The matched Slater kernel is not modified.

The implementation that SlaterExGenerator calls is slater_sr_hc_kernel in
junminchen/openmm-phyneo-plugin DMFF/dmff/admp/pairwise.py:

    b = sqrt(Bi * Bj)
    br = b * r
    x = 0.24 * br
    E_hc = m * Ai * Aj / x^14

added to the usual +Ai*Aj*P(br)*exp(-br). Inside that kernel r is Å and B is
Å^-1. Br is the same when r is nm and B is the XML value in nm^-1, so

    E_hc = m * A_i * A_j / (0.24 * sqrt(B_i * B_j) * r_nm)^14

Topological distances 1-2 through 1-6 are omitted, matching SlaterExForce
mScale12..16 = 0. Only this extra piece is added; the exponential exchange
stays in the matched kernel.
"""
import xml.etree.ElementTree as ET

import numpy as np

ALPHA = 0.24
FORCE_GROUP = 20


def load_AB(topology, xml):
    root = ET.parse(xml).getroot()
    table = {a.attrib["type"]: (float(a.attrib["A"]), float(a.attrib["B"]))
             for a in root.find("SlaterExForce") if a.tag == "Atom"}
    res = {(r.attrib["name"], a.attrib["name"]): a.attrib["type"]
           for r in root.find("Residues") for a in r if a.tag == "Atom"}
    rows = [table[res[(a.residue.name, a.name)]] for a in topology.atoms()]
    return (np.array([a for a, _ in rows], dtype=float),
            np.array([b for _, b in rows], dtype=float))


def excluded_pairs(topology, max_bonds=5):
    n = topology.getNumAtoms()
    adj = {i: set() for i in range(n)}
    for bond in topology.bonds():
        i, j = bond[0].index, bond[1].index
        adj[i].add(j)
        adj[j].add(i)
    pairs = []
    for i in range(n):
        seen, front = {i}, {i}
        for _ in range(max_bonds):
            front = {j for u in front for j in adj[u] if j not in seen}
            seen |= front
        pairs.extend((i, j) for j in seen if j > i)
    return pairs


def add_openmm_hardcore(system, topology, xml, cutoff_nm=1.0, max_bonds=5):
    import openmm as mm
    from openmm import unit
    force = mm.CustomNonbondedForce("A1*A2/((alpha*sqrt(B1*B2)*r)^14)")
    force.addGlobalParameter("alpha", ALPHA)
    force.addPerParticleParameter("A")
    force.addPerParticleParameter("B")
    A, B = load_AB(topology, xml)
    for a, b in zip(A, B):
        force.addParticle([float(a), float(b)])
    for i, j in excluded_pairs(topology, max_bonds):
        force.addExclusion(i, j)
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(cutoff_nm * unit.nanometer)
    force.setForceGroup(FORCE_GROUP)
    system.addForce(force)
    return force
