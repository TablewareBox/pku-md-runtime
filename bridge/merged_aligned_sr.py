"""One CustomNonbonded for the energy-aligned short-range terms.

Slater channels, QqTt, the form-5 hardcore and the dispersion g_p real space share
the 1-2..1-6 exclusion list, so they can be one kernel. Pairs on that list still
need the dispersion (1-g_p) CustomBond; a nonbonded kernel never sees them.
"""
import gc
import json
import math
import time
import xml.etree.ElementTree as ET

import numpy as np
import openmm as mm
from openmm import Platform, unit
from openmm.app import AllBonds, ForceField, HAngles, HBonds, PDBFile, PME

from dmff_dispersion import G6, G8, G10, add_dmff_dispersion, excluded_pairs
from exchange_hardcore import ALPHA
from stability_hypotheses import (
    BASE, CUTOFF, ECL, ETHRESH, KJ, NM, build, fix_pf6, zero_lj,
)

CONSTRAINTS = {"none": None, "hbonds": HBonds, "allbonds": AllBonds, "hangles": HAngles}
import mpidplugin

ONE_4PI_EPS0 = 138.935455846


def slater_types(root):
    fields = ["Aex", "Bex", "Aes", "Bes", "Apol", "Bpol", "Adhf", "Bdhf", "Asdisp", "Bsdisp",
              "Bdisp", "C6", "C8", "C10", "Bqq", "Qqq"]
    types = {name: {} for name in fields}
    for tag, fa, fb in (("SlaterExForce", "Aex", "Bex"), ("SlaterSrEsForce", "Aes", "Bes"),
                        ("SlaterSrPolForce", "Apol", "Bpol"), ("SlaterDhfForce", "Adhf", "Bdhf"),
                        ("SlaterSrDispForce", "Asdisp", "Bsdisp")):
        for atom in root.find(tag).findall("Atom"):
            types[fa][atom.get("type")] = float(atom.get("A", 0.0))
            types[fb][atom.get("type")] = float(atom.get("B", 0.0))
    for atom in root.find("SlaterDampingForce").findall("Atom"):
        types["Bdisp"][atom.get("type")] = float(atom.attrib["B"])
        for name in ("C6", "C8", "C10"):
            types[name][atom.get("type")] = float(atom.attrib[name])
    for atom in root.find("QqTtDampingForce").findall("Atom"):
        types["Bqq"][atom.get("type")] = float(atom.get("B", 0.0))
        types["Qqq"][atom.get("type")] = float(atom.get("Q", 0.0))
    return fields, types


def merged_expression(hardcore="r14"):
    def channel(a, b, sign):
        br = f"sqrt({b}1*{b}2)*r"
        return f"{sign}{a}1*{a}2*(1+({br})+({br})^2/3)*exp(-({br}))"
    ex = channel("Aex", "Bex", "")
    es = channel("Aes", "Bes", "-")
    ep = channel("Apol", "Bpol", "-")
    ed = channel("Asdisp", "Bsdisp", "-")
    eh = channel("Adhf", "Bdhf", "-")
    damp = ("f6*sqrt(C61*C62)/r^6 + f8*sqrt(C81*C82)/r^8 + f10*sqrt(C101*C102)/r^10")
    if hardcore == "r14":
        hc = "Aex1*Aex2/((alpha*sqrt(Bex1*Bex2)*r)^14)"
    elif hardcore == "r12":
        hc = "Aex1*Aex2*(4.3/(sqrt(Bex1*Bex2)*r))^12"
    else:
        raise ValueError("hardcore must be r14 or r12")
    qq = f"-{ONE_4PI_EPS0}*exp(-sqrt(Bqq1*Bqq2)*r)*(1+sqrt(Bqq1*Bqq2)*r)*Qqq1*Qqq2/r"
    gp = (f"-(sqrt(D61*D62)*({G6})/r^6 + sqrt(D81*D82)*({G8})/r^8"
          f" + sqrt(D101*D102)*({G10})/r^10)")
    return (
        f"{ex}{es}{ep}{ed}{eh}+({damp})+({hc})+({qq})+({gp})"
        "; f10=f8+exp(-z)*(z^9/362880+z^10/3628800)"
        "; f8=f6+exp(-z)*(z^7/5040+z^8/40320)"
        "; f6=exp(-z)*(1+z+z^2/2+z^3/6+z^4/24+z^5/120+z^6/720)"
        "; z=x-(2*x^2+3*x)/(x^2+3*x+3); x=sqrt(Bdisp1*Bdisp2)*r; x2=kappa2*r^2"
    )


def pme_parameters(box_nm, cutoff, ethresh, kappa=None, grid=None):
    if kappa is None:
        kappa = math.sqrt(-math.log(2.0 * ethresh)) / cutoff
    if grid is None:
        scale = 2.0 / (3.0 * ethresh ** 0.2)
        grid = max(int(math.ceil(scale * kappa * length)) for length in box_nm)
    return float(kappa), int(grid)


def dispersion_rows(topology, root):
    res_map = {(r.get("name"), a.get("name")): a.get("type")
               for r in root.find("Residues") for a in r.findall("Atom")}
    disp = {atom.get("type"): tuple(float(atom.get(name)) for name in ("C6", "C8", "C10"))
            for atom in root.find("ADMPDispPmeForce").findall("Atom")}
    return np.array([disp[res_map[(atom.residue.name, atom.name)]] for atom in topology.atoms()])


def particle_rows(topology, root):
    """Per-atom columns of the merged kernel, in ``merged_expression`` order.

    The last three columns are the ADMP dispersion C6/C8/C10. They are named
    D6/D8/D10 so they stay distinct from the Slater damping coefficients.
    """
    fields, types = slater_types(root)
    names = fields + ["D6", "D8", "D10"]
    disp = {atom.get("type"): tuple(float(atom.get(name)) for name in ("C6", "C8", "C10"))
            for atom in root.find("ADMPDispPmeForce").findall("Atom")}
    res = {(r.get("name"), a.get("name")): a.get("type") for r in root.find("Residues") for a in r.findall("Atom")}
    rows, kinds, same = [], [], True
    for atom in topology.atoms():
        kind = res[(atom.residue.name, atom.name)]
        row = [types[name].get(kind, 0.0) for name in fields]
        d6, d8, d10 = disp[kind]
        same = same and abs(d6 - types["C6"].get(kind, 0.0)) < 1e-8
        row.extend([d6, d8, d10])
        rows.append(row)
        kinds.append(kind)
    return names, np.asarray(rows, dtype=float), kinds, same


def type_order(root):
    return [atom.get("type") for atom in root.find("SlaterExForce").findall("Atom")]


def build_merged(pdb, ff_xml=None, ecl_xml=None, cutoff=None, ethresh=ETHRESH, kappa=None, grid=None,
                 thole_width=5.0, mutual_induced_target_epsilon=1e-8, mutual_induced_max_iterations=500,
                 constraints="HBonds", rigid_water=False, polarization="mutual", exclusion_shell=5,
                 hardcore_alpha=None, hardcore="r14", remove_lj=True, repair_pf6_trans=True, admp_group=1,
                 sr_group=17, disp_bond_group=22):
    """Build the frozen merged Hamiltonian. Defaults match the aligned production protocol."""
    ff_xml = ff_xml or (BASE + "/phyneo_hmtff_admp.xml")
    ecl_xml = ecl_xml or ECL
    cutoff = CUTOFF if cutoff is None else float(cutoff)
    box = np.array([float(v[i].value_in_unit(NM)) for i, v in enumerate(pdb.topology.getPeriodicBoxVectors())])
    kappa, k = pme_parameters(box, cutoff, ethresh, kappa, grid)
    system = ForceField(ff_xml).createSystem(
        pdb.topology, nonbondedMethod=PME, nonbondedCutoff=cutoff * NM,
        constraints=CONSTRAINTS[constraints.lower()], rigidWater=rigid_water, polarization=polarization,
        ewaldErrorTolerance=ethresh, defaultTholeWidth=thole_width,
        mutualInducedTargetEpsilon=mutual_induced_target_epsilon)
    if remove_lj:
        zero_lj(system)
    n_trans = fix_pf6(system, pdb) if repair_pf6_trans else 0
    admp = next(mpidplugin.ADMPPmeForce.cast(f) for f in system.getForces() if mpidplugin.ADMPPmeForce.isinstance(f))
    admp.setPMEParameters(kappa, k, k, k)
    admp.setMutualInducedMaxIterations(mutual_induced_max_iterations)
    admp.setForceGroup(admp_group)
    root = ET.parse(ecl_xml).getroot()
    c = dispersion_rows(pdb.topology, root)
    add_dmff_dispersion(system, admp, pdb.topology, c, kappa=kappa, cutoff=cutoff, max_bonds=exclusion_shell)
    for index in range(system.getNumForces() - 1, -1, -1):
        force = system.getForce(index)
        if force.getName() == "DispersionPmeReal" or (remove_lj and isinstance(force, mm.NonbondedForce)):
            system.removeForce(index)
        elif force.getName() == "DispersionPmeExclusionCorrection":
            force.setForceGroup(disp_bond_group)
    names, rows, _, same = particle_rows(pdb.topology, root)
    force = mm.CustomNonbondedForce(merged_expression(hardcore))
    force.setName("MergedShortRange")
    if hardcore == "r14":
        force.addGlobalParameter("alpha", ALPHA if hardcore_alpha is None else float(hardcore_alpha))
    force.addGlobalParameter("kappa2", kappa * kappa)
    for name in names:
        force.addPerParticleParameter(name)
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(cutoff * NM)
    force.setUseSwitchingFunction(False)
    force.setUseLongRangeCorrection(False)
    for row in rows:
        force.addParticle(row.tolist())
    excl = excluded_pairs(pdb.topology, exclusion_shell)
    for i, j in excl:
        force.addExclusion(i, j)
    force.setForceGroup(sr_group)
    system.addForce(force)
    return system, kappa, k, n_trans, same, box


def group_energy(ctx, groups):
    return ctx.getState(getEnergy=True, groups=set(groups)).getPotentialEnergy().value_in_unit(KJ)


def forces(ctx):
    return np.asarray(ctx.getState(getForces=True).getForces().value_in_unit(KJ / NM))


def open_context(system, pdb, precision):
    integ = mm.LangevinMiddleIntegrator(298.15 * unit.kelvin, 1.0 / unit.picosecond, 0.0005 * unit.picoseconds)
    integ.setRandomNumberSeed(20261008)
    integ.setConstraintTolerance(1e-5)
    ctx = mm.Context(system, integ, Platform.getPlatformByName("CUDA"),
                     {"Precision": precision, "DeviceIndex": "0"})
    ctx.setPositions(pdb.positions)
    ctx.applyConstraints(1e-5)
    return ctx


def main():
    plugin = __import__("sys").argv[1]
    Platform.loadPluginsFromDirectory(plugin)
    pdb = PDBFile(BASE + "/init.pdb")
    split, *_ = build(pdb, "nobonds")
    ctx = open_context(split, pdb, "double")
    ref_sr = group_energy(ctx, (17, 20, 21, 22, 23))
    ref_total = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
    ref_f = forces(ctx)
    del ctx, split
    gc.collect()
    system, kappa, k, n_trans, same_c, box = build_merged(pdb)
    ctx = open_context(system, pdb, "double")
    got_sr = group_energy(ctx, (17, 22))
    got_total = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
    got_f = forces(ctx)
    delta = got_f - ref_f
    print(json.dumps({
        "stage": "energy", "kappa": kappa, "grid": k, "pf6_trans_180": n_trans,
        "dispersion_C_equals_slater_C": same_c,
        "per_particle_parameters": 19,
        "split_short_range": ref_sr, "merged_short_range": got_sr,
        "short_range_difference": got_sr - ref_sr,
        "total_difference": got_total - ref_total,
        "force_rms": float(np.sqrt(np.mean(delta ** 2))),
        "force_max": float(np.max(np.abs(delta))),
    }), flush=True)
    ctx.setVelocitiesToTemperature(298.15 * unit.kelvin, 20261008)
    integ = ctx.getIntegrator()
    integ.step(200)
    t0 = time.time()
    integ.step(2000)
    dt = time.time() - t0
    pe = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
    print(json.dumps({"stage": "speed", "warmup": 200, "steps": 2000, "ms_per_step": 1000.0 * dt / 2000.0,
                      "pe": pe, "finite": bool(np.isfinite(pe))}), flush=True)


if __name__ == "__main__":
    main()
