"""Short production stability test. Form-5 hardcore is always on.

Two custom hypotheses, same cutoff, kappa, grid, QqTt, dispersion g_p and ADMP:

  bonds    Slater CustomNonbonded plus the distance-4/5 CustomBond correction
           (1-5 and 1-6 cancelled, the algebra that matched DMFF).
  nobonds  same CustomNonbonded, no correction bonds (1-5 and 1-6 stay full).

Native mode 3 is the other hypothesis and is launched by run_native_production.sh.
"""
import json
import math
import sys
import time

import numpy as np
import openmm as mm
from openmm import Platform, unit
from openmm.app import ForceField, HBonds, PDBFile, PME

sys.path.insert(0, "/home/changjh/MD_projects/PKUGraduateThesis/bridge")
sys.path.insert(0, "/home/changjh/MD_projects/ADMPPmeOpenMMPlugin")

import mpidplugin  # noqa: E402

from add_slater_custom import add_slater_custom  # noqa: E402
from dmff_dispersion import add_dmff_dispersion, excluded_pairs  # noqa: E402
from dmff_qqtt import add_qqtt_damping  # noqa: E402
from exchange_hardcore import ALPHA, FORCE_GROUP, add_openmm_hardcore, load_AB  # noqa: E402

BASE = "/home/changjh/MD_projects/PhyNEO-Electrolyte/examples/md_simulation"
ECL = BASE + "/phyneo_ecl.xml"
ETHRESH = 1e-4
CUTOFF = 1.0
KJ = unit.kilojoule_per_mole
NM = unit.nanometer


def ewald(box_nm):
    kappa = math.sqrt(-math.log(2.0 * ETHRESH)) / CUTOFF
    scale = 2.0 / (3.0 * ETHRESH ** 0.2)
    k = max(int(math.ceil(scale * kappa * length)) for length in box_nm)
    return kappa, k


def zero_lj(system):
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    for i in range(nb.getNumParticles()):
        q, sig, _ = nb.getParticleParameters(i)
        nb.setParticleParameters(i, q, sig, 0.0)
    for k in range(nb.getNumExceptions()):
        i, j, q, sig, _ = nb.getExceptionParameters(k)
        nb.setExceptionParameters(k, i, j, q, sig, 0.0)


def fix_pf6(system, pdb):
    x = np.asarray(pdb.positions.value_in_unit(NM))
    box = np.array([float(v[i].value_in_unit(NM)) for i, v in enumerate(pdb.topology.getPeriodicBoxVectors())])
    atoms = list(pdb.topology.atoms())
    af = next(f for f in system.getForces() if isinstance(f, mm.HarmonicAngleForce))
    n = 0
    for k in range(af.getNumAngles()):
        i, j, l, t0, kk = af.getAngleParameters(k)
        if atoms[j].residue.name != "PF6":
            continue
        u, v = x[i] - x[j], x[l] - x[j]
        u -= np.round(u / box) * box
        v -= np.round(v / box) * box
        if np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1.0, 1.0))) > 135.0:
            af.setAngleParameters(k, i, j, l, np.pi, kk)
            n += 1
    return n


def add_distance45_corrections(system, topology, C, kappa):
    """Cancel 1-5 and 1-6 on the forces that now only exclude 1-2..1-4.

    Dispersion nonbonded contributes -g on these pairs, so the bond adds the
    bare c/r^p and the sum is the m=0 result (1-g). QqTt and hardcore should
    be zero on these pairs, so their bonds are the negative of the nonbonded term.
    kappa is unused: the bare term has no Ewald split.
    """
    import xml.etree.ElementTree as ET
    del kappa
    pairs14 = set(excluded_pairs(topology, 3))
    extra = [p for p in excluded_pairs(topology, 5) if p not in pairs14]
    c = np.sqrt(np.asarray(C, dtype=float))
    bare = mm.CustomBondForce("step(rc-r)*(c6/r^6 + c8/r^8 + c10/r^10)")
    bare.setName("DispersionDistance45Bare")
    for name in ("c6", "c8", "c10"):
        bare.addPerBondParameter(name)
    bare.addGlobalParameter("rc", CUTOFF)
    bare.setUsesPeriodicBoundaryConditions(True)
    bare.setForceGroup(22)
    for i, j in extra:
        bare.addBond(i, j, list(map(float, c[i] * c[j])))
    system.addForce(bare)

    root = ET.parse(ECL).getroot()
    res_map = {(r.attrib["name"], a.attrib["name"]): a.attrib["type"]
               for r in root.find("Residues") for a in r if a.tag == "Atom"}
    qq = {a.attrib["type"]: (float(a.attrib["B"]), float(a.attrib["Q"]))
          for a in root.find("QqTtDampingForce") if a.tag == "Atom"}
    atoms = list(topology.atoms())
    bq = [qq[res_map[(a.residue.name, a.name)]] for a in atoms]
    qbond = mm.CustomBondForce("138.935455846*exp(-bij*r)*(1+bij*r)*q/r")
    qbond.setName("QqTtDistance45Cancel")
    qbond.addPerBondParameter("bij")
    qbond.addPerBondParameter("q")
    qbond.setUsesPeriodicBoundaryConditions(True)
    qbond.setForceGroup(23)
    for i, j in extra:
        bi, qi = bq[i]
        bj, qj = bq[j]
        qbond.addBond(i, j, [math.sqrt(bi * bj), qi * qj])
    system.addForce(qbond)

    A, B = load_AB(topology, ECL)
    hbond = mm.CustomBondForce("-A/((alpha*B*r)^14)")
    hbond.setName("HardcoreDistance45Cancel")
    hbond.addGlobalParameter("alpha", ALPHA)
    hbond.addPerBondParameter("A")
    hbond.addPerBondParameter("B")
    hbond.setUsesPeriodicBoundaryConditions(True)
    hbond.setForceGroup(20)
    for i, j in extra:
        hbond.addBond(i, j, [float(A[i] * A[j]), float(math.sqrt(B[i] * B[j]))])
    system.addForce(hbond)


def dispersion_coeff(topology):
    import xml.etree.ElementTree as ET
    root = ET.parse(ECL).getroot()
    res_map = {(r.attrib["name"], a.attrib["name"]): a.attrib["type"]
               for r in root.find("Residues") for a in r if a.tag == "Atom"}
    disp = {x.attrib["type"]: tuple(float(x.attrib[c]) for c in ("C6", "C8", "C10"))
            for x in root.find("ADMPDispPmeForce") if x.tag == "Atom"}
    return np.array([disp[res_map[(a.residue.name, a.name)]] for a in topology.atoms()])


def build(pdb, mode):
    box = np.array([float(v[i].value_in_unit(NM)) for i, v in enumerate(pdb.topology.getPeriodicBoxVectors())])
    kappa, k = ewald(box)
    system = ForceField(BASE + "/phyneo_hmtff_admp.xml").createSystem(
        pdb.topology, nonbondedMethod=PME, nonbondedCutoff=CUTOFF * NM, constraints=HBonds,
        rigidWater=False, polarization="mutual", ewaldErrorTolerance=ETHRESH, defaultTholeWidth=5.0,
        mutualInducedTargetEpsilon=1e-8)
    zero_lj(system)
    n_trans = fix_pf6(system, pdb)
    admp = next(mpidplugin.ADMPPmeForce.cast(f) for f in system.getForces() if mpidplugin.ADMPPmeForce.isinstance(f))
    admp.setPMEParameters(kappa, k, k, k)
    admp.setMutualInducedMaxIterations(500)
    # Mutual polarization puts 1-2..1-6 into NonbondedForce (10362 exceptions).
    # nobonds copies that list, so 1-5 and 1-6 never enter the Slater custom force.
    # bonds keeps the historical 1-4 list on every custom nonbonded force and
    # cancels distances 4 and 5 with CustomBondForces. The lists have to match
    # or CUDA context creation raises "All Forces must have identical exceptions".
    shell = 3 if mode == "bonds" else 5
    c = dispersion_coeff(pdb.topology)
    add_dmff_dispersion(system, admp, pdb.topology, c, kappa=kappa, cutoff=CUTOFF, max_bonds=shell)
    add_slater_custom(system, pdb.topology, ECL, cutoff=CUTOFF * NM,
                      exclusions=excluded_pairs(pdb.topology, shell), covalent_correction=(mode == "bonds"))
    add_qqtt_damping(system, pdb.topology, ECL, CUTOFF * NM, max_bonds=shell)
    add_openmm_hardcore(system, pdb.topology, ECL, CUTOFF, max_bonds=shell)
    if mode == "bonds":
        add_distance45_corrections(system, pdb.topology, c, kappa)
    corr = 0
    for force in system.getForces():
        name = force.getName()
        if name.startswith("SlaterCustomForce"):
            force.setForceGroup(17)
        elif name.startswith("SlaterCovalentCorrection"):
            force.setForceGroup(18)
            corr = force.getNumBonds()
        elif name == "DispersionPmeReal":
            force.setForceGroup(21)
        elif name == "DispersionPmeExclusionCorrection":
            force.setForceGroup(22)
        elif name == "QqTtDampingForce":
            force.setForceGroup(23)
        elif mpidplugin.ADMPPmeForce.isinstance(force):
            force.setForceGroup(1)
    return system, kappa, k, n_trans, corr, box


def snapshot(ctx):
    st = ctx.getState(getEnergy=True, getForces=True, getPositions=True)
    f = np.asarray(st.getForces().value_in_unit(KJ / NM))
    return (st.getPotentialEnergy().value_in_unit(KJ),
            float(np.max(np.abs(f))),
            np.asarray(st.getPositions().value_in_unit(NM)))


def relax(ctx, steps=300, max_disp=0.001, force_goal=500.0):
    disp = max_disp
    accepted = 0
    for it in range(steps):
        e, fmax, x = snapshot(ctx)
        if not np.isfinite(e):
            print(json.dumps({"stage": "relax_nan", "it": it}), flush=True)
            return e, fmax, accepted
        if fmax < force_goal:
            print(json.dumps({"stage": "relax_stop", "it": it, "E": e, "max_force": fmax}), flush=True)
            return e, fmax, accepted
        f = np.asarray(ctx.getState(getForces=True).getForces().value_in_unit(KJ / NM))
        trial = x + (disp / max(fmax, 1e-12)) * f
        ctx.setPositions(trial * NM)
        ctx.applyConstraints(1e-5)
        et, ftmax, _ = snapshot(ctx)
        ok = np.isfinite(et) and et < e and ftmax < max(5000.0, 3.0 * fmax)
        if ok:
            accepted += 1
            disp = min(max_disp, disp * (1.2 if ftmax < fmax else 0.8))
        else:
            ctx.setPositions(x * NM)
            disp *= 0.5
            if disp < 1e-6:
                print(json.dumps({"stage": "relax_stall", "it": it, "E": e, "max_force": fmax}), flush=True)
                break
        if it % 20 == 0:
            print(json.dumps({"stage": "relax", "it": it, "E": e if not ok else et,
                              "max_force": fmax if not ok else ftmax, "accepted": accepted}), flush=True)
    e, fmax, _ = snapshot(ctx)
    return e, fmax, accepted


def closest(x, box):
    d = x[:, None, :] - x[None, :, :]
    d -= box * np.round(d / box)
    r = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(r, np.inf)
    i, j = np.unravel_index(int(np.argmin(r)), r.shape)
    return float(r[i, j])


def group_energy(ctx, group):
    return ctx.getState(getEnergy=True, groups={group}).getPotentialEnergy().value_in_unit(KJ)


def main():
    out, plugin, mode, precision, nsteps = sys.argv[1:6]
    nsteps = int(nsteps)
    if mode not in ("bonds", "nobonds"):
        raise SystemExit("mode must be bonds or nobonds")
    Platform.loadPluginsFromDirectory(plugin)
    pdb = PDBFile(BASE + "/init.pdb")
    system, kappa, k, n_trans, corr, box = build(pdb, mode)
    expect = 2840 if mode == "bonds" else 0
    names = [{"name": f.getName(), "group": f.getForceGroup()} for f in system.getForces()]
    print(json.dumps({"stage": "built", "mode": mode, "precision": precision, "kappa": kappa, "grid": k,
                      "pf6_trans_180": n_trans, "slater_correction_bonds": corr, "expected_bonds": expect,
                      "hardcore": "A/(0.24*Br)^14", "forces": names}), flush=True)
    if corr != expect:
        raise SystemExit(f"correction bond count {corr} != {expect}")
    dt_fs = 0.5
    integ = mm.LangevinMiddleIntegrator(298.15 * unit.kelvin, 1.0 / unit.picosecond, dt_fs * unit.femtosecond)
    integ.setRandomNumberSeed(20261008)
    integ.setConstraintTolerance(1e-5)
    ctx = mm.Context(system, integ, Platform.getPlatformByName("CUDA"),
                     {"Precision": precision, "DeviceIndex": "0"})
    ctx.setPositions(pdb.positions)
    ctx.applyConstraints(1e-5)
    e0, f0, _ = snapshot(ctx)
    parts = {
        "slater_17_18": group_energy(ctx, 17) + group_energy(ctx, 18),
        "admp_1": group_energy(ctx, 1),
        "disp_21_22": group_energy(ctx, 21) + group_energy(ctx, 22),
        "qqtt_23": group_energy(ctx, 23),
        "hardcore_20": group_energy(ctx, FORCE_GROUP),
    }
    print(json.dumps({"stage": "initial", "E": e0, "max_force": f0, "parts": parts}), flush=True)
    t0 = time.time()
    e1, f1, n_acc = relax(ctx)
    print(json.dumps({"stage": "minimized", "E": e1, "max_force": f1, "accepted": n_acc,
                      "hardcore": group_energy(ctx, FORCE_GROUP), "wall_s": time.time() - t0}), flush=True)
    if not np.isfinite(e1):
        print(json.dumps({"status": "diverged during relax"}), flush=True)
        return
    ctx.setVelocitiesToTemperature(298.15 * unit.kelvin, 20261008)
    status = "running"
    step = 0
    try:
        while step < nsteps:
            chunk = min(100, nsteps - step)
            integ.step(chunk)
            step += chunk
            st = ctx.getState(getEnergy=True, getForces=True, getPositions=True)
            pe = st.getPotentialEnergy().value_in_unit(KJ)
            ke = st.getKineticEnergy().value_in_unit(KJ)
            fmax = float(np.max(np.abs(np.asarray(st.getForces().value_in_unit(KJ / NM)))))
            x = np.asarray(st.getPositions().value_in_unit(NM))
            dist = closest(x, box)
            hc = group_energy(ctx, FORCE_GROUP)
            print(json.dumps({"stage": "md", "step": step, "time_ps": step * dt_fs * 1e-3,
                              "pe": pe, "ke": ke, "max_force": fmax, "min_dist_nm": dist,
                              "hardcore": hc, "wall_s": time.time() - t0}), flush=True)
            if not np.isfinite(pe) or abs(pe) > 1e8 or fmax > 2e4 or dist < 0.05:
                status = f"diverged at step {step}"
                break
        else:
            status = "completed"
    except Exception as exc:
        status = f"exception after step {step}: {exc!r}"
        print(json.dumps({"stage": "md", "status": status}), flush=True)
    print(json.dumps({"status": status, "mode": mode, "precision": precision, "minimized_E": e1,
                      "minimized_max_force": f1, "wall_s": time.time() - t0}), flush=True)


if __name__ == "__main__":
    main()
