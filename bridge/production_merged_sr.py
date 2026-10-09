"""Merged short-range production. The system and the run are both set from flags or JSON.

Defaults are the frozen protocol: PhyNEO electrolyte, double precision, 0.5 fs,
298.15 K, NVT, cutoff 1 nm, Ewald tolerance 1e-4, form-5 hardcore, 1-2..1-6 exclusions.
A --config JSON uses the argparse destination names and is overridden by flags.
``--ensemble npt`` adds a Monte Carlo barostat at ``--pressure-bar`` and the run
temperature. A later leg with a different ensemble must resume from ``--state-in``;
a checkpoint belongs to one system, including whether that system has a barostat.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import openmm as mm
from openmm import Platform, unit
from openmm.app import PDBFile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exchange_hardcore import ALPHA
from merged_aligned_sr import KJ, NM, build_merged
from production_reporters import TransportRecorder, monopole_charges, particle_masses
from stability_hypotheses import BASE, CUTOFF, ECL, ETHRESH

DEFAULT_OUT = "/home/changjh/MD_projects/PKUGraduateThesis/gate/merged_prod"


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="JSON object of destination names; flags override it")
    ap.add_argument("--output-prefix", default=DEFAULT_OUT)
    ap.add_argument("--plugin-dir", default=os.environ.get("OPENMM_PLUGIN_DIR", ""))
    ap.add_argument("--pdb", default=BASE + "/init.pdb")
    ap.add_argument("--forcefield", default=BASE + "/phyneo_hmtff_admp.xml")
    ap.add_argument("--parameters", default=ECL, help="Slater, QqTt and dispersion XML")
    ap.add_argument("--box-nm", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="replace the PDB box with an orthorhombic box")
    ap.add_argument("--cutoff", type=float, default=CUTOFF, help="nm")
    ap.add_argument("--ethresh", type=float, default=ETHRESH, help="Ewald error tolerance; sets kappa and the grid")
    ap.add_argument("--kappa", type=float, default=None, help="nm^-1; overrides the value from --ethresh")
    ap.add_argument("--pme-grid", type=int, default=None, help="cubic PME grid; overrides the value from --ethresh")
    ap.add_argument("--thole-width", type=float, default=5.0)
    ap.add_argument("--mutual-induced-target-epsilon", type=float, default=1e-8)
    ap.add_argument("--mutual-induced-max-iterations", type=int, default=500)
    ap.add_argument("--constraints", choices=("HBonds", "AllBonds", "HAngles", "none"), default="HBonds")
    ap.add_argument("--rigid-water", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--polarization", default="mutual")
    ap.add_argument("--exclusion-shell", type=int, default=5,
                    help="graph distance of excluded pairs; 5 is 1-2 through 1-6 and must match ADMP")
    ap.add_argument("--hardcore", choices=("r14", "r12"), default="r14",
                    help="r14 is A/(0.24*Br)^14; r12 restores A*(4.3/Br)^12")
    ap.add_argument("--hardcore-alpha", type=float, default=ALPHA, help="used by r14")
    ap.add_argument("--zero-lj", action=argparse.BooleanOptionalAction, default=True,
                    help="remove the OpenMM Lennard-Jones term")
    ap.add_argument("--pf6-trans", action=argparse.BooleanOptionalAction, default=True,
                    help="set trans PF6 F-P-F equilibria to 180 degrees")
    ap.add_argument("--admp-group", type=int, default=1)
    ap.add_argument("--sr-group", type=int, default=17)
    ap.add_argument("--disp-bond-group", type=int, default=22)
    ap.add_argument("--ensemble", choices=("nvt", "npt"), default="nvt")
    ap.add_argument("--pressure-bar", type=float, default=1.0)
    ap.add_argument("--barostat-interval", type=int, default=25,
                    help="Monte Carlo barostat attempt interval, in steps")
    ap.add_argument("--temperature-K", type=float, default=298.15)
    ap.add_argument("--friction-per-ps", type=float, default=1.0)
    ap.add_argument("--timestep-fs", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--constraint-tolerance", type=float, default=1e-5)
    ap.add_argument("--precision", choices=("double", "single", "mixed"), default="double")
    ap.add_argument("--platform", choices=("CUDA", "CPU", "Reference"), default="CUDA")
    ap.add_argument("--device-index", default="0")
    ap.add_argument("--report-interval", type=int, default=100)
    ap.add_argument("--checkpoint-interval", type=int, default=0, help="0 writes a checkpoint only at the end")
    ap.add_argument("--frames", default=None,
                    help="production archive of positions and box, plus .vel.npy and .observables.csv "
                         "(charge current and pressure tensor) at each report")
    ap.add_argument("--positions", default=None, help="NumPy nm positions, shape (N, 3)")
    ap.add_argument("--state-in", default=None, help="OpenMM XML state")
    ap.add_argument("--checkpoint-in", default=None, help="OpenMM binary checkpoint")
    ap.add_argument("--velocities", choices=("temperature", "zero", "keep"), default=None,
                    help="default: temperature after a fresh start or a relax, otherwise keep")
    ap.add_argument("--relax", action=argparse.BooleanOptionalAction, default=None,
                    help="default: relax a fresh start, and skip it when a state or checkpoint is loaded")
    ap.add_argument("--relax-steps", type=int, default=300)
    ap.add_argument("--relax-max-disp-nm", type=float, default=0.001)
    ap.add_argument("--relax-force-goal", type=float, default=500.0)
    ap.add_argument("--relax-force-cap", type=float, default=5000.0)
    ap.add_argument("--relax-min-disp-nm", type=float, default=1e-6)
    ap.add_argument("--abort-abs-pe", type=float, default=1e8)
    ap.add_argument("--abort-max-force", type=float, default=2e4)
    return ap


def parse_args(argv):
    ap = parser()
    preliminary, _ = ap.parse_known_args(argv)
    if preliminary.config:
        with open(preliminary.config) as handle:
            loaded = json.load(handle)
        unknown = sorted(set(loaded) - {action.dest for action in ap._actions})
        if unknown:
            ap.error("unknown config keys: " + ", ".join(unknown))
        ap.set_defaults(**loaded)
    args = ap.parse_args(argv)
    if args.cutoff <= 0 or not 0 < args.ethresh < 1:
        ap.error("cutoff must be positive and ethresh must lie in (0, 1)")
    if args.pressure_bar <= 0 or args.barostat_interval < 1:
        ap.error("pressure must be positive and the barostat interval must be at least 1")
    if args.steps < 0 or args.report_interval < 1 or args.exclusion_shell < 1:
        ap.error("steps >= 0, report interval >= 1, exclusion shell >= 1")
    if args.checkpoint_interval and args.checkpoint_interval % args.report_interval != 0:
        ap.error("checkpoint interval must be a multiple of report interval")
    if args.ensemble == "npt" and args.report_interval % args.barostat_interval != 0:
        ap.error("report interval must be a multiple of the barostat interval")
    chosen = [name for name in (args.positions, args.state_in, args.checkpoint_in) if name]
    if len(chosen) > 1:
        ap.error("use only one of --positions, --state-in and --checkpoint-in")
    return args


def snapshot(ctx):
    state = ctx.getState(getEnergy=True, getForces=True, getPositions=True)
    forces = np.asarray(state.getForces().value_in_unit(KJ / NM))
    return (state.getPotentialEnergy().value_in_unit(KJ),
            float(np.max(np.abs(forces))),
            np.asarray(state.getPositions().value_in_unit(NM)))


def relax(ctx, args):
    disp = args.relax_max_disp_nm
    accepted = 0
    for it in range(args.relax_steps):
        energy, fmax, positions = snapshot(ctx)
        if not np.isfinite(energy):
            print(json.dumps({"stage": "relax_nan", "it": it}), flush=True)
            return energy, fmax, accepted
        if fmax < args.relax_force_goal:
            print(json.dumps({"stage": "relax_stop", "it": it, "E": energy, "max_force": fmax}), flush=True)
            return energy, fmax, accepted
        forces = np.asarray(ctx.getState(getForces=True).getForces().value_in_unit(KJ / NM))
        trial = positions + (disp / max(fmax, 1e-12)) * forces
        ctx.setPositions(trial * NM)
        ctx.applyConstraints(args.constraint_tolerance)
        trial_energy, trial_fmax, _ = snapshot(ctx)
        ok = (np.isfinite(trial_energy) and trial_energy < energy
              and trial_fmax < max(args.relax_force_cap, 3.0 * fmax))
        if ok:
            accepted += 1
            disp = min(args.relax_max_disp_nm, disp * (1.2 if trial_fmax < fmax else 0.8))
        else:
            ctx.setPositions(positions * NM)
            disp *= 0.5
            if disp < args.relax_min_disp_nm:
                break
        if it % 5 == 0:
            print(json.dumps({"stage": "relax", "it": it, "E": trial_energy if ok else energy,
                              "max_force": trial_fmax if ok else fmax, "accepted": accepted}), flush=True)
    energy, fmax, _ = snapshot(ctx)
    return energy, fmax, accepted


def place_coordinates(ctx, pdb, args):
    restarted = bool(args.checkpoint_in or args.state_in)
    if args.checkpoint_in:
        with open(args.checkpoint_in, "rb") as handle:
            ctx.loadCheckpoint(handle.read())
    elif args.state_in:
        with open(args.state_in) as handle:
            ctx.setState(mm.XmlSerializer.deserialize(handle.read()))
    else:
        if args.positions:
            coords = np.load(args.positions)
        else:
            coords = np.asarray(pdb.positions.value_in_unit(NM))
        ctx.setPositions(coords * NM)
        ctx.applyConstraints(args.constraint_tolerance)
    return restarted


def assign_velocities(ctx, args, restarted, did_relax):
    mode = args.velocities
    if mode is None:
        mode = "temperature" if did_relax or not restarted else "keep"
    if mode == "temperature":
        ctx.setVelocitiesToTemperature(args.temperature_K * unit.kelvin, args.seed)
    elif mode == "zero":
        ctx.setVelocities(np.zeros((ctx.getSystem().getNumParticles(), 3)) * (NM / unit.picosecond))
    return mode


def group_energy(ctx, group):
    return ctx.getState(getEnergy=True, groups={group}).getPotentialEnergy().value_in_unit(KJ)


def particle_mass_da(topology):
    total = 0.0
    for atom in topology.atoms():
        total += 0.0 if atom.element is None else atom.element.mass.value_in_unit(unit.dalton)
    return total


def box_lengths_and_density(ctx, mass_da):
    vectors = ctx.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(NM)
    volume = abs(float(np.dot(vectors[0], np.cross(vectors[1], vectors[2]))))
    lengths = [float(np.linalg.norm(edge)) for edge in vectors]
    density = None if volume == 0 else mass_da / volume * 0.001660539067
    return lengths, density


def write_checkpoint(ctx, path):
    with open(path, "wb") as handle:
        handle.write(ctx.createCheckpoint())


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.plugin_dir:
        Platform.loadPluginsFromDirectory(args.plugin_dir)
    pdb = PDBFile(args.pdb)
    if args.box_nm:
        x, y, z = args.box_nm
        pdb.topology.setPeriodicBoxVectors(mm.Vec3(x, 0, 0) * NM, mm.Vec3(0, y, 0) * NM, mm.Vec3(0, 0, z) * NM)
    system, kappa, grid, n_trans, same_c, box = build_merged(
        pdb, ff_xml=args.forcefield, ecl_xml=args.parameters, cutoff=args.cutoff, ethresh=args.ethresh,
        kappa=args.kappa, grid=args.pme_grid, thole_width=args.thole_width,
        mutual_induced_target_epsilon=args.mutual_induced_target_epsilon,
        mutual_induced_max_iterations=args.mutual_induced_max_iterations, constraints=args.constraints,
        rigid_water=args.rigid_water, polarization=args.polarization, exclusion_shell=args.exclusion_shell,
        hardcore_alpha=args.hardcore_alpha, hardcore=args.hardcore, remove_lj=args.zero_lj,
        repair_pf6_trans=args.pf6_trans,
        admp_group=args.admp_group, sr_group=args.sr_group, disp_bond_group=args.disp_bond_group)
    if args.ensemble == "npt":
        system.addForce(mm.MonteCarloBarostat(
            args.pressure_bar * unit.bar, args.temperature_K * unit.kelvin, args.barostat_interval))
    integrator = mm.LangevinMiddleIntegrator(
        args.temperature_K * unit.kelvin, args.friction_per_ps / unit.picosecond, args.timestep_fs * unit.femtosecond)
    integrator.setRandomNumberSeed(args.seed)
    integrator.setConstraintTolerance(args.constraint_tolerance)
    properties = {}
    if args.platform == "CUDA":
        properties = {"Precision": args.precision, "DeviceIndex": str(args.device_index)}
    context = mm.Context(system, integrator, Platform.getPlatformByName(args.platform), properties)
    restarted = place_coordinates(context, pdb, args)
    energy0, force0, _ = snapshot(context)
    out_dir = os.path.dirname(os.path.abspath(args.output_prefix))
    os.makedirs(out_dir, exist_ok=True)
    header = {
        "engine": "merged-short-range",
        "pdb": os.path.abspath(args.pdb), "forcefield": os.path.abspath(args.forcefield),
        "parameters": os.path.abspath(args.parameters), "box_nm": box.tolist(),
        "cutoff_nm": args.cutoff, "ethresh": args.ethresh, "kappa_nm": kappa, "grid": grid,
        "thole_width": args.thole_width, "polarization": args.polarization,
        "mutual_induced_target_epsilon": args.mutual_induced_target_epsilon,
        "mutual_induced_max_iterations": args.mutual_induced_max_iterations,
        "constraints": args.constraints, "constraint_tolerance": args.constraint_tolerance,
        "exclusion_shell": args.exclusion_shell, "hardcore": args.hardcore,
        "hardcore_alpha": args.hardcore_alpha,
        "zero_lj": args.zero_lj, "pf6_trans_180": n_trans, "dispersion_C_equals_slater_C": same_c,
        "ensemble": args.ensemble,
        "pressure_bar": args.pressure_bar if args.ensemble == "npt" else None,
        "barostat_interval": args.barostat_interval if args.ensemble == "npt" else None,
        "temperature_K": args.temperature_K, "friction_per_ps": args.friction_per_ps,
        "timestep_fs": args.timestep_fs, "steps": args.steps, "seed": args.seed,
        "precision": args.precision, "platform": args.platform, "device_index": str(args.device_index),
        "initial_E": energy0, "initial_max_force": force0,
        "forces": [force.getName() for force in system.getForces()],
    }
    mass_da = particle_mass_da(pdb.topology)
    print(json.dumps({"stage": "setup", **header}), flush=True)
    with open(args.output_prefix + ".manifest.json", "w") as handle:
        json.dump(header, handle, indent=2)
        handle.write("\n")
    started = time.time()
    do_relax = (not restarted) if args.relax is None else args.relax
    if do_relax:
        energy1, force1, accepted = relax(context, args)
        print(json.dumps({"stage": "minimized", "E": energy1, "max_force": force1, "accepted": accepted,
                          "wall_s": time.time() - started}), flush=True)
        if not np.isfinite(energy1):
            print(json.dumps({"status": "diverged during relax"}), flush=True)
            return
        np.save(args.output_prefix + "_min.npy", snapshot(context)[2])
    else:
        energy1, force1 = energy0, force0
    velocity_mode = assign_velocities(context, args, restarted, do_relax)
    print(json.dumps({"stage": "velocities", "mode": velocity_mode}), flush=True)
    status = "completed"
    step = 0
    frames = []
    recorder = None
    if args.frames:
        recorder = TransportRecorder(
            args.output_prefix, monopole_charges(pdb.topology, args.forcefield), particle_masses(system))
    try:
        while step < args.steps:
            # The barostat changes the box at the end of its interval, and ADMP
            # faults if another kernel then runs inside the same step() call.
            # A multi-step call also raises CUDA_ERROR_ILLEGAL_ADDRESS on the
            # experimental boxes after about 100 steps, so advance one step.
            integrator.step(1)
            step += 1
            if step % args.report_interval != 0 and step != args.steps:
                continue
            state = context.getState(
                getEnergy=True, getForces=True,
                getPositions=args.frames is not None, getVelocities=args.frames is not None)
            potential = state.getPotentialEnergy().value_in_unit(KJ)
            kinetic = state.getKineticEnergy().value_in_unit(KJ)
            fmax = float(np.max(np.abs(np.asarray(state.getForces().value_in_unit(KJ / NM)))))
            lengths, density = box_lengths_and_density(context, mass_da)
            sr = group_energy(context, args.sr_group)
            if args.frames:
                frames.append((
                    step,
                    np.array(state.getPositions(asNumpy=True).value_in_unit(NM), dtype=np.float64, copy=True),
                    np.array(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(NM), dtype=np.float64, copy=True),
                    float(potential), float(sr), float(density),
                ))
                recorder.write(step, step * args.timestep_fs * 1e-3, state)
            print(json.dumps({
                "stage": "md", "step": step, "time_ps": step * args.timestep_fs * 1e-3,
                "pe": potential, "ke": kinetic, "max_force": fmax,
                "density_g_ml": density, "box_lengths_nm": lengths,
                "sr": sr,
                "disp_bond": group_energy(context, args.disp_bond_group),
                "wall_s": time.time() - started,
            }), flush=True)
            if args.checkpoint_interval and step % args.checkpoint_interval == 0:
                write_checkpoint(context, args.output_prefix + ".chk")
            if not np.isfinite(potential) or abs(potential) > args.abort_abs_pe or fmax > args.abort_max_force:
                status = f"diverged at step {step}"
                break
    except Exception as exc:
        status = f"exception after step {step}: {exc!r}"
        print(json.dumps({"stage": "md", "status": status}), flush=True)
    if recorder is not None:
        recorder.close()
    if args.frames and frames:
        np.savez(
            args.frames,
            step=np.asarray([item[0] for item in frames]),
            positions=np.stack([item[1] for item in frames]),
            box=np.stack([item[2] for item in frames]),
            pe=np.asarray([item[3] for item in frames]),
            sr=np.asarray([item[4] for item in frames]),
            density=np.asarray([item[5] for item in frames]),
        )
    try:
        write_checkpoint(context, args.output_prefix + ".final.chk")
        with open(args.output_prefix + ".final.state.xml", "w") as handle:
            handle.write(mm.XmlSerializer.serialize(context.getState(
                getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True)))
    except Exception as exc:
        if status == "completed":
            status = f"exception writing restart: {exc!r}"
    print(json.dumps({"status": status, "minimized_E": energy1, "minimized_max_force": force1,
                      "wall_s": time.time() - started}), flush=True)


if __name__ == "__main__":
    main()
