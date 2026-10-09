"""Charge current and pressure tensor for a production frame.

Velocities are a stream of NumPy arrays, one ``(n_atoms, 3)`` record per frame
in nm/ps. Scalars go to CSV: current in e nm/ps, virials in kJ/mol, pressure
in bar. Charges are ADMP permanent monopoles. The configurational virial is
``-sum(r F)``, the same definition as the earlier CUDA smoke reporter.
"""
import csv
import json
import xml.etree.ElementTree as ET

import numpy as np
from openmm import unit


def monopole_charges(topology, ff_xml):
    """Permanent charges from ``ADMPPmeForce`` multipole ``c0``, in elementary charge."""
    root = ET.parse(ff_xml).getroot()
    types = {}
    for residue in root.find("Residues"):
        for atom in residue.findall("Atom"):
            types[(residue.get("name"), atom.get("name"))] = atom.get("type")
    monopoles = {atom.get("type"): float(atom.get("c0")) for atom in root.find("ADMPPmeForce").findall("Multipole")}
    charges = np.zeros(topology.getNumAtoms(), dtype=np.float64)
    for atom in topology.atoms():
        charges[atom.index] = monopoles[types[(atom.residue.name, atom.name)]]
    return charges


def particle_masses(system):
    dalton = unit.dalton
    return np.asarray([system.getParticleMass(i).value_in_unit(dalton) for i in range(system.getNumParticles())],
                      dtype=np.float64)


class TransportRecorder:
    """Append one production frame of velocities, current, and pressure."""

    def __init__(self, prefix, charges, masses):
        self._charges = np.asarray(charges, dtype=np.float64)
        self._masses = np.asarray(masses, dtype=np.float64)
        self._velocity = open(prefix + ".vel.npy", "wb")
        self._scalar = open(prefix + ".observables.csv", "w", newline="")
        self._writer = csv.writer(self._scalar)
        names = ["frame", "step", "time_ps", "volume_nm3", "current_x_e_nm_ps",
                 "current_y_e_nm_ps", "current_z_e_nm_ps", "kinetic_virial_kj_mol",
                 "config_virial_kj_mol"] + [f"pressure_{a}{b}_bar" for a in "xyz" for b in "xyz"]
        self._writer.writerow(names)
        self._frame = 0
        with open(prefix + ".observables.csv.meta.json", "w") as handle:
            json.dump({
                "velocity_unit": "nm/ps", "current_unit": "e*nm/ps", "virial_unit": "kJ/mol",
                "pressure_unit": "bar", "charge_source": "ADMPPmeForce Multipole c0",
                "config_virial": "-sum(r F)", "n_atoms": int(self._charges.size),
                "total_charge_e": float(self._charges.sum()),
                "total_mass_dalton": float(self._masses.sum()),
            }, handle, indent=2)
            handle.write("\n")

    def write(self, step, time_ps, state):
        vel = np.asarray(state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond), dtype=np.float64)
        pos = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=np.float64)
        frc = np.asarray(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer), dtype=np.float64)
        box = np.asarray(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer), dtype=np.float64)
        volume = float(abs(np.linalg.det(box)))
        # dalton*(nm/ps)^2 = kJ/mol. The one-half is the kinetic virial convention used previously.
        kinetic = 0.5 * np.einsum("i,ia,ib->ab", self._masses, vel, vel)
        configurational = -np.einsum("ia,ib->ab", pos, frc)
        pressure = (kinetic + configurational) / volume * 16.60539067
        current = self._charges @ vel
        self._writer.writerow([
            self._frame, int(step), float(time_ps), volume, *current,
            float(np.trace(kinetic)), float(np.trace(configurational)), *pressure.reshape(-1),
        ])
        self._scalar.flush()
        np.save(self._velocity, vel, allow_pickle=False)
        self._velocity.flush()
        self._frame += 1

    def close(self):
        self._scalar.close()
        self._velocity.close()
