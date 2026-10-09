"""DMFF QqTtDampingForce as an OpenMM CustomNonbondedForce.

DMFF TT_damping_qq_kernel (Å, kJ/mol): E = -DIELECTRIC * exp(-b r) (1 + b r) q_i q_j / r with
b = sqrt(B_i B_j); in nm units the prefactor becomes ONE_4PI_EPS0 = 138.935455846.
Pairs with mScale = 0 (graph distance 1..6 in phyneo_ecl.xml) are excluded.
"""
import xml.etree.ElementTree as ET

from openmm import CustomNonbondedForce

from dmff_dispersion import excluded_pairs

ONE_4PI_EPS0 = 138.935455846


def add_qqtt_damping(system, topology, xml_path, cutoff, max_bonds=5):
    root = ET.parse(xml_path).getroot()
    res_map = {(r.attrib["name"], a.attrib["name"]): a.attrib["type"]
               for r in root.find("Residues") for a in r if a.tag == "Atom"}
    params = {a.attrib["type"]: (float(a.attrib["B"]), float(a.attrib["Q"]))
              for a in root.find("QqTtDampingForce") if a.tag == "Atom"}
    f = CustomNonbondedForce(
        f"-{ONE_4PI_EPS0}*exp(-br)*(1+br)*q1*q2/r; br=sqrt(b1*b2)*r")
    f.setName("QqTtDampingForce")
    f.addPerParticleParameter("b")
    f.addPerParticleParameter("q")
    f.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    f.setCutoffDistance(cutoff)
    f.setUseSwitchingFunction(False)
    f.setUseLongRangeCorrection(False)
    for atom in topology.atoms():
        f.addParticle(list(params[res_map[(atom.residue.name, atom.name)]]))
    for i, j in excluded_pairs(topology, max_bonds):
        f.addExclusion(i, j)
    system.addForce(f)
    return f
