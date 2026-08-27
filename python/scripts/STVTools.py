"""
Python translation of STV_Tools.cxx
Original C++ class by Afroditi Papadopoulou

Computes Single Transverse Variables (STV) and related kinematic quantities
for a muon + proton final state, e.g. from CCQE-like neutrino interactions.

All 3-vectors are numpy arrays (or array-likes) of length 3: (x, y, z), with
z conventionally along the neutrino beam direction. Energies and masses are
in GeV. 4-vectors follow the ROOT TLorentzVector convention (px, py, pz, E).

Dependencies: numpy
"""

import numpy as np


class STVTools:

    MUON_MASS = 0.106        # GeV
    PROTON_MASS = 0.938272   # GeV
    NEUTRON_MASS = 0.939565  # GeV

    def __init__(self, muon_vector, proton_vector, muon_energy, proton_energy):
        muon_vector = np.asarray(muon_vector, dtype=float)
        proton_vector = np.asarray(proton_vector, dtype=float)

        Mm = self.MUON_MASS
        Mp = self.PROTON_MASS
        Mn = self.NEUTRON_MASS
        DeltaM2 = Mn**2 - Mp**2
        BE = 0.04  # GeV, binding energy for Ar

        # --- transverse / longitudinal split ---
        muon_trans = np.array([muon_vector[0], muon_vector[1], 0.])
        muon_trans_mag = np.linalg.norm(muon_trans)

        muon_long = np.array([0., 0., muon_vector[2]])

        proton_trans = np.array([proton_vector[0], proton_vector[1], 0.])
        proton_trans_mag = np.linalg.norm(proton_trans)

        proton_long = np.array([0., 0., proton_vector[2]])

        proton_KE = proton_energy - Mp

        # --- STV calculation ---
        pt_vector = muon_trans + proton_trans
        self.fPt = np.linalg.norm(pt_vector)

        cos_dat = np.dot(-muon_trans, pt_vector) / (muon_trans_mag * self.fPt)
        self.fDeltaAlphaT = np.degrees(np.arccos(np.clip(cos_dat, -1., 1.)))
        if self.fDeltaAlphaT > 180.: self.fDeltaAlphaT -= 180.
        if self.fDeltaAlphaT < 0.: self.fDeltaAlphaT += 180.

        cos_dpt = np.dot(-muon_trans, proton_trans) / (muon_trans_mag * proton_trans_mag)
        self.fDeltaPhiT = np.degrees(np.arccos(np.clip(cos_dpt, -1., 1.)))
        if self.fDeltaPhiT > 180.: self.fDeltaPhiT -= 180.
        if self.fDeltaPhiT < 0.: self.fDeltaPhiT += 180.

        # --- Calorimetric energy reconstruction ---
        self.fECal = muon_energy + proton_KE + BE

        # --- QE energy reconstruction ---
        muon_mag = np.linalg.norm(muon_vector)
        cos_theta_mu = muon_vector[2] / muon_mag  # TVector3::CosTheta() wrt z-axis
        EQE_num = 2 * (Mn - BE) * muon_energy - (BE**2 - 2 * Mn * BE + Mm**2 + DeltaM2)
        EQE_den = 2 * (Mn - BE - muon_energy + muon_mag * cos_theta_mu)
        self.fEQE = EQE_num / EQE_den

        # --- Reconstructed Q2 ---
        muon_4v = np.array([muon_vector[0], muon_vector[1], muon_vector[2], muon_energy])
        nu_4v = np.array([0., 0., self.fECal, self.fECal])
        q_4v = nu_4v - muon_4v
        self.fQ2 = -self._mag2(q_4v)

        # --- Ptx, Pty (PRD 101, 092001) ---
        unit_z = np.array([0., 0., 1.])
        self.fPtx = np.dot(np.cross(unit_z, muon_trans), pt_vector) / muon_trans_mag
        self.fPty = -np.dot(muon_trans, pt_vector) / muon_trans_mag

        # --- JLab light-cone variables ---
        proton_4v = np.array([proton_vector[0], proton_vector[1], proton_vector[2], proton_energy])
        miss_4v = muon_4v + proton_4v - nu_4v
        self.fEMiss = abs(miss_4v[3])
        self.fPMiss = np.linalg.norm(miss_4v[:3])

        # Suggestion from Jackson to avoid ECal assumption
        self.fPMissMinus = (muon_energy - muon_vector[2]) + (proton_energy - proton_vector[2])

        kMiss_num = self.fPt**2 + Mp**2
        kMiss_den = self.fPMissMinus * (2 * Mp - self.fPMissMinus)
        kMiss2 = Mp**2 * kMiss_num / kMiss_den - Mp**2
        self.fkMiss = np.sqrt(kMiss2)

        self.fA = self.fPMissMinus / Mp

        # --- MINERvA longitudinal & total variables ---
        # https://journals.aps.org/prc/pdf/10.1103/PhysRevC.95.065501
        MA = 22 * Mn + 18 * Mp - 0.34381  # GeV

        # https://doi.org/10.1140/epjc/s10052-019-6750-3 (table 7)
        MAPrime = MA - Mn + 0.0309  # GeV -- unused downstream, kept for fidelity

        # https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.121.022504, Eq. 8
        R = MA + np.linalg.norm(muon_long + proton_long) - muon_energy - proton_energy
        # -- unused downstream (original Eq. 7 expression for fPL was abandoned
        #    on Mar 6 2023 in favor of the "beyond transverse variables" fPL below);
        #    kept here only for fidelity with the original source.

        # --- Beyond-the-transverse-variables (Andy F., microboone-docdb 38090) ---
        self.fECalMB = muon_energy + proton_KE + 0.0309
        nu_4v_MB = np.array([0., 0., self.fECalMB, self.fECalMB])
        q_4v_MB = nu_4v_MB - muon_4v

        self.fPL = muon_vector[2] + proton_vector[2] - self.fECalMB
        pn_vector = np.array([pt_vector[0], pt_vector[1], self.fPL])

        q_vector = q_4v_MB[:3]
        # NOTE: reproduces the original code exactly, including its use of
        # qVector.X() twice when building qTVector -- this looks like a bug
        # in the original C++ (probably meant qVector.Y() for the second
        # component) but is kept here for fidelity with STV_Tools.cxx.
        qT_vector = np.array([q_vector[0], q_vector[0], 0.])
        q_vector_unit = self._safe_unit(q_vector)
        qT_vector_unit = self._safe_unit(qT_vector)

        self.fPn = np.sqrt(self.fPt**2 + self.fPL**2)

        q_mag = np.linalg.norm(q_vector)
        cos_a3dq = np.dot(q_vector, pn_vector) / (q_mag * self.fPn)
        self.fDeltaAlpha3Dq = np.degrees(np.arccos(np.clip(cos_a3dq, -1., 1.)))
        if self.fDeltaAlpha3Dq > 180.: self.fDeltaAlpha3Dq -= 180.
        if self.fDeltaAlpha3Dq < 0.: self.fDeltaAlpha3Dq += 180.

        cos_a3dmu = np.dot(-muon_vector, pn_vector) / (muon_mag * self.fPn)
        self.fDeltaAlpha3DMu = np.degrees(np.arccos(np.clip(cos_a3dmu, -1., 1.)))
        if self.fDeltaAlpha3DMu > 180.: self.fDeltaAlpha3DMu -= 180.
        if self.fDeltaAlpha3DMu < 0.: self.fDeltaAlpha3DMu += 180.

        proton_mag = np.linalg.norm(proton_vector)
        cos_p3d = np.dot(q_vector, proton_vector) / (q_mag * proton_mag)
        self.fDeltaPhi3D = np.degrees(np.arccos(np.clip(cos_p3d, -1., 1.)))
        if self.fDeltaPhi3D > 180.: self.fDeltaPhi3D -= 180.
        if self.fDeltaPhi3D < 0.: self.fDeltaPhi3D += 180.

        # Magnitudes
        self.fPnPerp = self.fPn * np.sin(np.radians(self.fDeltaAlpha3Dq))
        self.fPnPar = self.fPn * np.cos(np.radians(self.fDeltaAlpha3Dq))

        self.fPnPerpx = np.dot(np.cross(qT_vector_unit, unit_z), pn_vector)
        self.fPnPerpy = np.dot(np.cross(q_vector_unit, np.cross(qT_vector_unit, unit_z)), pn_vector)

    @classmethod
    def from_momenta(cls, muon_vector, proton_vector,
                    muon_mass=MUON_MASS, proton_mass=PROTON_MASS):
        """
        Convenience constructor for when you only have 3-momentum vectors.
        Computes energies assuming on-shell particles: E = sqrt(|p|^2 + m^2).
        """
        muon_vector = np.asarray(muon_vector, dtype=float)
        proton_vector = np.asarray(proton_vector, dtype=float)

        muon_energy = np.sqrt(np.dot(muon_vector, muon_vector) + muon_mass**2)
        proton_energy = np.sqrt(np.dot(proton_vector, proton_vector) + proton_mass**2)

        return cls(muon_vector, proton_vector, muon_energy, proton_energy)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _mag2(four_vector):
        """TLorentzVector::Mag2() = E^2 - |p|^2 for a (px,py,pz,E) 4-vector."""
        p = four_vector[:3]
        E = four_vector[3]
        return E**2 - np.dot(p, p)

    @staticmethod
    def _safe_unit(v):
        n = np.linalg.norm(v)
        return v / n if n != 0 else v

    # ------------------------------------------------------------------
    # getters, mirroring the original Return*() methods
    # ------------------------------------------------------------------
    def ReturnkMiss(self): return self.fkMiss
    def ReturnEMiss(self): return self.fEMiss
    def ReturnPMissMinus(self): return self.fPMissMinus
    def ReturnPMiss(self): return self.fPMiss
    def ReturnPt(self): return self.fPt
    def ReturnPtx(self): return self.fPtx
    def ReturnPty(self): return self.fPty
    def ReturnPnPerp(self): return self.fPnPerp
    def ReturnPnPerpx(self): return self.fPnPerpx
    def ReturnPnPerpy(self): return self.fPnPerpy
    def ReturnPnPar(self): return self.fPnPar
    def ReturnPL(self): return self.fPL
    def ReturnPn(self): return self.fPn
    def ReturnDeltaAlphaT(self): return self.fDeltaAlphaT
    def ReturnDeltaAlpha3Dq(self): return self.fDeltaAlpha3Dq
    def ReturnDeltaAlpha3DMu(self): return self.fDeltaAlpha3DMu
    def ReturnDeltaPhiT(self): return self.fDeltaPhiT
    def ReturnDeltaPhi3D(self): return self.fDeltaPhi3D
    def ReturnECal(self): return self.fECal
    def ReturnECalMB(self): return self.fECalMB
    def ReturnEQE(self): return self.fEQE
    def ReturnQ2(self): return self.fQ2
    def ReturnA(self): return self.fA