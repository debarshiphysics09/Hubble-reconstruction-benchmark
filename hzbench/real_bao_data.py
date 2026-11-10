"""
Radial-BAO H(z) data sets.

Three sets are provided.  All values were checked against the primary
papers (or, for DESI, the published DR2 table) during the revision; the
v1 header note that the table was "reproduced from memory" no longer applies.

1. "legacy_v1"  -- the 10-point table exactly as released with v1, kept ONLY
   so that the v1 numbers can be reproduced.  Its labels contained errors
   found by the referee and confirmed here:
     * z=0.43, 86.45+/-3.68 is Gaztanaga, Cabre & Hui (2009), not WiggleZ;
       WiggleZ (Blake+2012) reports three H(z) values (z=0.44, 0.60, 0.73).
     * z=0.57, 92.4+/-4.5 is Samushia+2013 (BOSS DR9 CMASS), not
       Anderson+2014 (DR11: 96.8+/-3.4).
     * z=2.33, 224+/-8 is the Busca+2013 value (quoted at z=2.3), not
       Delubac+2015 (222+/-7 at z=2.34).
     * Gaztanaga+2009 (two slices), Oka+2014 and Chuang & Wang (2013) all
       analyse SDSS DR7 LRGs; Delubac+2015 and Font-Ribera+2014 both use the
       BOSS DR11 Lya-forest spectra.  The v1 "independence audit" therefore
       did not remove all shared-sample dependence.

2. "legacy"  -- corrected legacy set (N=7), one measurement per independent
   galaxy/forest sample, most recent analysis of each sample:
     Gaztanaga+2009   z=0.24  79.69 +/- 2.65   SDSS DR6 LRG slice 0.15-0.30
     Gaztanaga+2009   z=0.43  86.45 +/- 3.68   SDSS DR6 LRG slice 0.40-0.47
     Blake+2012       z=0.44  82.6  +/- 7.8    WiggleZ
     Anderson+2014    z=0.57  96.8  +/- 3.4    BOSS DR11 CMASS
     Blake+2012       z=0.60  87.9  +/- 6.1    WiggleZ
     Blake+2012       z=0.73  97.3  +/- 7.0    WiggleZ
     Delubac+2015     z=2.34  222.0 +/- 7.0    BOSS DR11 Lya forest (auto)
   Oka+2014 and Chuang & Wang 2013 are dropped (same DR7 LRGs as
   Gaztanaga+2009 slices); Font-Ribera+2014 is dropped (same DR11 spectra as
   Delubac+2015).  The three WiggleZ points come from overlapping redshift
   slices and have a published correlation that we do not use (diagonal
   only).  Each legacy H(z) is quoted by its authors for their OWN fiducial
   sound horizon; the legacy set therefore carries heterogeneous, implicit
   r_d assumptions and is used only as a secondary, historical comparison.

3. "desi"  -- DESI DR2 (DESI Collaboration, Abdul Karim et al. 2025,
   PRD 112, 083515; arXiv:2503.14738, Table IV): the six D_H/r_d
   measurements (the BGS bin at z=0.295 constrains only D_V and is not
   used).  Different tracer bins are treated as independent by DESI; the
   D_M--D_H correlation within a bin is irrelevant here because only D_H is
   used.  Conversion: H = c / [(D_H/r_d) r_d], sigma_H = H * sigma(D_H/r_d)/(D_H/r_d),
   with r_d = 147.09 +/- 0.26 Mpc (Planck 2018 TT,TE,EE+lowE+lensing,
   flat LCDM).  The r_d uncertainty is fully correlated across points; it is
   returned separately (cov_rd) and propagated where stated.  Results using
   these points are CONDITIONAL on this early-Universe calibration.
"""
import numpy as np

C_KMS = 299792.458
RD_PLANCK = 147.09       # Mpc, Planck 2018 TT,TE,EE+lowE+lensing
RD_PLANCK_ERR = 0.26

_LEGACY_V1 = [  # as released in v1 (labels corrected in comments only)
    (0.24, 79.69, 2.65, "Gaztanaga+2009 (SDSS LRG)"),
    (0.30, 81.7, 6.22, "Oka+2014 (SDSS DR7 LRG)"),
    (0.35, 82.7, 8.4, "Chuang&Wang2013 (SDSS DR7 LRG)"),
    (0.43, 86.45, 3.68, "Gaztanaga+2009 [labelled WiggleZ in v1]"),
    (0.44, 82.6, 7.8, "Blake+2012 (WiggleZ)"),
    (0.57, 92.4, 4.5, "Samushia+2013 (BOSS DR9) [labelled Anderson+2014 in v1]"),
    (0.60, 87.9, 6.1, "Blake+2012 (WiggleZ)"),
    (0.73, 97.3, 7.0, "Blake+2012 (WiggleZ)"),
    (2.33, 224.0, 8.0, "Busca+2013 value [labelled Delubac+2015 in v1]"),
    (2.36, 226.0, 8.0, "Font-Ribera+2014 (BOSS DR11 Lya-QSO)"),
]

_LEGACY = [
    (0.24, 79.69, 2.65, "Gaztanaga+2009 (SDSS LRG, 0.15<z<0.30)"),
    (0.43, 86.45, 3.68, "Gaztanaga+2009 (SDSS LRG, 0.40<z<0.47)"),
    (0.44, 82.6, 7.8, "Blake+2012 (WiggleZ)"),
    (0.57, 96.8, 3.4, "Anderson+2014 (BOSS DR11 CMASS)"),
    (0.60, 87.9, 6.1, "Blake+2012 (WiggleZ)"),
    (0.73, 97.3, 7.0, "Blake+2012 (WiggleZ)"),
    (2.34, 222.0, 7.0, "Delubac+2015 (BOSS DR11 Lya forest)"),
]

# DESI DR2, Table IV of arXiv:2503.14738: z_eff, D_H/r_d, sigma, tracer
_DESI_DR2_DH = [
    (0.510, 21.863, 0.425, "DESI DR2 LRG1"),
    (0.706, 19.455, 0.330, "DESI DR2 LRG2"),
    (0.934, 17.641, 0.193, "DESI DR2 LRG3+ELG1"),
    (1.321, 14.176, 0.221, "DESI DR2 ELG2"),
    (1.484, 12.817, 0.516, "DESI DR2 QSO"),
    (2.330, 8.632, 0.101, "DESI DR2 Lya"),
]


def _arr(rows):
    rows = sorted(rows, key=lambda r: r[0])
    return (np.array([r[0] for r in rows], float), np.array([r[1] for r in rows], float),
            np.array([r[2] for r in rows], float), [r[3] for r in rows])


def load_bao_data(name="desi", rd=RD_PLANCK):
    """Return (z, H, sigma_H, sources).  For "desi", rd sets the conversion."""
    if name == "legacy_v1":
        return _arr(_LEGACY_V1)
    if name == "legacy":
        return _arr(_LEGACY)
    if name == "desi":
        z, x, sx, src = _arr(_DESI_DR2_DH)
        H = C_KMS / (x * rd)
        return z, H, H * sx / x, src
    raise ValueError(name)


def desi_rd_covariance(rd=RD_PLANCK, rd_err=RD_PLANCK_ERR):
    """Fully correlated covariance of the DESI-derived H from the r_d error."""
    z, H, s, _ = load_bao_data("desi", rd)
    v = H * rd_err / rd
    return np.outer(v, v)


def desi_dh_over_rd():
    z, x, sx, src = _arr(_DESI_DR2_DH)
    return z, x, sx, src


if __name__ == "__main__":
    for k in ("legacy_v1", "legacy", "desi"):
        z, H, s, src = load_bao_data(k)
        print(f"--- {k} (N={len(z)})")
        for a, b, c, d in zip(z, H, s, src):
            print(f"  z={a:.3f}  H={b:7.2f} +/- {c:5.2f}   {d}")
