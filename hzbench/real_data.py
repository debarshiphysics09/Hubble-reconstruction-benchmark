"""
Cosmic-chronometer (CC) H(z) compilations.

CC31  The 31-point differential-age compilation used in v1 and kept as the
      baseline (Jimenez+2003; Simon, Verde & Jimenez 2005; Stern+2010;
      Moresco+2012; Zhang+2014; Moresco 2015; Moresco+2016;
      Ratsimbazafy+2017), 0.07 <= z <= 1.965.  sigma_i are the quoted total
      (statistical + systematic) diagonal uncertainties.
CC32  CC31 + Borghi, Moresco & Cimatti (2022, ApJL 928, L4):
      H(0.75) = 98.8 +/- 33.6, i.e. the 32-point set reviewed by
      Moresco et al. (2022, Living Rev. Relativ. 25, 6).
CC34  CC32 + Jiao et al. (2023, ApJS 265, 48): H(0.80) = 113.1 +/- 15.1 (stat)
      +29.1/-11.3 (syst), symmetrised here to +/- 25.2 (stat and mean syst
      in quadrature); + Tomasetti et al. (2023, A&A 679, A96):
      H(1.26) = 135 +/- 65.  NOTE: Jiao+2023 and Borghi+2022 both analyse
      LEGA-C passive galaxies, so CC34 is NOT a set of independent points;
      it is used only as a sensitivity check.

Values verified against the primary papers / Moresco+2022 compilation in
the revision (November 2026).  Columns: z, H [km/s/Mpc], sigma_H [km/s/Mpc], source.
"""
import numpy as np

_CC31 = [
    (0.070, 69.0, 19.6, "Zhang+2014"),
    (0.090, 69.0, 12.0, "Jimenez+2003"),
    (0.120, 68.6, 26.2, "Zhang+2014"),
    (0.170, 83.0, 8.0, "Simon+2005"),
    (0.1791, 75.0, 4.0, "Moresco+2012"),
    (0.1993, 75.0, 5.0, "Moresco+2012"),
    (0.200, 72.9, 29.6, "Zhang+2014"),
    (0.270, 77.0, 14.0, "Simon+2005"),
    (0.280, 88.8, 36.6, "Zhang+2014"),
    (0.3519, 83.0, 14.0, "Moresco+2012"),
    (0.3802, 83.0, 13.5, "Moresco+2016"),
    (0.400, 95.0, 17.0, "Simon+2005"),
    (0.4004, 77.0, 10.2, "Moresco+2016"),
    (0.4247, 87.1, 11.2, "Moresco+2016"),
    (0.4497, 92.8, 12.9, "Moresco+2016"),
    (0.470, 89.0, 34.0, "Ratsimbazafy+2017"),
    (0.4783, 80.9, 9.0, "Moresco+2016"),
    (0.480, 97.0, 62.0, "Stern+2010"),
    (0.5929, 104.0, 13.0, "Moresco+2012"),
    (0.6797, 92.0, 8.0, "Moresco+2012"),
    (0.7812, 105.0, 12.0, "Moresco+2012"),
    (0.8754, 125.0, 17.0, "Moresco+2012"),
    (0.880, 90.0, 40.0, "Stern+2010"),
    (0.900, 117.0, 23.0, "Simon+2005"),
    (1.037, 154.0, 20.0, "Moresco+2012"),
    (1.300, 168.0, 17.0, "Simon+2005"),
    (1.363, 160.0, 33.6, "Moresco 2015"),
    (1.430, 177.0, 18.0, "Simon+2005"),
    (1.530, 140.0, 14.0, "Simon+2005"),
    (1.750, 202.0, 40.0, "Simon+2005"),
    (1.965, 186.5, 50.4, "Moresco 2015"),
]
_BORGHI22 = [(0.75, 98.8, 33.6, "Borghi+2022")]
_EXTRA34 = [(0.80, 113.1, 25.2, "Jiao+2023 (symmetrised)"),
            (1.26, 135.0, 65.0, "Tomasetti+2023")]

CC_SETS = {"CC31": _CC31, "CC32": _CC31 + _BORGHI22, "CC34": _CC31 + _BORGHI22 + _EXTRA34}


def load_cc_data(name="CC31", with_sources=False):
    """Return (z, H, sigma[, sources]) sorted by redshift."""
    rows = sorted(CC_SETS[name], key=lambda r: r[0])
    z = np.array([r[0] for r in rows], float)
    H = np.array([r[1] for r in rows], float)
    s = np.array([r[2] for r in rows], float)
    if with_sources:
        return z, H, s, [r[3] for r in rows]
    return z, H, s


if __name__ == "__main__":
    for k in CC_SETS:
        z, H, s = load_cc_data(k)
        print(f"{k}: N={len(z)}, z in [{z.min():.3f}, {z.max():.3f}]")
