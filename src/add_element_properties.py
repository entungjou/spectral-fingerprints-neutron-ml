from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_FILE = PROJECT_ROOT / "outputs" / "ml_dataset.csv"
OUT_FILE = PROJECT_ROOT / "outputs" / "ml_dataset_with_properties.csv"

# Basic elemental properties
# symbol: (Z, atomic_mass)
ELEMENT_INFO = {
    "H":  (1, 1.008),
    "He": (2, 4.0026),
    "Li": (3, 6.94),
    "Be": (4, 9.0122),
    "B":  (5, 10.81),
    "C":  (6, 12.011),
    "N":  (7, 14.007),
    "O":  (8, 15.999),
    "F":  (9, 18.998),
    "Ne": (10, 20.180),
    "Na": (11, 22.990),
    "Mg": (12, 24.305),
    "Al": (13, 26.982),
    "Si": (14, 28.085),
    "P":  (15, 30.974),
    "S":  (16, 32.06),
    "Cl": (17, 35.45),
    "Ar": (18, 39.948),
    "K":  (19, 39.098),
    "Ca": (20, 40.078),
    "Sc": (21, 44.956),
    "Ti": (22, 47.867),
    "V":  (23, 50.942),
    "Cr": (24, 51.996),
    "Mn": (25, 54.938),
    "Fe": (26, 55.845),
    "Co": (27, 58.933),
    "Ni": (28, 58.693),
    "Cu": (29, 63.546),
    "Zn": (30, 65.38),
    "Ga": (31, 69.723),
    "Ge": (32, 72.630),
    "As": (33, 74.922),
    "Se": (34, 78.971),
    "Br": (35, 79.904),
    "Kr": (36, 83.798),
    "Rb": (37, 85.468),
    "Sr": (38, 87.62),
    "Y":  (39, 88.906),
    "Zr": (40, 91.224),
    "Nb": (41, 92.906),
    "Mo": (42, 95.95),
    "Tc": (43, 98.0),
    "Ru": (44, 101.07),
    "Rh": (45, 102.91),
    "Pd": (46, 106.42),
    "Ag": (47, 107.87),
    "Cd": (48, 112.41),
    "In": (49, 114.82),
    "Sn": (50, 118.71),
    "Sb": (51, 121.76),
    "Te": (52, 127.60),
    "I":  (53, 126.90),
    "Xe": (54, 131.29),
    "Cs": (55, 132.91),
    "Ba": (56, 137.33),
    "La": (57, 138.91),
    "Ce": (58, 140.12),
    "Pr": (59, 140.91),
    "Nd": (60, 144.24),
    "Pm": (61, 145.0),
    "Sm": (62, 150.36),
    "Eu": (63, 151.96),
    "Gd": (64, 157.25),
    "Tb": (65, 158.93),
    "Dy": (66, 162.50),
    "Ho": (67, 164.93),
    "Er": (68, 167.26),
    "Tm": (69, 168.93),
    "Yb": (70, 173.05),
    "Lu": (71, 174.97),
    "Hf": (72, 178.49),
    "Ta": (73, 180.95),
    "W":  (74, 183.84),
    "Re": (75, 186.21),
    "Os": (76, 190.23),
    "Ir": (77, 192.22),
    "Pt": (78, 195.08),
    "Au": (79, 196.97),
    "Hg": (80, 200.59),
    "Tl": (81, 204.38),
    "Pb": (82, 207.2),
    "Bi": (83, 208.98),
}

def main():
    df = pd.read_csv(IN_FILE)

    if "element" not in df.columns:
        raise ValueError("ml_dataset.csv must contain an 'element' column")

    df["Z"] = df["element"].map(lambda x: ELEMENT_INFO.get(x, (None, None))[0])
    df["atomic_mass"] = df["element"].map(lambda x: ELEMENT_INFO.get(x, (None, None))[1])

    missing = df[df["Z"].isna() | df["atomic_mass"].isna()]

    if not missing.empty:
        print("Warning: some elements were not found in ELEMENT_INFO:")
        print(missing["element"].unique())

    df.to_csv(OUT_FILE, index=False)

    print("Loaded:", IN_FILE)
    print("Rows:", len(df))
    print("Saved ->", OUT_FILE)
    print("\nPreview:")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()