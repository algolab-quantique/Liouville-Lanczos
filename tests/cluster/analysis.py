#%%
from LiouvilleLanczos.Green import CF_Green, PolyLehmann_Green
import matplotlib.pyplot as plt

from pathlib import Path
import csv
import numpy as np

def read(folder, degen, id, analytic:bool):
    # Find the folder where this Python file lives.
    base_folder = Path(__file__).resolve().parent
    inner = "matrix" if analytic else "inner"
    # Build the folder where the CSV files were written.
    input_folder = base_folder / "results"

    # Build the exact CSV filename.
    input_file = input_folder / f"{degen}_{id}_{inner}_{folder}.csv"

    print(f"Reading coefficients from: {input_file}")
    # Check that the file exists before reading.
    if not input_file.exists():
        raise FileNotFoundError(f"Could not find file: {input_file}")

    # Store the a coefficients here.
    a_values = []

    # Store the b coefficients here.
    b_values = []

    # Store the mu rows here.
    mu_rows = []

    # Open the CSV file.
    with open(input_file, mode="r", newline="") as file:

        # Read the CSV using column names.
        reader = csv.DictReader(file)

        # Make sure the file has a header.
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {input_file}")

        # Find all mu columns, like mu_0, mu_1, ...
        mu_columns = [
            name for name in reader.fieldnames
            if name.startswith("mu_")
        ]

        # Sort mu columns by their numeric index.
        mu_columns = sorted(
            mu_columns,
            key=lambda name: int(name.split("_")[1])
        )

        # Loop through each row of the CSV.
        for row in reader:

            # Read raw a and b strings.
            a_raw = row["a"]
            b_raw = row["b"]

            # Save a if it is not blank.
            if a_raw != "":
                a_values.append(complex(a_raw))

            # Save b if it is not blank.
            if b_raw != "":
                b_values.append(complex(b_raw))

            # Store this row's mu values.
            mu_row = []

            # Loop through each mu column.
            for col in mu_columns:

                # Read raw mu string.
                mu_raw = row[col]

                # Save mu if it is not blank.
                if mu_raw != "":
                    mu_row.append(complex(mu_raw))

                # Otherwise save NaN.
                else:
                    mu_row.append(np.nan)

            # Add the mu row if there are mu columns.
            if len(mu_columns) > 0:
                mu_rows.append(mu_row)

    # Convert a to NumPy array.
    a = np.asarray(a_values, dtype=np.complex128)

    # Convert b to NumPy array.
    b = np.asarray(b_values, dtype=np.complex128)

    # Convert mi to NumPy array.
    mi = np.asarray(mu_rows, dtype=np.complex128)

    # If a is real, return it as real.
    if a.size > 0 and np.allclose(a.imag, 0):
        a = a.real

    # If b is real, return it as real.
    if b.size > 0 and np.allclose(b.imag, 0):
        b = b.real

    # If mi is real, return it as real.
    if mi.size > 0 and np.allclose(mi.imag, 0, equal_nan=True):
        mi = mi.real

    # Return the coefficients.
    return a, b, mi

def rebuild_matrix(G_upper):
    # Create a full 5 by 5 object matrix.
    G_full = np.empty((5, 5), dtype=object)

    # Fill everything with None first.
    G_full[:, :] = None

    # -------------------------
    # Diagonal entries
    # -------------------------

    # G_00 and G_44 are equivalent by reflection symmetry.
    G_full[0, 0] = G_upper[0, 0]
    G_full[4, 4] = G_upper[0, 0]

    # G_11 and G_33 are equivalent by reflection symmetry.
    G_full[1, 1] = G_upper[1, 1]
    G_full[3, 3] = G_upper[1, 1]

    # G_22 is the center-site diagonal Green's function.
    G_full[2, 2] = G_upper[2, 2]

    # -------------------------
    # Entries equivalent to G_01
    # -------------------------

    G_full[0, 1] = G_upper[0, 1]
    G_full[1, 0] = G_upper[0, 1]
    G_full[3, 4] = G_upper[0, 1]
    G_full[4, 3] = G_upper[0, 1]

    # -------------------------
    # Entries equivalent to G_02
    # -------------------------

    G_full[0, 2] = G_upper[0, 2]
    G_full[2, 0] = G_upper[0, 2]
    G_full[2, 4] = G_upper[0, 2]
    G_full[4, 2] = G_upper[0, 2]

    # -------------------------
    # Entries equivalent to G_03
    # -------------------------

    G_full[0, 3] = G_upper[0, 3]
    G_full[3, 0] = G_upper[0, 3]
    G_full[1, 4] = G_upper[0, 3]
    G_full[4, 1] = G_upper[0, 3]

    # -------------------------
    # Entries equivalent to G_04
    # -------------------------

    G_full[0, 4] = G_upper[0, 4]
    G_full[4, 0] = G_upper[0, 4]

    # -------------------------
    # Entries equivalent to G_12
    # -------------------------

    G_full[1, 2] = G_upper[1, 2]
    G_full[2, 1] = G_upper[1, 2]
    G_full[2, 3] = G_upper[1, 2]
    G_full[3, 2] = G_upper[1, 2]

    # -------------------------
    # Entries equivalent to G_13
    # -------------------------

    G_full[1, 3] = G_upper[1, 3]
    G_full[3, 1] = G_upper[1, 3]

    # Return the completed Green matrix.
    return G_full

def gm(G_full):
    n = 5
    hopping = np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    h_mu = (2) * np.diag(np.ones(n))
    h_0 = -hopping - h_mu
    
    def fermi(w):
        return w <= 0.0
    
    Kq = 0.0 + 0.0j
    
    for i in range(n):
        for j in range(n):
            g = G_full[j, i]
            if not hasattr(g, "integrate"):
                g = g.to_Lehmann()
            integral_G = g.integrate(lambda w: fermi(w))
            Kq += h_0[i, j] * integral_G

    wG = 0.0 + 0.0j


    for i in range(n):
        g = G_full[i, i]
        if not hasattr(g, "integrate"):
                g = g.to_Lehmann()
        integral_wG = g.integrate(lambda w: w * fermi(w))
        wG += integral_wG

    E_GM = 0.5 * (Kq + wG).real
    return E_GM



#%%
opset = ["0-1234","1-23","2-"]
iterations = [10,20,30,35,40,45,50,55,60,65,70,80,90]
analytic_greens = None
analytic_energies = None
inner_greens = None
inner_energies = None

for analytic in [True,False]:
    spin_greens_iter = []
    spin_energies_iter = []
    for iter in iterations: # slice iteration 90 is equivalent to list of iterations
        spin_greens = []
        spin_energies = []
        for c_fermi in ["up","down"]:
            degen_greens = []
            degen_energies = []
            for GS in range(2):
                G = np.empty((5,5), dtype=object)
                G[:, :] = None
                for i, op in enumerate(opset):
                    w = np.linspace(-5.5,5.5,1000)-1e-1j
                    main_i, other_i = op.split("-")
                    i = int(main_i)
                    alpha,beta,mm = read(c_fermi, degen = GS, id=i, analytic = analytic)
                    alpha = np.asarray(alpha[:iter])
                    beta = np.asarray(beta[:iter])
                    mm = np.asarray(mm[:iter,:])
                    
                    g = CF_Green(alpha,beta)
                    G[i][i]= g
                    
                    g_lehmann = g.to_Lehmann()

                    for moment_index, j_char in enumerate(other_i):
                        j = int(j_char)
                        G_ij = PolyLehmann_Green(
                            alpha,
                            beta,
                            mm[:,moment_index],
                            g_lehmann,
                        )
                        G[i, j] = G_ij
                G = rebuild_matrix(G)
                degen_greens.append(G)
                degen_energies.append(gm(G))

            spin_greens.append(degen_greens)
            spin_energies.append(degen_energies)
        spin_greens_iter.append(spin_greens)
        spin_energies_iter.append(spin_energies)
    if analytic:
        analytic_greens = spin_greens_iter
        analytic_energies = spin_energies_iter
    else:
        inner_greens = spin_greens_iter
        inner_energies = spin_energies_iter
    


# %%

alpha,beta,mm = read("up", degen = 0, id=0, analytic = False)
G00_1 = CF_Green(alpha,beta)
alpha,beta,mm = read("up", degen = 1, id=0, analytic = False)
G00_2 = CF_Green(alpha,beta)
w = np.linspace(-5.5,5.5,1000)-1e-1j
plt.figure()
plt.plot(w.real, np.imag(G00_1(w)))
plt.plot(w.real,np.imag(G00_2(w)))
plt.show

#%%
# spin_greens_iter [iteration index] [spin up or spin down] [degenerate state index] [i,j]
# spin_energies_iter [iteration index] [spin up or spin down] [degenerate state index]
plt.figure()
G00_1 = inner_greens[0][0][0][0][0](w)
G00_2 = inner_greens[0][0][1][0][0](w)
plt.plot(np.real(w),np.imag(G00_1+G00_2), label = "inner")
G00_1 = analytic_greens[0][0][0][0][0](w)
G00_2 = analytic_greens[0][0][1][0][0](w)
plt.plot(np.real(w),np.imag(G00_1+G00_2), label = "analytic")
plt.legend()
plt.show()

# %%
