#%%
from LiouvilleLanczos.Green import CF_Green, PolyLehmann_Green
import matplotlib.pyplot as plt
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.quantum_info import SparsePauliOp
from pathlib import Path
import csv
import numpy as np

n = 5


def to_sparse_pauli(op):
    MAPPER = JordanWignerMapper()
    """
    Convert FermionicOp -> SparsePauliOp.
    Leave SparsePauliOp unchanged.
    """
    if isinstance(op, SparsePauliOp):
        return op

    if isinstance(op, FermionicOp):
        return MAPPER.map(op)
    del MAPPER
    raise TypeError(f"Unsupported operator type: {type(op)}")

def hubbard(hop=np.array([[0, 1], [1, 0]]), u=4, mu=None):
    n = len(hop)
    if mu is None:
        mu = u / 2
    hopping = FermionicOp(
        {
            f"+_{i+m} -_{j+m}": hop[i, j]
            for m in [0, n]
            for i in range(n)
            for j in range(n)
            if hop[i, j] != 0
        },
        num_spin_orbitals=2 * n,
    )
    occupation = FermionicOp(
        {f"+_{i} -_{i}": 1 for i in range(2 * n)}, num_spin_orbitals=2 * n
    )
    interaction = FermionicOp(
        {f"+_{i} +_{i+n} -_{i+n} -_{i}": 1 for i in range(n)}, num_spin_orbitals=2 * n
    )
    HH = -hopping + u * interaction - mu * occupation
    
    C0 = FermionicOp(
        {
            "+_0": 1,
        },
        num_spin_orbitals=2*n,
    )
    return to_sparse_pauli(HH), to_sparse_pauli(C0)

hopping = np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
hamiltonian, C0 = hubbard(hopping, 4, mu=2)     #|up, up, up, up, up, down, down, down, down, down>

eigvals, eigvecs = np.linalg.eigh(hamiltonian.to_matrix())

true_gs_energy = eigvals[0]

def read(folder, degen, id, analytic:bool, site:str = "five"):
    # Find the folder where this Python file lives.
    base_folder = Path(__file__).resolve().parent
    inner = "matrix" if analytic else "inner"
    # Build the folder where the CSV files were written.
    input_folder = base_folder / "results"

    # Build the exact CSV filename.
    input_file = input_folder / f"{degen}_{id}_{inner}_{folder}_{site}.csv"
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
    if len(mu_columns) == 0:
        mi = np.empty((len(a_values), 0), dtype=np.complex128)
    else:
        mi = np.asarray(mu_rows, dtype=np.complex128)

        # Safety: force mu to stay 2D.
        if mi.ndim == 1:
            mi = mi.reshape(-1, len(mu_columns))


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

def rebuild_matrix_five(G_upper):
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

def rebuild_matrix_three(G_upper):

    G_full = np.empty((3, 3), dtype=object)
    G_full[:, :] = None

    # Diagonal: edge sites 0 and 2 are equivalent.
    G_full[0, 0] = G_upper[0, 0]
    G_full[2, 2] = G_upper[0, 0]

    # Center site.
    G_full[1, 1] = G_upper[1, 1]

    # Nearest-neighbor edge-center entries.
    G_full[0, 1] = G_upper[0, 1]
    G_full[1, 0] = G_upper[0, 1]
    G_full[1, 2] = G_upper[0, 1]
    G_full[2, 1] = G_upper[0, 1]

    # Edge-edge entries.
    G_full[0, 2] = G_upper[0, 2]
    G_full[2, 0] = G_upper[0, 2]

    return G_full

def gm(G_full, n = 5):
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



opset = ["0-1234","1-23","2-"] if n == 5 else ["0-12", "1-"]
analytic_greens = []
analytic_energies = []
inner_greens = []
inner_energies = []
iterations = range(1, 18 + 1) if n == 3 else range(1, 6+1)

for iter_cutoff in iterations:

    for analytic in [True, False]:

        green_iter = []
        energy_iter = []

        for c_fermi in ["up", "down"]:

            spin_greens = []
            spin_energies = []

            for GS in range(2):

                G = np.empty((n, n), dtype=object)
                G[:, :] = None

                for op in opset:
                    main_i, other_i = op.split("-")
                    row_idx = int(main_i)

                    alpha, beta, mm = read(
                        c_fermi,
                        degen=GS,
                        id=row_idx,
                        analytic=analytic,
                        site="five" if n == 5 else "three",
                    )

                    alpha = np.asarray(alpha[:iter_cutoff])
                    beta = np.asarray(beta[:iter_cutoff])
                    mm = np.asarray(mm[:iter_cutoff, :])

                    g = CF_Green(alpha, beta)
                    G[row_idx, row_idx] = g

                    g_lehmann = g.to_Lehmann()

                    for moment_index, j_char in enumerate(other_i):
                        col_idx = int(j_char)

                        G_ij = PolyLehmann_Green(
                            alpha,
                            beta,
                            mm[:, moment_index],
                            g_lehmann,
                        )

                        G[row_idx, col_idx] = G_ij

                G = rebuild_matrix_five(G) if len(opset[0].split("-")[1]) == 4 else rebuild_matrix_three(G)

                spin_greens.append(G)
                spin_energies.append(gm(G, n = n))

            green_iter.append(spin_greens)
            energy_iter.append(spin_energies)

        if analytic:
            analytic_greens.append(green_iter)
            analytic_energies.append(energy_iter)
        else:
            inner_greens.append(green_iter)
            inner_energies.append(energy_iter)
    


# %%
w = np.linspace(-5.5,5.5,1000)-1e-1j

# spin_greens_iter [iteration index] [spin up or spin down] [degenerate state index] [i,j]
# spin_energies_iter [iteration index] [spin up or spin down] [degenerate state index]

for i in range(len(iterations)):
    plt.figure()
    plt.title(f"Iteration {i+1}")
    G00_1 = inner_greens[i][0][0][0][0](w)
    G00_2 = inner_greens[i][0][1][0][0](w)
    plt.plot(np.real(w),np.imag(G00_1+G00_2),label = f"inner")
    G00_1 = analytic_greens[i][0][0][0][0](w)
    G00_2 = analytic_greens[i][0][1][0][0](w)
    plt.plot(np.real(w),np.imag(G00_1+G00_2)+0.5,  label = f"analytic")
    plt.legend()
    plt.savefig(f"tests/cluster/plots/five/iter_{i}.svg", format = "svg")

#%%
iters = list(iterations)

spin_labels = ["up", "down"]

analytic_spin_sum = np.zeros(len(iters), dtype=float)
inner_spin_sum = np.zeros(len(iters), dtype=float)

plt.figure(figsize=(8, 5))

for spin_idx, spin_label in enumerate(spin_labels):

    # Average over degenerate GS sectors, do NOT sum them.
    analytic_vals = [
        (
            analytic_energies[k][spin_idx][0]
            + analytic_energies[k][spin_idx][1]
        )
        for k in range(len(iters))
    ]

    inner_vals = [
        (
            inner_energies[k][spin_idx][0]
            + inner_energies[k][spin_idx][1]
        )
        for k in range(len(iters))
    ]

    analytic_vals = np.asarray(analytic_vals).real
    inner_vals = np.asarray(inner_vals).real

    # Physical spin sum: up + down.
    analytic_spin_sum += analytic_vals
    inner_spin_sum += inner_vals

    plt.plot(
        iters,
        analytic_vals,
        marker="o",
        linestyle="-",
        label=f"analytic {spin_label}, GS avg",
    )

    plt.plot(
        iters,
        inner_vals,
        marker="x",
        linestyle="--",
        label=f"inner {spin_label}, GS avg",
    )

plt.plot(
    iters,
    analytic_spin_sum*0.5,
    marker="o",
    linestyle="-",
    linewidth=3,
    label="analytic up + down, GS avg",
)

plt.plot(
    iters,
    inner_spin_sum*0.5,
    marker="x",
    linestyle="--",
    linewidth=3,
    label="inner up + down, GS avg",
)

plt.plot(
    iters,
    [true_gs_energy] * len(iters),
    color="black",
    linestyle=":",
    label="exact",
)

plt.xlabel("Lanczos iteration cutoff")
plt.ylabel("Galitskii-Migdal energy")
plt.title("Energy convergence: analytic vs inner Green's functions")
plt.xticks(iters)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"tests/cluster/plots/five/energy.svg", format = "svg")

# %%
