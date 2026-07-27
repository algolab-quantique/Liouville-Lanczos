#%%
from LiouvilleLanczos.Green import CF_Green, PolyLehmann_Green
import matplotlib.pyplot as plt
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.quantum_info import SparsePauliOp
from pathlib import Path
import csv
from qiskit.primitives import StatevectorEstimator
import numpy as np
from qiskit.circuit.library import real_amplitudes

vqe_hubbard_5sites = real_amplitudes(num_qubits=10, reps=10, entanglement='linear')
vqe_hubbard_5sites = vqe_hubbard_5sites.assign_parameters(np.array([
    2.3447832964919617, 1.4474302559938388, 4.713514444810458, 4.483509407584665, 4.400570018376737,
    3.1599828310535765, 1.4811927345460485, 4.401225733572901, 4.9385281687492535, 4.712960988556388,
    3.9345915095063786, 4.798725490293983, 1.6957962772935746, 4.18431690489243, 1.4314151293812927,
    3.6586171333767994, 2.0421913767282662, 4.861557641736541, 4.7384698726472285, 3.734752714234186,
    5.6471486574706375, 4.711408611937339, 4.747181489931575, 4.785705709657881, 5.773891260485987,
    3.164130952306159, 5.87030802541217, 4.62641468296399, 1.5715680039711153, 3.17431807296486,
    0.5479225386262592, 4.7125194894458895, 1.5707646540418294, 1.5708020948979795, 4.712386411886244,
    1.5707956120001854, 1.570800746348143, 4.712386246353408, 6.283183592137927, 6.283185024020641,
    6.283108640215091, -0.31483874554879887, 2.5684582292450258, 6.641278834606595, 1.595865508133264,
    3.5586234337709435, 1.6436999282262008, 1.5707985348741194, -2.0956374394721885e-06, -0.3551216158824308,
    3.1415901265735933, 4.413018428808857, 5.313840134725547, 4.478621591268556, 3.109855048054034,
    6.519653639112034, -0.14634612324162952, 4.712390805311075, 0.2152617755934342, 7.820990358428639,
    3.141596566559733, 4.315707796787452, 1.591058317291354, 1.5511314469306399, 4.654357447200541,
    4.708047365972247, 1.5707966504313826, 4.712389377489952, 4.710279668850846, 4.739490027393471,
    3.14158387406988, 4.71821903393061, 4.752704332942756, 4.680290507834347, 1.3224139337263787,
    3.6174009369047324, 1.5707939141135263, 1.5335737771548246, 4.690684139651661, 1.5408585947576945,
    9.324079134959437e-06, 1.6182910007664684, 1.6703109721089433, 1.4147301236346137, 1.5740252318995949,
    1.5707960492817332, 1.570798283653272, 1.5191349714558515, 1.6161172741980936, -0.31862288861427424,
    1.5707879571326253, 6.165949907011652, 6.6176881469443485, 1.2673449170192668, 3.6129609342647377,
    3.1415932962911555, 5.9963476937642755, 2.8726739404577013, 4.769961554828166, 2.9322930511089185,
    1.5707945163585162, 4.712389541091441, 4.712391063980809, 1.5707945358999216, 1.5707993205185902,
    4.7123863681724245, 1.5707962675025136, 4.712390122552536, 4.712389940305301, 4.712388510339969
]))

vqe_hubbard_3sites = real_amplitudes(num_qubits=6, reps=11, entanglement='linear')
vqe_hubbard_3sites = vqe_hubbard_3sites.assign_parameters(np.array([
    5.246829583426229, 1.8445563598975319, 6.249341224922228, 5.547426630865061, 4.204112886091582,
    4.7123853975763605, 0.27755320293694075, 0.32924783165591026, 1.5501725153481376, 0.354654347037912,
    1.4126950505240836, 7.524339982913663e-06, 1.5454883723953319, 4.008676262413297, 0.5471153440483539,
    0.6439116484454839, 5.416150750408826, 3.141590029386429, 1.121133004167295, 4.919561698711213,
    4.354025416990712, 4.348236205036469, 0.1627483139705849, 6.890458045940891e-06, 1.5750620500544459,
    3.925552761663296, 2.4735512439001406, 3.564021121427497, 2.1847082492649674, 3.1415875290047737,
    5.145406231818936, 3.082671747231, 4.755880471423207, 1.5314536798535008, 5.823307617043694,
    3.1416008591215223, 5.901892324799543, 1.4437881974849553, 4.645825634069647, 1.6387913496744664,
    5.2749649006268555, 1.1584242198623043e-05, 2.4788427008935474, 5.261826591438184, 5.982045041499495,
    2.24084148657242, 1.3918515401467277, 3.141590524117431, 2.6212585750135275, 0.0, 4.140488402169957,
    0.6887497275343819, 0.16071147559368984, 4.712382212081382, 5.6131238667553776, 0.2442058266515827,
    2.815716430300995, 1.696363607754504, 4.71239011907928, 6.283185307179586, 2.0669264858298853,
    2.441456446485303, 1.273233624886327, 2.646377569674503, 0.957628396457077, 1.8447208439653136,
    5.224543370694376, 1.583011829806925, 6.116083873171027, 6.0304071284479774e-06, 1.570785721175754,
    2.9229147479846883e-06
]))



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
    
    return to_sparse_pauli(HH)

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

for n in [3]: # if 5_site is also in the result folder add 5 for the plotting to run both

    hopping = np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    hamiltonian = hubbard(hopping, 4, mu=2)     #|up, up, up, up, up, down, down, down, down, down>

    eigvals, eigvecs = np.linalg.eigh(hamiltonian.to_matrix())

    true_gs_energy = eigvals[0]

    H = to_sparse_pauli(hamiltonian)

    estimator = StatevectorEstimator()
    if n == 5:
        job = estimator.run([(vqe_hubbard_5sites,H)])
    else:
        job = estimator.run([(vqe_hubbard_3sites,H)])

    job_result = job.result()[0].data.evs

    opset = ["0-1234","1-23","2-"] if n == 5 else ["0-12", "1-"]
    analytic_greens = []
    analytic_energies = []
    inner_greens = []
    inner_energies = []
    iterations = range(2, 18 + 1) if n == 3 else range(2, 6+1)

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
        
    #%%

    w = np.linspace(-5.5,5.5,1000)-1e-1j

    # spin_greens_iter [iteration index] [spin up or spin down] [degenerate state index] [i,j]
    # spin_energies_iter [iteration index] [spin up or spin down] [degenerate state index]

    
    for j in range(n):
        for k in range(n):
            plt.figure()
            plt.title(f"G_{j}{k}")
            G00_1 = inner_greens[-1][0][0][j][k](w)
            G00_2 = inner_greens[-1][0][1][j][k](w)
            plt.plot(np.real(w),np.imag(G00_1+G00_2),label = f"inner")
            G00_1 = analytic_greens[-1][0][0][j][k](w)
            G00_2 = analytic_greens[-1][0][1][j][k](w)
            plt.plot(np.real(w),np.imag(G00_1+G00_2),  label = f"analytic")
            plt.legend()
            if n == 5:
                plt.savefig(f"tests/cluster/plots/five/G_{j}{k}.svg", format = "svg")
            else:
                plt.savefig(f"tests/cluster/plots/three/G_{j}{k}.svg", format = "svg")
            plt.close()

    #%%

    iters = list(iterations)

    spin_labels = ["up", "down"]
    GS_labels = ["GS1","GS2"]


    plt.figure()

    gs_sums = {GS_idx: np.zeros(len(iters), dtype=float) for GS_idx in GS_labels}

    for i, spin_idx in enumerate(spin_labels):
        for j, GS_idx in enumerate(GS_labels):
            y_axis_analytical = np.array(
                [analytic_energies[k][i][j] for k in range(len(iters))] 
            ) * 0.5
            y_axis_inner = np.array(
                [inner_energies[k][i][j] for k in range(len(iters))] 
            ) * 0.5


            gs_sums[GS_idx] += y_axis_analytical

    # Plot the two GS sums
    plt.plot(
        iters,
        gs_sums["GS1"],
        label="mean_" + GS_labels[0],
        
    )
    plt.plot(
        iters,
        gs_sums["GS2"],
        label="mean_" + GS_labels[1],
        
    )

    # Plot sum of the two GS sums
    total_gs_sum = sum(gs_sums.values())

    plt.plot(
        iters,
        total_gs_sum ,
        label="sum_spin",
    )

    plt.plot(
        iters,
        [true_gs_energy] * len(iters),
        color="red",
        linestyle=":",
        label="exact diagonalization",
    )

    plt.plot(
        iters,
        [job_result] * len(iters),
        color="black",
        linestyle=":",
        label="VQE expectation value",
    )

    plt.xlabel("Lanczos iteration cutoff")
    plt.ylabel("Galitskii-Migdal energy")
    plt.xticks(iters)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if n == 5:
        plt.savefig(f"tests/cluster/plots/five/energy.svg", format = "svg")
    else:
        plt.savefig(f"tests/cluster/plots/three/energy.svg", format = "svg")
    
    plt.close()
