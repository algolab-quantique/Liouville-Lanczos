"""
ResultsManager — Gestion structurée des résultats de calculs quantiques.


Usage rapide :
    manager = ResultsManager()
    manager.save(molecule="H2", basis="sto-3g", n_qubits=4,
                 sampling="single", algorithm="Green",
                 data={"a": a, "true_gm_energy": true_gm_energy})
    manager.show(molecule="H2")
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from LiouvilleLanczos.Green import CF_Green, PolyLehmann_Green


# ── Constantes ────────────────────────────────────────────────────────────────

RESULTS_ROOT = Path("results")
RESULTS_FILE = "results.json"

VALID_BASES = {"sto-3g", "6-31g", "cc-pvdz"}
VALID_ALGORITHMS = {"Green", "Green+GM", "Green+GM+dis_curv"}
VALID_SAMPLINGS = {"VQE_SIM", "VQE_QPU", "Exact", "Top1", "Thresh1e-5"}


# ── Helpers internes ───────────────────────────────────────────────────────────


def _numpy_decoder(d: dict):
    # Complexe sérialisé
    if set(d.keys()) == {"real", "imag"}:
        return complex(d["real"], d["imag"])
    # CF_Green sérialisé
    if d.get("__type__") == "CF_Green":
        return CF_Green(
            a=np.array(d["a"]),
            b=np.array(d["b"]),
        )
    return d


def _build_path(molecule, basis, n_qubits, sampling, algorithm):
    algo_safe = algorithm.replace("+", "_")
    filename = f"{molecule.upper()}_{basis.lower()}_{n_qubits}q_{sampling.lower()}_{algo_safe}.json"
    return Path("results") / filename


def _load(path: Path) -> dict:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f, object_hook=_numpy_decoder)
        except json.JSONDecodeError:
            print(f"Fichier JSON corrompu ou vide, réinitialisé : {path}")
            return {}
    return {}


def _dump(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)  # crée results/ si besoin
    with path.open("w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _pretty_header(title: str, width: int = 60) -> str:
    bar = "─" * width
    return f"\n┌{bar}┐\n│ {title:<{width-1}}│\n└{bar}┘"


def make_green_list(a, b, mm, min_iter=1):
    mm = np.asarray(mm)
    num_iterations = len(a)
    num_k = max(num_iterations, min_iter)
    num_moments = len(mm[0])
    green_list = np.empty((num_k, num_moments + 1), dtype=object)
    for k in range(1, num_k + 1):
        kk = min(
            k, num_iterations
        )  ## fill the k > num_iteration with the converged CF_Green
        green_list[k - 1, 0] = CF_Green(a[:kk], b[:kk])
        green0 = green_list[k - 1, 0].to_Lehmann()
        for m in range(num_moments):
            green_list[k - 1, m + 1] = PolyLehmann_Green(
                a[:k], b[:k], mm[:k, m], green0
            )
    return green_list


# ── Classe principale ──────────────────────────────────────────────────────────


class ResultsManager:
    """
    Gestionnaire de résultats pour expériences de chimie quantique.

    Paramètres
    ----------
    root : str | Path
        Dossier racine pour tous les résultats (défaut : "results/").
    """

    def __init__(self, root: str | Path = "results") -> None:
        self.root = Path(root)

    # ── Écriture ──────────────────────────────────────────────────────────────

    def save(
        self,
        molecule: str,
        basis: str,
        n_qubits: int,
        sampling: str,
        algorithm: str,
        data: dict[str, Any],
        run_id: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Enregistre un résultat dans l'arborescence.

        Paramètres
        ----------
        molecule  : nom de la molécule  (ex. "H2", "LiH")
        basis     : base de calcul      (ex. "sto-3g", "6-31g")
        n_qubits  : nombre de qubits
        sampling  : type de sampling    (ex. "1bitstring", "IBM_QC", "Simulator")
        algorithm : algorithme utilisé  (ex. "Green", "Green+GM", "Green+GM+dis_curv")
        data      : dict quelconque contenant vos métriques
        run_id    : identifiant de run (défaut : timestamp ISO)
        overwrite : True pour écraser un run_id existant

        Retourne
        --------
        Path du dossier où le fichier a été écrit.
        """
        self._validate(basis, sampling, algorithm)

        path = _build_path(molecule, basis, n_qubits, sampling, algorithm)
        stored = _load(path)
        key = run_id or _timestamp()

        if key in stored and not overwrite:
            raise KeyError(
                f"Le run '{key}' existe déjà. "
                "Utilisez overwrite=True ou choisissez un autre run_id."
            )

        stored[key] = {
            "meta": {
                "molecule": molecule.upper(),
                "basis": basis.lower(),
                "n_qubits": n_qubits,
                "sampling": sampling.lower(),
                "algorithm": algorithm,
                "saved_at": _timestamp(),
            },
            "results": data,
        }

        _dump(path, stored)
        print(f"✔  Résultats sauvegardés → {path}  [run: {key}]")
        return path

    # ── Lecture / affichage ────────────────────────────────────────────────────

    def show(
        self,
        molecule: str | None = None,
        basis: str | None = None,
        n_qubits: int | None = None,
        sampling: str | None = None,
        algorithm: str | None = None,
        run_id: str | None = None,
        latest_only: bool = False,
    ) -> list[dict]:
        """
        Affiche et retourne les résultats filtrés.

        Tous les paramètres sont optionnels ; omettez ceux que vous voulez ignorer.
        Avec latest_only=True, seul le run le plus récent par fichier est affiché.

        Retourne
        --------
        Liste de dicts {meta, results, run_id, path}.
        """
        matches = self._collect(
            molecule=molecule,
            basis=basis,
            n_qubits=n_qubits,
            sampling=sampling,
            algorithm=algorithm,
            run_id=run_id,
            latest_only=latest_only,
        )

        if not matches:
            print("Aucun résultat trouvé pour ces critères.")
            return []

        for entry in matches:
            self._print_entry(entry)

        print(f"\n{'─'*62}")
        print(f"  {len(matches)} résultat(s) affiché(s).")
        return matches

    def get(
        self,
        molecule: str | None = None,
        basis: str | None = None,
        n_qubits: int | None = None,
        sampling: str | None = None,
        algorithm: str | None = None,
        run_id: str | None = None,
        latest_only: bool = False,
    ) -> list[dict]:
        """Comme show() mais sans affichage — retourne simplement la liste."""
        return self._collect(
            molecule=molecule,
            basis=basis,
            n_qubits=n_qubits,
            sampling=sampling,
            algorithm=algorithm,
            run_id=run_id,
            latest_only=latest_only,
        )

    def delete(
        self,
        molecule: str,
        basis: str,
        n_qubits: int,
        sampling: str,
        algorithm: str,
        run_id: str,
    ) -> bool:
        """Supprime un run précis. Retourne True si supprimé."""
        path = self._resolve(molecule, basis, n_qubits, sampling, algorithm)
        stored = _load(path)

        if run_id not in stored:
            print(f"✘  Run '{run_id}' introuvable.")
            return False

        del stored[run_id]
        _dump(path, stored)
        print(f"✔  Run '{run_id}' supprimé.")
        return True

    def tree(self, max_depth: int = 5) -> None:
        """Affiche l'arborescence des résultats existants."""
        print(_pretty_header(f"Arborescence : {self.root}"))
        self._print_tree(self.root, depth=0, max_depth=max_depth)

    # ── Dans la classe ResultsManager ─────────────────────────────────────────────

    def plot_green(
        self,
        molecule: str,
        basis: str,
        n_qubits: int,
        sampling: str,
        algorithm: str,
        run_id: str | None = None,
        w_range: tuple[float, float] = (-5.5, 5.5),
        eta: float = 1e-1,
        n_points: int = 1000,
        orbital: tuple[int, int] = (0, 0),
        savefig: str | None = None,
    ) -> None:
        entry = self._get_single(molecule, basis, n_qubits, sampling, algorithm, run_id)
        r = entry["results"]

        def rebuild_stack(abm_dict: dict):
            """Reconstruit le stacked_green depuis abm_dict, comme make_green_stack."""
            # min_iter = max des longueurs de a sur tous les opsets
            min_iter = max(len(a) for a, b, m in abm_dict.values())
            stacked = None
            for a, b, m in abm_dict.values():
                green_list = make_green_list(a, b, m, min_iter)
                if stacked is None:
                    stacked = green_list
                else:
                    stacked = np.hstack([stacked, green_list])
            return stacked

        true_stacked_green = rebuild_stack(r["true_green"])
        sqd_stacked_green = rebuild_stack(r["sqd_green"])

        i, j = orbital
        green_ed = true_stacked_green[i][j]
        green_sqd = sqd_stacked_green[i][j]

        w = np.linspace(*w_range, n_points) - 1j * eta
        w_real = np.real(w)
        label = f"{molecule.upper()} | {basis} | {n_qubits}q | {sampling}"

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            w_real,
            np.imag(green_ed(w)),
            color="steelblue",
            linewidth=1.8,
            label="Exact diagonalization",
        )
        ax.plot(
            w_real,
            np.imag(green_sqd(w)),
            color="firebrick",
            linewidth=1.5,
            linestyle="--",
            label="SQD approximation",
        )

        ax.set_xlabel(r"Frequency ($\omega$)", fontsize=13)
        ax.set_ylabel(rf"$\mathrm{{Im}}\, G_{{{i}{j}}}(\omega)$", fontsize=13)
        ax.set_title(f"Green Function — {label}", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=11, framealpha=0.9)
        ax.margins(x=0.02)
        fig.tight_layout()

        if savefig:
            fig.savefig(savefig, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_gm(
        self,
        molecule: str,
        basis: str,
        n_qubits: int,
        sampling: str,
        algorithm: str,
        run_id: str | None = None,
        savefig: str | None = None,
    ) -> None:
        """
        Plot la convergence de l'énergie GM iteration par iteration.
        Attend dans results : 'true_gm_energy' (array) et 'true_gs_energy' (float).
        """
        entry = self._get_single(molecule, basis, n_qubits, sampling, algorithm, run_id)
        r = entry["results"]

        gm_energy = np.array(r["true_gm_energy"])
        gs_energy = float(r["true_gs_energy"])
        iterations = np.arange(len(gm_energy))
        label = f"{molecule.upper()} | {basis} | {n_qubits}q | {sampling}"

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            iterations,
            gm_energy,
            color="steelblue",
            linewidth=1.8,
            marker="o",
            markersize=4,
            label=r"Computed energy $E_{\mathrm{SQD}}$",
        )
        ax.axhline(
            y=gs_energy,
            color="firebrick",
            linewidth=1.5,
            linestyle="--",
            label=rf"Exact ground state $E_0 = {gs_energy:.4f}$ Ha",
        )

        ax.set_xlabel("Iteration", fontsize=13)
        ax.set_ylabel("Energy (Ha)", fontsize=13)
        ax.set_title(f"GM Convergence — {label}", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=11, framealpha=0.9)
        ax.margins(x=0.02)
        fig.tight_layout()

        if savefig:
            fig.savefig(savefig, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_diss_curv(
        self,
        molecule: str,
        basis: str,
        n_qubits: int,
        sampling: str,
        run_id: str | None = None,
        savefig: str | None = None,
    ) -> None:
        """
        Plot la courbe de dissociation — exact, SQD et GM sur le même graphe.

        Récupère le run correspondant à algorithm="Green+GM+diss_curv" et trace
        les trois courbes d'énergie en fonction de la distance interatomique.

        Attend dans results : 'distances', 'gs_energie', 'sqd_energie', 'gm_energie'.

        Parameters
        ----------
        molecule : nom de la molécule
        basis    : base de calcul
        n_qubits : nombre de qubits
        sampling : type de sampling
        run_id   : identifiant de run (None = le plus récent)
        savefig  : chemin de sortie PDF/PNG (None = pas de sauvegarde)
        """
        entry = self._get_single(
            molecule,
            basis,
            n_qubits,
            sampling,
            algorithm="Green+GM+diss_curv",
            run_id=run_id,
        )
        r = entry["results"]

        distances = np.array(r["distances"])
        gs_energies = np.array(r["gs_energie"])
        sqd_energies = np.array(r["sqd_energie"])
        gm_energies = np.array(r["gm_energie"])

        label = f"{molecule.upper()} | {basis} | {n_qubits}q | {sampling}"

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            distances,
            gs_energies,
            color="firebrick",
            linewidth=1.8,
            marker="o",
            markersize=4,
            label="Exact ground state",
        )
        ax.plot(
            distances,
            sqd_energies,
            color="steelblue",
            linewidth=1.5,
            marker="o",
            markersize=4,
            linestyle="--",
            label="SQD",
        )
        ax.plot(
            distances,
            gm_energies,
            color="seagreen",
            linewidth=1.5,
            marker="o",
            markersize=4,
            linestyle="-.",
            label="SQD + GM",
        )

        ax.set_xlabel("Interatomic distance (Å)", fontsize=13)
        ax.set_ylabel("Energy (Ha)", fontsize=13)
        ax.set_title(f"Dissociation curve — {label}", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=11, framealpha=0.9)
        ax.margins(x=0.02)
        fig.tight_layout()

        if savefig:
            fig.savefig(savefig, dpi=300, bbox_inches="tight")
        plt.show()

        # ── Helper interne ────────────────────────────────────────────────────────────

    def _get_single(
        self,
        molecule: str,
        basis: str,
        n_qubits: int,
        sampling: str,
        algorithm: str,
        run_id: str | None,
    ) -> dict:
        """Récupère exactement un entry ou lève une erreur claire."""
        entries = self._collect(
            molecule=molecule,
            basis=basis,
            n_qubits=n_qubits,
            sampling=sampling,
            algorithm=algorithm,
            run_id=run_id,
            latest_only=(run_id is None),
        )
        if not entries:
            raise KeyError("Aucun résultat trouvé pour ces paramètres.")
        if len(entries) > 1:
            ids = [e["run_id"] for e in entries]
            raise KeyError(f"Plusieurs runs trouvés : {ids}. Précisez run_id=...")
        return entries[0]

    # ── Méthodes internes ─────────────────────────────────────────────────────

    def _resolve(
        self,
        molecule: str,
        basis: str,
        n_qubits: int,
        sampling: str,
        algorithm: str,
    ) -> Path:
        return _build_path(molecule, basis, n_qubits, sampling, algorithm)

    def _collect(
        self,
        molecule: str | None,
        basis: str | None,
        n_qubits: int | None,
        sampling: str | None,
        algorithm: str | None,
        run_id: str | None,
        latest_only: bool,
    ) -> list[dict]:
        """Parcourt l'arborescence et filtre les résultats."""
        matches = []

        for json_file in sorted(f for f in self.root.glob("*.json") if f.is_file()):
            stored = _load(json_file)
            for rid, entry in stored.items():
                m = entry["meta"]
                if molecule and m["molecule"].upper() != molecule.upper():
                    continue
                if basis and m["basis"] != basis.lower():
                    continue
                if n_qubits and m["n_qubits"] != n_qubits:
                    continue
                if sampling and m["sampling"].lower() != sampling.lower():
                    continue
                if algorithm and m["algorithm"] != algorithm:
                    continue
                if run_id and rid != run_id:
                    continue
                matches.append({**entry, "run_id": rid, "path": str(json_file)})

        if latest_only and matches:
            # On garde uniquement le run le plus récent par fichier
            by_file: dict[str, dict] = {}
            for e in matches:
                key = e["path"]
                if key not in by_file or e["run_id"] > by_file[key]["run_id"]:
                    by_file[key] = e
            matches = list(by_file.values())

        return matches

    @staticmethod
    def _print_entry(entry: dict) -> None:
        m = entry["meta"]
        r = entry["results"]
        rid = entry["run_id"]
        print(
            _pretty_header(
                f"{m['molecule']}  |  {m['basis']}  |  {m['n_qubits']}q  "
                f"|  {m['sampling']}  |  {m['algorithm']}"
            )
        )
        print(f"  Run ID   : {rid}")
        print(f"  Sauvegardé : {m['saved_at']}")
        print(f"  Fichier  : {entry['path']}")
        print(f"  Résultats :")
        for k, v in r.items():
            print(f"    {k:<25} {v}")

    @staticmethod
    def _print_tree(path: Path, depth: int, max_depth: int) -> None:
        if depth > max_depth or not path.exists():
            return
        indent = "    " * depth
        entries = sorted(path.iterdir()) if path.is_dir() else []
        for entry in entries:
            icon = "📁" if entry.is_dir() else "📄"
            print(f"{indent}{icon} {entry.name}")
            if entry.is_dir():
                ResultsManager._print_tree(entry, depth + 1, max_depth)

    @staticmethod
    def _validate(basis: str, sampling: str, algorithm: str) -> None:
        if basis.lower() not in VALID_BASES:
            print(f"Basis '{basis}' non reconnue. Valides : {VALID_BASES}")
        if sampling.lower() not in VALID_SAMPLINGS:
            print(f"Sampling '{sampling}' non reconnu. Valides : {VALID_SAMPLINGS}")
        if algorithm not in VALID_ALGORITHMS:
            print(f"Algorithme '{algorithm}' non reconnu. Valides : {VALID_ALGORITHMS}")


import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """Sérialise les types NumPy vers des types JSON natifs."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.complexfloating,)):
            return {"real": float(obj.real), "imag": float(obj.imag)}
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
