# simulation/cli.py
"""
CLI symétrique aux endpoints HTTP scénarios / simulations / galaxy.

Calque le pattern de ml/cli.py : pas de logique métier ici, on délègue
à la couche service. Sortie JSON sur stdout, pipable vers jq ou fichier.

Usage :
    python -m app.simulation.cli scenarios create --nom <n> --nb-systemes <N> [...]
    python -m app.simulation.cli scenarios list
    python -m app.simulation.cli scenarios show <id>
    python -m app.simulation.cli scenarios delete <id>
    python -m app.simulation.cli scenarios entites <id> [--seed <s>]
    python -m app.simulation.cli scenarios simulate <id> [--seed <s>]
    python -m app.simulation.cli scenarios full --nom <n> --nb-systemes <N> [...]

    python -m app.simulation.cli galaxy density <scenario_id> [--nx N --ny N --nz N]
    python -m app.simulation.cli galaxy events <simulation_id> [--n N]
"""

from __future__ import annotations

import argparse
import json
import sys

from app.db.connection import get_conn
from app.scenario import (
    ParametresAstrophysiques,
    ScenarioRequest,
    SimulationRequest,
)
from app.simulation import service
from app.simulation.galaxy_view import (
    calculer_densite,
    calculer_events_zoom_galaxie,
)


# ============================================================================
# HELPERS
# ============================================================================

def _print_json(payload: dict) -> None:
    """Affiche un dict en JSON UTF-8 sur stdout, indenté, default=str pour
    couvrir les types non-JSON natifs (Decimal, datetime, etc.)."""
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _build_scenario_request(args: argparse.Namespace) -> ScenarioRequest:
    """Construit une ScenarioRequest depuis les flags CLI."""
    parametres = ParametresAstrophysiques(
        masse_stellaire_moyenne=args.masse_stellaire_moyenne,
        indice_tellurique=args.indice_tellurique,
        planetes_par_systeme_moyen=args.planetes_par_systeme_moyen,
        duree_simulation_Ga=args.duree_simulation_Ga,
    )
    return ScenarioRequest(
        nom=args.nom,
        nb_systemes=args.nb_systemes,
        parametres=parametres,
    )


def _ajouter_args_parametres(parser: argparse.ArgumentParser) -> None:
    """Ajoute les flags des 4 paramètres astrophysiques avec leurs défauts."""
    defaults = ParametresAstrophysiques()
    parser.add_argument(
        "--masse-stellaire-moyenne",
        dest="masse_stellaire_moyenne",
        type=float,
        default=defaults.masse_stellaire_moyenne,
        help="M☉, défaut %(default)s",
    )
    parser.add_argument(
        "--indice-tellurique",
        dest="indice_tellurique",
        type=float,
        default=defaults.indice_tellurique,
        help="dans [0,1], défaut %(default)s",
    )
    parser.add_argument(
        "--planetes-par-systeme-moyen",
        dest="planetes_par_systeme_moyen",
        type=float,
        default=defaults.planetes_par_systeme_moyen,
        help="lambda Poisson, défaut %(default)s",
    )
    parser.add_argument(
        "--duree-simulation-Ga",
        dest="duree_simulation_Ga",
        type=float,
        default=defaults.duree_simulation_Ga,
        help="Ga, défaut %(default)s",
    )


# ============================================================================
# COMMANDES SCÉNARIOS
# ============================================================================

def cmd_scenarios_create(args: argparse.Namespace) -> None:
    req = _build_scenario_request(args)
    with get_conn() as conn:
        result = service.creer_scenario(conn, req)
    _print_json(result)


def cmd_scenarios_list(args: argparse.Namespace) -> None:
    with get_conn() as conn:
        result = service.lister_scenarios(conn)
    _print_json(result)


def cmd_scenarios_show(args: argparse.Namespace) -> None:
    with get_conn() as conn:
        try:
            result = service.lire_scenario(conn, args.scenario_id)
        except LookupError as exc:
            print(f"Erreur : {exc}", file=sys.stderr)
            sys.exit(1)
    _print_json(result)


def cmd_scenarios_delete(args: argparse.Namespace) -> None:
    with get_conn() as conn:
        try:
            result = service.supprimer_scenario(conn, args.scenario_id)
        except LookupError as exc:
            print(f"Erreur : {exc}", file=sys.stderr)
            sys.exit(1)
    _print_json(result)


def cmd_scenarios_entites(args: argparse.Namespace) -> None:
    with get_conn() as conn:
        result = service.generer_entites_scenario(conn, args.scenario_id, args.seed)
    _print_json(result)


def cmd_scenarios_simulate(args: argparse.Namespace) -> None:
    req = SimulationRequest(seed=args.seed)
    with get_conn() as conn:
        result = service.creer_simulation(conn, args.scenario_id, req)
    _print_json(result)


def cmd_scenarios_full(args: argparse.Namespace) -> None:
    req = _build_scenario_request(args)
    with get_conn() as conn:
        result = service.creer_scenario_full(conn, req)
    _print_json(result)


# ============================================================================
# COMMANDES GALAXY
# ============================================================================

def cmd_galaxy_density(args: argparse.Namespace) -> None:
    with get_conn() as conn:
        try:
            result = calculer_densite(conn, args.scenario_id, args.nx, args.ny, args.nz)
        except LookupError as exc:
            print(f"Erreur : {exc}", file=sys.stderr)
            sys.exit(1)
    _print_json(result)


def cmd_galaxy_events(args: argparse.Namespace) -> None:
    with get_conn() as conn:
        try:
            result = calculer_events_zoom_galaxie(conn, args.simulation_id, args.n)
        except LookupError as exc:
            print(f"Erreur : {exc}", file=sys.stderr)
            sys.exit(1)
    _print_json(result)


# ============================================================================
# PARSER
# ============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.simulation.cli",
        description="CLI symétrique aux endpoints scenarios / simulations / galaxy.",
    )
    sub = parser.add_subparsers(dest="domaine", required=True)

    # ---- scenarios ----
    p_sc = sub.add_parser("scenarios", help="Opérations sur les scénarios")
    sub_sc = p_sc.add_subparsers(dest="action", required=True)

    p_create = sub_sc.add_parser("create", help="Crée un scénario")
    p_create.add_argument("--nom", required=True)
    p_create.add_argument("--nb-systemes", dest="nb_systemes", type=int, required=True)
    _ajouter_args_parametres(p_create)
    p_create.set_defaults(func=cmd_scenarios_create)

    p_list = sub_sc.add_parser("list", help="Liste les scénarios")
    p_list.set_defaults(func=cmd_scenarios_list)

    p_show = sub_sc.add_parser("show", help="Détail d'un scénario")
    p_show.add_argument("scenario_id", type=int)
    p_show.set_defaults(func=cmd_scenarios_show)

    p_delete = sub_sc.add_parser("delete", help="Supprime un scénario (cascade)")
    p_delete.add_argument("scenario_id", type=int)
    p_delete.set_defaults(func=cmd_scenarios_delete)

    p_ent = sub_sc.add_parser("entites", help="Génère les entités d'un scénario")
    p_ent.add_argument("scenario_id", type=int)
    p_ent.add_argument("--seed", type=int, default=None)
    p_ent.set_defaults(func=cmd_scenarios_entites)

    p_sim = sub_sc.add_parser("simulate", help="Lance une simulation sur un scénario")
    p_sim.add_argument("scenario_id", type=int)
    p_sim.add_argument("--seed", type=int, default=None)
    p_sim.set_defaults(func=cmd_scenarios_simulate)

    p_full = sub_sc.add_parser("full", help="Pipeline complet (scénario + entités + simulation)")
    p_full.add_argument("--nom", required=True)
    p_full.add_argument("--nb-systemes", dest="nb_systemes", type=int, required=True)
    _ajouter_args_parametres(p_full)
    p_full.set_defaults(func=cmd_scenarios_full)

    # ---- galaxy ----
    p_gal = sub.add_parser("galaxy", help="Endpoints du zoom galaxie")
    sub_gal = p_gal.add_subparsers(dest="action", required=True)

    p_dens = sub_gal.add_parser("density", help="Grille 3D de densité d'un scénario")
    p_dens.add_argument("scenario_id", type=int)
    p_dens.add_argument("--nx", type=int, default=50)
    p_dens.add_argument("--ny", type=int, default=50)
    p_dens.add_argument("--nz", type=int, default=10)
    p_dens.set_defaults(func=cmd_galaxy_density)

    p_evt = sub_gal.add_parser("events", help="Événements zoom galaxie d'une simulation")
    p_evt.add_argument("simulation_id", type=int)
    p_evt.add_argument("--n", type=int, default=100,
                       help="Taux de rafraîchissement du compteur, défaut 100")
    p_evt.set_defaults(func=cmd_galaxy_events)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
