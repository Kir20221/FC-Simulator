"""
CLI symétrique des endpoints HTTP de ml/api.py.

À chaque endpoint correspond une commande. Utile pour le dev et pour
les opérations admin sans passer par HTTP.

Usage (depuis l'intérieur du conteneur api) :
    docker compose exec api python -m ml.cli datasets create --nom v1 --systemes 1000
    docker compose exec api python -m ml.cli datasets list
    docker compose exec api python -m ml.cli datasets show v1
    docker compose exec api python -m ml.cli datasets delete v1
"""

import argparse
import json
import sys

from app.db.connection import get_conn
from .generator.generate import DatasetGenerationRequest
from .service import (
    DatasetExisteDejaError,
    DatasetIntrouvableError,
    creer_dataset,
    lire_dataset,
    lister_datasets,
    supprimer_dataset,
)


def _print_json(obj) -> None:
    """Affichage JSON-friendly des modèles Pydantic."""
    if hasattr(obj, "model_dump"):
        data = obj.model_dump(mode="json")
    elif isinstance(obj, list):
        data = [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in obj]
    else:
        data = obj
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def cmd_datasets_create(args: argparse.Namespace) -> int:
    request = DatasetGenerationRequest(
        nom=args.nom,
        nb_systemes=args.systemes,
        seed=args.seed,
    )
    try:
        with get_conn() as conn:
            dataset = creer_dataset(conn, request)
        _print_json(dataset)
        return 0
    except DatasetExisteDejaError as e:
        print(f"Erreur : {e}", file=sys.stderr)
        return 1


def cmd_datasets_list(args: argparse.Namespace) -> int:
    with get_conn() as conn:
        datasets = lister_datasets(conn)
    _print_json(datasets)
    return 0


def cmd_datasets_show(args: argparse.Namespace) -> int:
    try:
        with get_conn() as conn:
            dataset = lire_dataset(conn, args.nom)
        _print_json(dataset)
        return 0
    except DatasetIntrouvableError as e:
        print(f"Erreur : {e}", file=sys.stderr)
        return 1


def cmd_datasets_delete(args: argparse.Namespace) -> int:
    try:
        with get_conn() as conn:
            supprimer_dataset(conn, args.nom)
        print(f"Dataset '{args.nom}' supprimé.")
        return 0
    except DatasetIntrouvableError as e:
        print(f"Erreur : {e}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="ml.cli")
    sub = parser.add_subparsers(dest="resource", required=True)

    # --- datasets ---
    p_ds = sub.add_parser("datasets", help="Gestion des datasets d'entraînement.")
    sub_ds = p_ds.add_subparsers(dest="action", required=True)

    p_create = sub_ds.add_parser("create", help="Génère un dataset.")
    p_create.add_argument("--nom", required=True)
    p_create.add_argument("--systemes", type=int, required=True)
    p_create.add_argument("--seed", type=int, default=42)
    p_create.set_defaults(func=cmd_datasets_create)

    p_list = sub_ds.add_parser("list", help="Liste les datasets.")
    p_list.set_defaults(func=cmd_datasets_list)

    p_show = sub_ds.add_parser("show", help="Affiche les métadonnées d'un dataset.")
    p_show.add_argument("nom")
    p_show.set_defaults(func=cmd_datasets_show)

    p_delete = sub_ds.add_parser("delete", help="Supprime un dataset.")
    p_delete.add_argument("nom")
    p_delete.set_defaults(func=cmd_datasets_delete)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
