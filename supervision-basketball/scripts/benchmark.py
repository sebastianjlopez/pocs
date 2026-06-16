"""
benchmark.py — Compara las estadísticas detectadas por el POC con los box scores oficiales.

Uso:
    # Comparar un video específico
    python benchmark.py --stats video_3_uJs693eNfuQ_stats.json --ground-truth ground_truth.json

    # Comparar todos los stats JSON encontrados en el directorio actual
    python benchmark.py --all --ground-truth ground_truth.json

    # Exportar reporte a JSON
    python benchmark.py --all --ground-truth ground_truth.json --output benchmark_report.json

Métricas calculadas (sólo a nivel de equipo):
  - Dominance accuracy: ¿el programa identifica correctamente qué equipo anotó/tiró más?
  - FG% error absoluto: |fg%_programa - fg%_oficial| por equipo (cuando disponible)
  - Shot ratio error: diferencia en la proporción de tiros de cada equipo
  - Basket ratio error: ídem para canastas

Limitaciones:
  - tracker_id ≠ número de camiseta → no se comparan stats individuales
  - Los videos son clips, no partidos completos → los conteos absolutos no coinciden con
    los box scores oficiales del partido entero
  - La asignación de equipos (brillo del jersey) puede estar invertida →
    se prueba ambas alineaciones y se reporta la mejor
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict | list:
    with open(path) as f:
        return json.load(f)


def fg_pct(made: int, attempted: int) -> float | None:
    if attempted and attempted > 0:
        return round(made / attempted * 100, 1)
    return None


def safe_abs_err(a: float | None, b: float | None) -> float | None:
    if a is not None and b is not None:
        return round(abs(a - b), 1)
    return None


def ratio(a: float | None, b: float | None) -> float | None:
    """a / (a + b), returns None if either is None or sum is 0."""
    if a is None or b is None:
        return None
    total = a + b
    if total == 0:
        return None
    return round(a / total, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Core comparison logic
# ──────────────────────────────────────────────────────────────────────────────

def compare(stats_json: dict, gt_entry: dict) -> dict:
    """
    Compare program output against one ground-truth entry.

    Because the program's team A/B may be swapped relative to the official
    team_1/team_2, we try both alignments and pick the one that best matches
    the official winner (by basket count). When neither is clearly better we
    report both.

    Returns a dict with all computed metrics.
    """
    program_teams = stats_json.get("teams", {})
    pa = program_teams.get("A", {})
    pb = program_teams.get("B", {})

    official = gt_entry.get("official")
    if not official:
        return {"skipped": True, "reason": "no_ground_truth"}

    o1 = official["team_1"]
    o2 = official["team_2"]
    official_winner = official.get("winner")  # "team_1" or "team_2"

    # Program-detected counts
    pa_baskets  = pa.get("baskets", 0)
    pb_baskets  = pb.get("baskets", 0)
    pa_shots    = pa.get("shots", 0)
    pb_shots    = pb.get("shots", 0)
    pa_fg_pct   = pa.get("shooting_pct")
    pb_fg_pct   = pb.get("shooting_pct")

    # Official counts
    o1_score    = o1.get("score")
    o2_score    = o2.get("score")
    o1_fga      = o1.get("fga")
    o2_fga      = o2.get("fga")
    o1_fg_pct   = o1.get("fg_pct")
    o2_fg_pct   = o2.get("fg_pct")

    # ── 1. Dominance: did the program pick the right winner? ──
    # "winner" = team with more detected baskets
    if pa_baskets > pb_baskets:
        program_winner = "A_is_team1"   # A maps to team_1
    elif pb_baskets > pa_baskets:
        program_winner = "B_is_team1"   # B maps to team_1
    else:
        program_winner = "tie"

    # Did it match? Try both alignments.
    # Alignment 1: A=team_1, B=team_2
    # Alignment 2: A=team_2, B=team_1
    if official_winner == "team_1":
        dominance_correct_align1 = pa_baskets > pb_baskets
        dominance_correct_align2 = pb_baskets > pa_baskets
    elif official_winner == "team_2":
        dominance_correct_align1 = pb_baskets > pa_baskets
        dominance_correct_align2 = pa_baskets > pb_baskets
    else:
        dominance_correct_align1 = None
        dominance_correct_align2 = None

    basket_dominance_correct = dominance_correct_align1 or dominance_correct_align2

    # ── 2. Shot dominance ──
    if o1_fga is not None and o2_fga is not None:
        official_shot_winner = "team_1" if o1_fga >= o2_fga else "team_2"
    else:
        official_shot_winner = None

    if pa_shots > pb_shots:
        program_shot_winner = "A"
    elif pb_shots > pa_shots:
        program_shot_winner = "B"
    else:
        program_shot_winner = "tie"

    if official_shot_winner == "team_1":
        shot_dominance_correct = pa_shots > pb_shots or pb_shots > pa_shots  # either alignment can be right
        # More specifically:
        shot_dom_align1 = pa_shots >= pb_shots  # A=team_1 shot more
        shot_dom_align2 = pb_shots >= pa_shots  # B=team_1 shot more
        shot_dominance_correct = shot_dom_align1 or shot_dom_align2
    elif official_shot_winner == "team_2":
        shot_dom_align1 = pb_shots >= pa_shots
        shot_dom_align2 = pa_shots >= pb_shots
        shot_dominance_correct = shot_dom_align1 or shot_dom_align2
    else:
        shot_dominance_correct = None

    # ── 3. FG% error (best alignment) ──
    # Try both alignments and pick the one with lower total FG% error
    fg_err_align1 = None
    fg_err_align2 = None

    if pa_fg_pct is not None and pb_fg_pct is not None:
        e1a = safe_abs_err(pa_fg_pct, o1_fg_pct)
        e1b = safe_abs_err(pb_fg_pct, o2_fg_pct)
        e2a = safe_abs_err(pa_fg_pct, o2_fg_pct)
        e2b = safe_abs_err(pb_fg_pct, o1_fg_pct)

        if e1a is not None and e1b is not None:
            fg_err_align1 = round((e1a + e1b) / 2, 1)
        if e2a is not None and e2b is not None:
            fg_err_align2 = round((e2a + e2b) / 2, 1)

    if fg_err_align1 is not None and fg_err_align2 is not None:
        best_fg_err = min(fg_err_align1, fg_err_align2)
        best_alignment = "A=team_1" if fg_err_align1 <= fg_err_align2 else "A=team_2"
    elif fg_err_align1 is not None:
        best_fg_err = fg_err_align1
        best_alignment = "A=team_1"
    elif fg_err_align2 is not None:
        best_fg_err = fg_err_align2
        best_alignment = "A=team_2"
    else:
        best_fg_err = None
        best_alignment = None

    # ── 4. Shot ratio error ──
    prog_shot_ratio_a = ratio(pa_shots, pb_shots)   # A's share of total shots
    off_shot_ratio_1  = ratio(o1_fga, o2_fga)       # team_1's share
    shot_ratio_err = None
    if prog_shot_ratio_a is not None and off_shot_ratio_1 is not None:
        # Try both alignments, take min error
        err1 = abs(prog_shot_ratio_a - off_shot_ratio_1)
        err2 = abs((1 - prog_shot_ratio_a) - off_shot_ratio_1)
        shot_ratio_err = round(min(err1, err2), 3)

    # ── 5. Basket ratio error ──
    prog_basket_ratio_a = ratio(pa_baskets, pb_baskets)
    if o1_score is not None and o2_score is not None:
        off_basket_ratio_1 = ratio(o1_score, o2_score)
    else:
        off_basket_ratio_1 = None

    basket_ratio_err = None
    if prog_basket_ratio_a is not None and off_basket_ratio_1 is not None:
        err1 = abs(prog_basket_ratio_a - off_basket_ratio_1)
        err2 = abs((1 - prog_basket_ratio_a) - off_basket_ratio_1)
        basket_ratio_err = round(min(err1, err2), 3)

    # ── 6. Raw program stats summary ──
    program_summary = {
        "team_A": {
            "shots":        pa_shots,
            "baskets":      pa_baskets,
            "fg_pct":       pa_fg_pct,
            "passes":       pa.get("passes", 0),
            "possessions":  pa.get("possessions", 0),
        },
        "team_B": {
            "shots":        pb_shots,
            "baskets":      pb_baskets,
            "fg_pct":       pb_fg_pct,
            "passes":       pb.get("passes", 0),
            "possessions":  pb.get("possessions", 0),
        },
        "events_total": stats_json.get("events_total", 0),
        "events_by_type": stats_json.get("events_by_type", {}),
    }

    return {
        "skipped": False,
        "video_id": gt_entry["video_id"],
        "game": gt_entry["game"],
        "official_score": f"{gt_entry['team_1_name']} {o1_score} — {gt_entry['team_2_name']} {o2_score}"
            if o1_score is not None else "unknown",
        "program_summary": program_summary,
        "metrics": {
            "basket_dominance_correct": basket_dominance_correct,
            "shot_dominance_correct":   shot_dominance_correct,
            "fg_pct_mae":               best_fg_err,
            "fg_pct_best_alignment":    best_alignment,
            "fg_err_align_A_team1":     fg_err_align1,
            "fg_err_align_A_team2":     fg_err_align2,
            "shot_ratio_err":           shot_ratio_err,
            "basket_ratio_err":         basket_ratio_err,
        },
        "notes": gt_entry.get("notes", ""),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def _yn(value: bool | None) -> str:
    if value is None:
        return "N/A"
    return "SI  [OK]" if value else "NO  [--]"


def _fmt(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value}{suffix}"


def print_report(results: list[dict]) -> None:
    print(f"\n{'='*72}")
    print("  BASKETBALL POC — BENCHMARK VS BOX SCORES OFICIALES")
    print(f"{'='*72}")
    print()
    print("  NOTA: Los videos son clips, no partidos completos.")
    print("  Los conteos absolutos NO se comparan con el box score completo.")
    print("  Sólo se comparan ratios y dominancias relativas.\n")

    for r in results:
        print(f"  {'-'*68}")
        if r.get("skipped"):
            vid = r.get("video_id", "?")
            print(f"  {vid}  →  SKIPPED ({r.get('reason', '')})\n")
            continue

        print(f"  {r['video_id']}")
        print(f"  {r['game']}")
        print(f"  Score oficial: {r['official_score']}")
        print()

        ps = r["program_summary"]
        ta = ps["team_A"]
        tb = ps["team_B"]

        print(f"  {'Programa':>12}   {'Equipo A':>10}   {'Equipo B':>10}")
        print(f"  {'Canastas':>12}   {ta['baskets']:>10}   {tb['baskets']:>10}")
        print(f"  {'Tiros':>12}   {ta['shots']:>10}   {tb['shots']:>10}")
        print(f"  {'FG%':>12}   {_fmt(ta['fg_pct'],'%'):>10}   {_fmt(tb['fg_pct'],'%'):>10}")
        print(f"  {'Poses.':>12}   {ta['possessions']:>10}   {tb['possessions']:>10}")
        print(f"  {'Pases':>12}   {ta['passes']:>10}   {tb['passes']:>10}")
        print(f"  {'Eventos total':>12}   {ps['events_total']:>10}")
        breakdown = ps.get("events_by_type", {})
        if breakdown:
            parts = "  ".join(f"{k}:{v}" for k, v in sorted(breakdown.items()))
            print(f"  {'Desglose':>12}   {parts}")
        print()

        m = r["metrics"]
        print(f"  Dominancia canastas correcta : {_yn(m['basket_dominance_correct'])}")
        print(f"  Dominancia tiros correcta    : {_yn(m['shot_dominance_correct'])}")
        print(f"  Error FG% medio (mejor alin.): {_fmt(m['fg_pct_mae'], ' pp')}  [{m['fg_pct_best_alignment'] or 'N/A'}]")
        print(f"    align A=team_1             : {_fmt(m['fg_err_align_A_team1'], ' pp')}")
        print(f"    align A=team_2             : {_fmt(m['fg_err_align_A_team2'], ' pp')}")
        print(f"  Error ratio tiros            : {_fmt(m['shot_ratio_err'])}")
        print(f"  Error ratio canastas         : {_fmt(m['basket_ratio_err'])}")
        if r.get("notes"):
            print(f"\n  Nota: {r['notes']}")
        print()

    # Aggregate
    valid = [r for r in results if not r.get("skipped")]
    if not valid:
        return

    basket_dom = [r["metrics"]["basket_dominance_correct"] for r in valid
                  if r["metrics"]["basket_dominance_correct"] is not None]
    shot_dom   = [r["metrics"]["shot_dominance_correct"] for r in valid
                  if r["metrics"]["shot_dominance_correct"] is not None]
    fg_errs    = [r["metrics"]["fg_pct_mae"] for r in valid
                  if r["metrics"]["fg_pct_mae"] is not None]
    shot_ratios = [r["metrics"]["shot_ratio_err"] for r in valid
                   if r["metrics"]["shot_ratio_err"] is not None]
    basket_ratios = [r["metrics"]["basket_ratio_err"] for r in valid
                     if r["metrics"]["basket_ratio_err"] is not None]

    print(f"  {'-'*68}")
    print("  RESUMEN AGREGADO")
    print(f"  {'-'*68}")
    print(f"  Videos comparados                : {len(valid)} / {len(results)}")
    if basket_dom:
        pct = sum(basket_dom) / len(basket_dom) * 100
        print(f"  Dominancia canastas (correcta)   : {sum(basket_dom)}/{len(basket_dom)}  ({pct:.0f}%)")
    if shot_dom:
        pct = sum(shot_dom) / len(shot_dom) * 100
        print(f"  Dominancia tiros (correcta)      : {sum(shot_dom)}/{len(shot_dom)}  ({pct:.0f}%)")
    if fg_errs:
        avg = round(sum(fg_errs) / len(fg_errs), 1)
        print(f"  FG% MAE promedio                 : {avg} pp  (n={len(fg_errs)})")
    if shot_ratios:
        avg = round(sum(shot_ratios) / len(shot_ratios), 3)
        print(f"  Error ratio tiros promedio       : {avg}  (n={len(shot_ratios)})")
    if basket_ratios:
        avg = round(sum(basket_ratios) / len(basket_ratios), 3)
        print(f"  Error ratio canastas promedio    : {avg}  (n={len(basket_ratios)})")
    print(f"  {'='*68}\n")



# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark del POC de baloncesto vs box scores oficiales"
    )
    ap.add_argument("--stats",        help="Ruta a un archivo *_stats.json del POC")
    ap.add_argument("--all",          action="store_true",
                    help="Busca todos los *_stats.json en el directorio actual")
    ap.add_argument("--ground-truth", required=True,
                    help="Ruta a ground_truth.json")
    ap.add_argument("--output",       default=None,
                    help="Exportar reporte a JSON (opcional)")
    args = ap.parse_args()

    if not args.stats and not args.all:
        ap.error("Especificá --stats <archivo> o --all")

    gt_list: list[dict] = load_json(args.ground_truth)
    gt_by_id = {entry["video_id"]: entry for entry in gt_list}

    # Collect stats files to process
    stats_files: list[Path] = []
    if args.all:
        stats_files = sorted(Path(".").glob("*_stats.json"))
        if not stats_files:
            print("No se encontraron archivos *_stats.json en el directorio actual.")
            return
    else:
        stats_files = [Path(args.stats)]

    results: list[dict] = []
    for stats_path in stats_files:
        # Derive video_id from filename: "video_3_uJs693eNfuQ_stats.json" → "video_3_uJs693eNfuQ"
        stem = stats_path.stem
        if stem.endswith("_stats"):
            video_id = stem[:-len("_stats")]
        else:
            video_id = stem

        print(f"Procesando {stats_path.name}  (video_id={video_id})")
        stats = load_json(str(stats_path))

        gt_entry = gt_by_id.get(video_id)
        if gt_entry is None:
            print(f"  ⚠  No se encontró entrada en ground_truth.json para '{video_id}' — saltando.")
            results.append({
                "skipped": True,
                "video_id": video_id,
                "reason": "not_in_ground_truth",
            })
            continue

        result = compare(stats, gt_entry)
        results.append(result)

    print_report(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Reporte exportado a: {args.output}")


if __name__ == "__main__":
    main()
