"""LLM-as-a-judge pairwise evaluation of translation quality.

Shuffles ALL segment×pair combinations (1128 segments × 45 pairs = 50,760),
then judges them one by one in random order.  Because pairs are globally
shuffled, every system combination gets coverage from the start, so win
rates converge fast.

Checkpoints at 100, 200, 300, 400, 500 pairs.  Auto-stops when win rates
stabilise (max Δ < 1pp between consecutive checkpoints).

Usage:
    python src/llm_judge.py                        # run (default: up to 500 pairs)
    python src/llm_judge.py --max-pairs 1000       # extend if needed
    python src/llm_judge.py --checkpoint-every 50  # finer checkpoints
    python src/llm_judge.py --report               # print stats, no API calls

Requirements:
    pip install openai
"""

import argparse
import json
import random
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from openai import OpenAI

# ── Paths ───────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
TRANSLATIONS_PATH = ROOT / "results" / "unite-models" / "parsed_translations.json"
SYSTEM_KEY_PATH = ROOT / "human_eval" / "system_key.json"
RESULTS_DIR = ROOT / "results" / "llm_judge"
RAW_JSONL = RESULTS_DIR / "raw_judgments.jsonl"
CHECKPOINTS_FILE = RESULTS_DIR / "checkpoints.json"

# ── LLM proxy config ───────────────────────────────────────────────

MODEL_NAME = "openai_direct_chat_gpt52"
MAX_TOKENS = 4096
API_TIMEOUT = 120.0

# ── Systems ─────────────────────────────────────────────────────────

EN_KEY = "EN_1945_George_Orwell_Animal_Farm"

HUMAN_SYSTEMS = [
    "UK_1947_Ivan_Cherniatynskyi_Kolhosp_tvaryn",
    "UK_1984_Iryna_Dybko_Khutir_tvaryn",
    "UK_1991_Oleksii_Drozdovskyi_Skotoferma",
    "UK_1991_Yurii_Shevchuk_Ferma_rai_dlia_tvaryn",
    "UK_1992_Natalia_Okolitenko_Skotokhutir",
    "UK_2020_Bohdana_Nosenok_Kolhosp_tvaryn",
    "UK_2021_Viacheslav_Stelmakh_Kolhosp_tvaryn",
]
AI_SYSTEMS = ["gpt_5_2", "lapa_translations_combine", "deepl"]
ALL_SYSTEMS = HUMAN_SYSTEMS + AI_SYSTEMS
SYSTEM_PAIRS = list(combinations(ALL_SYSTEMS, 2))  # 45 ordered pairs

SEED = 42
DEFAULT_MAX_PAIRS = 500
DEFAULT_CHECKPOINT = 100

# ── Prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert literary Ukrainian translator evaluating translation quality.
You will be given one English sentence and two Ukrainian translations.
Choose the better translation.

How to decide:
1. Meaning preservation — Does the translation convey the core meaning and intent \
of the English sentence? Minor additions or omissions are acceptable if they \
improve naturalness or style.
2. Fluency and literary quality — Which translation reads more natural, expressive, \
and appropriate for literary Ukrainian?

Rules:
• Prefer the translation that best balances intent and natural literary expression.
• Use "tie" only if you genuinely cannot decide.
• Judge each sentence independently.
• Ignore punctuation differences unless they affect readability.

Respond with EXACTLY one of these three words: system1, system2, tie"""

USER_TEMPLATE = """\
English: {english}

system1: {system1}

system2: {system2}"""


# ── Helpers ─────────────────────────────────────────────────────────

def get_client():
    return OpenAI(
        api_key="default",
        base_url="https://apigw.dplane.ppgr.io/clapi/api/v1",
        default_headers={
            "X-LLM-Proxy-Calling-Service": "ivan.kulynych@grammarly.com",
        },
        timeout=API_TIMEOUT,
    )


def load_system_key():
    with open(SYSTEM_KEY_PATH) as f:
        return json.load(f)


def load_translations():
    with open(TRANSLATIONS_PATH) as f:
        return json.load(f)["translations"]


def is_valid_segment(text):
    t = text.strip()
    return bool(t) and t != "---" and len(t.split()) >= 3


def get_valid_segments(trans):
    en = trans[EN_KEY]
    valid = []
    for i, seg in enumerate(en):
        if not is_valid_segment(seg):
            continue
        if all(is_valid_segment(trans[s][i]) for s in ALL_SYSTEMS):
            valid.append(i)
    return valid


def build_pair_pool(valid_segments):
    """Build and shuffle all (segment_index, sys_a, sys_b) triples."""
    pool = []
    for seg_idx in valid_segments:
        for sys_a, sys_b in SYSTEM_PAIRS:
            pool.append((seg_idx, sys_a, sys_b))
    rng = random.Random(SEED)
    rng.shuffle(pool)
    return pool


def call_judge(client, english, sys1_text, sys2_text, retries=3):
    """Call LLM and return 'system1', 'system2', or 'tie'."""
    user_msg = USER_TEMPLATE.format(
        english=english, system1=sys1_text, system2=sys2_text,
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=16,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip().lower()
            for valid in ("system1", "system2", "tie"):
                if valid in raw:
                    return valid
            print(f"    [warn] unparseable response: {raw!r}, retrying...")
        except Exception as e:
            print(f"    [err] attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return "error"


def load_existing_judgments():
    """Load previously saved judgments for resume support."""
    done = set()
    judgments = []
    if RAW_JSONL.exists():
        with open(RAW_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)
                judgments.append(j)
                canon = tuple(sorted([j["real_left"], j["real_right"]]))
                done.add((j["segment_index"], canon[0], canon[1]))
    return judgments, done


def append_judgment(record):
    with open(RAW_JSONL, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Aggregate stats ────────────────────────────────────────────────

def compute_stats(judgments):
    """Compute per-system win rates from a list of judgment records."""
    wins = defaultdict(int)
    losses = defaultdict(int)
    ties = defaultdict(int)

    for j in judgments:
        if j["verdict"] == "error":
            continue
        left = j["real_left"]
        right = j["real_right"]

        if j["verdict"] == "system1":
            wins[left] += 1
            losses[right] += 1
        elif j["verdict"] == "system2":
            wins[right] += 1
            losses[left] += 1
        else:
            ties[left] += 1
            ties[right] += 1

    system_stats = {}
    for sys in ALL_SYSTEMS:
        total = wins[sys] + losses[sys] + ties[sys]
        system_stats[sys] = {
            "wins": wins[sys],
            "losses": losses[sys],
            "ties": ties[sys],
            "total": total,
            "win_rate": wins[sys] / total if total else 0,
        }

    n_valid = len([j for j in judgments if j["verdict"] != "error"])
    n_segs = len(set(j["segment_index"] for j in judgments if j["verdict"] != "error"))
    n_pairs_covered = len(set(
        tuple(sorted([j["real_left"], j["real_right"]]))
        for j in judgments if j["verdict"] != "error"
    ))
    return {
        "n_judgments": n_valid,
        "n_segments_touched": n_segs,
        "n_system_pairs_covered": n_pairs_covered,
        "systems": system_stats,
    }


def short_name(s):
    return (
        s.replace("UK_", "")
        .replace("_Kolhosp_tvaryn", "")
        .replace("_Khutir_tvaryn", "")
        .replace("_Skotoferma", "")
        .replace("_Ferma_rai_dlia_tvaryn", "")
        .replace("_Skotokhutir", "")
        .replace("_translations_combine", "")
    )


def print_report(stats, prev_stats=None):
    """Print a concise report, optionally with deltas from previous checkpoint."""
    print(f"\n  Judgments: {stats['n_judgments']}  |  "
          f"Segments touched: {stats['n_segments_touched']}  |  "
          f"System pairs covered: {stats['n_system_pairs_covered']}/45")
    print(f"  {'System':<45} {'Win%':>6} {'W':>5} {'L':>5} {'T':>5}  {'Δ':>7}")
    print(f"  {'-' * 80}")

    ranked = sorted(
        stats["systems"].items(), key=lambda x: -x[1]["win_rate"],
    )
    max_delta = 0.0
    for sys, s in ranked:
        wr = s["win_rate"] * 100
        delta_str = ""
        if prev_stats and sys in prev_stats["systems"]:
            old_wr = prev_stats["systems"][sys]["win_rate"] * 100
            d = wr - old_wr
            max_delta = max(max_delta, abs(d))
            delta_str = f"{d:+.2f}pp"
        tag = " *" if sys in AI_SYSTEMS else ""
        name = short_name(sys)
        print(f"  {name:<45} {wr:5.1f}% {s['wins']:5} {s['losses']:5} {s['ties']:5}  {delta_str:>7}{tag}")

    if prev_stats:
        print(f"\n  Max Δ from previous checkpoint: {max_delta:.2f}pp")
    return max_delta


# ── Main loop ───────────────────────────────────────────────────────

def run_evaluation(max_pairs, checkpoint_every):
    client = get_client()
    trans = load_translations()
    system_key = load_system_key()
    valid_segments = get_valid_segments(trans)

    pool = build_pair_pool(valid_segments)
    pool = pool[:max_pairs]

    print(f"Total possible pairs: {len(valid_segments) * 45}")
    print(f"Pairs to evaluate (max): {len(pool)}")
    print(f"Checkpoint every: {checkpoint_every} pairs")

    judgments, done = load_existing_judgments()
    if done:
        print(f"Resuming: {len(done)} pairs already completed")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prev_stats = None
    checkpoints = []
    n_judged = len(judgments)
    pair_rng = random.Random(SEED + 7)

    for pool_idx, (seg_idx, sys_a, sys_b) in enumerate(pool):
        canon = tuple(sorted([sys_a, sys_b]))
        if (seg_idx, canon[0], canon[1]) in done:
            continue

        en_text = trans[EN_KEY][seg_idx]

        swap = pair_rng.random() < 0.5
        if swap:
            left, right = sys_b, sys_a
        else:
            left, right = sys_a, sys_b

        verdict = call_judge(client, en_text, trans[left][seg_idx], trans[right][seg_idx])

        record = {
            "segment_index": seg_idx,
            "real_left": left,
            "real_right": right,
            "anon_left": system_key.get(left, left),
            "anon_right": system_key.get(right, right),
            "swapped": swap,
            "verdict": verdict,
        }
        append_judgment(record)
        judgments.append(record)
        done.add((seg_idx, canon[0], canon[1]))
        n_judged += 1

        if n_judged % 100 == 0:
            print(f"  [{n_judged}/{len(pool)}]")

        if n_judged % checkpoint_every == 0:
            stats = compute_stats(judgments)
            delta = print_report(stats, prev_stats)

            checkpoints.append({
                "n_judgments": stats["n_judgments"],
                "n_segments_touched": stats["n_segments_touched"],
                "n_system_pairs_covered": stats["n_system_pairs_covered"],
                "max_delta_pp": delta if prev_stats else None,
                "systems": {
                    s: {"win_rate": d["win_rate"], "wins": d["wins"],
                        "losses": d["losses"], "ties": d["ties"]}
                    for s, d in stats["systems"].items()
                },
            })
            with open(CHECKPOINTS_FILE, "w") as f:
                json.dump(checkpoints, f, indent=2, ensure_ascii=False)

            if prev_stats and delta < 1.0 and n_judged >= 200:
                print(f"\n  ** Win rates stabilised (max Δ < 1pp) at {n_judged} pairs **")
                prev_stats = stats
                break

            prev_stats = stats

    # final checkpoint if not already saved
    stats = compute_stats(judgments)
    if not checkpoints or checkpoints[-1]["n_judgments"] != stats["n_judgments"]:
        print_report(stats, prev_stats)
        checkpoints.append({
            "n_judgments": stats["n_judgments"],
            "n_segments_touched": stats["n_segments_touched"],
            "n_system_pairs_covered": stats["n_system_pairs_covered"],
            "max_delta_pp": None,
            "systems": {
                s: {"win_rate": d["win_rate"], "wins": d["wins"],
                    "losses": d["losses"], "ties": d["ties"]}
                for s, d in stats["systems"].items()
            },
        })
        with open(CHECKPOINTS_FILE, "w") as f:
            json.dump(checkpoints, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Raw judgments → {RAW_JSONL}")
    print(f"Checkpoints      → {CHECKPOINTS_FILE}")


def report_only():
    """Print stats from existing results without calling the API."""
    if not RAW_JSONL.exists():
        print(f"No results found at {RAW_JSONL}")
        return

    judgments, _ = load_existing_judgments()
    stats = compute_stats(judgments)
    print_report(stats)

    if CHECKPOINTS_FILE.exists():
        with open(CHECKPOINTS_FILE) as f:
            cps = json.load(f)
        print(f"\n  Checkpoints recorded: {len(cps)}")
        for cp in cps:
            d = f"Δ={cp['max_delta_pp']:.2f}pp" if cp["max_delta_pp"] is not None else "—"
            print(f"    {cp['n_judgments']:>6} pairs  "
                  f"{cp['n_segments_touched']:>4} segs  "
                  f"{cp['n_system_pairs_covered']:>2}/45 sys-pairs  {d}")


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-a-judge pairwise translation evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Globally shuffles all segment×pair combinations, then judges one by one.
Checkpoints at 100, 200, 300, … pairs.  Auto-stops when max Δ < 1pp.

At 500 pairs, each of the 45 system pairs has ~11 judgments on average.
""",
    )
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS,
                        help=f"Max pairs to judge (default: {DEFAULT_MAX_PAIRS})")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT,
                        help=f"Stats every N pairs (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--report", action="store_true",
                        help="Print stats from saved results (no API calls)")
    args = parser.parse_args()

    if args.report:
        report_only()
    else:
        run_evaluation(
            max_pairs=args.max_pairs,
            checkpoint_every=args.checkpoint_every,
        )


if __name__ == "__main__":
    main()
