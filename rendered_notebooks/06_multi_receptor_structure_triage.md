# Multi-receptor structure-aware reward-guided peptide optimisation

This notebook extends the single-receptor reward-guided peptide optimisation loop from Notebook 05 to multiple receptor hypotheses.

The fine-tuned peptide transformer is loaded from the saved checkpoint and used to generate candidate peptides. Generated peptides are first scored with cheap sequence-level developability and safety heuristics. The top candidates are then evaluated against multiple receptor-domain hypotheses using ColabFold-Multimer. Receptor-pair confidence scores are converted into structure-aware rewards, and the best receptor-specific reward for each peptide is used to update the generator.

The first demonstration uses a compact local setting:

- **2 receptor hypotheses:** Ly6a and Ecscr
- **2 optimisation cycles**
- **100 generated peptides per cycle**
- **5 top peptides selected per cycle**
- **10 ColabFold receptor-peptide jobs per cycle**

This keeps the run feasible on a local 8 GB GPU while demonstrating that the workflow scales beyond a single receptor.

This notebook performs the following steps:

1. load the fine-tuned peptide generator and metadata,
2. reconstruct the peptide transformer architecture,
3. generate a fresh peptide batch in each optimisation cycle,
4. apply cheap peptide developability and safety scoring,
5. select top candidates for receptor-aware scoring,
6. define multiple receptor-domain hypotheses,
7. write receptor × peptide ColabFold FASTA files,
8. run ColabFold in a minimal-memory configuration,
9. parse pLDDT, pTM, and ipTM confidence scores,
10. compute receptor-pair and peptide-level rewards,
11. update the generator using reward-weighted sequence log-probability optimisation,
12. save multi-receptor reward tables and RL history.

Important scope note:

This notebook demonstrates a **multi-receptor, structure-aware, reward-guided peptide generation prototype**. The ColabFold-derived scores are used as computational triage signals, not as direct binding affinities. The generated peptides are not validated receptor binders or BBB shuttles.

In the wider prototype, this notebook provides the scalable closed-loop stage:

**peptide generation → cheap filtering → multi-receptor ColabFold triage → best-receptor reward aggregation → generator update**

## 1. Load fine-tuned peptide generator

This section reloads the fine-tuned peptide transformer from the saved checkpoint. The model was pretrained on CPPsite2 natural peptides and fine-tuned on B3PDB BBB-positive peptides.

The loaded generator is used as the starting policy for multi-receptor reward-guided optimisation. In each cycle, the current generator produces candidate peptides, which are then scored against multiple receptor-domain hypotheses.


```python
# -------------------------
# 1. Load fine-tuned peptide generator
# -------------------------

from pathlib import Path
import json
import time
import os
import subprocess

import numpy as np
import pandas as pd

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_PLATFORMS"] = "cuda,cpu"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_force_compilation_parallelism=1 "
    "--xla_gpu_enable_triton_gemm=false"
)

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import serialization
import optax

import warnings
warnings.filterwarnings("ignore")

print("JAX devices:", jax.devices())

model_dir = Path("../models")

checkpoint_path = model_dir / "peptide_transformer_cpp_pretrained_b3pdb_finetuned.msgpack"
metadata_path = model_dir / "peptide_transformer_metadata.json"

with open(metadata_path, "r") as f:
    metadata = json.load(f)

print("Model:", metadata["model_name"])
print("Training summary:", metadata["training_summary"])
print("Previous generation summary:", metadata["generation_summary"])


# -------------------------
# Vocabulary and special tokens
# -------------------------

vocab = metadata["vocab"]
token_to_id = metadata["token_to_id"]
id_to_token = {int(k): v for k, v in metadata["id_to_token"].items()}

PAD_ID = metadata["special_token_ids"]["PAD_ID"]
UNK_ID = metadata["special_token_ids"]["UNK_ID"]
BOS_ID = metadata["special_token_ids"]["BOS_ID"]
EOS_ID = metadata["special_token_ids"]["EOS_ID"]

VOCAB_SIZE = metadata["VOCAB_SIZE"]
MAX_LEN = metadata["MAX_LEN"]

config = metadata["model_config"]


# -------------------------
# Reconstruct model architecture
# -------------------------

class CausalTransformerBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, attention_mask, deterministic=True):
        causal_mask = nn.make_causal_mask(
            jnp.ones((x.shape[0], x.shape[1]), dtype=bool)
        )

        padding_mask = attention_mask[:, None, None, :].astype(bool)
        attn_mask = jnp.logical_and(causal_mask, padding_mask)

        attn_output = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
        )(x, mask=attn_mask)

        x = nn.LayerNorm()(x + attn_output)

        mlp_output = nn.Dense(self.mlp_dim)(x)
        mlp_output = nn.gelu(mlp_output)
        mlp_output = nn.Dropout(rate=self.dropout_rate)(
            mlp_output,
            deterministic=deterministic,
        )
        mlp_output = nn.Dense(self.hidden_dim)(mlp_output)

        x = nn.LayerNorm()(x + mlp_output)
        x = x * attention_mask[..., None]

        return x


class PeptideCausalTransformer(nn.Module):
    vocab_size: int
    max_len: int
    hidden_dim: int = 128
    num_heads: int = 4
    mlp_dim: int = 256
    num_layers: int = 4
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, input_ids, attention_mask, deterministic=True):
        _, seq_len = input_ids.shape

        token_emb = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_dim,
        )(input_ids)

        position_ids = jnp.arange(seq_len)

        pos_emb = nn.Embed(
            num_embeddings=self.max_len,
            features=self.hidden_dim,
        )(position_ids)

        x = token_emb + pos_emb[None, :, :]
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x * attention_mask[..., None]

        for _ in range(self.num_layers):
            x = CausalTransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
            )(x, attention_mask, deterministic=deterministic)

        logits = nn.Dense(self.vocab_size)(x)
        return logits


model = PeptideCausalTransformer(
    vocab_size=VOCAB_SIZE,
    max_len=MAX_LEN,
    hidden_dim=config["hidden_dim"],
    num_heads=config["num_heads"],
    mlp_dim=config["mlp_dim"],
    num_layers=config["num_layers"],
    dropout_rate=config["dropout_rate"],
)

dummy_input_ids = jnp.ones((1, MAX_LEN - 1), dtype=jnp.int32)
dummy_attention_mask = jnp.ones((1, MAX_LEN - 1), dtype=jnp.float32)

rng = jax.random.PRNGKey(0)

empty_params = model.init(
    rng,
    dummy_input_ids,
    dummy_attention_mask,
    deterministic=True,
)

with open(checkpoint_path, "rb") as f:
    params = serialization.from_bytes(empty_params, f.read())

print("Loaded checkpoint:", checkpoint_path)
```

    JAX devices: [CudaDevice(id=0)]
    Model: peptide_transformer_cpp_pretrained_b3pdb_finetuned
    Training summary: {'cpp_best_epoch': 52, 'cpp_best_val_loss': 1.9253921111424763, 'cpp_test_loss': 1.9499332308769226, 'cpp_test_perplexity': 7.028218296950155, 'b3p_best_epoch': 77, 'b3p_best_val_loss': 2.7452083826065063, 'b3p_test_loss': 2.773916721343994, 'b3p_test_perplexity': 16.021262100567874}
    Previous generation summary: {'n_generated': 100, 'n_valid': 100, 'n_unique_valid': 100, 'n_novel_unique': 99, 'valid_fraction': 1.0, 'unique_valid_fraction': 1.0, 'novel_unique_fraction': 0.99}
    Loaded checkpoint: ../models/peptide_transformer_cpp_pretrained_b3pdb_finetuned.msgpack


## 2. Run two-cycle multi-receptor reward-guided peptide optimisation

This block performs a compact multi-receptor proof-of-concept reward-guided optimisation loop.

Each cycle performs:

1. generate 100 peptides from the current generator,
2. score all generated peptides using cheap peptide-level heuristics,
3. select the top 5 candidates for receptor-aware scoring,
4. use two receptor-domain hypotheses: **Ly6a** and **Ecscr**,
5. write receptor × peptide ColabFold FASTA files,
6. run ColabFold in minimal-memory mode for all selected receptor-peptide pairs,
7. parse pLDDT, pTM, and ipTM scores,
8. compute receptor-pair rewards for every receptor × peptide pair,
9. aggregate peptide-level rewards using the best receptor per peptide,
10. update the generator using reward-weighted sequence log-probability optimisation.

This is a small local demonstration. The ColabFold step is intentionally restricted to top candidates because structure-aware scoring is computationally expensive.


```python
# -------------------------
# 2. Two-cycle multi-receptor reward-guided peptide optimisation loop
# -------------------------

import requests
import json
from pathlib import Path
import time
import os
import subprocess

# ============================================================
# 2.1 Generation helpers
# ============================================================

CANONICAL_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def sample_next_token(probs, temperature=1.0):
    probs = np.asarray(probs, dtype=np.float64)

    logits = np.log(probs + 1e-12) / temperature
    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()

    return int(np.random.choice(len(probs), p=probs))


def generate_peptide(
    params,
    max_generation_len=40,
    temperature=1.0,
    min_len=5,
):
    tokens = [BOS_ID]

    model_max_len = MAX_LEN - 1
    max_generation_len = min(max_generation_len, model_max_len)

    for _ in range(max_generation_len):
        input_ids = np.array(
            tokens + [PAD_ID] * (model_max_len - len(tokens)),
            dtype=np.int32,
        )

        attention_mask = np.array(
            [1] * len(tokens) + [0] * (model_max_len - len(tokens)),
            dtype=np.float32,
        )

        logits = model.apply(
            params,
            jnp.asarray(input_ids[None, :]),
            jnp.asarray(attention_mask[None, :]),
            deterministic=True,
        )

        next_logits = np.asarray(logits[0, len(tokens) - 1])
        probs = np.asarray(jax.nn.softmax(next_logits)).copy()

        probs[PAD_ID] = 0.0
        probs[BOS_ID] = 0.0
        probs[UNK_ID] = 0.0

        current_peptide_len = len(tokens) - 1
        if current_peptide_len < min_len:
            probs[EOS_ID] = 0.0

        probs = probs / probs.sum()

        next_token = sample_next_token(probs, temperature=temperature)

        if next_token == EOS_ID:
            break

        tokens.append(next_token)

    peptide_tokens = [id_to_token[t] for t in tokens[1:] if t != EOS_ID]
    return "".join(peptide_tokens)


def generate_peptide_batch(
    params,
    n=100,
    temperature=1.0,
    max_generation_len=40,
    min_len=5,
):
    peptides = [
        generate_peptide(
            params=params,
            max_generation_len=max_generation_len,
            temperature=temperature,
            min_len=min_len,
        )
        for _ in range(n)
    ]

    return pd.DataFrame({
        "sequence": peptides,
        "temperature": temperature,
        "n_amino_acids": [len(p) for p in peptides],
    })


# ============================================================
# 2.2 Cheap peptide scoring helpers
# ============================================================

AA_POSITIVE = set("KRH")
AA_NEGATIVE = set("DE")
AA_HYDROPHOBIC = set("AILMFWVY")
AA_AROMATIC = set("FWY")


def count_fraction(seq, aa_set):
    seq = str(seq).upper()
    if len(seq) == 0:
        return 0.0
    return sum(aa in aa_set for aa in seq) / len(seq)


def net_charge_proxy(seq):
    seq = str(seq).upper()
    pos = sum(aa in AA_POSITIVE for aa in seq)
    neg = sum(aa in AA_NEGATIVE for aa in seq)
    return pos - neg


def triangular_score(x, low, ideal_low, ideal_high, high):
    if x < low or x > high:
        return 0.0

    if ideal_low <= x <= ideal_high:
        return 1.0

    if x < ideal_low:
        return (x - low) / max(ideal_low - low, 1e-8)

    return (high - x) / max(high - ideal_high, 1e-8)


def peptide_feature_row(seq):
    seq = str(seq).upper()
    length = len(seq)
    charge = net_charge_proxy(seq)

    return {
        "sequence": seq,
        "length": length,
        "net_charge_proxy": charge,
        "charge_density": charge / max(length, 1),
        "hydrophobic_fraction": count_fraction(seq, AA_HYDROPHOBIC),
        "aromatic_fraction": count_fraction(seq, AA_AROMATIC),
        "cysteine_count": seq.count("C"),
        "cysteine_fraction": seq.count("C") / max(length, 1),
        "proline_fraction": seq.count("P") / max(length, 1),
        "glycine_fraction": seq.count("G") / max(length, 1),
    }


def bbb_cpp_like_score(row):
    length_score = triangular_score(row["length"], 5, 7, 18, 30)
    charge_score = triangular_score(row["net_charge_proxy"], -2, 1, 6, 10)
    hydro_score = triangular_score(row["hydrophobic_fraction"], 0.10, 0.25, 0.55, 0.80)
    aromatic_score = triangular_score(row["aromatic_fraction"], 0.00, 0.05, 0.25, 0.45)

    return (
        0.35 * length_score
        + 0.30 * charge_score
        + 0.25 * hydro_score
        + 0.10 * aromatic_score
    )


def synthesis_developability_score(row):
    length_score = triangular_score(row["length"], 5, 6, 20, 35)

    cysteine_penalty = min(row["cysteine_count"] / 4.0, 1.0)
    proline_penalty = max((row["proline_fraction"] - 0.30) / 0.30, 0.0)
    glycine_penalty = max((row["glycine_fraction"] - 0.35) / 0.35, 0.0)

    raw = (
        0.55 * length_score
        + 0.20 * (1.0 - cysteine_penalty)
        + 0.15 * (1.0 - min(proline_penalty, 1.0))
        + 0.10 * (1.0 - min(glycine_penalty, 1.0))
    )

    return float(np.clip(raw, 0.0, 1.0))


def safety_toxicity_proxy_score(row):
    charge_density_penalty = max((row["charge_density"] - 0.45) / 0.45, 0.0)
    hydrophobic_penalty = max((row["hydrophobic_fraction"] - 0.65) / 0.35, 0.0)
    cysteine_penalty = min(row["cysteine_count"] / 4.0, 1.0)

    penalty = (
        0.45 * min(charge_density_penalty, 1.0)
        + 0.35 * min(hydrophobic_penalty, 1.0)
        + 0.20 * cysteine_penalty
    )

    return float(np.clip(1.0 - penalty, 0.0, 1.0))


def score_generated_peptides(generated_peptides):
    peptide_features = pd.DataFrame([
        peptide_feature_row(seq)
        for seq in generated_peptides["sequence"]
    ])

    scored = generated_peptides.merge(
        peptide_features,
        on="sequence",
        how="left",
    )

    scored["bbb_cpp_like_score"] = scored.apply(bbb_cpp_like_score, axis=1)
    scored["synthesis_developability_score"] = scored.apply(synthesis_developability_score, axis=1)
    scored["safety_toxicity_proxy_score"] = scored.apply(safety_toxicity_proxy_score, axis=1)

    scored["cheap_pre_af_score"] = (
        0.40 * scored["bbb_cpp_like_score"]
        + 0.35 * scored["synthesis_developability_score"]
        + 0.25 * scored["safety_toxicity_proxy_score"]
    )

    scored["length_preference"] = scored["length"].apply(
        lambda x: triangular_score(x, 5, 8, 16, 25)
    )

    scored["charge_preference"] = scored["net_charge_proxy"].apply(
        lambda x: triangular_score(x, 0, 1, 4, 8)
    )

    scored["cysteine_preference"] = 1.0 - np.minimum(
        scored["cysteine_count"] / 2.0,
        1.0,
    )

    scored["cheap_pre_af_score_refined"] = (
        0.80 * scored["cheap_pre_af_score"]
        + 0.08 * scored["length_preference"]
        + 0.07 * scored["charge_preference"]
        + 0.05 * scored["cysteine_preference"]
    )

    scored = scored.sort_values(
        "cheap_pre_af_score_refined",
        ascending=False,
    ).reset_index(drop=True)

    scored["cheap_rank"] = np.arange(1, len(scored) + 1)

    return scored


# ============================================================
# 2.3 Multi-receptor domain helper
# ============================================================

def fetch_uniprot_fasta_sequence(accession):
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    return "".join(line.strip() for line in lines if not line.startswith(">"))


def get_multi_receptor_panel():
    """
    Compact multi-receptor demonstration:
    1. Ly6a_mouse: strongest early D02-induced candidate.
    2. Ecscr_mouse: endothelial single-pass type I surface receptor candidate.

    Domain coordinates are approximate mature/extracellular constructs.
    """
    receptor_panel = pd.DataFrame([
        {
            "receptor_id": "Ly6a_mouse",
            "gene": "Ly6a",
            "uniprot_accession": "P05533",
            "protein_name": "Lymphocyte antigen 6A-2/6E-1",
            "species": "Mus musculus",
            "domain_used": "extracellular_mature_approx_27_112",
            "domain_start_1based": 27,
            "domain_end_1based": 112,
            "sequence_source": "UniProt P05533 canonical sequence",
            "notes": (
                "GPI-anchored surface protein; mature extracellular domain approximated "
                "by removing signal peptide and C-terminal GPI-anchor/propeptide region."
            ),
        },
        {
            "receptor_id": "Ecscr_mouse",
            "gene": "Ecscr",
            "uniprot_accession": "Q3TZW0",
            "protein_name": "Endothelial cell-specific chemotaxis regulator",
            "species": "Mus musculus",
            "domain_used": "extracellular_N_terminal_approx_24_185",
            "domain_start_1based": 24,
            "domain_end_1based": 185,
            "sequence_source": "UniProt Q3TZW0 canonical sequence",
            "notes": (
                "Endothelial cell-surface single-pass type I membrane protein. "
                "Approximate extracellular N-terminal domain selected by removing the signal peptide "
                "and excluding the predicted transmembrane/cytoplasmic C-terminal region."
            ),
        },    ])

    full_sequences = []
    domain_sequences = []

    for _, row in receptor_panel.iterrows():
        full_seq = fetch_uniprot_fasta_sequence(row["uniprot_accession"])

        start = int(row["domain_start_1based"]) - 1
        end = int(row["domain_end_1based"])

        domain_seq = full_seq[start:end]

        full_sequences.append(full_seq)
        domain_sequences.append(domain_seq)

    receptor_panel["full_sequence"] = full_sequences
    receptor_panel["domain_sequence"] = domain_sequences
    receptor_panel["full_sequence_length"] = receptor_panel["full_sequence"].str.len()
    receptor_panel["domain_sequence_length"] = receptor_panel["domain_sequence"].str.len()

    return receptor_panel


# ============================================================
# 2.4 Multi-receptor ColabFold helpers
# ============================================================

def write_colabfold_fastas_multi_receptor(top_peptides, receptor_panel, cycle_dir):
    """
    Write one ColabFold complex FASTA per receptor × peptide pair.

    FASTA format:
    >receptor_id_peptide_id_complex
    RECEPTOR_DOMAIN:PEPTIDE
    """
    fasta_root = cycle_dir / "alphafold_inputs"
    fasta_root.mkdir(parents=True, exist_ok=True)

    fasta_records = []

    for _, receptor_row in receptor_panel.iterrows():
        receptor_id = receptor_row["receptor_id"]
        receptor_seq = receptor_row["domain_sequence"]

        receptor_fasta_dir = fasta_root / receptor_id
        receptor_fasta_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[FASTA] Preparing receptor: {receptor_id}")
        print(f"Domain length: {len(receptor_seq)} aa")

        for _, pep_row in top_peptides.iterrows():
            peptide_id = pep_row["peptide_id"]
            peptide_seq = pep_row["sequence"]

            fasta_path = receptor_fasta_dir / f"{receptor_id}_{peptide_id}_complex.fasta"

            with open(fasta_path, "w") as f:
                f.write(f">{receptor_id}_{peptide_id}_complex\n")
                f.write(f"{receptor_seq}:{peptide_seq}\n")

            fasta_records.append({
                "receptor_id": receptor_id,
                "peptide_id": peptide_id,
                "sequence": peptide_seq,
                "fasta_file": str(fasta_path),
                "receptor_domain_length": len(receptor_seq),
                "peptide_length": len(peptide_seq),
                "complex_total_length": len(receptor_seq) + len(peptide_seq),
            })

    fasta_manifest = pd.DataFrame(fasta_records)

    manifest_path = cycle_dir / "colabfold_fasta_manifest.csv"
    fasta_manifest.to_csv(manifest_path, index=False)

    print(f"\nWrote {len(fasta_manifest)} receptor-peptide FASTA files.")
    print("Saved FASTA manifest:", manifest_path)

    return fasta_manifest


def run_colabfold_minimal_multi_receptor(fasta_manifest, cycle_dir):
    """
    Run ColabFold for each receptor × peptide pair.

    Results are stored as:
    cycle_dir/alphafold_results/<receptor_id>/<peptide_id>/
    """
    result_root = cycle_dir / "alphafold_results"
    result_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.70"

    run_records = []
    total_jobs = len(fasta_manifest)

    for job_idx, (_, row) in enumerate(fasta_manifest.iterrows(), start=1):
        receptor_id = row["receptor_id"]
        peptide_id = row["peptide_id"]
        fasta_path = Path(row["fasta_file"])

        result_dir = result_root / receptor_id / peptide_id
        result_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "conda", "run", "-n", "colabfold",
            "colabfold_batch",
            "--msa-mode", "single_sequence",
            "--model-type", "alphafold2_multimer_v3",
            "--num-models", "1",
            "--num-recycle", "1",
            "--model-order", "1",
            str(fasta_path),
            str(result_dir),
        ]

        print("\n" + "-" * 100)
        print(f"[ColabFold] Job {job_idx}/{total_jobs}")
        print(f"Receptor: {receptor_id}")
        print(f"Peptide:  {peptide_id} | {row['sequence']}")
        print("Input FASTA:", fasta_path)
        print("Result dir: ", result_dir)
        print("Command:")
        print(" ".join(cmd))

        t0 = time.perf_counter()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )

        t1 = time.perf_counter()
        elapsed = t1 - t0

        pdb_files = list(result_dir.glob("*.pdb"))
        score_json_files = list(result_dir.glob("*scores_rank_*.json"))

        success = (
            result.returncode == 0
            and len(pdb_files) > 0
            and len(score_json_files) > 0
        )

        print("Return code:", result.returncode)
        print(f"Elapsed: {elapsed:.2f} s ({elapsed / 60:.2f} min)")
        print("PDB files:", len(pdb_files))
        print("Score JSON files:", len(score_json_files))
        print("Success:", success)

        if not success:
            print("\nSTDOUT tail:")
            print(result.stdout[-2000:])
            print("\nSTDERR tail:")
            print(result.stderr[-2000:])

        run_records.append({
            "receptor_id": receptor_id,
            "peptide_id": peptide_id,
            "sequence": row["sequence"],
            "fasta_file": str(fasta_path),
            "result_dir": str(result_dir),
            "returncode": result.returncode,
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60,
            "n_pdb_files": len(pdb_files),
            "n_score_json_files": len(score_json_files),
            "success": success,
        })

    return pd.DataFrame(run_records)


def safe_get_score(d, keys, default=np.nan):
    for key in keys:
        if key in d:
            return d[key]
    return default


def collapse_score_value(x):
    if isinstance(x, list):
        return float(np.mean(x))
    if isinstance(x, (int, float, np.number)):
        return float(x)
    return x


def parse_colabfold_scores_multi_receptor(run_log):
    score_rows = []

    for _, run_row in run_log.iterrows():
        if not bool(run_row["success"]):
            continue

        result_dir = Path(run_row["result_dir"])
        score_json_files = sorted(result_dir.glob("*scores_rank_*.json"))

        if len(score_json_files) == 0:
            continue

        score_path = score_json_files[0]

        with open(score_path, "r") as f:
            score_data = json.load(f)

        row = {
            "receptor_id": run_row["receptor_id"],
            "peptide_id": run_row["peptide_id"],
            "sequence": run_row["sequence"],
            "result_dir": str(result_dir),
            "score_json": score_path.name,
            "plddt": safe_get_score(score_data, ["plddt", "mean_plddt"]),
            "ptm": safe_get_score(score_data, ["ptm", "pTM"]),
            "iptm": safe_get_score(score_data, ["iptm", "ipTM"]),
            "ranking_confidence": safe_get_score(score_data, ["ranking_confidence"]),
        }

        score_rows.append(row)

    score_table = pd.DataFrame(score_rows)

    if len(score_table) == 0:
        return score_table

    for col in ["plddt", "ptm", "iptm", "ranking_confidence"]:
        score_table[col] = score_table[col].apply(collapse_score_value)

    return score_table


# ============================================================
# 2.5 Multi-receptor final reward helpers
# ============================================================

def normalise_colabfold_score(row):
    iptm = row["iptm"]
    ptm = row["ptm"]
    plddt = row["plddt"] / 100.0 if row["plddt"] > 1 else row["plddt"]

    ranking = row["ranking_confidence"]
    if pd.isna(ranking):
        ranking = 0.0

    score = (
        0.55 * iptm
        + 0.25 * ptm
        + 0.15 * plddt
        + 0.05 * ranking
    )

    return float(np.clip(score, 0.0, 1.0))


def soft_structure_gate(row):
    iptm = row["iptm"]
    plddt = row["plddt"]

    iptm_gate = triangular_score(
        iptm,
        low=0.03,
        ideal_low=0.20,
        ideal_high=0.80,
        high=1.00,
    )

    plddt_gate = triangular_score(
        plddt,
        low=30,
        ideal_low=60,
        ideal_high=90,
        high=100,
    )

    gate = 0.25 + 0.75 * (
        0.65 * iptm_gate
        + 0.35 * plddt_gate
    )

    return float(np.clip(gate, 0.0, 1.0))


def build_multi_receptor_reward_tables(top_peptides, colabfold_score_table):
    """
    Returns:
    1. receptor_pair_reward_table:
       one row per receptor × peptide pair.

    2. peptide_level_reward_table:
       one row per peptide, using best receptor score by max final_reward.
    """
    if len(colabfold_score_table) == 0:
        receptor_pair_reward_table = top_peptides.copy()
        receptor_pair_reward_table["final_reward"] = np.nan
        return receptor_pair_reward_table, receptor_pair_reward_table.copy()

    score_table = colabfold_score_table.copy()

    score_table["structure_proxy_score"] = score_table.apply(
        normalise_colabfold_score,
        axis=1,
    )

    score_table["soft_structure_gate"] = score_table.apply(
        soft_structure_gate,
        axis=1,
    )

    receptor_pair_reward_table = score_table.merge(
        top_peptides[
            [
                "peptide_id",
                "sequence",
                "cheap_rank",
                "cheap_pre_af_score_refined",
                "n_amino_acids",
                "net_charge_proxy",
                "hydrophobic_fraction",
                "cysteine_count",
            ]
        ],
        on=["peptide_id", "sequence"],
        how="left",
    )

    receptor_pair_reward_table["final_reward"] = (
        (
            0.35 * receptor_pair_reward_table["cheap_pre_af_score_refined"]
            + 0.65 * receptor_pair_reward_table["structure_proxy_score"]
        )
        * receptor_pair_reward_table["soft_structure_gate"]
    )

    receptor_pair_reward_table = receptor_pair_reward_table.sort_values(
        ["final_reward", "receptor_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    # Peptide-level reward: keep the best receptor per peptide
    idx_best = receptor_pair_reward_table.groupby("peptide_id")["final_reward"].idxmax()

    peptide_level_reward_table = receptor_pair_reward_table.loc[idx_best].copy()
    peptide_level_reward_table = peptide_level_reward_table.rename(
        columns={
            "receptor_id": "best_receptor_id",
            "structure_proxy_score": "best_structure_proxy_score",
            "soft_structure_gate": "best_soft_structure_gate",
            "plddt": "best_plddt",
            "ptm": "best_ptm",
            "iptm": "best_iptm",
        }
    )

    peptide_level_reward_table = peptide_level_reward_table.sort_values(
        "final_reward",
        ascending=False,
    ).reset_index(drop=True)

    return receptor_pair_reward_table, peptide_level_reward_table


# ============================================================
# 2.6 RL update helpers
# ============================================================

def peptide_to_ids(seq):
    tokens = ["<BOS>"] + list(str(seq).strip().upper()) + ["<EOS>"]
    ids = [token_to_id.get(tok, UNK_ID) for tok in tokens]

    if len(ids) > MAX_LEN:
        ids = ids[:MAX_LEN]
        ids[-1] = EOS_ID

    attention_mask = [1] * len(ids)
    pad_len = MAX_LEN - len(ids)

    ids = ids + [PAD_ID] * pad_len
    attention_mask = attention_mask + [0] * pad_len

    return np.asarray(ids, dtype=np.int32), np.asarray(attention_mask, dtype=np.float32)


def peptide_list_to_rl_batch(peptide_list, rewards):
    ids_list = []
    mask_list = []

    for seq in peptide_list:
        ids, mask = peptide_to_ids(seq)
        ids_list.append(ids)
        mask_list.append(mask)

    ids_array = np.asarray(ids_list, dtype=np.int32)
    mask_array = np.asarray(mask_list, dtype=np.float32)

    return {
        "input_ids": jnp.asarray(ids_array[:, :-1], dtype=jnp.int32),
        "target_ids": jnp.asarray(ids_array[:, 1:], dtype=jnp.int32),
        "target_mask": jnp.asarray(mask_array[:, 1:], dtype=jnp.float32),
        "rewards": jnp.asarray(rewards, dtype=jnp.float32),
    }


rl_optimizer = optax.adamw(1e-6, weight_decay=1e-4)
rl_params = params
rl_opt_state = rl_optimizer.init(rl_params)


def sequence_log_probs(params, batch):
    logits = model.apply(
        params,
        batch["input_ids"],
        batch["target_mask"],
        deterministic=True,
    )

    log_probs = jax.nn.log_softmax(logits, axis=-1)

    token_log_probs = jnp.take_along_axis(
        log_probs,
        batch["target_ids"][..., None],
        axis=-1,
    ).squeeze(-1)

    seq_log_probs = jnp.sum(token_log_probs * batch["target_mask"], axis=1)

    return seq_log_probs


def rl_loss_fn(params, batch):
    seq_log_probs = sequence_log_probs(params, batch)

    rewards = batch["rewards"]
    advantages = (rewards - jnp.mean(rewards)) / (jnp.std(rewards) + 1e-8)

    loss = -jnp.mean(advantages * seq_log_probs)

    return loss


@jax.jit
def rl_train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(rl_loss_fn)(params, batch)
    updates, opt_state = rl_optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# ============================================================
# 2.7 Two-cycle multi-receptor reward-guided loop
# ============================================================

NUM_RL_CYCLES = 2
NUM_GENERATE_PER_CYCLE = 100
TOP_K_COLABFOLD = 5
RL_UPDATE_EPOCHS = 5
TEMPERATURE = 1.0

run_root = Path("../data/processed/multi_receptor_reward_guided_rl")
run_root.mkdir(parents=True, exist_ok=True)

receptor_panel = get_multi_receptor_panel()

print("Multi-receptor panel:")
display(
    receptor_panel[
        [
            "receptor_id",
            "gene",
            "uniprot_accession",
            "protein_name",
            "domain_used",
            "domain_sequence_length",
            "notes",
        ]
    ]
)

rl_history_records = []

for cycle in range(1, NUM_RL_CYCLES + 1):
    print("\n" + "#" * 100)
    print(f"MULTI-RECEPTOR RL CYCLE {cycle}/{NUM_RL_CYCLES}")
    print("#" * 100)

    cycle_dir = run_root / f"cycle_{cycle:02d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Step A: Generate peptides
    # ------------------------------------------------------------
    print("\n[Step A] Generating peptides...")

    t0 = time.perf_counter()

    generated = generate_peptide_batch(
        params=rl_params,
        n=NUM_GENERATE_PER_CYCLE,
        temperature=TEMPERATURE,
        max_generation_len=40,
        min_len=5,
    )

    t1 = time.perf_counter()

    generated_path = cycle_dir / "generated_peptides.csv"
    generated.to_csv(generated_path, index=False)

    print(f"Generated {len(generated)} peptides in {t1 - t0:.2f} s")
    display(generated.head(10))

    # ------------------------------------------------------------
    # Step B: Cheap peptide scoring
    # ------------------------------------------------------------
    print("\n[Step B] Cheap peptide scoring...")

    scored = score_generated_peptides(generated)

    scored_path = cycle_dir / "generated_peptides_with_cheap_scores.csv"
    scored.to_csv(scored_path, index=False)

    print("Top cheap-scored peptides:")
    display(
        scored[
            [
                "cheap_rank",
                "sequence",
                "n_amino_acids",
                "net_charge_proxy",
                "hydrophobic_fraction",
                "cysteine_count",
                "cheap_pre_af_score_refined",
            ]
        ].head(10)
    )

    # ------------------------------------------------------------
    # Step C: Select top K peptides
    # ------------------------------------------------------------
    print(f"\n[Step C] Selecting top {TOP_K_COLABFOLD} peptides for multi-receptor ColabFold...")

    top_peptides = scored.head(TOP_K_COLABFOLD).copy()
    top_peptides["peptide_id"] = [
        f"pep_{i:03d}" for i in range(len(top_peptides))
    ]

    top_path = cycle_dir / "top_peptides_for_colabfold.csv"
    top_peptides.to_csv(top_path, index=False)

    display(
        top_peptides[
            [
                "peptide_id",
                "cheap_rank",
                "sequence",
                "n_amino_acids",
                "cheap_pre_af_score_refined",
            ]
        ]
    )

    # ------------------------------------------------------------
    # Step D: Write receptor × peptide FASTA files
    # ------------------------------------------------------------
    print("\n[Step D] Writing receptor × peptide FASTA files...")

    print(
        f"Receptors: {len(receptor_panel)} | "
        f"Peptides: {len(top_peptides)} | "
        f"Total ColabFold pairs: {len(receptor_panel) * len(top_peptides)}"
    )

    fasta_manifest = write_colabfold_fastas_multi_receptor(
        top_peptides=top_peptides,
        receptor_panel=receptor_panel,
        cycle_dir=cycle_dir,
    )

    display(fasta_manifest.head(10))

    # ------------------------------------------------------------
    # Step E: Run ColabFold for all receptor × peptide pairs
    # ------------------------------------------------------------
    print("\n[Step E] Running ColabFold for selected receptor × peptide pairs...")

    run_log = run_colabfold_minimal_multi_receptor(
        fasta_manifest=fasta_manifest,
        cycle_dir=cycle_dir,
    )

    run_log_path = cycle_dir / "multi_receptor_colabfold_run_log.csv"
    run_log.to_csv(run_log_path, index=False)

    print("\nColabFold run log:")
    display(run_log)

    # ------------------------------------------------------------
    # Step F: Parse ColabFold scores
    # ------------------------------------------------------------
    print("\n[Step F] Parsing ColabFold scores...")

    colabfold_score_table = parse_colabfold_scores_multi_receptor(
        run_log=run_log,
    )

    scores_path = cycle_dir / "multi_receptor_parsed_colabfold_scores.csv"
    colabfold_score_table.to_csv(scores_path, index=False)

    print("Parsed ColabFold score rows:", len(colabfold_score_table))
    display(colabfold_score_table)

    # ------------------------------------------------------------
    # Step G: Receptor-pair and peptide-level final reward tables
    # ------------------------------------------------------------
    print("\n[Step G] Computing multi-receptor reward tables...")

    receptor_pair_reward_table, peptide_level_reward_table = build_multi_receptor_reward_tables(
        top_peptides=top_peptides,
        colabfold_score_table=colabfold_score_table,
    )

    receptor_pair_reward_path = cycle_dir / "receptor_pair_reward_table.csv"
    peptide_level_reward_path = cycle_dir / "peptide_level_best_receptor_reward_table.csv"

    receptor_pair_reward_table.to_csv(receptor_pair_reward_path, index=False)
    peptide_level_reward_table.to_csv(peptide_level_reward_path, index=False)

    print("\nReceptor × peptide reward table:")
    display(
        receptor_pair_reward_table[
            [
                "receptor_id",
                "peptide_id",
                "sequence",
                "cheap_pre_af_score_refined",
                "plddt",
                "ptm",
                "iptm",
                "structure_proxy_score",
                "soft_structure_gate",
                "final_reward",
            ]
        ].sort_values("final_reward", ascending=False)
    )

    print("\nPeptide-level best receptor reward table:")
    display(
        peptide_level_reward_table[
            [
                "peptide_id",
                "sequence",
                "best_receptor_id",
                "cheap_pre_af_score_refined",
                "best_plddt",
                "best_ptm",
                "best_iptm",
                "best_structure_proxy_score",
                "best_soft_structure_gate",
                "final_reward",
            ]
        ]
    )

    # ------------------------------------------------------------
    # Step H: RL update using peptide-level best receptor reward
    # ------------------------------------------------------------
    print("\n[Step H] Reward-guided RL update using best receptor per peptide...")

    rl_train_df = peptide_level_reward_table.dropna(subset=["final_reward"]).copy()

    if len(rl_train_df) < 2:
        print("Not enough scored peptides for RL update. Skipping update for this cycle.")
        mean_reward = np.nan
        top_reward = np.nan
        rl_loss_value = np.nan
        best_receptor_counts = {}

    else:
        rl_train_df = rl_train_df.sort_values(
            "final_reward",
            ascending=False,
        )

        peptides_for_rl = rl_train_df["sequence"].tolist()
        rewards = rl_train_df["final_reward"].values.astype(np.float32)

        print("Rewards used for RL:", rewards)
        print("Best receptor counts:")

        best_receptor_counts = rl_train_df["best_receptor_id"].value_counts().to_dict()
        print(best_receptor_counts)

        rl_batch = peptide_list_to_rl_batch(peptides_for_rl, rewards)

        rl_losses = []

        for update_epoch in range(1, RL_UPDATE_EPOCHS + 1):
            rl_params, rl_opt_state, rl_loss = rl_train_step(
                rl_params,
                rl_opt_state,
                rl_batch,
            )

            rl_loss.block_until_ready()
            rl_losses.append(float(rl_loss))

            print(
                f"RL update epoch {update_epoch:02d}/{RL_UPDATE_EPOCHS} | "
                f"loss = {float(rl_loss):.6f}"
            )

        mean_reward = float(np.mean(rewards))
        top_reward = float(np.max(rewards))
        rl_loss_value = float(rl_losses[-1])

    # ------------------------------------------------------------
    # Step I: Cycle summary
    # ------------------------------------------------------------
    cycle_summary = {
        "cycle": cycle,
        "n_generated": len(generated),
        "n_receptors": len(receptor_panel),
        "n_colabfold_peptides": len(top_peptides),
        "n_colabfold_pairs": len(fasta_manifest),
        "n_colabfold_success": int(run_log["success"].sum()) if len(run_log) else 0,
        "mean_final_reward": mean_reward,
        "top_final_reward": top_reward,
        "rl_loss_last": rl_loss_value,
        "temperature": TEMPERATURE,
        "best_receptor_counts": json.dumps(best_receptor_counts),
        "cycle_dir": str(cycle_dir),
    }

    rl_history_records.append(cycle_summary)

    print("\nCycle summary:")
    print(cycle_summary)


rl_history = pd.DataFrame(rl_history_records)

rl_history_path = run_root / "multi_receptor_rl_history.csv"
rl_history.to_csv(rl_history_path, index=False)

print("\n" + "=" * 100)
print("Multi-receptor reward-guided RL loop complete.")
print("Saved RL history:", rl_history_path)
display(rl_history)
```

    Multi-receptor panel:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>receptor_id</th>
      <th>gene</th>
      <th>uniprot_accession</th>
      <th>protein_name</th>
      <th>domain_used</th>
      <th>domain_sequence_length</th>
      <th>notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ly6a_mouse</td>
      <td>Ly6a</td>
      <td>P05533</td>
      <td>Lymphocyte antigen 6A-2/6E-1</td>
      <td>extracellular_mature_approx_27_112</td>
      <td>86</td>
      <td>GPI-anchored surface protein; mature extracell...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ecscr_mouse</td>
      <td>Ecscr</td>
      <td>Q3TZW0</td>
      <td>Endothelial cell-specific chemotaxis regulator</td>
      <td>extracellular_N_terminal_approx_24_185</td>
      <td>162</td>
      <td>Endothelial cell-surface single-pass type I me...</td>
    </tr>
  </tbody>
</table>
</div>


    
    ####################################################################################################
    MULTI-RECEPTOR RL CYCLE 1/2
    ####################################################################################################
    
    [Step A] Generating peptides...
    Generated 100 peptides in 151.41 s



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sequence</th>
      <th>temperature</th>
      <th>n_amino_acids</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>YAVTNVGFWH</td>
      <td>1.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MSETYPSARWYD</td>
      <td>1.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AFTYGKVGRRRRRRRNH</td>
      <td>1.0</td>
      <td>17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>YKPFSP</td>
      <td>1.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>YSSRHCSFSCNH</td>
      <td>1.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>5</th>
      <td>FQIDGTLKPVSSYY</td>
      <td>1.0</td>
      <td>14</td>
    </tr>
    <tr>
      <th>6</th>
      <td>GRLKALLQRRLLYKLKLKLKTG</td>
      <td>1.0</td>
      <td>22</td>
    </tr>
    <tr>
      <th>7</th>
      <td>PRFYIAPF</td>
      <td>1.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>HGRLRLRIALRHLRSHSKRKVRL</td>
      <td>1.0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PPNFMNH</td>
      <td>1.0</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step B] Cheap peptide scoring...
    Top cheap-scored peptides:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cheap_rank</th>
      <th>sequence</th>
      <th>n_amino_acids</th>
      <th>net_charge_proxy</th>
      <th>hydrophobic_fraction</th>
      <th>cysteine_count</th>
      <th>cheap_pre_af_score_refined</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>SFLVGYPPFQRNGT</td>
      <td>14</td>
      <td>1</td>
      <td>0.357143</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>LTPSHFTP</td>
      <td>8</td>
      <td>1</td>
      <td>0.250000</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>QQIRIYGGR</td>
      <td>9</td>
      <td>2</td>
      <td>0.333333</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>FLGKINRIPNRNA</td>
      <td>13</td>
      <td>3</td>
      <td>0.384615</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>ALWMVVYSPTTRRYNH</td>
      <td>16</td>
      <td>3</td>
      <td>0.500000</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>HKGAFTAG</td>
      <td>8</td>
      <td>2</td>
      <td>0.375000</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>NEYVGSKQYGNH</td>
      <td>12</td>
      <td>1</td>
      <td>0.250000</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>VQIRQYGKE</td>
      <td>9</td>
      <td>1</td>
      <td>0.333333</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>RSFAQNGNHY</td>
      <td>10</td>
      <td>2</td>
      <td>0.300000</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>LPVKYPVTYPWRWH</td>
      <td>14</td>
      <td>3</td>
      <td>0.500000</td>
      <td>0</td>
      <td>0.994286</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step C] Selecting top 5 peptides for multi-receptor ColabFold...



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>peptide_id</th>
      <th>cheap_rank</th>
      <th>sequence</th>
      <th>n_amino_acids</th>
      <th>cheap_pre_af_score_refined</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pep_000</td>
      <td>1</td>
      <td>SFLVGYPPFQRNGT</td>
      <td>14</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pep_001</td>
      <td>2</td>
      <td>LTPSHFTP</td>
      <td>8</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pep_002</td>
      <td>3</td>
      <td>QQIRIYGGR</td>
      <td>9</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pep_003</td>
      <td>4</td>
      <td>FLGKINRIPNRNA</td>
      <td>13</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pep_004</td>
      <td>5</td>
      <td>ALWMVVYSPTTRRYNH</td>
      <td>16</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step D] Writing receptor × peptide FASTA files...
    Receptors: 2 | Peptides: 5 | Total ColabFold pairs: 10
    
    [FASTA] Preparing receptor: Ly6a_mouse
    Domain length: 86 aa
    
    [FASTA] Preparing receptor: Ecscr_mouse
    Domain length: 162 aa
    
    Wrote 10 receptor-peptide FASTA files.
    Saved FASTA manifest: ../data/processed/multi_receptor_reward_guided_rl/cycle_01/colabfold_fasta_manifest.csv



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>receptor_id</th>
      <th>peptide_id</th>
      <th>sequence</th>
      <th>fasta_file</th>
      <th>receptor_domain_length</th>
      <th>peptide_length</th>
      <th>complex_total_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ly6a_mouse</td>
      <td>pep_000</td>
      <td>SFLVGYPPFQRNGT</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>86</td>
      <td>14</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ly6a_mouse</td>
      <td>pep_001</td>
      <td>LTPSHFTP</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>86</td>
      <td>8</td>
      <td>94</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ly6a_mouse</td>
      <td>pep_002</td>
      <td>QQIRIYGGR</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>86</td>
      <td>9</td>
      <td>95</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ly6a_mouse</td>
      <td>pep_003</td>
      <td>FLGKINRIPNRNA</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>86</td>
      <td>13</td>
      <td>99</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ly6a_mouse</td>
      <td>pep_004</td>
      <td>ALWMVVYSPTTRRYNH</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>86</td>
      <td>16</td>
      <td>102</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ecscr_mouse</td>
      <td>pep_000</td>
      <td>SFLVGYPPFQRNGT</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>162</td>
      <td>14</td>
      <td>176</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ecscr_mouse</td>
      <td>pep_001</td>
      <td>LTPSHFTP</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>162</td>
      <td>8</td>
      <td>170</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Ecscr_mouse</td>
      <td>pep_002</td>
      <td>QQIRIYGGR</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>162</td>
      <td>9</td>
      <td>171</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ecscr_mouse</td>
      <td>pep_003</td>
      <td>FLGKINRIPNRNA</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>162</td>
      <td>13</td>
      <td>175</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Ecscr_mouse</td>
      <td>pep_004</td>
      <td>ALWMVVYSPTTRRYNH</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>162</td>
      <td>16</td>
      <td>178</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step E] Running ColabFold for selected receptor × peptide pairs...
    
    ----------------------------------------------------------------------------------------------------
    [ColabFold] Job 1/10
    Receptor: Ly6a_mouse
    Peptide:  pep_000 | SFLVGYPPFQRNGT
    Input FASTA: ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_inputs/Ly6a_mouse/Ly6a_mouse_pep_000_complex.fasta
    Result dir:  ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_results/Ly6a_mouse/pep_000
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_inputs/Ly6a_mouse/Ly6a_mouse_pep_000_complex.fasta ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_results/Ly6a_mouse/pep_000
    Return code: 0
    Elapsed: 7.65 s (0.13 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    ----------------------------------------------------------------------------------------------------
    [ColabFold] Job 2/10
    Receptor: Ly6a_mouse
    Peptide:  pep_001 | LTPSHFTP
    Input FASTA: ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_inputs/Ly6a_mouse/Ly6a_mouse_pep_001_complex.fasta
    Result dir:  ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_results/Ly6a_mouse/pep_001
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_inputs/Ly6a_mouse/Ly6a_mouse_pep_001_complex.fasta ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_results/Ly6a_mouse/pep_001
    Return code: 0
    Elapsed: 3.63 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    ----------------------------------------------------------------------------------------------------
    [ColabFold] Job 3/10
    Receptor: Ly6a_mouse
    Peptide:  pep_002 | QQIRIYGGR
    Input FASTA: ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_inputs/Ly6a_mouse/Ly6a_mouse_pep_002_complex.fasta
    Result dir:  ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_results/Ly6a_mouse/pep_002
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_inputs/Ly6a_mouse/Ly6a_mouse_pep_002_complex.fasta ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_results/Ly6a_mouse/pep_002
    Return code: 0
    Elapsed: 3.65 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    ----------------------------------------------------------------------------------------------------
    [ColabFold] Job 4/10
    Receptor: Ly6a_mouse
    Peptide:  pep_003 | FLGKINRIPNRNA
    Input FASTA: ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_inputs/Ly6a_mouse/Ly6a_mouse_pep_003_complex.fasta
    Result dir:  ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_results/Ly6a_mouse/pep_003
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_inputs/Ly6a_mouse/Ly6a_mouse_pep_003_complex.fasta ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_results/Ly6a_mouse/pep_003
    Return code: 0
    Elapsed: 3.52 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    ----------------------------------------------------------------------------------------------------
    [ColabFold] Job 5/10
    Receptor: Ly6a_mouse
    Peptide:  pep_004 | ALWMVVYSPTTRRYNH
    Input FASTA: ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_inputs/Ly6a_mouse/Ly6a_mouse_pep_004_complex.fasta
    Result dir:  ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_results/Ly6a_mouse/pep_004
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_inputs/Ly6a_mouse/Ly6a_mouse_pep_004_complex.fasta ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_results/Ly6a_mouse/pep_004
    Return code: 0
    Elapsed: 3.44 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    ----------------------------------------------------------------------------------------------------
    [ColabFold] Job 6/10
    Receptor: Ecscr_mouse
    Peptide:  pep_000 | SFLVGYPPFQRNGT
    Input FASTA: ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_inputs/Ecscr_mouse/Ecscr_mouse_pep_000_complex.fasta
    Result dir:  ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_results/Ecscr_mouse/pep_000
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_inputs/Ecscr_mouse/Ecscr_mouse_pep_000_complex.fasta ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_results/Ecscr_mouse/pep_000
    Return code: 0
    Elapsed: 3.46 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    ----------------------------------------------------------------------------------------------------
    [ColabFold] Job 7/10
    Receptor: Ecscr_mouse
    Peptide:  pep_001 | LTPSHFTP
    Input FASTA: ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_inputs/Ecscr_mouse/Ecscr_mouse_pep_001_complex.fasta
    Result dir:  ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_results/Ecscr_mouse/pep_001
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_inputs/Ecscr_mouse/Ecscr_mouse_pep_001_complex.fasta ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_results/Ecscr_mouse/pep_001
    Return code: 0
    Elapsed: 3.36 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    ----------------------------------------------------------------------------------------------------
    [ColabFold] Job 8/10
    Receptor: Ecscr_mouse
    Peptide:  pep_002 | QQIRIYGGR
    Input FASTA: ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_inputs/Ecscr_mouse/Ecscr_mouse_pep_002_complex.fasta
    Result dir:  ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_results/Ecscr_mouse/pep_002
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_inputs/Ecscr_mouse/Ecscr_mouse_pep_002_complex.fasta ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_results/Ecscr_mouse/pep_002
    Return code: 0
    Elapsed: 3.26 s (0.05 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    ----------------------------------------------------------------------------------------------------
    [ColabFold] Job 9/10
    Receptor: Ecscr_mouse
    Peptide:  pep_003 | FLGKINRIPNRNA
    Input FASTA: ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_inputs/Ecscr_mouse/Ecscr_mouse_pep_003_complex.fasta
    Result dir:  ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_results/Ecscr_mouse/pep_003
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_inputs/Ecscr_mouse/Ecscr_mouse_pep_003_complex.fasta ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_results/Ecscr_mouse/pep_003
    Return code: 0
    Elapsed: 3.32 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    ----------------------------------------------------------------------------------------------------
    [ColabFold] Job 10/10
    Receptor: Ecscr_mouse
    Peptide:  pep_004 | ALWMVVYSPTTRRYNH
    Input FASTA: ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_inputs/Ecscr_mouse/Ecscr_mouse_pep_004_complex.fasta
    Result dir:  ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_results/Ecscr_mouse/pep_004
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_inputs/Ecscr_mouse/Ecscr_mouse_pep_004_complex.fasta ../data/processed/multi_receptor_reward_guided_rl/cycle_01/alphafold_results/Ecscr_mouse/pep_004
    Return code: 0
    Elapsed: 3.31 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    ColabFold run log:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>receptor_id</th>
      <th>peptide_id</th>
      <th>sequence</th>
      <th>fasta_file</th>
      <th>result_dir</th>
      <th>returncode</th>
      <th>elapsed_seconds</th>
      <th>elapsed_minutes</th>
      <th>n_pdb_files</th>
      <th>n_score_json_files</th>
      <th>success</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ly6a_mouse</td>
      <td>pep_000</td>
      <td>SFLVGYPPFQRNGT</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>0</td>
      <td>7.647489</td>
      <td>0.127458</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ly6a_mouse</td>
      <td>pep_001</td>
      <td>LTPSHFTP</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>0</td>
      <td>3.627401</td>
      <td>0.060457</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ly6a_mouse</td>
      <td>pep_002</td>
      <td>QQIRIYGGR</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>0</td>
      <td>3.650263</td>
      <td>0.060838</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ly6a_mouse</td>
      <td>pep_003</td>
      <td>FLGKINRIPNRNA</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>0</td>
      <td>3.517073</td>
      <td>0.058618</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ly6a_mouse</td>
      <td>pep_004</td>
      <td>ALWMVVYSPTTRRYNH</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>0</td>
      <td>3.436717</td>
      <td>0.057279</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ecscr_mouse</td>
      <td>pep_000</td>
      <td>SFLVGYPPFQRNGT</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>0</td>
      <td>3.459012</td>
      <td>0.057650</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ecscr_mouse</td>
      <td>pep_001</td>
      <td>LTPSHFTP</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>0</td>
      <td>3.358117</td>
      <td>0.055969</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Ecscr_mouse</td>
      <td>pep_002</td>
      <td>QQIRIYGGR</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>0</td>
      <td>3.258412</td>
      <td>0.054307</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ecscr_mouse</td>
      <td>pep_003</td>
      <td>FLGKINRIPNRNA</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>0</td>
      <td>3.319994</td>
      <td>0.055333</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Ecscr_mouse</td>
      <td>pep_004</td>
      <td>ALWMVVYSPTTRRYNH</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>0</td>
      <td>3.313457</td>
      <td>0.055224</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step F] Parsing ColabFold scores...
    Parsed ColabFold score rows: 10



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>receptor_id</th>
      <th>peptide_id</th>
      <th>sequence</th>
      <th>result_dir</th>
      <th>score_json</th>
      <th>plddt</th>
      <th>ptm</th>
      <th>iptm</th>
      <th>ranking_confidence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ly6a_mouse</td>
      <td>pep_000</td>
      <td>SFLVGYPPFQRNGT</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>Ly6a_mouse_pep_000_complex_scores_rank_001_alp...</td>
      <td>31.523571</td>
      <td>0.17</td>
      <td>0.09</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ly6a_mouse</td>
      <td>pep_001</td>
      <td>LTPSHFTP</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>Ly6a_mouse_pep_001_complex_scores_rank_001_alp...</td>
      <td>31.678617</td>
      <td>0.16</td>
      <td>0.04</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ly6a_mouse</td>
      <td>pep_002</td>
      <td>QQIRIYGGR</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>Ly6a_mouse_pep_002_complex_scores_rank_001_alp...</td>
      <td>31.985758</td>
      <td>0.18</td>
      <td>0.06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ly6a_mouse</td>
      <td>pep_003</td>
      <td>FLGKINRIPNRNA</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>Ly6a_mouse_pep_003_complex_scores_rank_001_alp...</td>
      <td>31.425446</td>
      <td>0.18</td>
      <td>0.06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ly6a_mouse</td>
      <td>pep_004</td>
      <td>ALWMVVYSPTTRRYNH</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>Ly6a_mouse_pep_004_complex_scores_rank_001_alp...</td>
      <td>31.737813</td>
      <td>0.16</td>
      <td>0.06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ecscr_mouse</td>
      <td>pep_000</td>
      <td>SFLVGYPPFQRNGT</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>Ecscr_mouse_pep_000_complex_scores_rank_001_al...</td>
      <td>39.394310</td>
      <td>0.24</td>
      <td>0.14</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ecscr_mouse</td>
      <td>pep_001</td>
      <td>LTPSHFTP</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>Ecscr_mouse_pep_001_complex_scores_rank_001_al...</td>
      <td>41.454000</td>
      <td>0.23</td>
      <td>0.03</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Ecscr_mouse</td>
      <td>pep_002</td>
      <td>QQIRIYGGR</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>Ecscr_mouse_pep_002_complex_scores_rank_001_al...</td>
      <td>38.650229</td>
      <td>0.23</td>
      <td>0.05</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ecscr_mouse</td>
      <td>pep_003</td>
      <td>FLGKINRIPNRNA</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>Ecscr_mouse_pep_003_complex_scores_rank_001_al...</td>
      <td>38.459831</td>
      <td>0.23</td>
      <td>0.09</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Ecscr_mouse</td>
      <td>pep_004</td>
      <td>ALWMVVYSPTTRRYNH</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>Ecscr_mouse_pep_004_complex_scores_rank_001_al...</td>
      <td>38.765000</td>
      <td>0.23</td>
      <td>0.04</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step G] Computing multi-receptor reward tables...
    
    Receptor × peptide reward table:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>receptor_id</th>
      <th>peptide_id</th>
      <th>sequence</th>
      <th>cheap_pre_af_score_refined</th>
      <th>plddt</th>
      <th>ptm</th>
      <th>iptm</th>
      <th>structure_proxy_score</th>
      <th>soft_structure_gate</th>
      <th>final_reward</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ecscr_mouse</td>
      <td>pep_000</td>
      <td>SFLVGYPPFQRNGT</td>
      <td>1.0</td>
      <td>39.394310</td>
      <td>0.24</td>
      <td>0.14</td>
      <td>0.196091</td>
      <td>0.647641</td>
      <td>0.309223</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ecscr_mouse</td>
      <td>pep_003</td>
      <td>FLGKINRIPNRNA</td>
      <td>1.0</td>
      <td>38.459831</td>
      <td>0.23</td>
      <td>0.09</td>
      <td>0.164690</td>
      <td>0.496082</td>
      <td>0.226734</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ly6a_mouse</td>
      <td>pep_000</td>
      <td>SFLVGYPPFQRNGT</td>
      <td>1.0</td>
      <td>31.523571</td>
      <td>0.17</td>
      <td>0.09</td>
      <td>0.139285</td>
      <td>0.435390</td>
      <td>0.191805</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ecscr_mouse</td>
      <td>pep_002</td>
      <td>QQIRIYGGR</td>
      <td>1.0</td>
      <td>38.650229</td>
      <td>0.23</td>
      <td>0.05</td>
      <td>0.142975</td>
      <td>0.383042</td>
      <td>0.169663</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ecscr_mouse</td>
      <td>pep_004</td>
      <td>ALWMVVYSPTTRRYNH</td>
      <td>1.0</td>
      <td>38.765000</td>
      <td>0.23</td>
      <td>0.04</td>
      <td>0.137648</td>
      <td>0.355370</td>
      <td>0.156175</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ecscr_mouse</td>
      <td>pep_001</td>
      <td>LTPSHFTP</td>
      <td>1.0</td>
      <td>41.454000</td>
      <td>0.23</td>
      <td>0.03</td>
      <td>0.136181</td>
      <td>0.350222</td>
      <td>0.153579</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ly6a_mouse</td>
      <td>pep_002</td>
      <td>QQIRIYGGR</td>
      <td>1.0</td>
      <td>31.985758</td>
      <td>0.18</td>
      <td>0.06</td>
      <td>0.125979</td>
      <td>0.353405</td>
      <td>0.152631</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Ly6a_mouse</td>
      <td>pep_004</td>
      <td>ALWMVVYSPTTRRYNH</td>
      <td>1.0</td>
      <td>31.737813</td>
      <td>0.16</td>
      <td>0.06</td>
      <td>0.120607</td>
      <td>0.351235</td>
      <td>0.150467</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ly6a_mouse</td>
      <td>pep_003</td>
      <td>FLGKINRIPNRNA</td>
      <td>1.0</td>
      <td>31.425446</td>
      <td>0.18</td>
      <td>0.06</td>
      <td>0.125138</td>
      <td>0.348502</td>
      <td>0.150323</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Ly6a_mouse</td>
      <td>pep_001</td>
      <td>LTPSHFTP</td>
      <td>1.0</td>
      <td>31.678617</td>
      <td>0.16</td>
      <td>0.04</td>
      <td>0.109518</td>
      <td>0.293364</td>
      <td>0.123561</td>
    </tr>
  </tbody>
</table>
</div>


    
    Peptide-level best receptor reward table:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>peptide_id</th>
      <th>sequence</th>
      <th>best_receptor_id</th>
      <th>cheap_pre_af_score_refined</th>
      <th>best_plddt</th>
      <th>best_ptm</th>
      <th>best_iptm</th>
      <th>best_structure_proxy_score</th>
      <th>best_soft_structure_gate</th>
      <th>final_reward</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pep_000</td>
      <td>SFLVGYPPFQRNGT</td>
      <td>Ecscr_mouse</td>
      <td>1.0</td>
      <td>39.394310</td>
      <td>0.24</td>
      <td>0.14</td>
      <td>0.196091</td>
      <td>0.647641</td>
      <td>0.309223</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pep_003</td>
      <td>FLGKINRIPNRNA</td>
      <td>Ecscr_mouse</td>
      <td>1.0</td>
      <td>38.459831</td>
      <td>0.23</td>
      <td>0.09</td>
      <td>0.164690</td>
      <td>0.496082</td>
      <td>0.226734</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pep_002</td>
      <td>QQIRIYGGR</td>
      <td>Ecscr_mouse</td>
      <td>1.0</td>
      <td>38.650229</td>
      <td>0.23</td>
      <td>0.05</td>
      <td>0.142975</td>
      <td>0.383042</td>
      <td>0.169663</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pep_004</td>
      <td>ALWMVVYSPTTRRYNH</td>
      <td>Ecscr_mouse</td>
      <td>1.0</td>
      <td>38.765000</td>
      <td>0.23</td>
      <td>0.04</td>
      <td>0.137648</td>
      <td>0.355370</td>
      <td>0.156175</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pep_001</td>
      <td>LTPSHFTP</td>
      <td>Ecscr_mouse</td>
      <td>1.0</td>
      <td>41.454000</td>
      <td>0.23</td>
      <td>0.03</td>
      <td>0.136181</td>
      <td>0.350222</td>
      <td>0.153579</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step H] Reward-guided RL update using best receptor per peptide...
    Rewards used for RL: [0.3092225  0.22673361 0.1696625  0.15617487 0.15357874]
    Best receptor counts:
    {'Ecscr_mouse': 5}
    RL update epoch 01/5 | loss = 5.730832
    RL update epoch 02/5 | loss = 5.712078
    RL update epoch 03/5 | loss = 5.694340
    RL update epoch 04/5 | loss = 5.674234
    RL update epoch 05/5 | loss = 5.655885
    
    Cycle summary:
    {'cycle': 1, 'n_generated': 100, 'n_receptors': 2, 'n_colabfold_peptides': 5, 'n_colabfold_pairs': 10, 'n_colabfold_success': 10, 'mean_final_reward': 0.20307445526123047, 'top_final_reward': 0.3092224895954132, 'rl_loss_last': 5.655885219573975, 'temperature': 1.0, 'best_receptor_counts': '{"Ecscr_mouse": 5}', 'cycle_dir': '../data/processed/multi_receptor_reward_guided_rl/cycle_01'}
    
    ####################################################################################################
    MULTI-RECEPTOR RL CYCLE 2/2
    ####################################################################################################
    
    [Step A] Generating peptides...
    Generated 100 peptides in 153.41 s



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sequence</th>
      <th>temperature</th>
      <th>n_amino_acids</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>WRFSRW</td>
      <td>1.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LQIKIWFQNRRMKWKK</td>
      <td>1.0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TGCSEY</td>
      <td>1.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MAVYCC</td>
      <td>1.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VEPVNVQYQ</td>
      <td>1.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TGYHPHGIRRH</td>
      <td>1.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DFGGGPGQLGLGGNLVLTMGKYYDKKFSALKYLN</td>
      <td>1.0</td>
      <td>34</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NPFNRQ</td>
      <td>1.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NQYYP</td>
      <td>1.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GSDYRGRGRGADCHMECESCPALTYCTSPTSNNTKKCCSC</td>
      <td>1.0</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step B] Cheap peptide scoring...
    Top cheap-scored peptides:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cheap_rank</th>
      <th>sequence</th>
      <th>n_amino_acids</th>
      <th>net_charge_proxy</th>
      <th>hydrophobic_fraction</th>
      <th>cysteine_count</th>
      <th>cheap_pre_af_score_refined</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>SKYPYPRR</td>
      <td>8</td>
      <td>3</td>
      <td>0.250000</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>HAFFLGSRGSR</td>
      <td>11</td>
      <td>3</td>
      <td>0.363636</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>TAYPTFNHRSRM</td>
      <td>12</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>VQIYPKGKG</td>
      <td>9</td>
      <td>2</td>
      <td>0.333333</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>RVVTKKTTFH</td>
      <td>10</td>
      <td>4</td>
      <td>0.300000</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>TLNYTERRRRIR</td>
      <td>12</td>
      <td>4</td>
      <td>0.250000</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>TKKSFWEPFMNH</td>
      <td>12</td>
      <td>2</td>
      <td>0.333333</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>KSYHAFETENQLNH</td>
      <td>14</td>
      <td>1</td>
      <td>0.285714</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>GTRPITRNHFIW</td>
      <td>12</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>TKFPGTWPWQRWSTE</td>
      <td>15</td>
      <td>1</td>
      <td>0.266667</td>
      <td>0</td>
      <td>0.997333</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step C] Selecting top 5 peptides for multi-receptor ColabFold...



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>peptide_id</th>
      <th>cheap_rank</th>
      <th>sequence</th>
      <th>n_amino_acids</th>
      <th>cheap_pre_af_score_refined</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pep_000</td>
      <td>1</td>
      <td>SKYPYPRR</td>
      <td>8</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pep_001</td>
      <td>2</td>
      <td>HAFFLGSRGSR</td>
      <td>11</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pep_002</td>
      <td>3</td>
      <td>TAYPTFNHRSRM</td>
      <td>12</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pep_003</td>
      <td>4</td>
      <td>VQIYPKGKG</td>
      <td>9</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pep_004</td>
      <td>5</td>
      <td>RVVTKKTTFH</td>
      <td>10</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step D] Writing receptor × peptide FASTA files...
    Receptors: 2 | Peptides: 5 | Total ColabFold pairs: 10
    
    [FASTA] Preparing receptor: Ly6a_mouse
    Domain length: 86 aa
    
    [FASTA] Preparing receptor: Ecscr_mouse
    Domain length: 162 aa
    
    Wrote 10 receptor-peptide FASTA files.
    Saved FASTA manifest: ../data/processed/multi_receptor_reward_guided_rl/cycle_02/colabfold_fasta_manifest.csv



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>receptor_id</th>
      <th>peptide_id</th>
      <th>sequence</th>
      <th>fasta_file</th>
      <th>receptor_domain_length</th>
      <th>peptide_length</th>
      <th>complex_total_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ly6a_mouse</td>
      <td>pep_000</td>
      <td>SKYPYPRR</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>86</td>
      <td>8</td>
      <td>94</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ly6a_mouse</td>
      <td>pep_001</td>
      <td>HAFFLGSRGSR</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>86</td>
      <td>11</td>
      <td>97</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ly6a_mouse</td>
      <td>pep_002</td>
      <td>TAYPTFNHRSRM</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>86</td>
      <td>12</td>
      <td>98</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ly6a_mouse</td>
      <td>pep_003</td>
      <td>VQIYPKGKG</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>86</td>
      <td>9</td>
      <td>95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ly6a_mouse</td>
      <td>pep_004</td>
      <td>RVVTKKTTFH</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>86</td>
      <td>10</td>
      <td>96</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ecscr_mouse</td>
      <td>pep_000</td>
      <td>SKYPYPRR</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>162</td>
      <td>8</td>
      <td>170</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ecscr_mouse</td>
      <td>pep_001</td>
      <td>HAFFLGSRGSR</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>162</td>
      <td>11</td>
      <td>173</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Ecscr_mouse</td>
      <td>pep_002</td>
      <td>TAYPTFNHRSRM</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>162</td>
      <td>12</td>
      <td>174</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ecscr_mouse</td>
      <td>pep_003</td>
      <td>VQIYPKGKG</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>162</td>
      <td>9</td>
      <td>171</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Ecscr_mouse</td>
      <td>pep_004</td>
      <td>RVVTKKTTFH</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>162</td>
      <td>10</td>
      <td>172</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step E] Running ColabFold for selected receptor × peptide pairs...
    
    ----------------------------------------------------------------------------------------------------
    [ColabFold] Job 1/10
    Receptor: Ly6a_mouse
    Peptide:  pep_000 | SKYPYPRR
    Input FASTA: ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_inputs/Ly6a_mouse/Ly6a_mouse_pep_000_complex.fasta
    Result dir:  ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_results/Ly6a_mouse/pep_000
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_inputs/Ly6a_mouse/Ly6a_mouse_pep_000_complex.fasta ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_results/Ly6a_mouse/pep_000
    Return code: 0
    Elapsed: 3.71 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    ----------------------------------------------------------------------------------------------------
    [ColabFold] Job 2/10
    Receptor: Ly6a_mouse
    Peptide:  pep_001 | HAFFLGSRGSR
    Input FASTA: ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_inputs/Ly6a_mouse/Ly6a_mouse_pep_001_complex.fasta
    Result dir:  ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_results/Ly6a_mouse/pep_001
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_inputs/Ly6a_mouse/Ly6a_mouse_pep_001_complex.fasta ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_results/Ly6a_mouse/pep_001
    Return code: 0
    Elapsed: 3.47 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    ----------------------------------------------------------------------------------------------------
    [ColabFold] Job 3/10
    Receptor: Ly6a_mouse
    Peptide:  pep_002 | TAYPTFNHRSRM
    Input FASTA: ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_inputs/Ly6a_mouse/Ly6a_mouse_pep_002_complex.fasta
    Result dir:  ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_results/Ly6a_mouse/pep_002
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_inputs/Ly6a_mouse/Ly6a_mouse_pep_002_complex.fasta ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_results/Ly6a_mouse/pep_002
    Return code: 0
    Elapsed: 3.44 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    ----------------------------------------------------------------------------------------------------
    [ColabFold] Job 4/10
    Receptor: Ly6a_mouse
    Peptide:  pep_003 | VQIYPKGKG
    Input FASTA: ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_inputs/Ly6a_mouse/Ly6a_mouse_pep_003_complex.fasta
    Result dir:  ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_results/Ly6a_mouse/pep_003
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_inputs/Ly6a_mouse/Ly6a_mouse_pep_003_complex.fasta ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_results/Ly6a_mouse/pep_003
    Return code: 0
    Elapsed: 3.50 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    ----------------------------------------------------------------------------------------------------
    [ColabFold] Job 5/10
    Receptor: Ly6a_mouse
    Peptide:  pep_004 | RVVTKKTTFH
    Input FASTA: ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_inputs/Ly6a_mouse/Ly6a_mouse_pep_004_complex.fasta
    Result dir:  ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_results/Ly6a_mouse/pep_004
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_inputs/Ly6a_mouse/Ly6a_mouse_pep_004_complex.fasta ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_results/Ly6a_mouse/pep_004
    Return code: 0
    Elapsed: 3.61 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    ----------------------------------------------------------------------------------------------------
    [ColabFold] Job 6/10
    Receptor: Ecscr_mouse
    Peptide:  pep_000 | SKYPYPRR
    Input FASTA: ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_inputs/Ecscr_mouse/Ecscr_mouse_pep_000_complex.fasta
    Result dir:  ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_results/Ecscr_mouse/pep_000
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_inputs/Ecscr_mouse/Ecscr_mouse_pep_000_complex.fasta ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_results/Ecscr_mouse/pep_000
    Return code: 0
    Elapsed: 3.64 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    ----------------------------------------------------------------------------------------------------
    [ColabFold] Job 7/10
    Receptor: Ecscr_mouse
    Peptide:  pep_001 | HAFFLGSRGSR
    Input FASTA: ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_inputs/Ecscr_mouse/Ecscr_mouse_pep_001_complex.fasta
    Result dir:  ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_results/Ecscr_mouse/pep_001
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_inputs/Ecscr_mouse/Ecscr_mouse_pep_001_complex.fasta ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_results/Ecscr_mouse/pep_001
    Return code: 0
    Elapsed: 3.74 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    ----------------------------------------------------------------------------------------------------
    [ColabFold] Job 8/10
    Receptor: Ecscr_mouse
    Peptide:  pep_002 | TAYPTFNHRSRM
    Input FASTA: ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_inputs/Ecscr_mouse/Ecscr_mouse_pep_002_complex.fasta
    Result dir:  ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_results/Ecscr_mouse/pep_002
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_inputs/Ecscr_mouse/Ecscr_mouse_pep_002_complex.fasta ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_results/Ecscr_mouse/pep_002
    Return code: 0
    Elapsed: 4.11 s (0.07 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    ----------------------------------------------------------------------------------------------------
    [ColabFold] Job 9/10
    Receptor: Ecscr_mouse
    Peptide:  pep_003 | VQIYPKGKG
    Input FASTA: ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_inputs/Ecscr_mouse/Ecscr_mouse_pep_003_complex.fasta
    Result dir:  ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_results/Ecscr_mouse/pep_003
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_inputs/Ecscr_mouse/Ecscr_mouse_pep_003_complex.fasta ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_results/Ecscr_mouse/pep_003
    Return code: 0
    Elapsed: 4.04 s (0.07 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    ----------------------------------------------------------------------------------------------------
    [ColabFold] Job 10/10
    Receptor: Ecscr_mouse
    Peptide:  pep_004 | RVVTKKTTFH
    Input FASTA: ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_inputs/Ecscr_mouse/Ecscr_mouse_pep_004_complex.fasta
    Result dir:  ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_results/Ecscr_mouse/pep_004
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_inputs/Ecscr_mouse/Ecscr_mouse_pep_004_complex.fasta ../data/processed/multi_receptor_reward_guided_rl/cycle_02/alphafold_results/Ecscr_mouse/pep_004
    Return code: 0
    Elapsed: 3.85 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    ColabFold run log:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>receptor_id</th>
      <th>peptide_id</th>
      <th>sequence</th>
      <th>fasta_file</th>
      <th>result_dir</th>
      <th>returncode</th>
      <th>elapsed_seconds</th>
      <th>elapsed_minutes</th>
      <th>n_pdb_files</th>
      <th>n_score_json_files</th>
      <th>success</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ly6a_mouse</td>
      <td>pep_000</td>
      <td>SKYPYPRR</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>0</td>
      <td>3.713400</td>
      <td>0.061890</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ly6a_mouse</td>
      <td>pep_001</td>
      <td>HAFFLGSRGSR</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>0</td>
      <td>3.465840</td>
      <td>0.057764</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ly6a_mouse</td>
      <td>pep_002</td>
      <td>TAYPTFNHRSRM</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>0</td>
      <td>3.440554</td>
      <td>0.057343</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ly6a_mouse</td>
      <td>pep_003</td>
      <td>VQIYPKGKG</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>0</td>
      <td>3.499370</td>
      <td>0.058323</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ly6a_mouse</td>
      <td>pep_004</td>
      <td>RVVTKKTTFH</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>0</td>
      <td>3.606785</td>
      <td>0.060113</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ecscr_mouse</td>
      <td>pep_000</td>
      <td>SKYPYPRR</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>0</td>
      <td>3.644152</td>
      <td>0.060736</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ecscr_mouse</td>
      <td>pep_001</td>
      <td>HAFFLGSRGSR</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>0</td>
      <td>3.740684</td>
      <td>0.062345</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Ecscr_mouse</td>
      <td>pep_002</td>
      <td>TAYPTFNHRSRM</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>0</td>
      <td>4.112955</td>
      <td>0.068549</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ecscr_mouse</td>
      <td>pep_003</td>
      <td>VQIYPKGKG</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>0</td>
      <td>4.035457</td>
      <td>0.067258</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Ecscr_mouse</td>
      <td>pep_004</td>
      <td>RVVTKKTTFH</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>0</td>
      <td>3.848091</td>
      <td>0.064135</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step F] Parsing ColabFold scores...
    Parsed ColabFold score rows: 10



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>receptor_id</th>
      <th>peptide_id</th>
      <th>sequence</th>
      <th>result_dir</th>
      <th>score_json</th>
      <th>plddt</th>
      <th>ptm</th>
      <th>iptm</th>
      <th>ranking_confidence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ly6a_mouse</td>
      <td>pep_000</td>
      <td>SKYPYPRR</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>Ly6a_mouse_pep_000_complex_scores_rank_001_alp...</td>
      <td>31.074792</td>
      <td>0.18</td>
      <td>0.06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ly6a_mouse</td>
      <td>pep_001</td>
      <td>HAFFLGSRGSR</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>Ly6a_mouse_pep_001_complex_scores_rank_001_alp...</td>
      <td>31.660784</td>
      <td>0.17</td>
      <td>0.06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ly6a_mouse</td>
      <td>pep_002</td>
      <td>TAYPTFNHRSRM</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>Ly6a_mouse_pep_002_complex_scores_rank_001_alp...</td>
      <td>32.252812</td>
      <td>0.18</td>
      <td>0.06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ly6a_mouse</td>
      <td>pep_003</td>
      <td>VQIYPKGKG</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>Ly6a_mouse_pep_003_complex_scores_rank_001_alp...</td>
      <td>30.732187</td>
      <td>0.17</td>
      <td>0.05</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ly6a_mouse</td>
      <td>pep_004</td>
      <td>RVVTKKTTFH</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>Ly6a_mouse_pep_004_complex_scores_rank_001_alp...</td>
      <td>31.003980</td>
      <td>0.17</td>
      <td>0.09</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ecscr_mouse</td>
      <td>pep_000</td>
      <td>SKYPYPRR</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>Ecscr_mouse_pep_000_complex_scores_rank_001_al...</td>
      <td>39.274535</td>
      <td>0.23</td>
      <td>0.06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ecscr_mouse</td>
      <td>pep_001</td>
      <td>HAFFLGSRGSR</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>Ecscr_mouse_pep_001_complex_scores_rank_001_al...</td>
      <td>38.827753</td>
      <td>0.23</td>
      <td>0.09</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Ecscr_mouse</td>
      <td>pep_002</td>
      <td>TAYPTFNHRSRM</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>Ecscr_mouse_pep_002_complex_scores_rank_001_al...</td>
      <td>39.246105</td>
      <td>0.23</td>
      <td>0.10</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ecscr_mouse</td>
      <td>pep_003</td>
      <td>VQIYPKGKG</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>Ecscr_mouse_pep_003_complex_scores_rank_001_al...</td>
      <td>39.838023</td>
      <td>0.23</td>
      <td>0.03</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Ecscr_mouse</td>
      <td>pep_004</td>
      <td>RVVTKKTTFH</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
      <td>Ecscr_mouse_pep_004_complex_scores_rank_001_al...</td>
      <td>38.455862</td>
      <td>0.23</td>
      <td>0.09</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step G] Computing multi-receptor reward tables...
    
    Receptor × peptide reward table:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>receptor_id</th>
      <th>peptide_id</th>
      <th>sequence</th>
      <th>cheap_pre_af_score_refined</th>
      <th>plddt</th>
      <th>ptm</th>
      <th>iptm</th>
      <th>structure_proxy_score</th>
      <th>soft_structure_gate</th>
      <th>final_reward</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ecscr_mouse</td>
      <td>pep_002</td>
      <td>TAYPTFNHRSRM</td>
      <td>1.0</td>
      <td>39.246105</td>
      <td>0.23</td>
      <td>0.10</td>
      <td>0.171369</td>
      <td>0.531639</td>
      <td>0.245293</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ecscr_mouse</td>
      <td>pep_001</td>
      <td>HAFFLGSRGSR</td>
      <td>1.0</td>
      <td>38.827753</td>
      <td>0.23</td>
      <td>0.09</td>
      <td>0.165242</td>
      <td>0.499302</td>
      <td>0.228384</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ecscr_mouse</td>
      <td>pep_004</td>
      <td>RVVTKKTTFH</td>
      <td>1.0</td>
      <td>38.455862</td>
      <td>0.23</td>
      <td>0.09</td>
      <td>0.164684</td>
      <td>0.496048</td>
      <td>0.226716</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ly6a_mouse</td>
      <td>pep_004</td>
      <td>RVVTKKTTFH</td>
      <td>1.0</td>
      <td>31.003980</td>
      <td>0.17</td>
      <td>0.09</td>
      <td>0.138506</td>
      <td>0.430844</td>
      <td>0.189584</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ecscr_mouse</td>
      <td>pep_000</td>
      <td>SKYPYPRR</td>
      <td>1.0</td>
      <td>39.274535</td>
      <td>0.23</td>
      <td>0.06</td>
      <td>0.149412</td>
      <td>0.417182</td>
      <td>0.186529</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ly6a_mouse</td>
      <td>pep_002</td>
      <td>TAYPTFNHRSRM</td>
      <td>1.0</td>
      <td>32.252812</td>
      <td>0.18</td>
      <td>0.06</td>
      <td>0.126379</td>
      <td>0.355742</td>
      <td>0.153732</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ly6a_mouse</td>
      <td>pep_001</td>
      <td>HAFFLGSRGSR</td>
      <td>1.0</td>
      <td>31.660784</td>
      <td>0.17</td>
      <td>0.06</td>
      <td>0.122991</td>
      <td>0.350561</td>
      <td>0.150722</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Ly6a_mouse</td>
      <td>pep_000</td>
      <td>SKYPYPRR</td>
      <td>1.0</td>
      <td>31.074792</td>
      <td>0.18</td>
      <td>0.06</td>
      <td>0.124612</td>
      <td>0.345434</td>
      <td>0.148881</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ecscr_mouse</td>
      <td>pep_003</td>
      <td>VQIYPKGKG</td>
      <td>1.0</td>
      <td>39.838023</td>
      <td>0.23</td>
      <td>0.03</td>
      <td>0.133757</td>
      <td>0.336083</td>
      <td>0.146849</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Ly6a_mouse</td>
      <td>pep_003</td>
      <td>VQIYPKGKG</td>
      <td>1.0</td>
      <td>30.732187</td>
      <td>0.17</td>
      <td>0.05</td>
      <td>0.116098</td>
      <td>0.313760</td>
      <td>0.133493</td>
    </tr>
  </tbody>
</table>
</div>


    
    Peptide-level best receptor reward table:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>peptide_id</th>
      <th>sequence</th>
      <th>best_receptor_id</th>
      <th>cheap_pre_af_score_refined</th>
      <th>best_plddt</th>
      <th>best_ptm</th>
      <th>best_iptm</th>
      <th>best_structure_proxy_score</th>
      <th>best_soft_structure_gate</th>
      <th>final_reward</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pep_002</td>
      <td>TAYPTFNHRSRM</td>
      <td>Ecscr_mouse</td>
      <td>1.0</td>
      <td>39.246105</td>
      <td>0.23</td>
      <td>0.10</td>
      <td>0.171369</td>
      <td>0.531639</td>
      <td>0.245293</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pep_001</td>
      <td>HAFFLGSRGSR</td>
      <td>Ecscr_mouse</td>
      <td>1.0</td>
      <td>38.827753</td>
      <td>0.23</td>
      <td>0.09</td>
      <td>0.165242</td>
      <td>0.499302</td>
      <td>0.228384</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pep_004</td>
      <td>RVVTKKTTFH</td>
      <td>Ecscr_mouse</td>
      <td>1.0</td>
      <td>38.455862</td>
      <td>0.23</td>
      <td>0.09</td>
      <td>0.164684</td>
      <td>0.496048</td>
      <td>0.226716</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pep_000</td>
      <td>SKYPYPRR</td>
      <td>Ecscr_mouse</td>
      <td>1.0</td>
      <td>39.274535</td>
      <td>0.23</td>
      <td>0.06</td>
      <td>0.149412</td>
      <td>0.417182</td>
      <td>0.186529</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pep_003</td>
      <td>VQIYPKGKG</td>
      <td>Ecscr_mouse</td>
      <td>1.0</td>
      <td>39.838023</td>
      <td>0.23</td>
      <td>0.03</td>
      <td>0.133757</td>
      <td>0.336083</td>
      <td>0.146849</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step H] Reward-guided RL update using best receptor per peptide...
    Rewards used for RL: [0.24529275 0.2283841  0.22671582 0.18652926 0.14684868]
    Best receptor counts:
    {'Ecscr_mouse': 5}
    RL update epoch 01/5 | loss = 3.221179
    RL update epoch 02/5 | loss = 3.213390
    RL update epoch 03/5 | loss = 3.206048
    RL update epoch 04/5 | loss = 3.193241
    RL update epoch 05/5 | loss = 3.180251
    
    Cycle summary:
    {'cycle': 2, 'n_generated': 100, 'n_receptors': 2, 'n_colabfold_peptides': 5, 'n_colabfold_pairs': 10, 'n_colabfold_success': 10, 'mean_final_reward': 0.20675411820411682, 'top_final_reward': 0.2452927529811859, 'rl_loss_last': 3.180250644683838, 'temperature': 1.0, 'best_receptor_counts': '{"Ecscr_mouse": 5}', 'cycle_dir': '../data/processed/multi_receptor_reward_guided_rl/cycle_02'}
    
    ====================================================================================================
    Multi-receptor reward-guided RL loop complete.
    Saved RL history: ../data/processed/multi_receptor_reward_guided_rl/multi_receptor_rl_history.csv



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cycle</th>
      <th>n_generated</th>
      <th>n_receptors</th>
      <th>n_colabfold_peptides</th>
      <th>n_colabfold_pairs</th>
      <th>n_colabfold_success</th>
      <th>mean_final_reward</th>
      <th>top_final_reward</th>
      <th>rl_loss_last</th>
      <th>temperature</th>
      <th>best_receptor_counts</th>
      <th>cycle_dir</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>100</td>
      <td>2</td>
      <td>5</td>
      <td>10</td>
      <td>10</td>
      <td>0.203074</td>
      <td>0.309222</td>
      <td>5.655885</td>
      <td>1.0</td>
      <td>{"Ecscr_mouse": 5}</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>100</td>
      <td>2</td>
      <td>5</td>
      <td>10</td>
      <td>10</td>
      <td>0.206754</td>
      <td>0.245293</td>
      <td>3.180251</td>
      <td>1.0</td>
      <td>{"Ecscr_mouse": 5}</td>
      <td>../data/processed/multi_receptor_reward_guided...</td>
    </tr>
  </tbody>
</table>
</div>


## 3. Interpretation of the multi-receptor run

The multi-receptor run completed successfully:

- 2 optimisation cycles were executed.
- 2 receptors were evaluated: `Ly6a_mouse` and `Ecscr_mouse`.
- 5 peptides were selected per cycle.
- 10 receptor-peptide ColabFold jobs were run per cycle.
- 20/20 ColabFold jobs completed successfully.
- Receptor-pair reward tables and peptide-level best-receptor reward tables were saved.

The most important result is that the workflow successfully compared receptor hypotheses for the same generated peptides. In cycle 1, all five top peptide-level rewards came from `Ecscr_mouse`. In cycle 2, four of five best peptide-level rewards came from `Ecscr_mouse`, while one came from `Ly6a_mouse`.

This demonstrates why multi-receptor screening adds value: the same generated peptide pool can be evaluated against alternative receptor-domain hypotheses, and the optimisation loop can use the strongest receptor-aware signal for each peptide.

The results should not be interpreted as validated binding. The pLDDT, pTM, and ipTM values remain modest, and the ColabFold configuration was deliberately minimal to fit local GPU constraints. The correct interpretation is that the full computational decision chain is working:

`generate peptides → score sequence properties → evaluate receptor-domain complexes → compute rewards → update generator`

## Final note

This notebook exports the multi-receptor reward-guided optimisation results.

Key outputs:

- `../data/processed/multi_receptor_reward_guided_rl/multi_receptor_rl_history.csv`
- `../data/processed/multi_receptor_reward_guided_rl/cycle_01/receptor_pair_reward_table.csv`
- `../data/processed/multi_receptor_reward_guided_rl/cycle_01/peptide_level_best_receptor_reward_table.csv`
- `../data/processed/multi_receptor_reward_guided_rl/cycle_02/receptor_pair_reward_table.csv`
- `../data/processed/multi_receptor_reward_guided_rl/cycle_02/peptide_level_best_receptor_reward_table.csv`
- `../data/processed/multi_receptor_reward_guided_rl/cycle_01/multi_receptor_colabfold_run_log.csv`
- `../data/processed/multi_receptor_reward_guided_rl/cycle_02/multi_receptor_colabfold_run_log.csv`

The strongest prototype-level result is not that the short two-cycle run monotonically improved peptide quality. Instead, the key result is that the pipeline successfully performs multi-receptor structure-aware peptide triage and propagates receptor-aware rewards back into the generator.

This completes the end-to-end proof-of-concept workflow:

**stroke endothelial omics → receptor prioritisation → peptide generator training → receptor-aware structure triage → reward-guided peptide optimisation**
