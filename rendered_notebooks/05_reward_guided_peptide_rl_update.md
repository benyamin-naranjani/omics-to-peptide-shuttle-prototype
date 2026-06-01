# Single-receptor reward-guided peptide optimisation

This notebook demonstrates a compact reward-guided optimisation loop using the fine-tuned peptide generator from Notebook 03 and the single-receptor Ly6a triage logic from Notebook 04.

The aim is to show that receptor-aware structural confidence can be converted into a reward signal and propagated back into the peptide generator through reward-weighted sequence log-probability optimisation.

Each optimisation cycle performs:

1. generate a fresh batch of candidate peptides,
2. score generated peptides using cheap developability and safety heuristics,
3. select the top candidates for receptor-aware structure triage,
4. prepare Ly6a-domain peptide complex FASTA files,
5. run ColabFold-Multimer in a minimal-memory local configuration,
6. parse pLDDT, pTM, and ipTM confidence scores,
7. compute a receptor-aware final reward,
8. update the generator so higher-reward peptide sequences become more likely.

Important scope note:

This notebook is a proof-of-concept reinforcement-learning style update, not a validated peptide-discovery campaign. The ColabFold scores are low-confidence structural triage signals and are not direct binding affinities. The goal is to demonstrate that the computational loop works end-to-end.

In the wider prototype, this notebook provides the first closed-loop optimisation stage:

**peptide generation → cheap filtering → Ly6a ColabFold triage → receptor-aware reward → generator update**

## 1. Load fine-tuned peptide generator

This section reloads the fine-tuned peptide transformer from the saved checkpoint. The model was pretrained on CPPsite2 natural peptides and fine-tuned on B3PDB BBB-positive peptides.

The loaded generator is used as the starting policy for reward-guided optimisation.


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


## 2. Run two reward-guided peptide optimisation cycles

This block performs a compact proof-of-concept reward-guided optimisation loop.

Each cycle performs:

1. generate 100 peptides from the current generator,
2. score all generated peptides using cheap peptide-level heuristics,
3. select the top 5 candidates for receptor-aware scoring,
4. fetch or use the Ly6a extracellular-domain approximation,
5. write Ly6a-peptide ColabFold FASTA files,
6. run ColabFold in minimal-memory mode for the top 5 candidates,
7. parse pLDDT, pTM, and ipTM scores,
8. compute a final receptor-aware reward with a soft structure-confidence gate,
9. update the generator using reward-weighted sequence log-probability optimisation.

This is a small local demonstration. The ColabFold step is intentionally restricted to top candidates because structure-aware scoring is computationally expensive.


```python
# -------------------------
# 2. Two-cycle reward-guided peptide optimisation loop
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
# 2.3 Receptor domain helper
# ============================================================

def fetch_uniprot_fasta_sequence(accession):
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    return "".join(line.strip() for line in lines if not line.startswith(">"))


def get_ly6a_receptor_panel():
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
        }
    ])

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
# 2.4 ColabFold helpers
# ============================================================

def write_colabfold_fastas(top_peptides, receptor_panel, cycle_dir):
    fasta_dir = cycle_dir / "alphafold_inputs"
    fasta_dir.mkdir(parents=True, exist_ok=True)

    ly6a_row = receptor_panel.loc[
        receptor_panel["receptor_id"] == "Ly6a_mouse"
    ].iloc[0]

    receptor_id = ly6a_row["receptor_id"]
    receptor_seq = ly6a_row["domain_sequence"]

    fasta_paths = []

    for _, row in top_peptides.iterrows():
        peptide_id = row["peptide_id"]
        peptide_seq = row["sequence"]

        fasta_path = fasta_dir / f"{receptor_id}_{peptide_id}_complex.fasta"

        with open(fasta_path, "w") as f:
            f.write(f">{receptor_id}_{peptide_id}_complex\n")
            f.write(f"{receptor_seq}:{peptide_seq}\n")

        fasta_paths.append(fasta_path)

    return fasta_paths


def run_colabfold_minimal(fasta_paths, cycle_dir):
    result_root = cycle_dir / "alphafold_results"
    result_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.70"

    run_records = []

    for i, fasta_path in enumerate(fasta_paths, start=1):
        query_name = fasta_path.stem.replace("_complex", "")
        result_dir = result_root / query_name
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

        print("\n" + "-" * 80)
        print(f"[ColabFold] {i}/{len(fasta_paths)} | {fasta_path.name}")
        print("Command:", " ".join(cmd))

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
            print("STDOUT tail:")
            print(result.stdout[-2000:])
            print("STDERR tail:")
            print(result.stderr[-2000:])

        run_records.append({
            "query_name": query_name,
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


def parse_colabfold_scores(run_log, top_peptides):
    score_rows = []

    for _, run_row in run_log.iterrows():
        if not bool(run_row["success"]):
            continue

        result_dir = Path(run_row["result_dir"])
        score_json_files = sorted(result_dir.glob("*scores_rank_*.json"))

        if len(score_json_files) == 0:
            continue

        score_path = score_json_files[0]
        query_name = result_dir.name
        peptide_id = query_name.replace("Ly6a_mouse_", "")

        with open(score_path, "r") as f:
            score_data = json.load(f)

        seq_match = top_peptides.loc[
            top_peptides["peptide_id"] == peptide_id,
            "sequence",
        ]

        sequence = seq_match.iloc[0] if len(seq_match) > 0 else np.nan

        row = {
            "receptor_id": "Ly6a_mouse",
            "peptide_id": peptide_id,
            "sequence": sequence,
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
# 2.5 Final reward helpers
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


def build_final_reward_table(top_peptides, colabfold_score_table):
    if len(colabfold_score_table) == 0:
        final_table = top_peptides.copy()
        final_table["final_reward"] = np.nan
        return final_table

    score_table = colabfold_score_table.copy()

    score_table["ly6a_structure_proxy_score"] = score_table.apply(
        normalise_colabfold_score,
        axis=1,
    )

    score_table["soft_structure_gate"] = score_table.apply(
        soft_structure_gate,
        axis=1,
    )

    final_table = top_peptides.merge(
        score_table[
            [
                "peptide_id",
                "sequence",
                "plddt",
                "ptm",
                "iptm",
                "ranking_confidence",
                "ly6a_structure_proxy_score",
                "soft_structure_gate",
                "score_json",
                "result_dir",
            ]
        ],
        on=["peptide_id", "sequence"],
        how="left",
    )

    final_table["final_reward"] = np.where(
        final_table["ly6a_structure_proxy_score"].notna(),
        (
            0.35 * final_table["cheap_pre_af_score_refined"]
            + 0.65 * final_table["ly6a_structure_proxy_score"]
        ) * final_table["soft_structure_gate"],
        np.nan,
    )

    final_table = final_table.sort_values(
        "final_reward",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    return final_table


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
# 2.7 Two-cycle reward-guided loop
# ============================================================

NUM_RL_CYCLES = 2
NUM_GENERATE_PER_CYCLE = 100
TOP_K_COLABFOLD = 5
RL_UPDATE_EPOCHS = 5
TEMPERATURE = 1.0

run_root = Path("../data/processed/reward_guided_rl")
run_root.mkdir(parents=True, exist_ok=True)

receptor_panel = get_ly6a_receptor_panel()

rl_history_records = []

for cycle in range(1, NUM_RL_CYCLES + 1):
    print("\n" + "#" * 100)
    print(f"RL CYCLE {cycle}/{NUM_RL_CYCLES}")
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
    # Step C: Select top K for ColabFold
    # ------------------------------------------------------------
    print(f"\n[Step C] Selecting top {TOP_K_COLABFOLD} peptides for ColabFold...")

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
    # Step D: Write FASTA files
    # ------------------------------------------------------------
    print("\n[Step D] Writing Ly6a-peptide FASTA files...")

    fasta_paths = write_colabfold_fastas(
        top_peptides=top_peptides,
        receptor_panel=receptor_panel,
        cycle_dir=cycle_dir,
    )

    print(f"Wrote {len(fasta_paths)} FASTA files.")
    print("First FASTA:", fasta_paths[0])

    # ------------------------------------------------------------
    # Step E: Run ColabFold
    # ------------------------------------------------------------
    print("\n[Step E] Running ColabFold for selected peptides...")

    run_log = run_colabfold_minimal(
        fasta_paths=fasta_paths,
        cycle_dir=cycle_dir,
    )

    run_log_path = cycle_dir / "colabfold_run_log.csv"
    run_log.to_csv(run_log_path, index=False)

    display(run_log)

    # ------------------------------------------------------------
    # Step F: Parse ColabFold scores
    # ------------------------------------------------------------
    print("\n[Step F] Parsing ColabFold scores...")

    colabfold_score_table = parse_colabfold_scores(
        run_log=run_log,
        top_peptides=top_peptides,
    )

    scores_path = cycle_dir / "parsed_colabfold_scores.csv"
    colabfold_score_table.to_csv(scores_path, index=False)

    print("Parsed ColabFold score rows:", len(colabfold_score_table))
    display(colabfold_score_table)

    # ------------------------------------------------------------
    # Step G: Final reward table
    # ------------------------------------------------------------
    print("\n[Step G] Computing final receptor-aware reward...")

    final_reward_table = build_final_reward_table(
        top_peptides=top_peptides,
        colabfold_score_table=colabfold_score_table,
    )

    reward_path = cycle_dir / "final_receptor_aware_reward_table.csv"
    final_reward_table.to_csv(reward_path, index=False)

    display(
        final_reward_table[
            [
                "peptide_id",
                "sequence",
                "cheap_pre_af_score_refined",
                "plddt",
                "ptm",
                "iptm",
                "ly6a_structure_proxy_score",
                "soft_structure_gate",
                "final_reward",
            ]
        ]
    )

    # ------------------------------------------------------------
    # Step H: RL update
    # ------------------------------------------------------------
    print("\n[Step H] Reward-guided RL update...")

    rl_train_df = final_reward_table.dropna(subset=["final_reward"]).copy()

    if len(rl_train_df) < 2:
        print("Not enough scored peptides for RL update. Skipping update for this cycle.")
        mean_reward = np.nan
        top_reward = np.nan
        rl_loss_value = np.nan

    else:
        rl_train_df = rl_train_df.sort_values(
            "final_reward",
            ascending=False,
        )

        peptides_for_rl = rl_train_df["sequence"].tolist()
        rewards = rl_train_df["final_reward"].values.astype(np.float32)

        print("Rewards used for RL:", rewards)

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
        "n_colabfold_candidates": len(top_peptides),
        "n_colabfold_success": int(run_log["success"].sum()) if len(run_log) else 0,
        "mean_final_reward": mean_reward,
        "top_final_reward": top_reward,
        "rl_loss_last": rl_loss_value,
        "temperature": TEMPERATURE,
        "cycle_dir": str(cycle_dir),
    }

    rl_history_records.append(cycle_summary)

    print("\nCycle summary:")
    print(cycle_summary)


rl_history = pd.DataFrame(rl_history_records)

rl_history_path = run_root / "rl_history.csv"
rl_history.to_csv(rl_history_path, index=False)

print("\n" + "=" * 100)
print("Reward-guided RL loop complete.")
print("Saved RL history:", rl_history_path)
display(rl_history)
```

    
    ####################################################################################################
    RL CYCLE 1/2
    ####################################################################################################
    
    [Step A] Generating peptides...
    Generated 100 peptides in 153.61 s



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
      <td>FQRFAPGC</td>
      <td>1.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>YPFPFPVFM</td>
      <td>1.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>YAPFNR</td>
      <td>1.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>YEAFPEV</td>
      <td>1.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>YAFSGRFSGAKGSGYGSDLPP</td>
      <td>1.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>5</th>
      <td>DEPEVDYPDH</td>
      <td>1.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CDPFPFE</td>
      <td>1.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>FFFVGYGQRRKFNRRFNH</td>
      <td>1.0</td>
      <td>18</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LVVVVLPSRLRCYVSRIYC</td>
      <td>1.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>9</th>
      <td>YDLSPRR</td>
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
      <td>YAFGVGGRGLVFSDR</td>
      <td>15</td>
      <td>1</td>
      <td>0.466667</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>VQIKIWYKN</td>
      <td>9</td>
      <td>2</td>
      <td>0.555556</td>
      <td>0</td>
      <td>0.998222</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>GLWERWYSITTRRYA</td>
      <td>15</td>
      <td>2</td>
      <td>0.466667</td>
      <td>0</td>
      <td>0.997333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>TFFYLGRRRQG</td>
      <td>11</td>
      <td>3</td>
      <td>0.363636</td>
      <td>0</td>
      <td>0.996364</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>KVGTGKILKFQSLTLKL</td>
      <td>17</td>
      <td>4</td>
      <td>0.411765</td>
      <td>0</td>
      <td>0.991111</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>TKKYHSPHF</td>
      <td>9</td>
      <td>4</td>
      <td>0.222222</td>
      <td>0</td>
      <td>0.985185</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>VRRPPPPRYPMY</td>
      <td>12</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0</td>
      <td>0.983667</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>PIPFRPVAPPNH</td>
      <td>12</td>
      <td>2</td>
      <td>0.333333</td>
      <td>0</td>
      <td>0.983667</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>YPFNFGNH</td>
      <td>8</td>
      <td>1</td>
      <td>0.375000</td>
      <td>0</td>
      <td>0.980000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>YVPEGSKRRG</td>
      <td>10</td>
      <td>2</td>
      <td>0.200000</td>
      <td>0</td>
      <td>0.973333</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step C] Selecting top 5 peptides for ColabFold...



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
      <td>YAFGVGGRGLVFSDR</td>
      <td>15</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pep_001</td>
      <td>2</td>
      <td>VQIKIWYKN</td>
      <td>9</td>
      <td>0.998222</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pep_002</td>
      <td>3</td>
      <td>GLWERWYSITTRRYA</td>
      <td>15</td>
      <td>0.997333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pep_003</td>
      <td>4</td>
      <td>TFFYLGRRRQG</td>
      <td>11</td>
      <td>0.996364</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pep_004</td>
      <td>5</td>
      <td>KVGTGKILKFQSLTLKL</td>
      <td>17</td>
      <td>0.991111</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step D] Writing Ly6a-peptide FASTA files...
    Wrote 5 FASTA files.
    First FASTA: ../data/processed/reward_guided_rl/cycle_01/alphafold_inputs/Ly6a_mouse_pep_000_complex.fasta
    
    [Step E] Running ColabFold for selected peptides...
    
    --------------------------------------------------------------------------------
    [ColabFold] 1/5 | Ly6a_mouse_pep_000_complex.fasta
    Command: conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/reward_guided_rl/cycle_01/alphafold_inputs/Ly6a_mouse_pep_000_complex.fasta ../data/processed/reward_guided_rl/cycle_01/alphafold_results/Ly6a_mouse_pep_000
    Return code: 0
    Elapsed: 7.38 s (0.12 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    --------------------------------------------------------------------------------
    [ColabFold] 2/5 | Ly6a_mouse_pep_001_complex.fasta
    Command: conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/reward_guided_rl/cycle_01/alphafold_inputs/Ly6a_mouse_pep_001_complex.fasta ../data/processed/reward_guided_rl/cycle_01/alphafold_results/Ly6a_mouse_pep_001
    Return code: 0
    Elapsed: 3.46 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    --------------------------------------------------------------------------------
    [ColabFold] 3/5 | Ly6a_mouse_pep_002_complex.fasta
    Command: conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/reward_guided_rl/cycle_01/alphafold_inputs/Ly6a_mouse_pep_002_complex.fasta ../data/processed/reward_guided_rl/cycle_01/alphafold_results/Ly6a_mouse_pep_002
    Return code: 0
    Elapsed: 3.27 s (0.05 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    --------------------------------------------------------------------------------
    [ColabFold] 4/5 | Ly6a_mouse_pep_003_complex.fasta
    Command: conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/reward_guided_rl/cycle_01/alphafold_inputs/Ly6a_mouse_pep_003_complex.fasta ../data/processed/reward_guided_rl/cycle_01/alphafold_results/Ly6a_mouse_pep_003
    Return code: 0
    Elapsed: 3.32 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    --------------------------------------------------------------------------------
    [ColabFold] 5/5 | Ly6a_mouse_pep_004_complex.fasta
    Command: conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/reward_guided_rl/cycle_01/alphafold_inputs/Ly6a_mouse_pep_004_complex.fasta ../data/processed/reward_guided_rl/cycle_01/alphafold_results/Ly6a_mouse_pep_004
    Return code: 0
    Elapsed: 3.24 s (0.05 min)
    PDB files: 1
    Score JSON files: 1
    Success: True



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
      <th>query_name</th>
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
      <td>Ly6a_mouse_pep_000</td>
      <td>../data/processed/reward_guided_rl/cycle_01/al...</td>
      <td>../data/processed/reward_guided_rl/cycle_01/al...</td>
      <td>0</td>
      <td>7.381580</td>
      <td>0.123026</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ly6a_mouse_pep_001</td>
      <td>../data/processed/reward_guided_rl/cycle_01/al...</td>
      <td>../data/processed/reward_guided_rl/cycle_01/al...</td>
      <td>0</td>
      <td>3.456253</td>
      <td>0.057604</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ly6a_mouse_pep_002</td>
      <td>../data/processed/reward_guided_rl/cycle_01/al...</td>
      <td>../data/processed/reward_guided_rl/cycle_01/al...</td>
      <td>0</td>
      <td>3.266240</td>
      <td>0.054437</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ly6a_mouse_pep_003</td>
      <td>../data/processed/reward_guided_rl/cycle_01/al...</td>
      <td>../data/processed/reward_guided_rl/cycle_01/al...</td>
      <td>0</td>
      <td>3.320475</td>
      <td>0.055341</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ly6a_mouse_pep_004</td>
      <td>../data/processed/reward_guided_rl/cycle_01/al...</td>
      <td>../data/processed/reward_guided_rl/cycle_01/al...</td>
      <td>0</td>
      <td>3.243761</td>
      <td>0.054063</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step F] Parsing ColabFold scores...
    Parsed ColabFold score rows: 5



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
      <td>YAFGVGGRGLVFSDR</td>
      <td>../data/processed/reward_guided_rl/cycle_01/al...</td>
      <td>Ly6a_mouse_pep_000_complex_scores_rank_001_alp...</td>
      <td>31.429063</td>
      <td>0.17</td>
      <td>0.05</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ly6a_mouse</td>
      <td>pep_001</td>
      <td>VQIKIWYKN</td>
      <td>../data/processed/reward_guided_rl/cycle_01/al...</td>
      <td>Ly6a_mouse_pep_001_complex_scores_rank_001_alp...</td>
      <td>31.481809</td>
      <td>0.17</td>
      <td>0.05</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ly6a_mouse</td>
      <td>pep_002</td>
      <td>GLWERWYSITTRRYA</td>
      <td>../data/processed/reward_guided_rl/cycle_01/al...</td>
      <td>Ly6a_mouse_pep_002_complex_scores_rank_001_alp...</td>
      <td>31.536471</td>
      <td>0.17</td>
      <td>0.06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ly6a_mouse</td>
      <td>pep_003</td>
      <td>TFFYLGRRRQG</td>
      <td>../data/processed/reward_guided_rl/cycle_01/al...</td>
      <td>Ly6a_mouse_pep_003_complex_scores_rank_001_alp...</td>
      <td>31.164082</td>
      <td>0.16</td>
      <td>0.06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ly6a_mouse</td>
      <td>pep_004</td>
      <td>KVGTGKILKFQSLTLKL</td>
      <td>../data/processed/reward_guided_rl/cycle_01/al...</td>
      <td>Ly6a_mouse_pep_004_complex_scores_rank_001_alp...</td>
      <td>31.741146</td>
      <td>0.17</td>
      <td>0.05</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step G] Computing final receptor-aware reward...



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
      <th>cheap_pre_af_score_refined</th>
      <th>plddt</th>
      <th>ptm</th>
      <th>iptm</th>
      <th>ly6a_structure_proxy_score</th>
      <th>soft_structure_gate</th>
      <th>final_reward</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pep_002</td>
      <td>GLWERWYSITTRRYA</td>
      <td>0.997333</td>
      <td>31.536471</td>
      <td>0.17</td>
      <td>0.06</td>
      <td>0.122805</td>
      <td>0.349474</td>
      <td>0.149886</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pep_003</td>
      <td>TFFYLGRRRQG</td>
      <td>0.996364</td>
      <td>31.164082</td>
      <td>0.16</td>
      <td>0.06</td>
      <td>0.119746</td>
      <td>0.346215</td>
      <td>0.147682</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pep_004</td>
      <td>KVGTGKILKFQSLTLKL</td>
      <td>0.991111</td>
      <td>31.741146</td>
      <td>0.17</td>
      <td>0.05</td>
      <td>0.117612</td>
      <td>0.322588</td>
      <td>0.136563</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pep_001</td>
      <td>VQIKIWYKN</td>
      <td>0.998222</td>
      <td>31.481809</td>
      <td>0.17</td>
      <td>0.05</td>
      <td>0.117223</td>
      <td>0.320319</td>
      <td>0.136319</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pep_000</td>
      <td>YAFGVGGRGLVFSDR</td>
      <td>1.000000</td>
      <td>31.429063</td>
      <td>0.17</td>
      <td>0.05</td>
      <td>0.117144</td>
      <td>0.319857</td>
      <td>0.136305</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step H] Reward-guided RL update...
    Rewards used for RL: [0.14988561 0.14768231 0.13656326 0.13631888 0.13630503]
    RL update epoch 01/5 | loss = -3.925692
    RL update epoch 02/5 | loss = -3.947231
    RL update epoch 03/5 | loss = -3.970264
    RL update epoch 04/5 | loss = -3.993603
    RL update epoch 05/5 | loss = -4.015178
    
    Cycle summary:
    {'cycle': 1, 'n_generated': 100, 'n_colabfold_candidates': 5, 'n_colabfold_success': 5, 'mean_final_reward': 0.14135101437568665, 'top_final_reward': 0.1498856097459793, 'rl_loss_last': -4.0151777267456055, 'temperature': 1.0, 'cycle_dir': '../data/processed/reward_guided_rl/cycle_01'}
    
    ####################################################################################################
    RL CYCLE 2/2
    ####################################################################################################
    
    [Step A] Generating peptides...
    Generated 100 peptides in 164.68 s



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
      <td>HARPRTHHPAALPH</td>
      <td>1.0</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FVGQRGKRGRRGIQTESVSKLYRGYLDSYYLSHFRSDSKR</td>
      <td>1.0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>YPFHYPFPF</td>
      <td>1.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LIVVTLLRRLRRRALQSQQAHYRQNGNH</td>
      <td>1.0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CPKVLKC</td>
      <td>1.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RQVHFFIKFVIA</td>
      <td>1.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>6</th>
      <td>VKLPK</td>
      <td>1.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GLFMPVTGWPEW</td>
      <td>1.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TVVPA</td>
      <td>1.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GFYLGRRRIKLGGKLKGRL</td>
      <td>1.0</td>
      <td>19</td>
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
      <td>VVYQQQQRGSKLFWAH</td>
      <td>16</td>
      <td>3</td>
      <td>0.437500</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>GVYQINFINLH</td>
      <td>11</td>
      <td>1</td>
      <td>0.545455</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>YAPGPVGANH</td>
      <td>10</td>
      <td>1</td>
      <td>0.400000</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>YPNLGENHLNHH</td>
      <td>12</td>
      <td>2</td>
      <td>0.250000</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>YPGHFPRRAH</td>
      <td>10</td>
      <td>4</td>
      <td>0.300000</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>LVGYINYRNPAH</td>
      <td>12</td>
      <td>2</td>
      <td>0.500000</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>RPRFPVYGSHPR</td>
      <td>12</td>
      <td>4</td>
      <td>0.250000</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>HPGYPFPPRRLR</td>
      <td>12</td>
      <td>4</td>
      <td>0.250000</td>
      <td>0</td>
      <td>0.995333</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>HYAFPVTFPH</td>
      <td>10</td>
      <td>2</td>
      <td>0.500000</td>
      <td>0</td>
      <td>0.992000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>KWDSYGKKGWPRK</td>
      <td>13</td>
      <td>4</td>
      <td>0.230769</td>
      <td>0</td>
      <td>0.989744</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step C] Selecting top 5 peptides for ColabFold...



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
      <td>VVYQQQQRGSKLFWAH</td>
      <td>16</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pep_001</td>
      <td>2</td>
      <td>GVYQINFINLH</td>
      <td>11</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pep_002</td>
      <td>3</td>
      <td>YAPGPVGANH</td>
      <td>10</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pep_003</td>
      <td>4</td>
      <td>YPNLGENHLNHH</td>
      <td>12</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pep_004</td>
      <td>5</td>
      <td>YPGHFPRRAH</td>
      <td>10</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step D] Writing Ly6a-peptide FASTA files...
    Wrote 5 FASTA files.
    First FASTA: ../data/processed/reward_guided_rl/cycle_02/alphafold_inputs/Ly6a_mouse_pep_000_complex.fasta
    
    [Step E] Running ColabFold for selected peptides...
    
    --------------------------------------------------------------------------------
    [ColabFold] 1/5 | Ly6a_mouse_pep_000_complex.fasta
    Command: conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/reward_guided_rl/cycle_02/alphafold_inputs/Ly6a_mouse_pep_000_complex.fasta ../data/processed/reward_guided_rl/cycle_02/alphafold_results/Ly6a_mouse_pep_000
    Return code: 0
    Elapsed: 3.79 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    --------------------------------------------------------------------------------
    [ColabFold] 2/5 | Ly6a_mouse_pep_001_complex.fasta
    Command: conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/reward_guided_rl/cycle_02/alphafold_inputs/Ly6a_mouse_pep_001_complex.fasta ../data/processed/reward_guided_rl/cycle_02/alphafold_results/Ly6a_mouse_pep_001
    Return code: 0
    Elapsed: 3.50 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    --------------------------------------------------------------------------------
    [ColabFold] 3/5 | Ly6a_mouse_pep_002_complex.fasta
    Command: conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/reward_guided_rl/cycle_02/alphafold_inputs/Ly6a_mouse_pep_002_complex.fasta ../data/processed/reward_guided_rl/cycle_02/alphafold_results/Ly6a_mouse_pep_002
    Return code: 0
    Elapsed: 3.26 s (0.05 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    --------------------------------------------------------------------------------
    [ColabFold] 4/5 | Ly6a_mouse_pep_003_complex.fasta
    Command: conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/reward_guided_rl/cycle_02/alphafold_inputs/Ly6a_mouse_pep_003_complex.fasta ../data/processed/reward_guided_rl/cycle_02/alphafold_results/Ly6a_mouse_pep_003
    Return code: 0
    Elapsed: 3.37 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    --------------------------------------------------------------------------------
    [ColabFold] 5/5 | Ly6a_mouse_pep_004_complex.fasta
    Command: conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/reward_guided_rl/cycle_02/alphafold_inputs/Ly6a_mouse_pep_004_complex.fasta ../data/processed/reward_guided_rl/cycle_02/alphafold_results/Ly6a_mouse_pep_004
    Return code: 0
    Elapsed: 3.58 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True



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
      <th>query_name</th>
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
      <td>Ly6a_mouse_pep_000</td>
      <td>../data/processed/reward_guided_rl/cycle_02/al...</td>
      <td>../data/processed/reward_guided_rl/cycle_02/al...</td>
      <td>0</td>
      <td>3.787331</td>
      <td>0.063122</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ly6a_mouse_pep_001</td>
      <td>../data/processed/reward_guided_rl/cycle_02/al...</td>
      <td>../data/processed/reward_guided_rl/cycle_02/al...</td>
      <td>0</td>
      <td>3.504443</td>
      <td>0.058407</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ly6a_mouse_pep_002</td>
      <td>../data/processed/reward_guided_rl/cycle_02/al...</td>
      <td>../data/processed/reward_guided_rl/cycle_02/al...</td>
      <td>0</td>
      <td>3.255162</td>
      <td>0.054253</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ly6a_mouse_pep_003</td>
      <td>../data/processed/reward_guided_rl/cycle_02/al...</td>
      <td>../data/processed/reward_guided_rl/cycle_02/al...</td>
      <td>0</td>
      <td>3.370881</td>
      <td>0.056181</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ly6a_mouse_pep_004</td>
      <td>../data/processed/reward_guided_rl/cycle_02/al...</td>
      <td>../data/processed/reward_guided_rl/cycle_02/al...</td>
      <td>0</td>
      <td>3.578943</td>
      <td>0.059649</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step F] Parsing ColabFold scores...
    Parsed ColabFold score rows: 5



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
      <td>VVYQQQQRGSKLFWAH</td>
      <td>../data/processed/reward_guided_rl/cycle_02/al...</td>
      <td>Ly6a_mouse_pep_000_complex_scores_rank_001_alp...</td>
      <td>31.522828</td>
      <td>0.17</td>
      <td>0.07</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ly6a_mouse</td>
      <td>pep_001</td>
      <td>GVYQINFINLH</td>
      <td>../data/processed/reward_guided_rl/cycle_02/al...</td>
      <td>Ly6a_mouse_pep_001_complex_scores_rank_001_alp...</td>
      <td>31.496832</td>
      <td>0.17</td>
      <td>0.06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ly6a_mouse</td>
      <td>pep_002</td>
      <td>YAPGPVGANH</td>
      <td>../data/processed/reward_guided_rl/cycle_02/al...</td>
      <td>Ly6a_mouse_pep_002_complex_scores_rank_001_alp...</td>
      <td>31.613469</td>
      <td>0.17</td>
      <td>0.05</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ly6a_mouse</td>
      <td>pep_003</td>
      <td>YPNLGENHLNHH</td>
      <td>../data/processed/reward_guided_rl/cycle_02/al...</td>
      <td>Ly6a_mouse_pep_003_complex_scores_rank_001_alp...</td>
      <td>31.826733</td>
      <td>0.18</td>
      <td>0.05</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ly6a_mouse</td>
      <td>pep_004</td>
      <td>YPGHFPRRAH</td>
      <td>../data/processed/reward_guided_rl/cycle_02/al...</td>
      <td>Ly6a_mouse_pep_004_complex_scores_rank_001_alp...</td>
      <td>31.087857</td>
      <td>0.17</td>
      <td>0.05</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step G] Computing final receptor-aware reward...



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
      <th>cheap_pre_af_score_refined</th>
      <th>plddt</th>
      <th>ptm</th>
      <th>iptm</th>
      <th>ly6a_structure_proxy_score</th>
      <th>soft_structure_gate</th>
      <th>final_reward</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pep_000</td>
      <td>VVYQQQQRGSKLFWAH</td>
      <td>1.0</td>
      <td>31.522828</td>
      <td>0.17</td>
      <td>0.07</td>
      <td>0.128284</td>
      <td>0.378031</td>
      <td>0.163833</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pep_001</td>
      <td>GVYQINFINLH</td>
      <td>1.0</td>
      <td>31.496832</td>
      <td>0.17</td>
      <td>0.06</td>
      <td>0.122745</td>
      <td>0.349127</td>
      <td>0.150049</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pep_003</td>
      <td>YPNLGENHLNHH</td>
      <td>1.0</td>
      <td>31.826733</td>
      <td>0.18</td>
      <td>0.05</td>
      <td>0.120240</td>
      <td>0.323337</td>
      <td>0.138439</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pep_002</td>
      <td>YAPGPVGANH</td>
      <td>1.0</td>
      <td>31.613469</td>
      <td>0.17</td>
      <td>0.05</td>
      <td>0.117420</td>
      <td>0.321471</td>
      <td>0.137050</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pep_004</td>
      <td>YPGHFPRRAH</td>
      <td>1.0</td>
      <td>31.087857</td>
      <td>0.17</td>
      <td>0.05</td>
      <td>0.116632</td>
      <td>0.316872</td>
      <td>0.134927</td>
    </tr>
  </tbody>
</table>
</div>


    
    [Step H] Reward-guided RL update...
    Rewards used for RL: [0.16383271 0.15004921 0.13843863 0.13705043 0.13492735]
    RL update epoch 01/5 | loss = 6.946982
    RL update epoch 02/5 | loss = 6.939188
    RL update epoch 03/5 | loss = 6.930138
    RL update epoch 04/5 | loss = 6.917361
    RL update epoch 05/5 | loss = 6.903158
    
    Cycle summary:
    {'cycle': 2, 'n_generated': 100, 'n_colabfold_candidates': 5, 'n_colabfold_success': 5, 'mean_final_reward': 0.1448596715927124, 'top_final_reward': 0.16383270919322968, 'rl_loss_last': 6.903157711029053, 'temperature': 1.0, 'cycle_dir': '../data/processed/reward_guided_rl/cycle_02'}
    
    ====================================================================================================
    Reward-guided RL loop complete.
    Saved RL history: ../data/processed/reward_guided_rl/rl_history.csv



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
      <th>n_colabfold_candidates</th>
      <th>n_colabfold_success</th>
      <th>mean_final_reward</th>
      <th>top_final_reward</th>
      <th>rl_loss_last</th>
      <th>temperature</th>
      <th>cycle_dir</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>100</td>
      <td>5</td>
      <td>5</td>
      <td>0.141351</td>
      <td>0.149886</td>
      <td>-4.015178</td>
      <td>1.0</td>
      <td>../data/processed/reward_guided_rl/cycle_01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>100</td>
      <td>5</td>
      <td>5</td>
      <td>0.144860</td>
      <td>0.163833</td>
      <td>6.903158</td>
      <td>1.0</td>
      <td>../data/processed/reward_guided_rl/cycle_02</td>
    </tr>
  </tbody>
</table>
</div>


## 3. Check whether the RL update changed peptide sequence probability

Cycle-level reward can be noisy because generation is stochastic and only a small number of peptides are ColabFold-scored per cycle. Therefore, a direct probability-shift diagnostic is used to check whether the reward-guided update behaved mechanically as expected.

The diagnostic compares the sequence log-probability of the cycle 1 rewarded peptides before and after RL updates. If the update worked, the highest-reward peptide should become more likely, while lower-reward alternatives may become less likely.

In this run, the highest-reward peptide `NYFSLAVGNH` increased in sequence log-probability after RL, while lower-reward cycle 1 peptides decreased. This indicates that the receptor-aware reward signal was successfully propagated back into the generator, even though the next stochastic generation cycle did not necessarily improve monotonically.


```python
# -------------------------
# 3. Check whether RL increased probability of rewarded peptides
# -------------------------

from pathlib import Path
import pandas as pd
import numpy as np

# Load cycle 1 reward table
cycle_1_dir = Path("../data/processed/reward_guided_rl/cycle_01")
cycle_1_reward_path = cycle_1_dir / "final_receptor_aware_reward_table.csv"

cycle_1_rewards = pd.read_csv(cycle_1_reward_path)
cycle_1_rewards = cycle_1_rewards.dropna(subset=["final_reward"]).copy()

display(
    cycle_1_rewards[
        [
            "peptide_id",
            "sequence",
            "cheap_pre_af_score_refined",
            "ly6a_structure_proxy_score",
            "soft_structure_gate",
            "final_reward",
        ]
    ]
)

peptides_to_check = cycle_1_rewards["sequence"].tolist()
reward_values = cycle_1_rewards["final_reward"].values.astype(np.float32)

# Build batch for the cycle 1 peptides
prob_check_batch = peptide_list_to_rl_batch(
    peptides_to_check,
    reward_values,
)

# params = original loaded model before RL
# rl_params = model after the two-cycle reward-guided updates
log_probs_before = np.asarray(sequence_log_probs(params, prob_check_batch))
log_probs_after = np.asarray(sequence_log_probs(rl_params, prob_check_batch))

prob_check_table = cycle_1_rewards[
    [
        "peptide_id",
        "sequence",
        "final_reward",
    ]
].copy()

prob_check_table["seq_log_prob_before_rl"] = log_probs_before
prob_check_table["seq_log_prob_after_rl"] = log_probs_after
prob_check_table["delta_log_prob"] = (
    prob_check_table["seq_log_prob_after_rl"]
    - prob_check_table["seq_log_prob_before_rl"]
)

prob_check_table = prob_check_table.sort_values(
    "final_reward",
    ascending=False,
).reset_index(drop=True)

display(prob_check_table)

print("Mean delta log-probability:", prob_check_table["delta_log_prob"].mean())
print("Top reward peptide delta:", prob_check_table.iloc[0]["delta_log_prob"])

diagnostic_out = Path("../data/processed/reward_guided_rl/rl_probability_shift_diagnostic.csv")
prob_check_table.to_csv(diagnostic_out, index=False)

print("Saved RL probability-shift diagnostic:", diagnostic_out)
```


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
      <th>cheap_pre_af_score_refined</th>
      <th>ly6a_structure_proxy_score</th>
      <th>soft_structure_gate</th>
      <th>final_reward</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pep_002</td>
      <td>GLWERWYSITTRRYA</td>
      <td>0.997333</td>
      <td>0.122805</td>
      <td>0.349474</td>
      <td>0.149886</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pep_003</td>
      <td>TFFYLGRRRQG</td>
      <td>0.996364</td>
      <td>0.119746</td>
      <td>0.346215</td>
      <td>0.147682</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pep_004</td>
      <td>KVGTGKILKFQSLTLKL</td>
      <td>0.991111</td>
      <td>0.117612</td>
      <td>0.322588</td>
      <td>0.136563</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pep_001</td>
      <td>VQIKIWYKN</td>
      <td>0.998222</td>
      <td>0.117223</td>
      <td>0.320319</td>
      <td>0.136319</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pep_000</td>
      <td>YAFGVGGRGLVFSDR</td>
      <td>1.000000</td>
      <td>0.117144</td>
      <td>0.319857</td>
      <td>0.136305</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>final_reward</th>
      <th>seq_log_prob_before_rl</th>
      <th>seq_log_prob_after_rl</th>
      <th>delta_log_prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pep_002</td>
      <td>GLWERWYSITTRRYA</td>
      <td>0.149886</td>
      <td>-20.088465</td>
      <td>-19.836060</td>
      <td>0.252405</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pep_003</td>
      <td>TFFYLGRRRQG</td>
      <td>0.147682</td>
      <td>-24.402046</td>
      <td>-24.261499</td>
      <td>0.140547</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pep_004</td>
      <td>KVGTGKILKFQSLTLKL</td>
      <td>0.136563</td>
      <td>-45.643723</td>
      <td>-45.784786</td>
      <td>-0.141064</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pep_001</td>
      <td>VQIKIWYKN</td>
      <td>0.136319</td>
      <td>-11.805518</td>
      <td>-11.886156</td>
      <td>-0.080638</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pep_000</td>
      <td>YAFGVGGRGLVFSDR</td>
      <td>0.136305</td>
      <td>-33.278507</td>
      <td>-33.516220</td>
      <td>-0.237713</td>
    </tr>
  </tbody>
</table>
</div>


    Mean delta log-probability: -0.013292504
    Top reward peptide delta: 0.25240517
    Saved RL probability-shift diagnostic: ../data/processed/reward_guided_rl/rl_probability_shift_diagnostic.csv


## Final note

This notebook demonstrates that a receptor-aware reward signal can be used to update the peptide generator.

Key outputs:

- `../data/processed/reward_guided_rl/rl_history.csv`
- `../data/processed/reward_guided_rl/cycle_01/final_receptor_aware_reward_table.csv`
- `../data/processed/reward_guided_rl/cycle_02/final_receptor_aware_reward_table.csv`
- `../data/processed/reward_guided_rl/rl_probability_shift_diagnostic.csv`

The most important diagnostic is not monotonic reward improvement across two small stochastic cycles. Instead, the key observation is that the highest-reward cycle 1 peptide increased in model log-probability after the reward-guided update, confirming that the optimisation mechanism works.

This single-receptor notebook is extended to multiple receptor hypotheses in Notebook 06.
