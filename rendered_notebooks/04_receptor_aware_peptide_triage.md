# Receptor-aware peptide triage with ColabFold

This notebook reloads the fine-tuned peptide generator from Notebook 03 and performs the first receptor-aware triage step. A fresh batch of candidate peptides is generated, filtered using cheap developability and safety heuristics, and then evaluated against an omics-derived receptor-domain hypothesis using ColabFold-Multimer.

The first receptor-aware example focuses on **Ly6a**, the strongest early D02-induced surface-accessible candidate from the endothelial receptor-mining workflow. Ly6a is used here as a proof-of-concept receptor-domain hypothesis, not as a validated peptide-binding receptor.

This notebook performs the following steps:

1. load the fine-tuned peptide generator and metadata,
2. reconstruct the peptide transformer architecture,
3. generate a fresh batch of candidate peptides,
4. score generated peptides using cheap developability and safety heuristics,
5. select top-ranked peptides for receptor-aware structure triage,
6. define the Ly6a receptor-domain hypothesis and prepare ColabFold FASTA files,
7. run a single minimal-memory ColabFold test case,
8. run ColabFold for all selected Ly6a-peptide candidates,
9. parse ColabFold confidence scores,
10. compute a Ly6a structure proxy score and final receptor-aware reward.

Important scope note:

The ColabFold-derived scores are used as **structure-aware confidence signals**, not as direct binding affinities. Low pLDDT, pTM, or ipTM values indicate uncertain complex predictions. This notebook demonstrates the computational triage workflow rather than claiming experimentally validated receptor binding.

In the wider prototype, this notebook provides the first receptor-aware scoring stage:

**fine-tuned peptide generator → candidate peptides → cheap peptide filtering → Ly6a-domain ColabFold triage → receptor-aware reward table**

## 1. Load fine-tuned peptide generator and metadata

This section reloads the peptide transformer trained in Notebook 03. The model was pretrained on CPPsite2 natural peptides and fine-tuned on B3PDB BBB-positive peptides. Here, the saved model is used to generate a new batch of candidate shuttle peptides for receptor-aware triage.


```python
from pathlib import Path
import json
import time

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

import warnings
warnings.filterwarnings("ignore")

print("JAX devices:", jax.devices())

model_dir = Path("../models")
data_dir = Path("../data/processed/peptides")

checkpoint_path = model_dir / "peptide_transformer_cpp_pretrained_b3pdb_finetuned.msgpack"
metadata_path = model_dir / "peptide_transformer_metadata.json"

with open(metadata_path, "r") as f:
    metadata = json.load(f)

print(metadata["model_name"])
print(metadata["training_summary"])
print(metadata["generation_summary"])
```

    JAX devices: [CudaDevice(id=0)]
    peptide_transformer_cpp_pretrained_b3pdb_finetuned
    {'cpp_best_epoch': 52, 'cpp_best_val_loss': 1.9253921111424763, 'cpp_test_loss': 1.9499332308769226, 'cpp_test_perplexity': 7.028218296950155, 'b3p_best_epoch': 77, 'b3p_best_val_loss': 2.7452083826065063, 'b3p_test_loss': 2.773916721343994, 'b3p_test_perplexity': 16.021262100567874}
    {'n_generated': 100, 'n_valid': 100, 'n_unique_valid': 100, 'n_novel_unique': 99, 'valid_fraction': 1.0, 'unique_valid_fraction': 1.0, 'novel_unique_fraction': 0.99}


## 2. Reconstruct peptide transformer architecture

Flax checkpoints store model parameters, but the model architecture must be reconstructed before the parameters can be loaded. The architecture below matches the transformer configuration used during training.


```python
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
        batch_size, seq_len = input_ids.shape

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

    Loaded checkpoint: ../models/peptide_transformer_cpp_pretrained_b3pdb_finetuned.msgpack


## 3. Generate candidate peptides

A fresh batch of candidate shuttle peptides is generated from the fine-tuned transformer. These sequences form the starting pool for receptor-aware triage. Temperature-controlled sampling is used to balance conservativeness and diversity.


```python
# -------------------------
# 3. Generate candidate peptides
# -------------------------

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


t0 = time.perf_counter()

generated_peptides = generate_peptide_batch(
    params=params,
    n=1000,
    temperature=1.0,
    max_generation_len=40,
    min_len=5,
)

t1 = time.perf_counter()

print(f"Generated {len(generated_peptides)} peptides in {t1 - t0:.2f} s")
display(generated_peptides.head(20))
```

    Generated 1000 peptides in 1625.19 s



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
      <td>FPGGRRRGGPESCGLGGPRSRRKLFPKLKLRFCRLRLSG</td>
      <td>1.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PVRPQQPRPLYYYH</td>
      <td>1.0</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>VGKKKRRRRQ</td>
      <td>1.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PLPCPPVPGIMRKPHVGPRPENNINGMGVRCATY</td>
      <td>1.0</td>
      <td>34</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KLFVVVDTVYATRIATITTKV</td>
      <td>1.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>5</th>
      <td>YGYLAEL</td>
      <td>1.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>MVAQQH</td>
      <td>1.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>YAFYC</td>
      <td>1.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>HALYPPW</td>
      <td>1.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>9</th>
      <td>YGRKKRRQRRAAA</td>
      <td>1.0</td>
      <td>13</td>
    </tr>
    <tr>
      <th>10</th>
      <td>TPEFDPWGPNL</td>
      <td>1.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>11</th>
      <td>CYASAFFSNH</td>
      <td>1.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SAYHVGSKNYQTPEATSEECSSRCTCRSY</td>
      <td>1.0</td>
      <td>29</td>
    </tr>
    <tr>
      <th>13</th>
      <td>YDFGG</td>
      <td>1.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CGLGMDFH</td>
      <td>1.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>15</th>
      <td>VQTEYSW</td>
      <td>1.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>16</th>
      <td>EEAGGGG</td>
      <td>1.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>17</th>
      <td>VQIIRPM</td>
      <td>1.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>18</th>
      <td>KLVVKYLTPT</td>
      <td>1.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>19</th>
      <td>DCCPPPYPRPGKCRPRKCCACPRHCCHENRGMLPH</td>
      <td>1.0</td>
      <td>35</td>
    </tr>
  </tbody>
</table>
</div>


## 4. Cheap peptide developability and safety scoring

Before running expensive receptor-aware ColabFold prediction, generated peptides are filtered using lightweight sequence-level heuristics. These scores are not validated toxicity or synthesizability predictors. They are early triage proxies designed to prioritise peptides with favourable length, charge, hydrophobicity, cysteine burden, and overall composition.

Three interpretable scores are calculated:

- **BBB/CPP-like score:** rewards moderate length, mild/moderate positive charge, and balanced hydrophobic/aromatic content.
- **Synthesis/developability score:** penalises excessive length, cysteine-rich sequences, and extreme proline/glycine burden.
- **Safety/toxicity proxy:** penalises extreme cationic charge density, excessive hydrophobicity, and cysteine-rich sequences.

A refined cheap pre-ColabFold score is then calculated using these three scores plus soft tie-breakers for length, charge, and cysteine burden. This prevents many peptides from receiving identical perfect scores and provides a more useful ranking before structure-aware triage.


```python
# -------------------------
# 4. Cheap peptide developability and safety scoring
# -------------------------

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
    """
    Simple charge proxy at physiological pH.
    K/R/H counted positive; D/E counted negative.
    """
    seq = str(seq).upper()
    pos = sum(aa in AA_POSITIVE for aa in seq)
    neg = sum(aa in AA_NEGATIVE for aa in seq)
    return pos - neg


def triangular_score(x, low, ideal_low, ideal_high, high):
    """
    Score between 0 and 1.
    Full score inside the ideal range.
    Linear penalty outside the ideal range.
    """
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
    charge_density = charge / max(length, 1)

    hydrophobic_fraction = count_fraction(seq, AA_HYDROPHOBIC)
    aromatic_fraction = count_fraction(seq, AA_AROMATIC)

    cysteine_count = seq.count("C")
    cysteine_fraction = cysteine_count / max(length, 1)

    proline_fraction = seq.count("P") / max(length, 1)
    glycine_fraction = seq.count("G") / max(length, 1)

    return {
        "sequence": seq,
        "length": length,
        "net_charge_proxy": charge,
        "charge_density": charge_density,
        "hydrophobic_fraction": hydrophobic_fraction,
        "aromatic_fraction": aromatic_fraction,
        "cysteine_count": cysteine_count,
        "cysteine_fraction": cysteine_fraction,
        "proline_fraction": proline_fraction,
        "glycine_fraction": glycine_fraction,
    }


def bbb_cpp_like_score(row):
    """
    Rewards peptide-like BBB/CPP plausibility.
    """
    length_score = triangular_score(
        row["length"],
        low=5,
        ideal_low=7,
        ideal_high=18,
        high=30,
    )

    charge_score = triangular_score(
        row["net_charge_proxy"],
        low=-2,
        ideal_low=1,
        ideal_high=6,
        high=10,
    )

    hydro_score = triangular_score(
        row["hydrophobic_fraction"],
        low=0.10,
        ideal_low=0.25,
        ideal_high=0.55,
        high=0.80,
    )

    aromatic_score = triangular_score(
        row["aromatic_fraction"],
        low=0.00,
        ideal_low=0.05,
        ideal_high=0.25,
        high=0.45,
    )

    return (
        0.35 * length_score
        + 0.30 * charge_score
        + 0.25 * hydro_score
        + 0.10 * aromatic_score
    )


def synthesis_developability_score(row):
    """
    Higher score = more favourable early synthesis/developability profile.
    This is a heuristic, not a validated synthesis model.
    """
    length_score = triangular_score(
        row["length"],
        low=5,
        ideal_low=6,
        ideal_high=20,
        high=35,
    )

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
    """
    Higher score = safer by simple sequence heuristics.
    Penalises extreme cationic charge density, excessive hydrophobicity,
    and cysteine-rich sequences.
    """
    charge_density_penalty = max((row["charge_density"] - 0.45) / 0.45, 0.0)
    hydrophobic_penalty = max((row["hydrophobic_fraction"] - 0.65) / 0.35, 0.0)
    cysteine_penalty = min(row["cysteine_count"] / 4.0, 1.0)

    penalty = (
        0.45 * min(charge_density_penalty, 1.0)
        + 0.35 * min(hydrophobic_penalty, 1.0)
        + 0.20 * cysteine_penalty
    )

    return float(np.clip(1.0 - penalty, 0.0, 1.0))


# Build feature table
peptide_features = pd.DataFrame([
    peptide_feature_row(seq)
    for seq in generated_peptides["sequence"]
])

scored_peptides = generated_peptides.merge(
    peptide_features,
    on="sequence",
    how="left",
)

# Main interpretable scores
scored_peptides["bbb_cpp_like_score"] = scored_peptides.apply(
    bbb_cpp_like_score,
    axis=1,
)

scored_peptides["synthesis_developability_score"] = scored_peptides.apply(
    synthesis_developability_score,
    axis=1,
)

scored_peptides["safety_toxicity_proxy_score"] = scored_peptides.apply(
    safety_toxicity_proxy_score,
    axis=1,
)

# Base cheap pre-ColabFold score
scored_peptides["cheap_pre_af_score"] = (
    0.40 * scored_peptides["bbb_cpp_like_score"]
    + 0.35 * scored_peptides["synthesis_developability_score"]
    + 0.25 * scored_peptides["safety_toxicity_proxy_score"]
)

# Soft tie-breakers to avoid many identical 1.0 scores
scored_peptides["length_preference"] = scored_peptides["length"].apply(
    lambda x: triangular_score(
        x,
        low=5,
        ideal_low=8,
        ideal_high=16,
        high=25,
    )
)

scored_peptides["charge_preference"] = scored_peptides["net_charge_proxy"].apply(
    lambda x: triangular_score(
        x,
        low=0,
        ideal_low=1,
        ideal_high=4,
        high=8,
    )
)

scored_peptides["cysteine_preference"] = 1.0 - np.minimum(
    scored_peptides["cysteine_count"] / 2.0,
    1.0,
)

# Refined score used for ranking before ColabFold
scored_peptides["cheap_pre_af_score_refined"] = (
    0.80 * scored_peptides["cheap_pre_af_score"]
    + 0.08 * scored_peptides["length_preference"]
    + 0.07 * scored_peptides["charge_preference"]
    + 0.05 * scored_peptides["cysteine_preference"]
)

scored_peptides = scored_peptides.sort_values(
    "cheap_pre_af_score_refined",
    ascending=False,
).reset_index(drop=True)

scored_peptides["cheap_rank"] = np.arange(1, len(scored_peptides) + 1)

display(
    scored_peptides[
        [
            "cheap_rank",
            "sequence",
            "n_amino_acids",
            "net_charge_proxy",
            "hydrophobic_fraction",
            "aromatic_fraction",
            "cysteine_count",
            "bbb_cpp_like_score",
            "synthesis_developability_score",
            "safety_toxicity_proxy_score",
            "cheap_pre_af_score",
            "length_preference",
            "charge_preference",
            "cysteine_preference",
            "cheap_pre_af_score_refined",
        ]
    ].head(20)
)

processed_dir = Path("../data/processed/peptides")
processed_dir.mkdir(parents=True, exist_ok=True)

scored_path = processed_dir / "notebook04_generated_peptides_with_cheap_scores.csv"
scored_peptides.to_csv(scored_path, index=False)

print("Saved cheap-scored peptides:", scored_path)
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
      <th>cheap_rank</th>
      <th>sequence</th>
      <th>n_amino_acids</th>
      <th>net_charge_proxy</th>
      <th>hydrophobic_fraction</th>
      <th>aromatic_fraction</th>
      <th>cysteine_count</th>
      <th>bbb_cpp_like_score</th>
      <th>synthesis_developability_score</th>
      <th>safety_toxicity_proxy_score</th>
      <th>cheap_pre_af_score</th>
      <th>length_preference</th>
      <th>charge_preference</th>
      <th>cysteine_preference</th>
      <th>cheap_pre_af_score_refined</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>PVRPQQPRPLYYYH</td>
      <td>14</td>
      <td>3</td>
      <td>0.357143</td>
      <td>0.214286</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>RLFMPMVYPTTG</td>
      <td>12</td>
      <td>1</td>
      <td>0.500000</td>
      <td>0.166667</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>YPRGALPH</td>
      <td>8</td>
      <td>2</td>
      <td>0.375000</td>
      <td>0.125000</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>TRPPWVRPYLNH</td>
      <td>12</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.166667</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>GYFAAGPERRRGY</td>
      <td>13</td>
      <td>2</td>
      <td>0.384615</td>
      <td>0.230769</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>TKFKGEQFSPYP</td>
      <td>12</td>
      <td>1</td>
      <td>0.250000</td>
      <td>0.250000</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>VYRPPRLFVRNH</td>
      <td>12</td>
      <td>4</td>
      <td>0.416667</td>
      <td>0.166667</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>PQIYRPRN</td>
      <td>8</td>
      <td>2</td>
      <td>0.250000</td>
      <td>0.125000</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>IPAYLPTNH</td>
      <td>9</td>
      <td>1</td>
      <td>0.444444</td>
      <td>0.111111</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>VLQQKLPWQQNMKGYP</td>
      <td>16</td>
      <td>2</td>
      <td>0.375000</td>
      <td>0.125000</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>SYGHFPHSPFARP</td>
      <td>13</td>
      <td>3</td>
      <td>0.307692</td>
      <td>0.230769</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>QEFVGYLYGRRGRRLT</td>
      <td>16</td>
      <td>3</td>
      <td>0.375000</td>
      <td>0.187500</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>YAFGEVGRNGR</td>
      <td>11</td>
      <td>1</td>
      <td>0.363636</td>
      <td>0.181818</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>TYYPPKNH</td>
      <td>8</td>
      <td>2</td>
      <td>0.250000</td>
      <td>0.250000</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>TGLPPVWH</td>
      <td>8</td>
      <td>1</td>
      <td>0.375000</td>
      <td>0.125000</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>HAPWEHTVYH</td>
      <td>10</td>
      <td>2</td>
      <td>0.400000</td>
      <td>0.200000</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>VYASKPVTRY</td>
      <td>10</td>
      <td>2</td>
      <td>0.500000</td>
      <td>0.200000</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>KTHKGFPFLNDDVGY</td>
      <td>15</td>
      <td>1</td>
      <td>0.333333</td>
      <td>0.200000</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>TADYDRTRMNH</td>
      <td>11</td>
      <td>1</td>
      <td>0.272727</td>
      <td>0.090909</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>NTKTTFKYAH</td>
      <td>10</td>
      <td>3</td>
      <td>0.300000</td>
      <td>0.200000</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


    Saved cheap-scored peptides: ../data/processed/peptides/notebook04_generated_peptides_with_cheap_scores.csv


## 5. Select top peptides for receptor-aware ColabFold triage

Only the top-ranked peptides from the cheap scoring layer are selected for receptor-aware ColabFold prediction. This keeps the structure-aware step computationally feasible while preserving the most promising candidates.

The selected peptides are not claimed to be binders at this stage. They are candidates with favourable sequence-level properties that will next be evaluated against the Ly6a extracellular-domain hypothesis using ColabFold-Multimer.


```python
# -------------------------
# 5. Select top peptides for receptor-aware ColabFold triage
# -------------------------

N_AF_CANDIDATES = 10

top_af_peptides = scored_peptides.head(N_AF_CANDIDATES).copy()

top_af_peptides["peptide_id"] = [
    f"pep_{i:03d}" for i in range(len(top_af_peptides))
]

display(
    top_af_peptides[
        [
            "peptide_id",
            "cheap_rank",
            "sequence",
            "n_amino_acids",
            "net_charge_proxy",
            "hydrophobic_fraction",
            "aromatic_fraction",
            "cysteine_count",
            "bbb_cpp_like_score",
            "synthesis_developability_score",
            "safety_toxicity_proxy_score",
            "cheap_pre_af_score",
            "length_preference",
            "charge_preference",
            "cysteine_preference",
            "cheap_pre_af_score_refined",
        ]
    ]
)

top_af_path = processed_dir / "top10_peptides_for_ly6a_colabfold.csv"
top_af_peptides.to_csv(top_af_path, index=False)

print("Saved top ColabFold candidates:", top_af_path)
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
      <th>cheap_rank</th>
      <th>sequence</th>
      <th>n_amino_acids</th>
      <th>net_charge_proxy</th>
      <th>hydrophobic_fraction</th>
      <th>aromatic_fraction</th>
      <th>cysteine_count</th>
      <th>bbb_cpp_like_score</th>
      <th>synthesis_developability_score</th>
      <th>safety_toxicity_proxy_score</th>
      <th>cheap_pre_af_score</th>
      <th>length_preference</th>
      <th>charge_preference</th>
      <th>cysteine_preference</th>
      <th>cheap_pre_af_score_refined</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pep_000</td>
      <td>1</td>
      <td>PVRPQQPRPLYYYH</td>
      <td>14</td>
      <td>3</td>
      <td>0.357143</td>
      <td>0.214286</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pep_001</td>
      <td>2</td>
      <td>RLFMPMVYPTTG</td>
      <td>12</td>
      <td>1</td>
      <td>0.500000</td>
      <td>0.166667</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pep_002</td>
      <td>3</td>
      <td>YPRGALPH</td>
      <td>8</td>
      <td>2</td>
      <td>0.375000</td>
      <td>0.125000</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pep_003</td>
      <td>4</td>
      <td>TRPPWVRPYLNH</td>
      <td>12</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.166667</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pep_004</td>
      <td>5</td>
      <td>GYFAAGPERRRGY</td>
      <td>13</td>
      <td>2</td>
      <td>0.384615</td>
      <td>0.230769</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>pep_005</td>
      <td>6</td>
      <td>TKFKGEQFSPYP</td>
      <td>12</td>
      <td>1</td>
      <td>0.250000</td>
      <td>0.250000</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pep_006</td>
      <td>7</td>
      <td>VYRPPRLFVRNH</td>
      <td>12</td>
      <td>4</td>
      <td>0.416667</td>
      <td>0.166667</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>pep_007</td>
      <td>8</td>
      <td>PQIYRPRN</td>
      <td>8</td>
      <td>2</td>
      <td>0.250000</td>
      <td>0.125000</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>pep_008</td>
      <td>9</td>
      <td>IPAYLPTNH</td>
      <td>9</td>
      <td>1</td>
      <td>0.444444</td>
      <td>0.111111</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>pep_009</td>
      <td>10</td>
      <td>VLQQKLPWQQNMKGYP</td>
      <td>16</td>
      <td>2</td>
      <td>0.375000</td>
      <td>0.125000</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


    Saved top ColabFold candidates: ../data/processed/peptides/top10_peptides_for_ly6a_colabfold.csv


## 6. Define Ly6a receptor domain and prepare ColabFold FASTA files

The generated peptides and top ColabFold-selected candidates are saved for reproducibility. The Ly6a receptor-domain hypothesis is prepared from the canonical mouse UniProt sequence by extracting an approximate extracellular mature domain.

For receptor-aware structure prediction, each selected peptide is paired with the Ly6a extracellular-domain approximation in ColabFold complex FASTA format. The colon separates chains: chain A is the Ly6a receptor domain and chain B is the generated peptide.


```python
# -------------------------
# 6. Save generated peptides and prepare receptor-peptide FASTA inputs
# -------------------------

from pathlib import Path
import pandas as pd
import requests

peptide_out_dir = Path("../data/processed/peptides")
receptor_out_dir = Path("../data/processed/receptors")
fasta_dir = Path("../data/processed/alphafold_inputs/ly6a_top10")

peptide_out_dir.mkdir(parents=True, exist_ok=True)
receptor_out_dir.mkdir(parents=True, exist_ok=True)
fasta_dir.mkdir(parents=True, exist_ok=True)

# Save the freshly generated peptides from Notebook 04
generated_peptides_path = peptide_out_dir / "notebook04_generated_peptides.csv"
generated_peptides.to_csv(generated_peptides_path, index=False)

# Save the cheap-scored top peptides selected for ColabFold
top_af_path = peptide_out_dir / "top10_peptides_for_ly6a_colabfold.csv"
top_af_peptides.to_csv(top_af_path, index=False)

print("Saved generated peptides:", generated_peptides_path)
print("Saved top ColabFold candidates:", top_af_path)


# -------------------------
# Define Ly6a receptor-domain hypothesis
# -------------------------

def fetch_uniprot_fasta_sequence(accession):
    """
    Fetch canonical UniProt protein sequence in FASTA format.
    Returns the sequence as a one-letter amino-acid string.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    lines = r.text.strip().splitlines()
    return "".join(line.strip() for line in lines if not line.startswith(">"))


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

receptor_panel_path = receptor_out_dir / "receptor_panel_with_domain_sequences.csv"
receptor_panel.to_csv(receptor_panel_path, index=False)

print("Saved receptor panel:", receptor_panel_path)

display(
    receptor_panel[
        [
            "receptor_id",
            "gene",
            "uniprot_accession",
            "protein_name",
            "domain_used",
            "full_sequence_length",
            "domain_sequence_length",
            "domain_sequence",
            "notes",
        ]
    ]
)


# -------------------------
# Prepare Ly6a receptor-peptide FASTA files
# -------------------------

ly6a_row = receptor_panel.loc[
    receptor_panel["receptor_id"] == "Ly6a_mouse"
].iloc[0]

receptor_id = ly6a_row["receptor_id"]
receptor_seq = ly6a_row["domain_sequence"]

# Save receptor domain alone for reference
receptor_fasta_path = fasta_dir / "Ly6a_mouse_domain_27_112.fasta"

with open(receptor_fasta_path, "w") as f:
    f.write(f">{receptor_id}_domain_27_112\n")
    f.write(f"{receptor_seq}\n")

print("Saved receptor FASTA:", receptor_fasta_path)

# Save one receptor-peptide complex FASTA per selected top peptide
complex_fasta_paths = []

for _, row in top_af_peptides.iterrows():
    peptide_id = row["peptide_id"]
    peptide_seq = row["sequence"]

    complex_fasta_path = fasta_dir / f"{receptor_id}_{peptide_id}_complex.fasta"

    with open(complex_fasta_path, "w") as f:
        f.write(f">{receptor_id}_{peptide_id}_complex\n")
        f.write(f"{receptor_seq}:{peptide_seq}\n")

    complex_fasta_paths.append(complex_fasta_path)

print(f"Saved {len(complex_fasta_paths)} receptor-peptide complex FASTA files.")

print("\nFirst complex FASTA preview:")
first_path = complex_fasta_paths[0]

with open(first_path, "r") as f:
    print(f.read())

print("Receptor domain length:", len(receptor_seq))
print("First peptide length:", len(top_af_peptides.iloc[0]["sequence"]))
print("First complex total length:", len(receptor_seq) + len(top_af_peptides.iloc[0]["sequence"]))
```

    Saved generated peptides: ../data/processed/peptides/notebook04_generated_peptides.csv
    Saved top ColabFold candidates: ../data/processed/peptides/top10_peptides_for_ly6a_colabfold.csv
    Saved receptor panel: ../data/processed/receptors/receptor_panel_with_domain_sequences.csv



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
      <th>full_sequence_length</th>
      <th>domain_sequence_length</th>
      <th>domain_sequence</th>
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
      <td>134</td>
      <td>86</td>
      <td>LECYQCYGVPFETSCPSITCPYPDGVCVTQEAAVIVDSQTRKVKNN...</td>
      <td>GPI-anchored surface protein; mature extracell...</td>
    </tr>
  </tbody>
</table>
</div>


    Saved receptor FASTA: ../data/processed/alphafold_inputs/ly6a_top10/Ly6a_mouse_domain_27_112.fasta
    Saved 10 receptor-peptide complex FASTA files.
    
    First complex FASTA preview:
    >Ly6a_mouse_pep_000_complex
    LECYQCYGVPFETSCPSITCPYPDGVCVTQEAAVIVDSQTRKVKNNLCLPICPPNIESMEILGTKVNVKTSCCQEDLCNVAVPNGG:PVRPQQPRPLYYYH
    
    Receptor domain length: 86
    First peptide length: 14
    First complex total length: 100


## 7. Run one minimal-memory ColabFold receptor-peptide test

A single Ly6a-domain/generated-peptide FASTA file is submitted to ColabFold as a minimal structure-aware triage test. This checks that the separate ColabFold environment can be called from the notebook and that output files are written correctly before scaling to multiple peptides.

To fit the local 8 GB GPU, ColabFold is run in a minimal triage configuration: single-sequence mode, one AlphaFold-Multimer model, one recycle, and reduced JAX memory preallocation.


```python
# -------------------------
# 7. Run one minimal-memory ColabFold receptor-peptide test
# -------------------------

import subprocess
import time
import os
from pathlib import Path

# Use the first selected top-10 FASTA generated in Block 7
single_fasta = sorted(Path("../data/processed/alphafold_inputs/ly6a_top10").glob("*_complex.fasta"))[0]

single_result_dir = Path("../data/processed/alphafold_results/ly6a_top10_test_single")
single_result_dir.mkdir(parents=True, exist_ok=True)

cmd = [
    "conda", "run", "-n", "colabfold",
    "colabfold_batch",
    "--msa-mode", "single_sequence",
    "--model-type", "alphafold2_multimer_v3",
    "--num-models", "1",
    "--num-recycle", "1",
    "--model-order", "1",
    str(single_fasta),
    str(single_result_dir),
]

env = os.environ.copy()
env["MPLBACKEND"] = "Agg"

# Reduce JAX memory pressure on local 8 GB GPU
env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.70"

print("Input FASTA:", single_fasta)
print("Result directory:", single_result_dir)

print("\nRunning command:")
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

print("Return code:", result.returncode)
print(f"Elapsed time: {elapsed:.2f} s ({elapsed / 60:.2f} min)")

print("\nSTDOUT tail:")
print(result.stdout[-5000:])

print("\nSTDERR tail:")
print(result.stderr[-5000:])

output_files = sorted(single_result_dir.glob("*"))

print("\nOutput files:")
for p in output_files:
    print(p.name)

pdb_files = list(single_result_dir.glob("*.pdb"))
json_files = [
    p for p in single_result_dir.glob("*scores_rank_*.json")
]

print("\nPDB files:", len(pdb_files))
print("Score JSON files:", len(json_files))

if len(pdb_files) > 0 and len(json_files) > 0:
    print("ColabFold structure prediction appears successful.")
else:
    print("No structure/score files yet.")
```

    Input FASTA: ../data/processed/alphafold_inputs/ly6a_top10/Ly6a_mouse_pep_000_complex.fasta
    Result directory: ../data/processed/alphafold_results/ly6a_top10_test_single
    
    Running command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/alphafold_inputs/ly6a_top10/Ly6a_mouse_pep_000_complex.fasta ../data/processed/alphafold_results/ly6a_top10_test_single
    Return code: 0
    Elapsed time: 9.44 s (0.16 min)
    
    STDOUT tail:
    2026-05-31 23:07:36,077 Running colabfold 1.6.1
    2026-05-31 23:07:40,952 Running on GPU
    2026-05-31 23:07:41,060 Found 2 citations for tools or databases
    2026-05-31 23:07:41,060 Skipping Ly6a_mouse_pep_000_complex (already done)
    2026-05-31 23:07:41,060 Done
    
    
    STDERR tail:
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261657.619439    8379 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261660.014490    8379 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    
    
    Output files:
    Ly6a_mouse_pep_000_complex.a3m
    Ly6a_mouse_pep_000_complex.done.txt
    Ly6a_mouse_pep_000_complex_coverage.png
    Ly6a_mouse_pep_000_complex_pae.png
    Ly6a_mouse_pep_000_complex_plddt.png
    Ly6a_mouse_pep_000_complex_predicted_aligned_error_v1.json
    Ly6a_mouse_pep_000_complex_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json
    Ly6a_mouse_pep_000_complex_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb
    cite.bibtex
    config.json
    log.txt
    
    PDB files: 1
    Score JSON files: 1
    ColabFold structure prediction appears successful.


## 8. Run ColabFold for all top Ly6a-peptide candidates

After confirming that a single Ly6a-peptide test case runs successfully, the same minimal-memory ColabFold configuration is applied to all selected top peptides. Each receptor-peptide pair is run as a separate FASTA file and written to a separate result folder.

This local run uses a lightweight triage configuration suitable for an 8 GB GPU: single-sequence mode, one AlphaFold-Multimer model, one recycle, and reduced JAX memory preallocation. Existing completed predictions are skipped automatically.


```python
# -------------------------
# 8. Run ColabFold for all top Ly6a-peptide candidates
# -------------------------

import subprocess
import time
import os
from pathlib import Path
import pandas as pd

fasta_dir = Path("../data/processed/alphafold_inputs/ly6a_top10")
batch_result_root = Path("../data/processed/alphafold_results/ly6a_top10_batch")

batch_result_root.mkdir(parents=True, exist_ok=True)

fasta_files = sorted(fasta_dir.glob("*_complex.fasta"))

print(f"Found {len(fasta_files)} FASTA files.")
for f in fasta_files:
    print(f.name)

env = os.environ.copy()
env["MPLBACKEND"] = "Agg"

# Reduce JAX memory pressure on local 8 GB GPU
env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.70"

run_records = []

t_batch_start = time.perf_counter()

for i, fasta_path in enumerate(fasta_files, start=1):
    query_name = fasta_path.stem.replace("_complex", "")
    result_dir = batch_result_root / query_name
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

    print("\n" + "=" * 80)
    print(f"Running {i}/{len(fasta_files)}:", fasta_path.name)
    print("Result directory:", result_dir)
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
    print(f"Elapsed time: {elapsed:.2f} s ({elapsed / 60:.2f} min)")
    print("PDB files:", len(pdb_files))
    print("Score JSON files:", len(score_json_files))
    print("Success:", success)

    print("\nSTDOUT tail:")
    print(result.stdout[-2500:])

    print("\nSTDERR tail:")
    print(result.stderr[-2500:])

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

t_batch_end = time.perf_counter()
batch_elapsed = t_batch_end - t_batch_start

run_log = pd.DataFrame(run_records)

display(run_log)

run_log_path = batch_result_root / "ly6a_top10_colabfold_run_log.csv"
run_log.to_csv(run_log_path, index=False)

print("\nBatch elapsed time:")
print(f"{batch_elapsed:.2f} s ({batch_elapsed / 60:.2f} min)")

print("Saved run log:", run_log_path)
print("Successful runs:", int(run_log["success"].sum()), "/", len(run_log))
```

    Found 10 FASTA files.
    Ly6a_mouse_pep_000_complex.fasta
    Ly6a_mouse_pep_001_complex.fasta
    Ly6a_mouse_pep_002_complex.fasta
    Ly6a_mouse_pep_003_complex.fasta
    Ly6a_mouse_pep_004_complex.fasta
    Ly6a_mouse_pep_005_complex.fasta
    Ly6a_mouse_pep_006_complex.fasta
    Ly6a_mouse_pep_007_complex.fasta
    Ly6a_mouse_pep_008_complex.fasta
    Ly6a_mouse_pep_009_complex.fasta
    
    ================================================================================
    Running 1/10: Ly6a_mouse_pep_000_complex.fasta
    Result directory: ../data/processed/alphafold_results/ly6a_top10_batch/Ly6a_mouse_pep_000
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/alphafold_inputs/ly6a_top10/Ly6a_mouse_pep_000_complex.fasta ../data/processed/alphafold_results/ly6a_top10_batch/Ly6a_mouse_pep_000
    Return code: 0
    Elapsed time: 4.31 s (0.07 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    STDOUT tail:
    2026-05-31 23:07:43,207 Running colabfold 1.6.1
    2026-05-31 23:07:44,969 Running on GPU
    2026-05-31 23:07:45,019 Found 2 citations for tools or databases
    2026-05-31 23:07:45,019 Skipping Ly6a_mouse_pep_000_complex (already done)
    2026-05-31 23:07:45,020 Done
    
    
    STDERR tail:
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261663.708842    8493 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261664.560975    8493 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    
    
    ================================================================================
    Running 2/10: Ly6a_mouse_pep_001_complex.fasta
    Result directory: ../data/processed/alphafold_results/ly6a_top10_batch/Ly6a_mouse_pep_001
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/alphafold_inputs/ly6a_top10/Ly6a_mouse_pep_001_complex.fasta ../data/processed/alphafold_results/ly6a_top10_batch/Ly6a_mouse_pep_001
    Return code: 0
    Elapsed time: 4.16 s (0.07 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    STDOUT tail:
    2026-05-31 23:07:47,480 Running colabfold 1.6.1
    2026-05-31 23:07:48,760 Running on GPU
    2026-05-31 23:07:48,807 Found 2 citations for tools or databases
    2026-05-31 23:07:48,807 Skipping Ly6a_mouse_pep_001_complex (already done)
    2026-05-31 23:07:48,807 Done
    
    
    STDERR tail:
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261667.527389    8607 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261668.359845    8607 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    
    
    ================================================================================
    Running 3/10: Ly6a_mouse_pep_002_complex.fasta
    Result directory: ../data/processed/alphafold_results/ly6a_top10_batch/Ly6a_mouse_pep_002
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/alphafold_inputs/ly6a_top10/Ly6a_mouse_pep_002_complex.fasta ../data/processed/alphafold_results/ly6a_top10_batch/Ly6a_mouse_pep_002
    Return code: 0
    Elapsed time: 4.00 s (0.07 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    STDOUT tail:
    2026-05-31 23:07:51,079 Running colabfold 1.6.1
    2026-05-31 23:07:52,790 Running on GPU
    2026-05-31 23:07:52,834 Found 2 citations for tools or databases
    2026-05-31 23:07:52,836 Skipping Ly6a_mouse_pep_002_complex (already done)
    2026-05-31 23:07:52,836 Done
    
    
    STDERR tail:
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261671.517425    8721 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261688.356754    8721 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    
    
    ================================================================================
    Running 4/10: Ly6a_mouse_pep_003_complex.fasta
    Result directory: ../data/processed/alphafold_results/ly6a_top10_batch/Ly6a_mouse_pep_003
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/alphafold_inputs/ly6a_top10/Ly6a_mouse_pep_003_complex.fasta ../data/processed/alphafold_results/ly6a_top10_batch/Ly6a_mouse_pep_003
    Return code: 0
    Elapsed time: 3.51 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    STDOUT tail:
    2026-05-31 23:07:54,750 Running colabfold 1.6.1
    2026-05-31 23:07:56,300 Running on GPU
    2026-05-31 23:07:56,347 Found 2 citations for tools or databases
    2026-05-31 23:07:56,353 Skipping Ly6a_mouse_pep_003_complex (already done)
    2026-05-31 23:07:56,353 Done
    
    
    STDERR tail:
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261675.068833    8835 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261675.925700    8835 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    
    
    ================================================================================
    Running 5/10: Ly6a_mouse_pep_004_complex.fasta
    Result directory: ../data/processed/alphafold_results/ly6a_top10_batch/Ly6a_mouse_pep_004
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/alphafold_inputs/ly6a_top10/Ly6a_mouse_pep_004_complex.fasta ../data/processed/alphafold_results/ly6a_top10_batch/Ly6a_mouse_pep_004
    Return code: 0
    Elapsed time: 3.67 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    STDOUT tail:
    2026-05-31 23:07:58,350 Running colabfold 1.6.1
    2026-05-31 23:07:59,933 Running on GPU
    2026-05-31 23:07:59,981 Found 2 citations for tools or databases
    2026-05-31 23:07:59,983 Skipping Ly6a_mouse_pep_004_complex (already done)
    2026-05-31 23:07:59,983 Done
    
    
    STDERR tail:
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261678.662636    8949 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261679.527611    8949 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    
    
    ================================================================================
    Running 6/10: Ly6a_mouse_pep_005_complex.fasta
    Result directory: ../data/processed/alphafold_results/ly6a_top10_batch/Ly6a_mouse_pep_005
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/alphafold_inputs/ly6a_top10/Ly6a_mouse_pep_005_complex.fasta ../data/processed/alphafold_results/ly6a_top10_batch/Ly6a_mouse_pep_005
    Return code: 0
    Elapsed time: 3.72 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    STDOUT tail:
    2026-05-31 23:08:02,075 Running colabfold 1.6.1
    2026-05-31 23:08:03,657 Running on GPU
    2026-05-31 23:08:03,707 Found 2 citations for tools or databases
    2026-05-31 23:08:03,713 Skipping Ly6a_mouse_pep_005_complex (already done)
    2026-05-31 23:08:03,713 Done
    
    
    STDERR tail:
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261698.314571    9061 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261683.224563    9061 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    
    
    ================================================================================
    Running 7/10: Ly6a_mouse_pep_006_complex.fasta
    Result directory: ../data/processed/alphafold_results/ly6a_top10_batch/Ly6a_mouse_pep_006
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/alphafold_inputs/ly6a_top10/Ly6a_mouse_pep_006_complex.fasta ../data/processed/alphafold_results/ly6a_top10_batch/Ly6a_mouse_pep_006
    Return code: 0
    Elapsed time: 3.53 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    STDOUT tail:
    2026-05-31 23:08:05,750 Running colabfold 1.6.1
    2026-05-31 23:08:07,222 Running on GPU
    2026-05-31 23:08:07,267 Found 2 citations for tools or databases
    2026-05-31 23:08:07,273 Skipping Ly6a_mouse_pep_006_complex (already done)
    2026-05-31 23:08:07,273 Done
    
    
    STDERR tail:
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261686.051084    9175 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261686.850642    9175 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    
    
    ================================================================================
    Running 8/10: Ly6a_mouse_pep_007_complex.fasta
    Result directory: ../data/processed/alphafold_results/ly6a_top10_batch/Ly6a_mouse_pep_007
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/alphafold_inputs/ly6a_top10/Ly6a_mouse_pep_007_complex.fasta ../data/processed/alphafold_results/ly6a_top10_batch/Ly6a_mouse_pep_007
    Return code: 0
    Elapsed time: 3.46 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    STDOUT tail:
    2026-05-31 23:08:09,151 Running colabfold 1.6.1
    2026-05-31 23:08:10,635 Running on GPU
    2026-05-31 23:08:10,677 Found 2 citations for tools or databases
    2026-05-31 23:08:10,678 Skipping Ly6a_mouse_pep_007_complex (already done)
    2026-05-31 23:08:10,678 Done
    
    
    STDERR tail:
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261689.499752    9289 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261690.290861    9289 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    
    
    ================================================================================
    Running 9/10: Ly6a_mouse_pep_008_complex.fasta
    Result directory: ../data/processed/alphafold_results/ly6a_top10_batch/Ly6a_mouse_pep_008
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/alphafold_inputs/ly6a_top10/Ly6a_mouse_pep_008_complex.fasta ../data/processed/alphafold_results/ly6a_top10_batch/Ly6a_mouse_pep_008
    Return code: 0
    Elapsed time: 3.42 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    STDOUT tail:
    2026-05-31 23:08:12,587 Running colabfold 1.6.1
    2026-05-31 23:08:14,075 Running on GPU
    2026-05-31 23:08:14,119 Found 2 citations for tools or databases
    2026-05-31 23:08:14,125 Skipping Ly6a_mouse_pep_008_complex (already done)
    2026-05-31 23:08:14,125 Done
    
    
    STDERR tail:
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261692.922476    9401 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261693.696584    9401 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    
    
    ================================================================================
    Running 10/10: Ly6a_mouse_pep_009_complex.fasta
    Result directory: ../data/processed/alphafold_results/ly6a_top10_batch/Ly6a_mouse_pep_009
    Command:
    conda run -n colabfold colabfold_batch --msa-mode single_sequence --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 1 --model-order 1 ../data/processed/alphafold_inputs/ly6a_top10/Ly6a_mouse_pep_009_complex.fasta ../data/processed/alphafold_results/ly6a_top10_batch/Ly6a_mouse_pep_009
    Return code: 0
    Elapsed time: 3.56 s (0.06 min)
    PDB files: 1
    Score JSON files: 1
    Success: True
    
    STDOUT tail:
    2026-05-31 23:08:16,013 Running colabfold 1.6.1
    2026-05-31 23:08:17,574 Running on GPU
    2026-05-31 23:08:17,624 Found 2 citations for tools or databases
    2026-05-31 23:08:17,624 Skipping Ly6a_mouse_pep_009_complex (already done)
    2026-05-31 23:08:17,624 Done
    
    
    STDERR tail:
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261696.367929    9515 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1780261697.167518    9515 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    



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
      <td>../data/processed/alphafold_inputs/ly6a_top10/...</td>
      <td>../data/processed/alphafold_results/ly6a_top10...</td>
      <td>0</td>
      <td>4.313333</td>
      <td>0.071889</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ly6a_mouse_pep_001</td>
      <td>../data/processed/alphafold_inputs/ly6a_top10/...</td>
      <td>../data/processed/alphafold_results/ly6a_top10...</td>
      <td>0</td>
      <td>4.161797</td>
      <td>0.069363</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ly6a_mouse_pep_002</td>
      <td>../data/processed/alphafold_inputs/ly6a_top10/...</td>
      <td>../data/processed/alphafold_results/ly6a_top10...</td>
      <td>0</td>
      <td>4.000218</td>
      <td>0.066670</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ly6a_mouse_pep_003</td>
      <td>../data/processed/alphafold_inputs/ly6a_top10/...</td>
      <td>../data/processed/alphafold_results/ly6a_top10...</td>
      <td>0</td>
      <td>3.513583</td>
      <td>0.058560</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ly6a_mouse_pep_004</td>
      <td>../data/processed/alphafold_inputs/ly6a_top10/...</td>
      <td>../data/processed/alphafold_results/ly6a_top10...</td>
      <td>0</td>
      <td>3.666805</td>
      <td>0.061113</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ly6a_mouse_pep_005</td>
      <td>../data/processed/alphafold_inputs/ly6a_top10/...</td>
      <td>../data/processed/alphafold_results/ly6a_top10...</td>
      <td>0</td>
      <td>3.718125</td>
      <td>0.061969</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ly6a_mouse_pep_006</td>
      <td>../data/processed/alphafold_inputs/ly6a_top10/...</td>
      <td>../data/processed/alphafold_results/ly6a_top10...</td>
      <td>0</td>
      <td>3.532804</td>
      <td>0.058880</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Ly6a_mouse_pep_007</td>
      <td>../data/processed/alphafold_inputs/ly6a_top10/...</td>
      <td>../data/processed/alphafold_results/ly6a_top10...</td>
      <td>0</td>
      <td>3.461332</td>
      <td>0.057689</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ly6a_mouse_pep_008</td>
      <td>../data/processed/alphafold_inputs/ly6a_top10/...</td>
      <td>../data/processed/alphafold_results/ly6a_top10...</td>
      <td>0</td>
      <td>3.416491</td>
      <td>0.056942</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Ly6a_mouse_pep_009</td>
      <td>../data/processed/alphafold_inputs/ly6a_top10/...</td>
      <td>../data/processed/alphafold_results/ly6a_top10...</td>
      <td>0</td>
      <td>3.564295</td>
      <td>0.059405</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    
    Batch elapsed time:
    37.35 s (0.62 min)
    Saved run log: ../data/processed/alphafold_results/ly6a_top10_batch/ly6a_top10_colabfold_run_log.csv
    Successful runs: 10 / 10


## 9. Parse ColabFold receptor-peptide confidence scores

The ColabFold run produces a structure file and a score JSON file for each receptor-peptide pair. Here, the score JSON files are parsed to extract confidence metrics such as pLDDT, pTM, ipTM, and ranking confidence.

These values are retained as receptor-aware structure confidence signals, not as direct binding affinities.


```python
# -------------------------
# 9. Parse ColabFold scores for all top Ly6a-peptide candidates
# -------------------------

import json
from pathlib import Path
import pandas as pd
import numpy as np

batch_result_root = Path("../data/processed/alphafold_results/ly6a_top10_batch")

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

score_rows = []

for result_dir in sorted(batch_result_root.iterdir()):
    if not result_dir.is_dir():
        continue

    score_json_files = sorted(result_dir.glob("*scores_rank_*.json"))

    if len(score_json_files) == 0:
        continue

    score_path = score_json_files[0]
    query_name = result_dir.name

    # expected: Ly6a_mouse_pep_000
    peptide_id = query_name.split("_")[-2] + "_" + query_name.split("_")[-1] if "pep_" in query_name else query_name
    # safer direct extraction
    peptide_id = query_name.replace("Ly6a_mouse_", "")

    with open(score_path, "r") as f:
        score_data = json.load(f)

    seq_match = top_af_peptides.loc[
        top_af_peptides["peptide_id"] == peptide_id,
        "sequence"
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

colabfold_score_table = pd.DataFrame(score_rows)

for col in ["plddt", "ptm", "iptm", "ranking_confidence"]:
    if col in colabfold_score_table.columns:
        colabfold_score_table[col] = colabfold_score_table[col].apply(collapse_score_value)

display(colabfold_score_table)

parsed_score_out = Path("../data/processed/alphafold_results/parsed_ly6a_top10_colabfold_raw_scores.csv")
colabfold_score_table.to_csv(parsed_score_out, index=False)

print("Parsed score rows:", len(colabfold_score_table))
print("Saved parsed raw ColabFold scores:", parsed_score_out)
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
      <td>PVRPQQPRPLYYYH</td>
      <td>../data/processed/alphafold_results/ly6a_top10...</td>
      <td>Ly6a_mouse_pep_000_complex_scores_rank_001_alp...</td>
      <td>31.350306</td>
      <td>0.17</td>
      <td>0.04</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ly6a_mouse</td>
      <td>pep_001</td>
      <td>RLFMPMVYPTTG</td>
      <td>../data/processed/alphafold_results/ly6a_top10...</td>
      <td>Ly6a_mouse_pep_001_complex_scores_rank_001_alp...</td>
      <td>32.175319</td>
      <td>0.18</td>
      <td>0.05</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ly6a_mouse</td>
      <td>pep_002</td>
      <td>YPRGALPH</td>
      <td>../data/processed/alphafold_results/ly6a_top10...</td>
      <td>Ly6a_mouse_pep_002_complex_scores_rank_001_alp...</td>
      <td>31.028163</td>
      <td>0.17</td>
      <td>0.06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ly6a_mouse</td>
      <td>pep_003</td>
      <td>TRPPWVRPYLNH</td>
      <td>../data/processed/alphafold_results/ly6a_top10...</td>
      <td>Ly6a_mouse_pep_003_complex_scores_rank_001_alp...</td>
      <td>30.883131</td>
      <td>0.16</td>
      <td>0.06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ly6a_mouse</td>
      <td>pep_004</td>
      <td>GYFAAGPERRRGY</td>
      <td>../data/processed/alphafold_results/ly6a_top10...</td>
      <td>Ly6a_mouse_pep_004_complex_scores_rank_001_alp...</td>
      <td>31.624479</td>
      <td>0.17</td>
      <td>0.05</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ly6a_mouse</td>
      <td>pep_005</td>
      <td>TKFKGEQFSPYP</td>
      <td>../data/processed/alphafold_results/ly6a_top10...</td>
      <td>Ly6a_mouse_pep_005_complex_scores_rank_001_alp...</td>
      <td>31.635300</td>
      <td>0.17</td>
      <td>0.06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ly6a_mouse</td>
      <td>pep_006</td>
      <td>VYRPPRLFVRNH</td>
      <td>../data/processed/alphafold_results/ly6a_top10...</td>
      <td>Ly6a_mouse_pep_006_complex_scores_rank_001_alp...</td>
      <td>31.412737</td>
      <td>0.18</td>
      <td>0.08</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Ly6a_mouse</td>
      <td>pep_007</td>
      <td>PQIYRPRN</td>
      <td>../data/processed/alphafold_results/ly6a_top10...</td>
      <td>Ly6a_mouse_pep_007_complex_scores_rank_001_alp...</td>
      <td>30.643367</td>
      <td>0.17</td>
      <td>0.06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ly6a_mouse</td>
      <td>pep_008</td>
      <td>IPAYLPTNH</td>
      <td>../data/processed/alphafold_results/ly6a_top10...</td>
      <td>Ly6a_mouse_pep_008_complex_scores_rank_001_alp...</td>
      <td>31.130729</td>
      <td>0.16</td>
      <td>0.05</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Ly6a_mouse</td>
      <td>pep_009</td>
      <td>VLQQKLPWQQNMKGYP</td>
      <td>../data/processed/alphafold_results/ly6a_top10...</td>
      <td>Ly6a_mouse_pep_009_complex_scores_rank_001_alp...</td>
      <td>31.273093</td>
      <td>0.18</td>
      <td>0.06</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    Parsed score rows: 10
    Saved parsed raw ColabFold scores: ../data/processed/alphafold_results/parsed_ly6a_top10_colabfold_raw_scores.csv


## 10. Compute Ly6a structure proxy score and final receptor-aware reward

The parsed ColabFold metrics are converted into a conservative Ly6a structure proxy score. This proxy is not interpreted as binding affinity. It is a confidence-weighted structure-aware triage signal.

To avoid rewarding the “best bad” candidate, the final reward preserves absolute structure confidence using a soft structure gate. Low-confidence receptor-peptide predictions are strongly downweighted even if they are the best available result in a small candidate batch.


```python
# -------------------------
# 10. Ly6a structure proxy score and final receptor-aware reward
# -------------------------

def normalise_colabfold_score(row):
    """
    Conservative receptor-aware proxy score.

    ipTM is weighted most strongly because it reflects interface confidence
    in AlphaFold-Multimer/ColabFold predictions.

    This is not binding affinity.
    """
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
    """
    Soft absolute confidence gate.

    Very low ipTM/pLDDT predictions are not treated as strong receptor hits,
    even if they are relatively best within a small batch.
    """
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

    # Floor keeps weak predictions usable for toy-loop demonstration,
    # while still strongly downweighting them.
    gate = 0.25 + 0.75 * (
        0.65 * iptm_gate
        + 0.35 * plddt_gate
    )

    return float(np.clip(gate, 0.0, 1.0))


colabfold_score_table["ly6a_structure_proxy_score"] = colabfold_score_table.apply(
    normalise_colabfold_score,
    axis=1,
)

colabfold_score_table["soft_structure_gate"] = colabfold_score_table.apply(
    soft_structure_gate,
    axis=1,
)

# Merge structure score with the top peptides selected by cheap scoring
final_reward_table = top_af_peptides.merge(
    colabfold_score_table[
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

# Final reward is only calculated for peptides that have been ColabFold-scored.
# Other top-10 candidates remain NaN until their structure predictions are run.
final_reward_table["final_reward"] = np.where(
    final_reward_table["ly6a_structure_proxy_score"].notna(),
    (
        0.35 * final_reward_table["cheap_pre_af_score_refined"]
        + 0.65 * final_reward_table["ly6a_structure_proxy_score"]
    ) * final_reward_table["soft_structure_gate"],
    np.nan,
)

final_reward_table = final_reward_table.sort_values(
    "final_reward",
    ascending=False,
    na_position="last",
).reset_index(drop=True)

display(
    final_reward_table[
        [
            "peptide_id",
            "cheap_rank",
            "sequence",
            "cheap_pre_af_score_refined",
            "plddt",
            "ptm",
            "iptm",
            "ranking_confidence",
            "ly6a_structure_proxy_score",
            "soft_structure_gate",
            "final_reward",
        ]
    ]
)

reward_out = Path("../data/processed/peptides/ly6a_receptor_aware_reward_table.csv")
final_reward_table.to_csv(reward_out, index=False)

print("Saved receptor-aware reward table:", reward_out)
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
      <th>cheap_rank</th>
      <th>sequence</th>
      <th>cheap_pre_af_score_refined</th>
      <th>plddt</th>
      <th>ptm</th>
      <th>iptm</th>
      <th>ranking_confidence</th>
      <th>ly6a_structure_proxy_score</th>
      <th>soft_structure_gate</th>
      <th>final_reward</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pep_006</td>
      <td>7</td>
      <td>VYRPPRLFVRNH</td>
      <td>1.0</td>
      <td>31.412737</td>
      <td>0.18</td>
      <td>0.08</td>
      <td>NaN</td>
      <td>0.136119</td>
      <td>0.405744</td>
      <td>0.177909</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pep_005</td>
      <td>6</td>
      <td>TKFKGEQFSPYP</td>
      <td>1.0</td>
      <td>31.635300</td>
      <td>0.17</td>
      <td>0.06</td>
      <td>NaN</td>
      <td>0.122953</td>
      <td>0.350338</td>
      <td>0.150617</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pep_009</td>
      <td>10</td>
      <td>VLQQKLPWQQNMKGYP</td>
      <td>1.0</td>
      <td>31.273093</td>
      <td>0.18</td>
      <td>0.06</td>
      <td>NaN</td>
      <td>0.124910</td>
      <td>0.347169</td>
      <td>0.149696</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pep_002</td>
      <td>3</td>
      <td>YPRGALPH</td>
      <td>1.0</td>
      <td>31.028163</td>
      <td>0.17</td>
      <td>0.06</td>
      <td>NaN</td>
      <td>0.122042</td>
      <td>0.345026</td>
      <td>0.148129</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pep_003</td>
      <td>4</td>
      <td>TRPPWVRPYLNH</td>
      <td>1.0</td>
      <td>30.883131</td>
      <td>0.16</td>
      <td>0.06</td>
      <td>NaN</td>
      <td>0.119325</td>
      <td>0.343757</td>
      <td>0.146977</td>
    </tr>
    <tr>
      <th>5</th>
      <td>pep_007</td>
      <td>8</td>
      <td>PQIYRPRN</td>
      <td>1.0</td>
      <td>30.643367</td>
      <td>0.17</td>
      <td>0.06</td>
      <td>NaN</td>
      <td>0.121465</td>
      <td>0.341659</td>
      <td>0.146555</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pep_001</td>
      <td>2</td>
      <td>RLFMPMVYPTTG</td>
      <td>1.0</td>
      <td>32.175319</td>
      <td>0.18</td>
      <td>0.05</td>
      <td>NaN</td>
      <td>0.120763</td>
      <td>0.326387</td>
      <td>0.139855</td>
    </tr>
    <tr>
      <th>7</th>
      <td>pep_004</td>
      <td>5</td>
      <td>GYFAAGPERRRGY</td>
      <td>1.0</td>
      <td>31.624479</td>
      <td>0.17</td>
      <td>0.05</td>
      <td>NaN</td>
      <td>0.117437</td>
      <td>0.321567</td>
      <td>0.137095</td>
    </tr>
    <tr>
      <th>8</th>
      <td>pep_008</td>
      <td>9</td>
      <td>IPAYLPTNH</td>
      <td>1.0</td>
      <td>31.130729</td>
      <td>0.16</td>
      <td>0.05</td>
      <td>NaN</td>
      <td>0.114196</td>
      <td>0.317247</td>
      <td>0.134585</td>
    </tr>
    <tr>
      <th>9</th>
      <td>pep_000</td>
      <td>1</td>
      <td>PVRPQQPRPLYYYH</td>
      <td>1.0</td>
      <td>31.350306</td>
      <td>0.17</td>
      <td>0.04</td>
      <td>NaN</td>
      <td>0.111525</td>
      <td>0.290492</td>
      <td>0.122730</td>
    </tr>
  </tbody>
</table>
</div>


    Saved receptor-aware reward table: ../data/processed/peptides/ly6a_receptor_aware_reward_table.csv


## Final note

This notebook exports a Ly6a receptor-aware peptide reward table for downstream reward-guided optimisation.

Key outputs:

- `../data/processed/peptides/notebook04_generated_peptides_with_cheap_scores.csv`
- `../data/processed/peptides/top10_peptides_for_ly6a_colabfold.csv`
- `../data/processed/peptides/ly6a_receptor_aware_reward_table.csv`

The final reward is a computational triage signal that combines cheap peptide-level properties with ColabFold-derived structural confidence. It is not a measured binding score and should not be interpreted as experimental validation.
