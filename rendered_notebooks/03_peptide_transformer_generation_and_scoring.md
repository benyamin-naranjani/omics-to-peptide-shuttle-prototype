# 03. Peptide transformer training, fine-tuning, and generation

This notebook trains and saves a small autoregressive peptide transformer used as the generator in the downstream receptor-aware optimisation workflow.

The model is trained in two stages:

1. **CPPsite2 pretraining:** the model first learns general peptide sequence grammar from CPPsite2 natural cell-penetrating peptides.
2. **B3PDB fine-tuning:** the pretrained model is then fine-tuned on B3PDB BBB-positive peptide sequences to bias generation toward BBB-penetrating peptide-like sequence space.

The trained generator is later reused by the receptor-aware notebooks to generate candidate peptides, score them using cheap developability/safety heuristics, evaluate selected candidates with ColabFold, and update the generator through reward-guided optimisation.

This notebook performs the following steps:

1. import libraries, configure paths, and load cleaned peptide datasets,
2. define peptide tokenisation and vocabulary,
3. encode peptide sequences and create train/validation/test splits,
4. build causal language-model batches,
5. define the autoregressive peptide transformer,
6. define next-token loss and training steps,
7. pretrain the model on CPPsite2 natural peptides,
8. evaluate CPPsite2 pretraining on a held-out test set,
9. fine-tune the model on B3PDB BBB-positive peptides,
10. evaluate B3PDB fine-tuning on a held-out test set,
11. generate peptides from the fine-tuned transformer,
12. evaluate generated peptides for validity, uniqueness, and novelty,
13. save the fine-tuned generator checkpoint and metadata.

Important scope note:

This notebook trains a generative peptide language model. It does not claim that generated peptides bind a receptor, cross the BBB, are non-toxic, or are experimentally validated. Those properties are addressed only as downstream prioritisation signals in later notebooks.

In the wider prototype, this notebook provides the generative modelling stage:

**clean peptide corpus → peptide tokenisation → CPPsite2 pretraining → B3PDB fine-tuning → peptide generation → saved generator for receptor-aware optimisation**

## 1. Imports and configuration

This section imports the modelling libraries, defines project paths, loads the cleaned peptide datasets, and prepares the notebook configuration. The model is trained as a small causal peptide transformer that first learns general peptide sequence grammar from CPPsite2 natural peptides and is then fine-tuned on B3PDB BBB-positive peptides.


```python
import numpy as np
import pandas as pd
from pathlib import Path

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_PLATFORMS"] = "cuda,cpu"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_force_compilation_parallelism=1 "
    "--xla_gpu_enable_triton_gemm=false"
)

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")
```


```python
# Checking JAX compute device
print("JAX devices:", jax.devices())
```

    JAX devices: [CudaDevice(id=0)]



```python
# Loading the toy training (CPPSITE) and retraining (b3pdb) combined dataset
data_path = Path("../data/processed/peptides/cppsite_b3pdb_training_sequences.csv")
combined_peptides = pd.read_csv(data_path)

display(combined_peptides.head())
print(combined_peptides.shape)
print(combined_peptides["source"].value_counts())
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
      <th>sequence</th>
      <th>source</th>
      <th>bbb_positive</th>
      <th>n_amino_acids</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RRRRRRRGGIYLATALAKWALKQGF</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IYLATALAKWALKQGFGGRRRRRRR</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RRRRRRRGGIYLATALAKWALKQ</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>IYLATALAKWALKQGGRRRRRRR</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RRRRRRRGGKLAKLAKKLAKLAK</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>


    (1311, 4)
    source
    CPPsite2_natural    1180
    B3PDB_Chemical       131
    Name: count, dtype: int64



```python
cpp_train_df = combined_peptides[
    combined_peptides["source"] == "CPPsite2_natural"
].copy()

b3p_finetune_df = combined_peptides[
    combined_peptides["source"] == "B3PDB_Chemical"
].copy()

print("CPP pretrain:", cpp_train_df.shape)
print("B3PDB fine-tune:", b3p_finetune_df.shape)

print(cpp_train_df["n_amino_acids"].describe())
print(b3p_finetune_df["n_amino_acids"].describe())
```

    CPP pretrain: (1180, 4)
    B3PDB fine-tune: (131, 4)
    count    1180.000000
    mean       16.951695
    std         8.021507
    min         3.000000
    25%        12.000000
    50%        16.000000
    75%        20.000000
    max        61.000000
    Name: n_amino_acids, dtype: float64
    count    131.000000
    mean      13.053435
    std        8.953137
    min        2.000000
    25%        7.000000
    50%       11.000000
    75%       15.500000
    max       52.000000
    Name: n_amino_acids, dtype: float64


## 2. Peptide tokenisation and vocabulary

Peptide sequences are represented as one-letter amino-acid strings. The vocabulary contains the 20 canonical amino acids plus four special tokens: padding, unknown residues, beginning-of-sequence, and end-of-sequence.

The `<EOS>` token is important because it teaches the generator when to stop sequence generation.


```python
# -------------------------
# 2. Peptide tokenisation
# -------------------------

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
CANONICAL_AAS = list("ACDEFGHIKLMNPQRSTVWY")

def tokenize_peptide(seq):
    """
    Tokenise a peptide sequence into one-letter amino-acid tokens.
    """
    return list(str(seq).strip().upper())


# Build vocabulary from canonical amino acids only
vocab = SPECIAL_TOKENS + CANONICAL_AAS

token_to_id = {tok: i for i, tok in enumerate(vocab)}
id_to_token = {i: tok for tok, i in token_to_id.items()}

PAD_ID = token_to_id["<PAD>"]
UNK_ID = token_to_id["<UNK>"]
BOS_ID = token_to_id["<BOS>"]
EOS_ID = token_to_id["<EOS>"]

VOCAB_SIZE = len(vocab)

# Use full combined dataset to define maximum sequence length
# +2 because we add <BOS> and <EOS>
MAX_LEN = int(combined_peptides["sequence"].astype(str).str.len().max()) + 2

print("VOCAB_SIZE:", VOCAB_SIZE)
print("MAX_LEN:", MAX_LEN)
print("vocab:", vocab)
```

    VOCAB_SIZE: 24
    MAX_LEN: 63
    vocab: ['<PAD>', '<UNK>', '<BOS>', '<EOS>', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


## 3. Peptide encoding and train/validation/test splits

For causal peptide generation, each sequence is encoded as:

`<BOS> + amino acids + <EOS>`

and then padded to a fixed maximum length. The model is trained to predict the next token at each position.

Separate train/validation/test splits are created for CPPsite2 pretraining and B3PDB BBB-positive fine-tuning.


```python
# -------------------------
# 3. Peptide -> token IDs + train/val/test split
# -------------------------

def peptide_to_ids(seq):
    tokens = ["<BOS>"] + tokenize_peptide(seq) + ["<EOS>"]
    ids = [token_to_id.get(tok, UNK_ID) for tok in tokens]

    if len(ids) > MAX_LEN:
        ids = ids[:MAX_LEN]
        ids[-1] = EOS_ID

    attention_mask = [1] * len(ids)

    pad_len = MAX_LEN - len(ids)
    ids = ids + [PAD_ID] * pad_len
    attention_mask = attention_mask + [0] * pad_len

    return np.asarray(ids, dtype=np.int32), np.asarray(attention_mask, dtype=np.float32)


def encode_dataframe(df, seq_col="sequence"):
    X_ids = []
    X_mask = []

    for seq in df[seq_col].values:
        ids, mask = peptide_to_ids(seq)
        X_ids.append(ids)
        X_mask.append(mask)

    X_ids = np.asarray(X_ids, dtype=np.int32)
    X_mask = np.asarray(X_mask, dtype=np.float32)

    return X_ids, X_mask


def make_splits(X_ids, X_mask, test_size=0.2, val_size=0.2, random_state=3141592):
    idx = np.arange(len(X_ids))

    idx_trainval, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
    )

    split = {
        "train_ids": X_ids[idx_train],
        "train_mask": X_mask[idx_train],
        "val_ids": X_ids[idx_val],
        "val_mask": X_mask[idx_val],
        "test_ids": X_ids[idx_test],
        "test_mask": X_mask[idx_test],
    }

    return split


# Encode CPPsite pretraining data
X_cpp_ids, X_cpp_mask = encode_dataframe(cpp_train_df)

cpp_split = make_splits(
    X_cpp_ids,
    X_cpp_mask,
    test_size=0.2,
    val_size=0.2,
    random_state=3141592,
)

# Encode B3PDB fine-tuning data
X_b3p_ids, X_b3p_mask = encode_dataframe(b3p_finetune_df)

b3p_split = make_splits(
    X_b3p_ids,
    X_b3p_mask,
    test_size=0.2,
    val_size=0.2,
    random_state=3141592,
)


print("CPP pretrain split:")
print("train:", cpp_split["train_ids"].shape)
print("val:  ", cpp_split["val_ids"].shape)
print("test: ", cpp_split["test_ids"].shape)

print("\nB3PDB fine-tune split:")
print("train:", b3p_split["train_ids"].shape)
print("val:  ", b3p_split["val_ids"].shape)
print("test: ", b3p_split["test_ids"].shape)
```

    CPP pretrain split:
    train: (755, 63)
    val:   (189, 63)
    test:  (236, 63)
    
    B3PDB fine-tune split:
    train: (83, 63)
    val:   (21, 63)
    test:  (27, 63)


## 4. Causal language-model batches

For autoregressive peptide generation, the model predicts the next token from previous tokens. Each encoded peptide is shifted into an input sequence and a target sequence.

Example:

`<BOS> A G Y L <EOS>`

becomes:

`input_ids:  <BOS> A G Y L`

`target_ids: A G Y L <EOS>`


```python
# -------------------------
# 4. Batching helpers for causal language modelling
# -------------------------

BATCH_SIZE = 32


def make_batches(
    X_ids,
    X_mask,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=0,
    drop_last=True,
):
    idx = np.arange(len(X_ids))

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]

        if drop_last and len(batch_idx) < batch_size:
            continue

        full_ids = X_ids[batch_idx]
        full_mask = X_mask[batch_idx]

        yield {
            # Input sequence excludes the final token
            "input_ids": jnp.asarray(full_ids[:, :-1], dtype=jnp.int32),

            # Target sequence excludes the first token
            "target_ids": jnp.asarray(full_ids[:, 1:], dtype=jnp.int32),

            # Target mask ignores padded target positions
            "target_mask": jnp.asarray(full_mask[:, 1:], dtype=jnp.float32),
        }
```

## qucik sanity check:


```python
batch = next(make_batches(
    cpp_split["train_ids"],
    cpp_split["train_mask"],
    batch_size=4,
    shuffle=False,
    drop_last=False,
))

print("input_ids:", batch["input_ids"].shape)
print("target_ids:", batch["target_ids"].shape)
print("target_mask:", batch["target_mask"].shape)

print("First input tokens:")
print([id_to_token[int(i)] for i in np.array(batch["input_ids"][0])[:20]])

print("First target tokens:")
print([id_to_token[int(i)] for i in np.array(batch["target_ids"][0])[:20]])
```

    input_ids: (4, 62)
    target_ids: (4, 62)
    target_mask: (4, 62)
    First input tokens:
    ['<BOS>', 'F', 'L', 'I', 'F', 'I', 'R', 'V', 'I', 'C', 'I', 'V', 'I', 'A', 'K', 'L', 'K', 'A', 'N', 'L']
    First target tokens:
    ['F', 'L', 'I', 'F', 'I', 'R', 'V', 'I', 'C', 'I', 'V', 'I', 'A', 'K', 'L', 'K', 'A', 'N', 'L', 'M']


## 5. Causal peptide transformer model

The model is an autoregressive transformer. A causal attention mask prevents each token from seeing future tokens, so the model learns genuine next-token generation. Padding positions are also masked so that variable-length peptides can be trained in fixed-size batches.


```python
# -------------------------
# 5. Causal peptide transformer model
# -------------------------

class CausalTransformerBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, attention_mask, deterministic=True):
        # Causal mask: each token can attend only to itself and previous tokens
        causal_mask = nn.make_causal_mask(
            jnp.ones((x.shape[0], x.shape[1]), dtype=bool)
        )

        # Padding mask: ignore padded positions
        padding_mask = attention_mask[:, None, None, :].astype(bool)

        # Combined attention mask
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

        # Zero out padded token representations
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

        # Next-token logits for every sequence position
        logits = nn.Dense(self.vocab_size)(x)

        return logits
```

## Instantiating for CPP pretraining 


```python
model = PeptideCausalTransformer(
    vocab_size=VOCAB_SIZE,
    max_len=MAX_LEN,
    hidden_dim=128,
    num_heads=4,
    mlp_dim=256,
    num_layers=4,
    dropout_rate=0.2,
)

dummy_batch = next(make_batches(
    cpp_split["train_ids"],
    cpp_split["train_mask"],
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True,
))

# input attention mask corresponds to input_ids, so exclude final mask position
dummy_input_mask = dummy_batch["target_mask"]  # same length as input_ids after shift

rng = jax.random.PRNGKey(42)

params = model.init(
    rng,
    dummy_batch["input_ids"],
    dummy_input_mask,
    deterministic=True,
)

optimizer = optax.adamw(3e-4, weight_decay=1e-4)
opt_state = optimizer.init(params)

logits = model.apply(
    params,
    dummy_batch["input_ids"],
    dummy_input_mask,
    deterministic=True,
)

print("logits shape:", logits.shape)
print("vocab size:", VOCAB_SIZE)
```

    logits shape: (32, 62, 24)
    vocab size: 24


## 6. Causal language-model loss and training steps

The generator is trained with token-level softmax cross-entropy. For each peptide, the model predicts the next amino-acid token at every position. Padding positions are ignored using the target mask.


```python
# -------------------------
# 6. Loss, train step, and evaluation step
# -------------------------

def loss_fn(params, batch, rng):
    logits = model.apply(
        params,
        batch["input_ids"],
        batch["target_mask"],
        deterministic=False,
        rngs={"dropout": rng},
    )

    target_ids = batch["target_ids"]
    target_mask = batch["target_mask"]

    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits,
        target_ids,
    )

    loss = jnp.sum(per_token_loss * target_mask) / jnp.sum(target_mask)

    return loss


@jax.jit
def train_step(params, opt_state, batch, rng):
    loss, grads = jax.value_and_grad(loss_fn)(
        params,
        batch,
        rng,
    )

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss


@jax.jit
def eval_step(params, batch):
    logits = model.apply(
        params,
        batch["input_ids"],
        batch["target_mask"],
        deterministic=True,
    )

    target_ids = batch["target_ids"]
    target_mask = batch["target_mask"]

    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits,
        target_ids,
    )

    loss = jnp.sum(per_token_loss * target_mask) / jnp.sum(target_mask)

    return loss


def evaluate_loss(params, X_ids, X_mask, batch_size=BATCH_SIZE):
    losses = []

    for batch in make_batches(
        X_ids,
        X_mask,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    ):
        loss = eval_step(params, batch)
        loss.block_until_ready()
        losses.append(float(loss))

    return float(np.mean(losses))
```

## sanity check:


```python
initial_train_loss = evaluate_loss(
    params,
    cpp_split["train_ids"],
    cpp_split["train_mask"],
)

initial_val_loss = evaluate_loss(
    params,
    cpp_split["val_ids"],
    cpp_split["val_mask"],
)

print("Initial train loss:", initial_train_loss)
print("Initial val loss:", initial_val_loss)
print("Random baseline approx log(vocab):", np.log(VOCAB_SIZE))
```

    Initial train loss: 4.270669877529144
    Initial val loss: 4.295222679773967
    Random baseline approx log(vocab): 3.1780538303479458


## 7. CPPsite2 pretraining loop

The transformer is first pretrained on CPPsite2 natural peptides to learn general cell-penetrating and uptake-related peptide sequence grammar. Early stopping is based on validation next-token cross-entropy loss, where lower loss indicates better language modelling.


```python
import time

# -------------------------
# 7. CPPsite pretraining loop
# -------------------------

num_epochs = 300
patience = 30
selection_start_epoch = 20

best_val_loss = np.inf
best_epoch = 0
best_params = None
epochs_without_improvement = 0

rng = jax.random.PRNGKey(123)

train_history = []

t_start = time.perf_counter()

for epoch in range(1, num_epochs + 1):
    epoch_losses = []

    for batch in make_batches(
        cpp_split["train_ids"],
        cpp_split["train_mask"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=epoch,
        drop_last=True,
    ):
        rng, step_rng = jax.random.split(rng)

        params, opt_state, loss = train_step(
            params,
            opt_state,
            batch,
            step_rng,
        )

        loss.block_until_ready()
        epoch_losses.append(float(loss))

    train_loss = float(np.mean(epoch_losses))

    val_loss = evaluate_loss(
        params,
        cpp_split["val_ids"],
        cpp_split["val_mask"],
    )

    train_history.append({
        "stage": "CPPsite_pretraining",
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    })

    print(
        f"Epoch {epoch:03d} | "
        f"train_loss = {train_loss:.4f} | "
        f"val_loss = {val_loss:.4f}"
    )

    if epoch >= selection_start_epoch:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

    if epoch >= selection_start_epoch and epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
        break

t_end = time.perf_counter()

params = best_params

print(f"\nBest CPPsite validation loss: {best_val_loss:.4f}")
print(f"Best CPPsite epoch: {best_epoch}")
print(f"Total pretraining time: {t_end - t_start:.2f} s")

train_history_df = pd.DataFrame(train_history)
display(train_history_df.tail())
```

    Epoch 001 | train_loss = 2.9531 | val_loss = 2.6941
    Epoch 002 | train_loss = 2.6778 | val_loss = 2.6080
    Epoch 003 | train_loss = 2.5993 | val_loss = 2.5202
    Epoch 004 | train_loss = 2.5517 | val_loss = 2.4713
    Epoch 005 | train_loss = 2.4954 | val_loss = 2.4295
    Epoch 006 | train_loss = 2.4344 | val_loss = 2.3833
    Epoch 007 | train_loss = 2.4063 | val_loss = 2.3666
    Epoch 008 | train_loss = 2.3608 | val_loss = 2.3246
    Epoch 009 | train_loss = 2.3193 | val_loss = 2.3141
    Epoch 010 | train_loss = 2.2913 | val_loss = 2.2825
    Epoch 011 | train_loss = 2.2595 | val_loss = 2.2694
    Epoch 012 | train_loss = 2.2376 | val_loss = 2.2513
    Epoch 013 | train_loss = 2.1998 | val_loss = 2.2237
    Epoch 014 | train_loss = 2.1579 | val_loss = 2.2065
    Epoch 015 | train_loss = 2.1524 | val_loss = 2.1860
    Epoch 016 | train_loss = 2.1186 | val_loss = 2.1833
    Epoch 017 | train_loss = 2.1055 | val_loss = 2.1715
    Epoch 018 | train_loss = 2.0818 | val_loss = 2.1552
    Epoch 019 | train_loss = 2.0523 | val_loss = 2.1447
    Epoch 020 | train_loss = 2.0297 | val_loss = 2.1197
    Epoch 021 | train_loss = 2.0086 | val_loss = 2.1141
    Epoch 022 | train_loss = 2.0079 | val_loss = 2.1029
    Epoch 023 | train_loss = 1.9781 | val_loss = 2.1005
    Epoch 024 | train_loss = 1.9541 | val_loss = 2.0931
    Epoch 025 | train_loss = 1.9519 | val_loss = 2.0828
    Epoch 026 | train_loss = 1.9318 | val_loss = 2.0820
    Epoch 027 | train_loss = 1.9088 | val_loss = 2.0602
    Epoch 028 | train_loss = 1.8791 | val_loss = 2.0465
    Epoch 029 | train_loss = 1.8663 | val_loss = 2.0390
    Epoch 030 | train_loss = 1.8493 | val_loss = 2.0439
    Epoch 031 | train_loss = 1.8357 | val_loss = 2.0324
    Epoch 032 | train_loss = 1.8378 | val_loss = 2.0282
    Epoch 033 | train_loss = 1.8192 | val_loss = 2.0111
    Epoch 034 | train_loss = 1.8001 | val_loss = 2.0139
    Epoch 035 | train_loss = 1.7792 | val_loss = 2.0128
    Epoch 036 | train_loss = 1.7630 | val_loss = 2.0097
    Epoch 037 | train_loss = 1.7504 | val_loss = 1.9965
    Epoch 038 | train_loss = 1.7369 | val_loss = 1.9762
    Epoch 039 | train_loss = 1.7246 | val_loss = 1.9939
    Epoch 040 | train_loss = 1.7086 | val_loss = 1.9746
    Epoch 041 | train_loss = 1.6908 | val_loss = 1.9753
    Epoch 042 | train_loss = 1.6890 | val_loss = 1.9588
    Epoch 043 | train_loss = 1.6664 | val_loss = 1.9625
    Epoch 044 | train_loss = 1.6380 | val_loss = 1.9910
    Epoch 045 | train_loss = 1.6392 | val_loss = 1.9467
    Epoch 046 | train_loss = 1.6386 | val_loss = 1.9557
    Epoch 047 | train_loss = 1.6089 | val_loss = 1.9512
    Epoch 048 | train_loss = 1.6100 | val_loss = 1.9591
    Epoch 049 | train_loss = 1.5841 | val_loss = 1.9447
    Epoch 050 | train_loss = 1.5814 | val_loss = 1.9435
    Epoch 051 | train_loss = 1.5719 | val_loss = 1.9554
    Epoch 052 | train_loss = 1.5732 | val_loss = 1.9254
    Epoch 053 | train_loss = 1.5395 | val_loss = 1.9405
    Epoch 054 | train_loss = 1.5296 | val_loss = 1.9534
    Epoch 055 | train_loss = 1.5130 | val_loss = 1.9443
    Epoch 056 | train_loss = 1.5106 | val_loss = 1.9346
    Epoch 057 | train_loss = 1.4882 | val_loss = 1.9385
    Epoch 058 | train_loss = 1.4799 | val_loss = 1.9603
    Epoch 059 | train_loss = 1.4695 | val_loss = 1.9278
    Epoch 060 | train_loss = 1.4595 | val_loss = 1.9422
    Epoch 061 | train_loss = 1.4471 | val_loss = 1.9411
    Epoch 062 | train_loss = 1.4334 | val_loss = 1.9397
    Epoch 063 | train_loss = 1.4281 | val_loss = 1.9485
    Epoch 064 | train_loss = 1.4271 | val_loss = 1.9462
    Epoch 065 | train_loss = 1.4207 | val_loss = 1.9562
    Epoch 066 | train_loss = 1.4023 | val_loss = 1.9371
    Epoch 067 | train_loss = 1.3873 | val_loss = 1.9480
    Epoch 068 | train_loss = 1.3789 | val_loss = 1.9488
    Epoch 069 | train_loss = 1.3665 | val_loss = 1.9458
    Epoch 070 | train_loss = 1.3612 | val_loss = 1.9401
    Epoch 071 | train_loss = 1.3348 | val_loss = 1.9557
    Epoch 072 | train_loss = 1.3258 | val_loss = 1.9384
    Epoch 073 | train_loss = 1.3259 | val_loss = 1.9472
    Epoch 074 | train_loss = 1.3210 | val_loss = 1.9386
    Epoch 075 | train_loss = 1.3029 | val_loss = 1.9435
    Epoch 076 | train_loss = 1.3012 | val_loss = 1.9469
    Epoch 077 | train_loss = 1.3080 | val_loss = 1.9650
    Epoch 078 | train_loss = 1.2815 | val_loss = 1.9533
    Epoch 079 | train_loss = 1.2750 | val_loss = 1.9631
    Epoch 080 | train_loss = 1.2749 | val_loss = 1.9823
    Epoch 081 | train_loss = 1.2610 | val_loss = 1.9714
    Epoch 082 | train_loss = 1.2370 | val_loss = 1.9578
    Early stopping at epoch 82. Best epoch: 52
    
    Best CPPsite validation loss: 1.9254
    Best CPPsite epoch: 52
    Total pretraining time: 25.28 s



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
      <th>stage</th>
      <th>epoch</th>
      <th>train_loss</th>
      <th>val_loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>77</th>
      <td>CPPsite_pretraining</td>
      <td>78</td>
      <td>1.281524</td>
      <td>1.953262</td>
    </tr>
    <tr>
      <th>78</th>
      <td>CPPsite_pretraining</td>
      <td>79</td>
      <td>1.274977</td>
      <td>1.963100</td>
    </tr>
    <tr>
      <th>79</th>
      <td>CPPsite_pretraining</td>
      <td>80</td>
      <td>1.274920</td>
      <td>1.982268</td>
    </tr>
    <tr>
      <th>80</th>
      <td>CPPsite_pretraining</td>
      <td>81</td>
      <td>1.261008</td>
      <td>1.971360</td>
    </tr>
    <tr>
      <th>81</th>
      <td>CPPsite_pretraining</td>
      <td>82</td>
      <td>1.237035</td>
      <td>1.957761</td>
    </tr>
  </tbody>
</table>
</div>


## 8. CPPsite2 held-out test evaluation

The pretrained generator is evaluated on the held-out CPPsite2 test set using next-token cross-entropy and perplexity. Lower values indicate better peptide language modelling.


```python
# -------------------------
# 8. CPPsite test evaluation
# -------------------------

cpp_test_loss = evaluate_loss(
    params,
    cpp_split["test_ids"],
    cpp_split["test_mask"],
)

cpp_test_perplexity = float(np.exp(cpp_test_loss))

print("Best CPPsite epoch:", best_epoch)
print("Best CPPsite validation loss:", best_val_loss)
print("CPPsite test loss:", cpp_test_loss)
print("CPPsite test perplexity:", cpp_test_perplexity)
```

    Best CPPsite epoch: 52
    Best CPPsite validation loss: 1.9253921111424763
    CPPsite test loss: 1.9499332308769226
    CPPsite test perplexity: 7.028218296950155


## 9. B3PDB BBB-positive fine-tuning

The CPPsite2-pretrained model is fine-tuned on B3PDB BBB-positive peptides. This shifts the generator from general uptake-related peptide grammar toward BBB-penetrating peptide-like sequence space.


```python
# -------------------------
# 9. B3PDB BBB-positive fine-tuning
# -------------------------

# Lower learning rate for small fine-tuning set
optimizer = optax.adamw(5e-5, weight_decay=1e-4)
opt_state = optimizer.init(params)

num_epochs_ft = 150
patience_ft = 25
selection_start_epoch_ft = 10

best_b3p_val_loss = np.inf
best_b3p_epoch = 0
best_b3p_params = None
epochs_without_improvement = 0

rng = jax.random.PRNGKey(456)

finetune_history = []

t_start = time.perf_counter()

for epoch in range(1, num_epochs_ft + 1):
    epoch_losses = []

    for batch in make_batches(
        b3p_split["train_ids"],
        b3p_split["train_mask"],
        batch_size=16,          # smaller batch for small fine-tune set
        shuffle=True,
        seed=epoch,
        drop_last=True,
    ):
        rng, step_rng = jax.random.split(rng)

        params, opt_state, loss = train_step(
            params,
            opt_state,
            batch,
            step_rng,
        )

        loss.block_until_ready()
        epoch_losses.append(float(loss))

    train_loss = float(np.mean(epoch_losses))

    val_loss = evaluate_loss(
        params,
        b3p_split["val_ids"],
        b3p_split["val_mask"],
        batch_size=16,
    )

    finetune_history.append({
        "stage": "B3PDB_finetuning",
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    })

    print(
        f"Fine-tune epoch {epoch:03d} | "
        f"train_loss = {train_loss:.4f} | "
        f"val_loss = {val_loss:.4f}"
    )

    if epoch >= selection_start_epoch_ft:
        if val_loss < best_b3p_val_loss:
            best_b3p_val_loss = val_loss
            best_b3p_epoch = epoch
            best_b3p_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

    if epoch >= selection_start_epoch_ft and epochs_without_improvement >= patience_ft:
        print(f"Early stopping at fine-tune epoch {epoch}. Best epoch: {best_b3p_epoch}")
        break

t_end = time.perf_counter()

params = best_b3p_params

print(f"\nBest B3PDB validation loss: {best_b3p_val_loss:.4f}")
print(f"Best B3PDB epoch: {best_b3p_epoch}")
print(f"Total fine-tuning time: {t_end - t_start:.2f} s")

finetune_history_df = pd.DataFrame(finetune_history)
display(finetune_history_df.tail())
```

    Fine-tune epoch 001 | train_loss = 3.4154 | val_loss = 3.5916
    Fine-tune epoch 002 | train_loss = 3.3403 | val_loss = 3.4386
    Fine-tune epoch 003 | train_loss = 3.2121 | val_loss = 3.3137
    Fine-tune epoch 004 | train_loss = 3.0819 | val_loss = 3.2217
    Fine-tune epoch 005 | train_loss = 3.0376 | val_loss = 3.1519
    Fine-tune epoch 006 | train_loss = 2.9477 | val_loss = 3.1014
    Fine-tune epoch 007 | train_loss = 2.9555 | val_loss = 3.0627
    Fine-tune epoch 008 | train_loss = 2.8996 | val_loss = 3.0341
    Fine-tune epoch 009 | train_loss = 2.8800 | val_loss = 3.0117
    Fine-tune epoch 010 | train_loss = 2.8855 | val_loss = 2.9928
    Fine-tune epoch 011 | train_loss = 2.8283 | val_loss = 2.9747
    Fine-tune epoch 012 | train_loss = 2.8425 | val_loss = 2.9594
    Fine-tune epoch 013 | train_loss = 2.8277 | val_loss = 2.9495
    Fine-tune epoch 014 | train_loss = 2.8018 | val_loss = 2.9383
    Fine-tune epoch 015 | train_loss = 2.7983 | val_loss = 2.9276
    Fine-tune epoch 016 | train_loss = 2.7666 | val_loss = 2.9171
    Fine-tune epoch 017 | train_loss = 2.7580 | val_loss = 2.9101
    Fine-tune epoch 018 | train_loss = 2.7383 | val_loss = 2.9045
    Fine-tune epoch 019 | train_loss = 2.7528 | val_loss = 2.8995
    Fine-tune epoch 020 | train_loss = 2.7167 | val_loss = 2.8943
    Fine-tune epoch 021 | train_loss = 2.7133 | val_loss = 2.8894
    Fine-tune epoch 022 | train_loss = 2.7326 | val_loss = 2.8851
    Fine-tune epoch 023 | train_loss = 2.7079 | val_loss = 2.8816
    Fine-tune epoch 024 | train_loss = 2.7008 | val_loss = 2.8763
    Fine-tune epoch 025 | train_loss = 2.6700 | val_loss = 2.8719
    Fine-tune epoch 026 | train_loss = 2.6878 | val_loss = 2.8668
    Fine-tune epoch 027 | train_loss = 2.6936 | val_loss = 2.8620
    Fine-tune epoch 028 | train_loss = 2.6488 | val_loss = 2.8588
    Fine-tune epoch 029 | train_loss = 2.6712 | val_loss = 2.8557
    Fine-tune epoch 030 | train_loss = 2.6828 | val_loss = 2.8532
    Fine-tune epoch 031 | train_loss = 2.6894 | val_loss = 2.8502
    Fine-tune epoch 032 | train_loss = 2.6772 | val_loss = 2.8466
    Fine-tune epoch 033 | train_loss = 2.6477 | val_loss = 2.8397
    Fine-tune epoch 034 | train_loss = 2.6182 | val_loss = 2.8322
    Fine-tune epoch 035 | train_loss = 2.6111 | val_loss = 2.8287
    Fine-tune epoch 036 | train_loss = 2.5805 | val_loss = 2.8249
    Fine-tune epoch 037 | train_loss = 2.5661 | val_loss = 2.8238
    Fine-tune epoch 038 | train_loss = 2.5685 | val_loss = 2.8190
    Fine-tune epoch 039 | train_loss = 2.5797 | val_loss = 2.8136
    Fine-tune epoch 040 | train_loss = 2.5861 | val_loss = 2.8073
    Fine-tune epoch 041 | train_loss = 2.5911 | val_loss = 2.8023
    Fine-tune epoch 042 | train_loss = 2.5373 | val_loss = 2.7994
    Fine-tune epoch 043 | train_loss = 2.5128 | val_loss = 2.7977
    Fine-tune epoch 044 | train_loss = 2.4864 | val_loss = 2.7990
    Fine-tune epoch 045 | train_loss = 2.5314 | val_loss = 2.7991
    Fine-tune epoch 046 | train_loss = 2.4760 | val_loss = 2.7943
    Fine-tune epoch 047 | train_loss = 2.4846 | val_loss = 2.7915
    Fine-tune epoch 048 | train_loss = 2.4773 | val_loss = 2.7812
    Fine-tune epoch 049 | train_loss = 2.4687 | val_loss = 2.7790
    Fine-tune epoch 050 | train_loss = 2.4244 | val_loss = 2.7807
    Fine-tune epoch 051 | train_loss = 2.4310 | val_loss = 2.7842
    Fine-tune epoch 052 | train_loss = 2.4312 | val_loss = 2.7862
    Fine-tune epoch 053 | train_loss = 2.4576 | val_loss = 2.7891
    Fine-tune epoch 054 | train_loss = 2.4225 | val_loss = 2.7816
    Fine-tune epoch 055 | train_loss = 2.4099 | val_loss = 2.7747
    Fine-tune epoch 056 | train_loss = 2.3831 | val_loss = 2.7670
    Fine-tune epoch 057 | train_loss = 2.3817 | val_loss = 2.7599
    Fine-tune epoch 058 | train_loss = 2.3539 | val_loss = 2.7588
    Fine-tune epoch 059 | train_loss = 2.3935 | val_loss = 2.7582
    Fine-tune epoch 060 | train_loss = 2.3482 | val_loss = 2.7588
    Fine-tune epoch 061 | train_loss = 2.3768 | val_loss = 2.7613
    Fine-tune epoch 062 | train_loss = 2.3570 | val_loss = 2.7636
    Fine-tune epoch 063 | train_loss = 2.3292 | val_loss = 2.7658
    Fine-tune epoch 064 | train_loss = 2.3088 | val_loss = 2.7677
    Fine-tune epoch 065 | train_loss = 2.2832 | val_loss = 2.7618
    Fine-tune epoch 066 | train_loss = 2.2729 | val_loss = 2.7567
    Fine-tune epoch 067 | train_loss = 2.2450 | val_loss = 2.7543
    Fine-tune epoch 068 | train_loss = 2.2915 | val_loss = 2.7599
    Fine-tune epoch 069 | train_loss = 2.2655 | val_loss = 2.7605
    Fine-tune epoch 070 | train_loss = 2.3240 | val_loss = 2.7613
    Fine-tune epoch 071 | train_loss = 2.2179 | val_loss = 2.7658
    Fine-tune epoch 072 | train_loss = 2.2106 | val_loss = 2.7644
    Fine-tune epoch 073 | train_loss = 2.2089 | val_loss = 2.7568
    Fine-tune epoch 074 | train_loss = 2.2346 | val_loss = 2.7493
    Fine-tune epoch 075 | train_loss = 2.2196 | val_loss = 2.7494
    Fine-tune epoch 076 | train_loss = 2.2112 | val_loss = 2.7474
    Fine-tune epoch 077 | train_loss = 2.1985 | val_loss = 2.7452
    Fine-tune epoch 078 | train_loss = 2.1437 | val_loss = 2.7500
    Fine-tune epoch 079 | train_loss = 2.1918 | val_loss = 2.7588
    Fine-tune epoch 080 | train_loss = 2.1409 | val_loss = 2.7642
    Fine-tune epoch 081 | train_loss = 2.1686 | val_loss = 2.7666
    Fine-tune epoch 082 | train_loss = 2.1557 | val_loss = 2.7670
    Fine-tune epoch 083 | train_loss = 2.0977 | val_loss = 2.7643
    Fine-tune epoch 084 | train_loss = 2.0989 | val_loss = 2.7568
    Fine-tune epoch 085 | train_loss = 2.0799 | val_loss = 2.7556
    Fine-tune epoch 086 | train_loss = 2.0842 | val_loss = 2.7611
    Fine-tune epoch 087 | train_loss = 2.1144 | val_loss = 2.7688
    Fine-tune epoch 088 | train_loss = 2.0939 | val_loss = 2.7744
    Fine-tune epoch 089 | train_loss = 2.0901 | val_loss = 2.7722
    Fine-tune epoch 090 | train_loss = 2.0332 | val_loss = 2.7554
    Fine-tune epoch 091 | train_loss = 2.0701 | val_loss = 2.7467
    Fine-tune epoch 092 | train_loss = 2.0299 | val_loss = 2.7487
    Fine-tune epoch 093 | train_loss = 2.0133 | val_loss = 2.7619
    Fine-tune epoch 094 | train_loss = 2.0111 | val_loss = 2.7751
    Fine-tune epoch 095 | train_loss = 2.0366 | val_loss = 2.7761
    Fine-tune epoch 096 | train_loss = 2.0075 | val_loss = 2.7701
    Fine-tune epoch 097 | train_loss = 1.9965 | val_loss = 2.7596
    Fine-tune epoch 098 | train_loss = 1.9344 | val_loss = 2.7536
    Fine-tune epoch 099 | train_loss = 1.9880 | val_loss = 2.7532
    Fine-tune epoch 100 | train_loss = 1.9283 | val_loss = 2.7527
    Fine-tune epoch 101 | train_loss = 1.9658 | val_loss = 2.7563
    Fine-tune epoch 102 | train_loss = 1.9277 | val_loss = 2.7545
    Early stopping at fine-tune epoch 102. Best epoch: 77
    
    Best B3PDB validation loss: 2.7452
    Best B3PDB epoch: 77
    Total fine-tuning time: 14.30 s



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
      <th>stage</th>
      <th>epoch</th>
      <th>train_loss</th>
      <th>val_loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>97</th>
      <td>B3PDB_finetuning</td>
      <td>98</td>
      <td>1.934356</td>
      <td>2.753624</td>
    </tr>
    <tr>
      <th>98</th>
      <td>B3PDB_finetuning</td>
      <td>99</td>
      <td>1.987990</td>
      <td>2.753189</td>
    </tr>
    <tr>
      <th>99</th>
      <td>B3PDB_finetuning</td>
      <td>100</td>
      <td>1.928346</td>
      <td>2.752697</td>
    </tr>
    <tr>
      <th>100</th>
      <td>B3PDB_finetuning</td>
      <td>101</td>
      <td>1.965849</td>
      <td>2.756337</td>
    </tr>
    <tr>
      <th>101</th>
      <td>B3PDB_finetuning</td>
      <td>102</td>
      <td>1.927702</td>
      <td>2.754535</td>
    </tr>
  </tbody>
</table>
</div>


## 10. B3PDB held-out test evaluation

The fine-tuned generator is evaluated on held-out B3PDB BBB-positive peptides using next-token loss and perplexity. This checks whether the model retains reasonable sequence-modelling performance after BBB-focused fine-tuning.


```python
# -------------------------
# 10. B3PDB test evaluation
# -------------------------

b3p_test_loss = evaluate_loss(
    params,
    b3p_split["test_ids"],
    b3p_split["test_mask"],
    batch_size=16,
)

b3p_test_perplexity = float(np.exp(b3p_test_loss))

print("Best B3PDB fine-tune epoch:", best_b3p_epoch)
print("Best B3PDB validation loss:", best_b3p_val_loss)
print("B3PDB test loss:", b3p_test_loss)
print("B3PDB test perplexity:", b3p_test_perplexity)
```

    Best B3PDB fine-tune epoch: 77
    Best B3PDB validation loss: 2.7452083826065063
    B3PDB test loss: 2.773916721343994
    B3PDB test perplexity: 16.021262100567874


## 11. Peptide generation from the fine-tuned transformer

The fine-tuned autoregressive model starts from `<BOS>` and samples amino-acid tokens one by one until `<EOS>` is produced or the maximum length is reached.

Temperature controls diversity: lower values produce safer and more conservative peptides, while higher values increase diversity and novelty but may reduce sequence quality.


```python
# -------------------------
# 11. Generate peptides
# -------------------------

def sample_next_token(probs, temperature=1.0):
    probs = np.asarray(probs, dtype=np.float64)

    # Temperature scaling
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
    """
    Generate one peptide sequence from the fine-tuned causal transformer.
    """
    tokens = [BOS_ID]

    # model input length is MAX_LEN - 1 during training after causal shift
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

        # Predict token after the last real token
        next_logits = np.asarray(logits[0, len(tokens) - 1])
        probs = np.asarray(jax.nn.softmax(next_logits)).copy()
        
        # Avoid generating PAD/BOS/UNK
        probs[PAD_ID] = 0.0
        probs[BOS_ID] = 0.0
        probs[UNK_ID] = 0.0
        
        # Do not allow EOS before minimum peptide length
        current_peptide_len = len(tokens) - 1
        if current_peptide_len < min_len:
            probs[EOS_ID] = 0.0
        
        probs = probs / probs.sum()
        
        next_token = sample_next_token(probs, temperature=temperature)

        if next_token == EOS_ID:
            break

        tokens.append(next_token)

    peptide_tokens = [id_to_token[t] for t in tokens[1:] if t != EOS_ID]
    peptide = "".join(peptide_tokens)

    return peptide


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


t_0 = time.perf_counter()

generated_peptides = generate_peptide_batch(
    params=params,              # fine-tuned B3PDB params
    n=100,
    temperature=1.0,
    max_generation_len=40,
    min_len=5,
)

t_1 = time.perf_counter()

print(f"Took {t_1 - t_0:.4f} s")
print("Generated peptides:", generated_peptides.shape)
display(generated_peptides.head(20))
```

    Took 146.2727 s
    Generated peptides: (100, 3)



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
      <td>LVVGYPRRRRNRR</td>
      <td>1.0</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CKEFGPLE</td>
      <td>1.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DAAKLAIASHNMKWKDDY</td>
      <td>1.0</td>
      <td>18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SIPLPRPRPRLPMETH</td>
      <td>1.0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FKFVVGRGGGGGSGCYLNHYK</td>
      <td>1.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>5</th>
      <td>WYFQAVFQ</td>
      <td>1.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TKPFV</td>
      <td>1.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AVQQKQQQP</td>
      <td>1.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>8</th>
      <td>YNFFP</td>
      <td>1.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>IKVKK</td>
      <td>1.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>YPFENC</td>
      <td>1.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>11</th>
      <td>TKYNFPVTENPV</td>
      <td>1.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>12</th>
      <td>LLIILRRRIRKQAHAHSK</td>
      <td>1.0</td>
      <td>18</td>
    </tr>
    <tr>
      <th>13</th>
      <td>AVLPVPVCWPVP</td>
      <td>1.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SRASARPTYNTNKPERYNH</td>
      <td>1.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>15</th>
      <td>VKYHFPV</td>
      <td>1.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>16</th>
      <td>GTVASFTEEQQYQQ</td>
      <td>1.0</td>
      <td>14</td>
    </tr>
    <tr>
      <th>17</th>
      <td>VLNYQYP</td>
      <td>1.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>18</th>
      <td>CTKTRRKWHR</td>
      <td>1.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>19</th>
      <td>YAFNSDFPE</td>
      <td>1.0</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>


## 12. Peptide validity, uniqueness, and novelty check

Generated peptide sequences are evaluated using simple sequence-level filters before downstream reward scoring. Unlike SMILES generation, peptide validity does not require RDKit parsing. A generated peptide is considered valid if it contains only canonical amino-acid residues and falls within the selected length range.

Three basic quality metrics are reported:

- **Validity:** fraction of generated sequences containing only canonical amino acids and acceptable length.
- **Uniqueness:** fraction of valid sequences that are not duplicates within the generated set.
- **Novelty:** fraction of unique valid sequences that were not already present in the CPPsite2 or B3PDB training data.

This provides a lightweight quality-control layer before moving to multi-objective scoring, receptor-aware prioritisation, and structure-aware triage.


```python
# -------------------------
# 12. Peptide validity, uniqueness, and novelty
# -------------------------

from pathlib import Path

# If only one generation batch exists, use it as the full generated set
generated_peptides_all = generated_peptides.copy()

canonical_aas = set("ACDEFGHIKLMNPQRSTVWY")

training_sequence_set = set(
    combined_peptides["sequence"].astype(str).str.upper()
)

def is_valid_peptide(seq, min_len=5, max_len=40):
    seq = str(seq).strip().upper()
    
    if len(seq) < min_len or len(seq) > max_len:
        return False
    
    if not set(seq).issubset(canonical_aas):
        return False
    
    return True

generated_peptides_all["is_valid"] = generated_peptides_all["sequence"].apply(
    lambda s: is_valid_peptide(s, min_len=5, max_len=40)
)

generated_peptides_all["is_unique"] = ~generated_peptides_all["sequence"].duplicated()

generated_peptides_all["is_novel"] = ~generated_peptides_all["sequence"].isin(
    training_sequence_set
)

valid_peptides = generated_peptides_all[
    generated_peptides_all["is_valid"]
].copy()

unique_valid_peptides = valid_peptides.drop_duplicates(
    subset=["sequence"]
).copy()

novel_unique_peptides = unique_valid_peptides[
    unique_valid_peptides["is_novel"]
].copy()

print("Generated:", len(generated_peptides_all))
print("Valid:", len(valid_peptides), len(valid_peptides) / len(generated_peptides_all))
print("Unique valid:", len(unique_valid_peptides), len(unique_valid_peptides) / max(len(valid_peptides), 1))
print("Novel unique valid:", len(novel_unique_peptides), len(novel_unique_peptides) / max(len(unique_valid_peptides), 1))

print("\nLength summary for novel unique peptides:")
display(novel_unique_peptides["n_amino_acids"].describe())

print("\nSample novel valid peptides:")
display(novel_unique_peptides.head(20))

processed_dir = Path("../data/processed/peptides")
processed_dir.mkdir(parents=True, exist_ok=True)

generated_out = processed_dir / "generated_peptides_finetuned_transformer.csv"
novel_out = processed_dir / "novel_unique_generated_peptides.csv"

generated_peptides_all.to_csv(generated_out, index=False)
novel_unique_peptides.to_csv(novel_out, index=False)

print("\nSaved:", generated_out)
print("Saved:", novel_out)
```

    Generated: 100
    Valid: 100 1.0
    Unique valid: 100 1.0
    Novel unique valid: 99 0.99
    
    Length summary for novel unique peptides:



    count    99.000000
    mean     13.181818
    std       7.922280
    min       5.000000
    25%       8.000000
    50%      10.000000
    75%      16.000000
    max      38.000000
    Name: n_amino_acids, dtype: float64


    
    Sample novel valid peptides:



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
      <th>is_valid</th>
      <th>is_unique</th>
      <th>is_novel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LVVGYPRRRRNRR</td>
      <td>1.0</td>
      <td>13</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CKEFGPLE</td>
      <td>1.0</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DAAKLAIASHNMKWKDDY</td>
      <td>1.0</td>
      <td>18</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SIPLPRPRPRLPMETH</td>
      <td>1.0</td>
      <td>16</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FKFVVGRGGGGGSGCYLNHYK</td>
      <td>1.0</td>
      <td>21</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>WYFQAVFQ</td>
      <td>1.0</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TKPFV</td>
      <td>1.0</td>
      <td>5</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AVQQKQQQP</td>
      <td>1.0</td>
      <td>9</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>YNFFP</td>
      <td>1.0</td>
      <td>5</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>IKVKK</td>
      <td>1.0</td>
      <td>5</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>YPFENC</td>
      <td>1.0</td>
      <td>6</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>TKYNFPVTENPV</td>
      <td>1.0</td>
      <td>12</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>AVLPVPVCWPVP</td>
      <td>1.0</td>
      <td>12</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SRASARPTYNTNKPERYNH</td>
      <td>1.0</td>
      <td>19</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>VKYHFPV</td>
      <td>1.0</td>
      <td>7</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16</th>
      <td>GTVASFTEEQQYQQ</td>
      <td>1.0</td>
      <td>14</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>17</th>
      <td>VLNYQYP</td>
      <td>1.0</td>
      <td>7</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>CTKTRRKWHR</td>
      <td>1.0</td>
      <td>10</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>YAFNSDFPE</td>
      <td>1.0</td>
      <td>9</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20</th>
      <td>YAFAVAVGNG</td>
      <td>1.0</td>
      <td>10</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    
    Saved: ../data/processed/peptides/generated_peptides_finetuned_transformer.csv
    Saved: ../data/processed/peptides/novel_unique_generated_peptides.csv


## 13. Save fine-tuned peptide generator and metadata

The fine-tuned transformer parameters, vocabulary, model configuration, training summary, and generation summary are saved so that later notebooks can reload the generator without retraining.

The saved model checkpoint and metadata are used directly by the receptor-aware scoring and reward-guided optimisation notebooks.


```python
# -------------------------
# 13. Save fine-tuned model and metadata
# -------------------------

from pathlib import Path
from flax import serialization
import json

model_dir = Path("../models")
model_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = model_dir / "peptide_transformer_cpp_pretrained_b3pdb_finetuned.msgpack"

with open(checkpoint_path, "wb") as f:
    f.write(serialization.to_bytes(params))

print("Saved model checkpoint:", checkpoint_path)


metadata = {
    "model_name": "peptide_transformer_cpp_pretrained_b3pdb_finetuned",
    "description": (
        "Small causal peptide transformer pretrained on CPPsite2 natural peptides "
        "and fine-tuned on B3PDB chemically synthesized BBB-positive peptides."
    ),
    "vocab": vocab,
    "token_to_id": token_to_id,
    "id_to_token": {str(k): v for k, v in id_to_token.items()},
    "special_token_ids": {
        "PAD_ID": int(PAD_ID),
        "UNK_ID": int(UNK_ID),
        "BOS_ID": int(BOS_ID),
        "EOS_ID": int(EOS_ID),
    },
    "VOCAB_SIZE": int(VOCAB_SIZE),
    "MAX_LEN": int(MAX_LEN),
    "model_config": {
        "hidden_dim": 128,
        "num_heads": 4,
        "mlp_dim": 256,
        "num_layers": 4,
        "dropout_rate": 0.2,
    },
    "training_data": {
        "cppsite_pretrain_sequences": int(len(cpp_train_df)),
        "b3pdb_finetune_sequences": int(len(b3p_finetune_df)),
        "combined_unique_sequences": int(len(combined_peptides)),
    },
    "training_summary": {
        "cpp_best_epoch": int(best_epoch),
        "cpp_best_val_loss": float(best_val_loss),
        "cpp_test_loss": float(cpp_test_loss),
        "cpp_test_perplexity": float(cpp_test_perplexity),
        "b3p_best_epoch": int(best_b3p_epoch),
        "b3p_best_val_loss": float(best_b3p_val_loss),
        "b3p_test_loss": float(b3p_test_loss),
        "b3p_test_perplexity": float(b3p_test_perplexity),
    },
    "generation_summary": {
        "n_generated": int(len(generated_peptides_all)),
        "n_valid": int(generated_peptides_all["is_valid"].sum()),
        "n_unique_valid": int(len(unique_valid_peptides)),
        "n_novel_unique": int(len(novel_unique_peptides)),
        "valid_fraction": float(generated_peptides_all["is_valid"].mean()),
        "unique_valid_fraction": float(
            len(unique_valid_peptides) / max(int(generated_peptides_all["is_valid"].sum()), 1)
        ),
        "novel_unique_fraction": float(
            len(novel_unique_peptides) / max(len(unique_valid_peptides), 1)
        ),
    },
    "output_files": {
        "generated_peptides": "../data/processed/peptides/generated_peptides_finetuned_transformer.csv",
        "novel_unique_peptides": "../data/processed/peptides/novel_unique_generated_peptides.csv",
    },
}

metadata_path = model_dir / "peptide_transformer_metadata.json"

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved metadata:", metadata_path)
```

    Saved model checkpoint: ../models/peptide_transformer_cpp_pretrained_b3pdb_finetuned.msgpack
    Saved metadata: ../models/peptide_transformer_metadata.json


### Reloading sanity check:


```python
with open(metadata_path, "r") as f:
    loaded_metadata = json.load(f)

print(loaded_metadata["model_name"])
print(loaded_metadata["training_summary"])
print(loaded_metadata["generation_summary"])
```

    peptide_transformer_cpp_pretrained_b3pdb_finetuned
    {'cpp_best_epoch': 52, 'cpp_best_val_loss': 1.9253921111424763, 'cpp_test_loss': 1.9499332308769226, 'cpp_test_perplexity': 7.028218296950155, 'b3p_best_epoch': 77, 'b3p_best_val_loss': 2.7452083826065063, 'b3p_test_loss': 2.773916721343994, 'b3p_test_perplexity': 16.021262100567874}
    {'n_generated': 100, 'n_valid': 100, 'n_unique_valid': 100, 'n_novel_unique': 99, 'valid_fraction': 1.0, 'unique_valid_fraction': 1.0, 'novel_unique_fraction': 0.99}


## Final note

This notebook exports the fine-tuned peptide generator used by downstream receptor-aware notebooks.

Key outputs:

- `../models/peptide_transformer_cpp_pretrained_b3pdb_finetuned.msgpack`
- `../models/peptide_transformer_metadata.json`
- `../data/processed/peptides/generated_peptides_finetuned_transformer.csv`
- `../data/processed/peptides/novel_unique_generated_peptides.csv`

The generated peptides are candidate sequences for computational triage only. They are not claimed to be validated BBB shuttles or receptor binders.
