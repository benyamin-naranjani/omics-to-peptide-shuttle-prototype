# B3Pdb and CPPsite2 peptide dataset acquisition and preprocessing

This notebook retrieves and prepares peptide sequence data used for fine-tuning the peptide generator. The aim is to build a clean peptide corpus from public peptide resources and convert it into a format suitable for tokenisation, language-model training, and downstream peptide generation.

The notebook uses two public peptide resources:

1. **B3Pdb chemical-modification peptide entries**  
   URL: `https://webs.iiitd.edu.in/raghava/b3pdb/browse_sub1.php?token=Chemical&col=10`  
   Rationale: B3Pdb provides blood-brain barrier peptide entries. The chemical-modification subset is used here as a BBB-relevant peptide source for the fine-tuning corpus.

2. **CPPsite2 natural peptide FASTA file**  
   URL: `http://crdd.osdd.net/raghava/cppsite/downloads.php`  
   Rationale: CPPsite2 provides cell-penetrating peptide sequences. These peptides are not treated as BBB-positive labels, but they enrich the peptide language-model corpus with uptake-related peptide patterns.

The notebook performs the following steps:

1. define local raw and processed data paths,
2. retrieve public B3Pdb peptide data from available online resources,
3. parse downloaded files or HTML tables when needed,
4. read CPPsite2 peptide FASTA sequences,
5. standardise peptide sequences to canonical one-letter amino-acid format,
6. remove invalid, duplicated, or out-of-scope sequences,
7. save cleaned peptide tables for downstream model training.

Important scope note:

This notebook is a **data acquisition and cleaning step**, not a biological validation analysis. The resulting peptide corpus is used as a training/fine-tuning resource for sequence generation. Peptide function, receptor specificity, toxicity, synthesizability, and BBB transport relevance are not assumed from this dataset alone.

In the wider prototype, this notebook provides the peptide-data preparation stage:

**public peptide data retrieval → sequence cleaning → peptide corpus construction → transformer fine-tuning → peptide generation**

## 1. Retrieve B3Pdb chemical-modification peptide pages

The B3Pdb chemical-modification browse pages are retrieved from the public B3Pdb web interface. Each page is parsed as an HTML table, the largest table is retained, and all pages are concatenated into one raw table. The raw table is saved before sequence cleaning so the acquisition step remains reproducible.


```python
# -------------------------
# 1.1 Retrieve all B3Pdb chemical-modification pages
# -------------------------

from io import StringIO
from pathlib import Path
import pandas as pd
import requests

raw_dir = Path("../data/raw/peptides/b3pdb")
raw_dir.mkdir(parents=True, exist_ok=True)

base_url = "https://webs.iiitd.edu.in/raghava/b3pdb/browse_sub1.php?token=Chemical&col=10"

all_pages = []

for page in range(1, 12):
    url = base_url if page == 1 else f"{base_url}&page={page}"
    print("Reading:", url)

    html = requests.get(url, timeout=20).text
    tables = pd.read_html(StringIO(html))

    table = max(tables, key=lambda x: x.shape[0] * x.shape[1])
    table["source_page"] = page
    table["source_url"] = url

    all_pages.append(table)

raw_b3pdb_chemical = (
    pd.concat(all_pages, ignore_index=True)
    .drop_duplicates()
    .reset_index(drop=True)
)

print("Raw B3Pdb chemical table shape:", raw_b3pdb_chemical.shape)
display(raw_b3pdb_chemical.head())
```

    Reading: https://webs.iiitd.edu.in/raghava/b3pdb/browse_sub1.php?token=Chemical&col=10
    Reading: https://webs.iiitd.edu.in/raghava/b3pdb/browse_sub1.php?token=Chemical&col=10&page=2
    Reading: https://webs.iiitd.edu.in/raghava/b3pdb/browse_sub1.php?token=Chemical&col=10&page=3
    Reading: https://webs.iiitd.edu.in/raghava/b3pdb/browse_sub1.php?token=Chemical&col=10&page=4
    Reading: https://webs.iiitd.edu.in/raghava/b3pdb/browse_sub1.php?token=Chemical&col=10&page=5
    Reading: https://webs.iiitd.edu.in/raghava/b3pdb/browse_sub1.php?token=Chemical&col=10&page=6
    Reading: https://webs.iiitd.edu.in/raghava/b3pdb/browse_sub1.php?token=Chemical&col=10&page=7
    Reading: https://webs.iiitd.edu.in/raghava/b3pdb/browse_sub1.php?token=Chemical&col=10&page=8
    Reading: https://webs.iiitd.edu.in/raghava/b3pdb/browse_sub1.php?token=Chemical&col=10&page=9
    Reading: https://webs.iiitd.edu.in/raghava/b3pdb/browse_sub1.php?token=Chemical&col=10&page=10
    Reading: https://webs.iiitd.edu.in/raghava/b3pdb/browse_sub1.php?token=Chemical&col=10&page=11
    Raw B3Pdb chemical table shape: (507, 32)



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
      <th>B3pdbID</th>
      <th>PEPTIDE NAME</th>
      <th>PEPTIDE SEQUENCE (1-letter)</th>
      <th>PEPTIDE SEQUENCE (3-letter)</th>
      <th>N-TERMINAL MODIFICATION</th>
      <th>C-TERMINAL MODIFICATION</th>
      <th>CHEMICAL MODIFICATION</th>
      <th>PEPTIDE LENGTH</th>
      <th>PEPTIDE CONFORMATION</th>
      <th>PEPTIDE NATURE</th>
      <th>...</th>
      <th>TRANSPORT TYPE</th>
      <th>SUBCELLULAR LOCALISATION</th>
      <th>COMBINATION</th>
      <th>PHYSICAL CONDITION</th>
      <th>RESPONSE</th>
      <th>RESULT</th>
      <th>LABEL</th>
      <th>PMID</th>
      <th>source_page</th>
      <th>source_url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b3pdb_0002</td>
      <td>Tf-PFV</td>
      <td>PFVYLI</td>
      <td>ProPheValTyrLeuIle</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>Linear</td>
      <td>Cationic</td>
      <td>...</td>
      <td>Transcytosis</td>
      <td>NaN</td>
      <td>Combination with the transferrin protein</td>
      <td>Glioblastoma multiforme</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30261346b3pdb_0003K16ApoENANANANANA16NACationi...</td>
      <td>1</td>
      <td>https://webs.iiitd.edu.in/raghava/b3pdb/browse...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b3pdb_0003</td>
      <td>K16ApoE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16.0</td>
      <td>NaN</td>
      <td>Cationic</td>
      <td>...</td>
      <td>Transcytosis</td>
      <td>intracellular</td>
      <td>It is given in the combination with IgG4.1</td>
      <td>Alzheimer’s disease</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30300748b3pdb_0007D-T7HRPYIAHCHisArgProTyrIleA...</td>
      <td>1</td>
      <td>https://webs.iiitd.edu.in/raghava/b3pdb/browse...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b3pdb_0007</td>
      <td>D-T7</td>
      <td>HRPYIAHC</td>
      <td>HisArgProTyrIleAlaHisCys</td>
      <td>NaN</td>
      <td>Cysteine on C-terminal</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>Linear</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Combined with Cediranib and Paclitaxel.</td>
      <td>Glioma</td>
      <td>Stability/half life</td>
      <td>3.31-fold higher than saline</td>
      <td>NaN</td>
      <td>30525386b3pdb_0008Transportan 10AGYLLGKINLKALA...</td>
      <td>1</td>
      <td>https://webs.iiitd.edu.in/raghava/b3pdb/browse...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b3pdb_0008</td>
      <td>Transportan 10</td>
      <td>AGYLLGKINLKALAALAKKIL</td>
      <td>AlaGlyTyrLeuLeuGlyLysIleAsnLeuLysAlaLeuAlaAlaL...</td>
      <td>NaN</td>
      <td>Amide group is attached</td>
      <td>NaN</td>
      <td>21.0</td>
      <td>Linear</td>
      <td>Cationic</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Alone</td>
      <td>NaN</td>
      <td>MIC</td>
      <td>6.3 ± 1.3 micromolar</td>
      <td>NaN</td>
      <td>30824786b3pdb_0009Transportan 10AGYLLGKINLKALA...</td>
      <td>1</td>
      <td>https://webs.iiitd.edu.in/raghava/b3pdb/browse...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b3pdb_0009</td>
      <td>Transportan 10</td>
      <td>AGYLLGKINLKALAALAKKIL</td>
      <td>AlaGlyTyrLeuLeuGlyLysIleAsnLeuLysAlaLeuAlaAlaL...</td>
      <td>N-terminal Fmoc group was removed and the prop...</td>
      <td>Amide group is attached</td>
      <td>NaN</td>
      <td>21.0</td>
      <td>Linear</td>
      <td>Cationic</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Alone</td>
      <td>NaN</td>
      <td>MIC</td>
      <td>1.6 ± 0.9micromolar</td>
      <td>NaN</td>
      <td>30824786b3pdb_0010Transportan 10AGYLLGKINLKALA...</td>
      <td>1</td>
      <td>https://webs.iiitd.edu.in/raghava/b3pdb/browse...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>



```python
# -------------------------
# 1.2 Save raw B3Pdb table
# -------------------------

raw_b3pdb_path = raw_dir / "b3pdb_chemical_raw.csv"

raw_b3pdb_chemical.to_csv(
    raw_b3pdb_path,
    index=False,
)

print("Saved raw B3Pdb chemical table:", raw_b3pdb_path)
```

    Saved raw B3Pdb chemical table: ../data/raw/peptides/b3pdb/b3pdb_chemical_raw.csv


## 2. Clean B3Pdb peptide sequences

The raw B3Pdb table contains peptide sequences in one-letter amino-acid format. These sequences are standardised by converting to uppercase, removing non-letter characters, retaining only canonical amino acids, dropping missing values, and removing duplicates. The cleaned B3Pdb sequence table is saved for downstream model training.


```python
# -------------------------
# 2.1 Define peptide-cleaning helper
# -------------------------

import re

processed_dir = Path("../data/processed/peptides")
processed_dir.mkdir(parents=True, exist_ok=True)

canonical_aas = set("ACDEFGHIKLMNPQRSTVWY")


def clean_peptide(seq):
    if seq is None or pd.isna(seq):
        return None

    seq = str(seq).strip().upper()
    seq = re.sub(r"[^A-Z]", "", seq)

    if seq in {"", "NA", "NAN", "NONE", "NULL"}:
        return None

    if not set(seq).issubset(canonical_aas):
        return None

    return seq
```


```python
# -------------------------
# 2.2 Extract clean B3Pdb sequences
# -------------------------

seq_col = "PEPTIDE SEQUENCE (1-letter)"

b3pdb_chemical_sequences = (
    raw_b3pdb_chemical[seq_col]
    .apply(clean_peptide)
    .dropna()
    .drop_duplicates()
    .reset_index(drop=True)
    .to_frame("sequence")
)

b3pdb_chemical_sequences["source"] = "B3PDB_Chemical"
b3pdb_chemical_sequences["bbb_positive"] = 1
b3pdb_chemical_sequences["n_amino_acids"] = (
    b3pdb_chemical_sequences["sequence"].str.len()
)

print("Clean unique B3Pdb chemical sequences:", b3pdb_chemical_sequences.shape)
display(b3pdb_chemical_sequences.head(20))
```

    Clean unique B3Pdb chemical sequences: (135, 4)



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
      <td>PFVYLI</td>
      <td>B3PDB_Chemical</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HRPYIAHC</td>
      <td>B3PDB_Chemical</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AGYLLGKINLKALAALAKKIL</td>
      <td>B3PDB_Chemical</td>
      <td>1</td>
      <td>21</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CAQK</td>
      <td>B3PDB_Chemical</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CCAQK</td>
      <td>B3PDB_Chemical</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HGLASTLTRWAHYNALIRAF</td>
      <td>B3PDB_Chemical</td>
      <td>1</td>
      <td>20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>YAFDVVG</td>
      <td>B3PDB_Chemical</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SLSHSPQ</td>
      <td>B3PDB_Chemical</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>VAARTGEIYVPW</td>
      <td>B3PDB_Chemical</td>
      <td>1</td>
      <td>12</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GLHTSATNLYLH</td>
      <td>B3PDB_Chemical</td>
      <td>1</td>
      <td>12</td>
    </tr>
    <tr>
      <th>10</th>
      <td>THRPPMWSPVWP</td>
      <td>B3PDB_Chemical</td>
      <td>1</td>
      <td>12</td>
    </tr>
    <tr>
      <th>11</th>
      <td>HAIYPRH</td>
      <td>B3PDB_Chemical</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>12</th>
      <td>TFFYGGSRGKRNNFKTEEY</td>
      <td>B3PDB_Chemical</td>
      <td>1</td>
      <td>19</td>
    </tr>
    <tr>
      <th>13</th>
      <td>VRLPPPVKLPPPVRLPPP</td>
      <td>B3PDB_Chemical</td>
      <td>1</td>
      <td>18</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CTFFYGGSRGKRNNFKTEE</td>
      <td>B3PDB_Chemical</td>
      <td>1</td>
      <td>19</td>
    </tr>
    <tr>
      <th>15</th>
      <td>SHAVSS</td>
      <td>B3PDB_Chemical</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>16</th>
      <td>SHAVS</td>
      <td>B3PDB_Chemical</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>17</th>
      <td>TPPVSHAV</td>
      <td>B3PDB_Chemical</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>18</th>
      <td>CDTPPVC</td>
      <td>B3PDB_Chemical</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>19</th>
      <td>CNCKAPETALCARRCQQH</td>
      <td>B3PDB_Chemical</td>
      <td>1</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>



```python
# -------------------------
# 2.3 Save clean B3Pdb sequences
# -------------------------

b3pdb_sequences_path = processed_dir / "b3pdb_chemical_unique_sequences.csv"

b3pdb_chemical_sequences.to_csv(
    b3pdb_sequences_path,
    index=False,
)

print("Saved clean B3Pdb sequences:", b3pdb_sequences_path)
```

    Saved clean B3Pdb sequences: ../data/processed/peptides/b3pdb_chemical_unique_sequences.csv


## 3. Read and clean CPPsite2 natural peptide sequences

A natural CPPsite2 peptide FASTA file is used as an additional peptide corpus. These sequences are uptake-related peptide examples, not BBB-positive labels. They are cleaned using the same canonical amino-acid filter as B3Pdb and retained as a broader peptide pretraining/fine-tuning resource.


```python
# -------------------------
# 3.1 Define FASTA reader
# -------------------------

def read_fasta_sequences(path):
    sequences = []
    current = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)

    if current:
        sequences.append("".join(current))

    return sequences
```


```python
# -------------------------
# 3.2 Read and clean CPPsite2 natural peptides
# -------------------------

cpp_path = Path("../data/raw/peptides/cppsite2/natural_pep.fa")

cpp_sequences = (
    pd.Series(read_fasta_sequences(cpp_path), name="sequence")
    .apply(clean_peptide)
    .dropna()
    .drop_duplicates()
    .reset_index(drop=True)
    .to_frame()
)

cpp_sequences["source"] = "CPPsite2_natural"
cpp_sequences["bbb_positive"] = 0  # uptake-positive, not necessarily BBB-positive
cpp_sequences["n_amino_acids"] = cpp_sequences["sequence"].str.len()

print("Clean unique CPPsite2 sequences:", cpp_sequences.shape)
display(cpp_sequences.head(20))
```

    Clean unique CPPsite2 sequences: (1180, 4)



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
    <tr>
      <th>5</th>
      <td>KLAKLAKKLAKLAKGGRRRRRRR</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>6</th>
      <td>RWRWRWRW</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>WRWKKKKA</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>YGRKKRRQRRR</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>9</th>
      <td>KKALLAHALHLLALLALHLAHALKKA</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>26</td>
    </tr>
    <tr>
      <th>10</th>
      <td>RRRRRRRR</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GALFLGFLGAAGSTMGAWSQPKSKRKV</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>27</td>
    </tr>
    <tr>
      <th>12</th>
      <td>GLWRALWRLLRSLWRLLWRA</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>20</td>
    </tr>
    <tr>
      <th>13</th>
      <td>KAFAKLAARLYRKALARQLGVAA</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>14</th>
      <td>GSRVQIRCRFRNSTR</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>15</th>
      <td>GRKKRRQRRRPQ</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>16</th>
      <td>CRNGRGPDC</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>17</th>
      <td>CNGRC</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>18</th>
      <td>CRGDC</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>19</th>
      <td>CRGDKGDPC</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>


## 4. Combine B3Pdb and CPPsite2 peptide corpora

The cleaned B3Pdb and CPPsite2 sequences are combined into one unique peptide corpus for downstream language-model training. Duplicate sequences are removed. The `source` column records the first retained source label, and `bbb_positive` should be interpreted cautiously because CPPsite2 sequences are uptake-related rather than confirmed BBB-shuttle peptides.


```python
# -------------------------
# 4.1 Combine B3Pdb and CPPsite2 peptide datasets
# -------------------------

b3pdb_sequences = pd.read_csv(
    processed_dir / "b3pdb_chemical_unique_sequences.csv"
)

combined_peptides = pd.concat(
    [cpp_sequences, b3pdb_sequences],
    ignore_index=True,
)

combined_peptides = (
    combined_peptides
    .drop_duplicates(subset=["sequence"])
    .reset_index(drop=True)
)

print("Combined unique peptides:", combined_peptides.shape)

print("Source counts:")
display(combined_peptides["source"].value_counts().to_frame("n_sequences"))

display(combined_peptides.head(20))
```

    Combined unique peptides: (1311, 4)
    Source counts:



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
      <th>n_sequences</th>
    </tr>
    <tr>
      <th>source</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CPPsite2_natural</th>
      <td>1180</td>
    </tr>
    <tr>
      <th>B3PDB_Chemical</th>
      <td>131</td>
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
    <tr>
      <th>5</th>
      <td>KLAKLAKKLAKLAKGGRRRRRRR</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>6</th>
      <td>RWRWRWRW</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>WRWKKKKA</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>YGRKKRRQRRR</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>9</th>
      <td>KKALLAHALHLLALLALHLAHALKKA</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>26</td>
    </tr>
    <tr>
      <th>10</th>
      <td>RRRRRRRR</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GALFLGFLGAAGSTMGAWSQPKSKRKV</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>27</td>
    </tr>
    <tr>
      <th>12</th>
      <td>GLWRALWRLLRSLWRLLWRA</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>20</td>
    </tr>
    <tr>
      <th>13</th>
      <td>KAFAKLAARLYRKALARQLGVAA</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>14</th>
      <td>GSRVQIRCRFRNSTR</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>15</th>
      <td>GRKKRRQRRRPQ</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>16</th>
      <td>CRNGRGPDC</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>17</th>
      <td>CNGRC</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>18</th>
      <td>CRGDC</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>19</th>
      <td>CRGDKGDPC</td>
      <td>CPPsite2_natural</td>
      <td>0</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



```python
# -------------------------
# 4.2 Save combined peptide corpus
# -------------------------

combined_peptides_path = processed_dir / "cppsite_b3pdb_training_sequences.csv"

combined_peptides.to_csv(
    combined_peptides_path,
    index=False,
)

print("Saved combined peptide corpus:", combined_peptides_path)
```

    Saved combined peptide corpus: ../data/processed/peptides/cppsite_b3pdb_training_sequences.csv


## Final note

This notebook exports the cleaned peptide corpus used by downstream peptide language-model notebooks. The corpus is intended for generative model training and fine-tuning, not as a validated functional peptide benchmark.

Output file:

`../data/processed/peptides/cppsite_b3pdb_training_sequences.csv`
