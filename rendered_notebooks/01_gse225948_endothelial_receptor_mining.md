# Endothelial receptor mining from GSE225948

This notebook inspects the public single-cell transcriptomic dataset associated with the StrokeVis resource and extracts gene-level endothelial receptor candidates for downstream peptide-shuttle design.

The aim is not to perform full formal differential-expression modelling. Instead, this notebook builds a practical omics-to-peptide-target workflow:

1. inspect the downloaded raw metadata and count files,
2. identify brain endothelial cell populations and experimental groups,
3. verify endothelial/BBB identity using canonical marker genes,
4. build EC pseudobulk expression profiles,
5. rank early stroke-responsive endothelial genes,
6. annotate candidate proteins with UniProt accessions and subcellular localisation,
7. classify putative surface accessibility,
8. export a receptor shortlist for downstream peptide generation and structure-aware triage.

Dataset/resource:

- **Title:** Brain and blood single-cell transcriptomic analysis in acute and subacute phases after experimental stroke  
- **Web resource:** https://anratherlab.shinyapps.io/strokevis/  
- **Public release:** June 14, 2023  
- **Local raw-data folder:** `../data/raw/GSE225948_RAW/`

Important scope note:

This analysis identifies **gene-level, stroke-responsive endothelial surface/membrane-associated candidates**, not validated receptor proteins. The exported candidates require follow-up validation of isoform-specific topology, extracellular-domain exposure, receptor complex state, internalisation capacity, mouse-human translation, and experimental binding.

In the wider prototype, this notebook provides the first stage of the computational decision chain:

**transcriptomic mining → receptor prioritisation → peptide-design constraints → candidate generation → receptor-aware structure triage → experimental handoff**

## 1. Inspect downloaded raw files

The raw StrokeVis/GSE225948 files were downloaded into the local project directory. This first step lists the available compressed CSV files and previews the first rows of each table to understand their structure before selecting the relevant metadata and expression matrices.


```python
# -------------------------
# 1.1 Imports and paths
# -------------------------

import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path("../data/raw/GSE225948_RAW/")
files = sorted(data_dir.glob("*.csv.gz"))

print(f"Raw data directory: {data_dir}")
print(f"Number of .csv.gz files found: {len(files)}")
```

    Raw data directory: ../data/raw/GSE225948_RAW
    Number of .csv.gz files found: 58



```python
# -------------------------
# 1.2 List available files
# -------------------------

for file in files:
    print("-", file.name)
```

    - GSM7060815_Brain_GR180716_counts.csv.gz
    - GSM7060815_Brain_GR180716_metadata.csv.gz
    - GSM7060816_Brain_GR181128_counts.csv.gz
    - GSM7060816_Brain_GR181128_metadata.csv.gz
    - GSM7060817_Brain_GR181212_counts.csv.gz
    - GSM7060817_Brain_GR181212_metadata.csv.gz
    - GSM7060818_Brain_GR190110_counts.csv.gz
    - GSM7060818_Brain_GR190110_metadata.csv.gz
    - GSM7060819_Brain_GR180426_counts.csv.gz
    - GSM7060819_Brain_GR180426_metadata.csv.gz
    - GSM7060820_Brain_GR180614_counts.csv.gz
    - GSM7060820_Brain_GR180614_metadata.csv.gz
    - GSM7060821_Brain_GR180919_counts.csv.gz
    - GSM7060821_Brain_GR180919_metadata.csv.gz
    - GSM7060822_Brain_GR181024_counts.csv.gz
    - GSM7060822_Brain_GR181024_metadata.csv.gz
    - GSM7060823_Brain_GR180125_counts.csv.gz
    - GSM7060823_Brain_GR180125_metadata.csv.gz
    - GSM7060824_Brain_GR180613_counts.csv.gz
    - GSM7060824_Brain_GR180613_metadata.csv.gz
    - GSM7060825_Brain_GR180905_counts.csv.gz
    - GSM7060825_Brain_GR180905_metadata.csv.gz
    - GSM7060826_Brain_GR181114_counts.csv.gz
    - GSM7060826_Brain_GR181114_metadata.csv.gz
    - GSM7060827_Brain_aged_GR210708_counts.csv.gz
    - GSM7060827_Brain_aged_GR210708_metadata.csv.gz
    - GSM7060828_Brain_aged_GR200728_counts.csv.gz
    - GSM7060828_Brain_aged_GR200728_metadata.csv.gz
    - GSM7060829_Brain_aged_GR200723_counts.csv.gz
    - GSM7060829_Brain_aged_GR200723_metadata.csv.gz
    - GSM7060830_Brain_aged_GR200716_counts.csv.gz
    - GSM7060830_Brain_aged_GR200716_metadata.csv.gz
    - GSM7060831_Brain_aged_GR210225_counts.csv.gz
    - GSM7060831_Brain_aged_GR210225_metadata.csv.gz
    - GSM7060832_Brain_aged_GR200812_counts.csv.gz
    - GSM7060832_Brain_aged_GR200812_metadata.csv.gz
    - GSM7060833_Blood_GR18110701_counts.csv.gz
    - GSM7060833_Blood_GR18110701_metadata.csv.gz
    - GSM7060834_Blood_GR18121901_counts.csv.gz
    - GSM7060834_Blood_GR18121901_metadata.csv.gz
    - GSM7060835_Blood_GR190417_counts.csv.gz
    - GSM7060835_Blood_GR190417_metadata.csv.gz
    - GSM7060836_Blood_GR190522_counts.csv.gz
    - GSM7060836_Blood_GR190522_metadata.csv.gz
    - GSM7060837_Blood_GR18110702_counts.csv.gz
    - GSM7060837_Blood_GR18110702_metadata.csv.gz
    - GSM7060838_Blood_GR18121902_counts.csv.gz
    - GSM7060838_Blood_GR18121902_metadata.csv.gz
    - GSM7060839_Blood_GR190116_counts.csv.gz
    - GSM7060839_Blood_GR190116_metadata.csv.gz
    - GSM7060840_Blood_GR190410_counts.csv.gz
    - GSM7060840_Blood_GR190410_metadata.csv.gz
    - GSM7060841_Blood_GR181120_counts.csv.gz
    - GSM7060841_Blood_GR181120_metadata.csv.gz
    - GSM7060842_Blood_GR190415_counts.csv.gz
    - GSM7060842_Blood_GR190415_metadata.csv.gz
    - GSM7060843_Blood_GR190529_counts.csv.gz
    - GSM7060843_Blood_GR190529_metadata.csv.gz



```python
# -------------------------
# 1.3 Preview the first files
# -------------------------

for file in files[:5]:
    print("\n" + "=" * 80)
    print(file.name)

    df_preview = pd.read_csv(file, nrows=5)

    print("Preview shape:", df_preview.shape)
    display(df_preview)
```

    
    ================================================================================
    GSM7060815_Brain_GR180716_counts.csv.gz
    Preview shape: (5, 3466)



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
      <th>Unnamed: 0</th>
      <th>BRS02R1GGGATCGATATG</th>
      <th>BRS02R1ATTCAATATCAC</th>
      <th>BRS02R1ATTTCCGTCCGC</th>
      <th>BRS02R1TAGCGCGAGACC</th>
      <th>BRS02R1GCTGATTATTTG</th>
      <th>BRS02R1CTATCTCTCGAA</th>
      <th>BRS02R1CTGAATTTCCCC</th>
      <th>BRS02R1ATTCACACACTT</th>
      <th>BRS02R1ATTTGCGAGTGA</th>
      <th>...</th>
      <th>BRS02R1CATCTGTCGGAC</th>
      <th>BRS02R1ACGGGGTTAATG</th>
      <th>BRS02R1CACTACTTGTCC</th>
      <th>BRS02R1AGTCGGCTATAG</th>
      <th>BRS02R1AGAGCACTCGGC</th>
      <th>BRS02R1GGTCGCGTTGAT</th>
      <th>BRS02R1GGCGTTGGTACC</th>
      <th>BRS02R1TGTTCCGAGAAG</th>
      <th>BRS02R1GTGATGAAAAGT</th>
      <th>BRS02R1TTTTTTTTTCGN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0610009B22Rik</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.992199</td>
      <td>0</td>
      <td>0.97048</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0610009E02Rik</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.00000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0610009L18Rik</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.00000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0610010F05Rik</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.00000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0610010K14Rik</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.00000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 3466 columns</p>
</div>


    
    ================================================================================
    GSM7060815_Brain_GR180716_metadata.csv.gz
    Preview shape: (5, 15)



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
      <th>Unnamed: 0</th>
      <th>study.ident</th>
      <th>nCount_RNA</th>
      <th>nFeature_RNA</th>
      <th>percent.mt</th>
      <th>tissue</th>
      <th>organism</th>
      <th>strain</th>
      <th>sex</th>
      <th>age</th>
      <th>treatment</th>
      <th>Replicate</th>
      <th>Sample_description</th>
      <th>sub.celltype</th>
      <th>parent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BRS02R1GGGATCGATATG</td>
      <td>GR180716</td>
      <td>2988.583761</td>
      <td>973</td>
      <td>5.347042</td>
      <td>brain</td>
      <td>mus musculus</td>
      <td>C57BL/6J</td>
      <td>male</td>
      <td>W8</td>
      <td>Sham</td>
      <td>1</td>
      <td>2d sham; hemisphere; CD45hi + MG + EC</td>
      <td>Epi</td>
      <td>Epi</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BRS02R1ATTCAATATCAC</td>
      <td>GR180716</td>
      <td>3825.015376</td>
      <td>1138</td>
      <td>0.340638</td>
      <td>brain</td>
      <td>mus musculus</td>
      <td>C57BL/6J</td>
      <td>male</td>
      <td>W8</td>
      <td>Sham</td>
      <td>1</td>
      <td>2d sham; hemisphere; CD45hi + MG + EC</td>
      <td>Gran5</td>
      <td>Gran</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BRS02R1ATTTCCGTCCGC</td>
      <td>GR180716</td>
      <td>3426.051680</td>
      <td>1915</td>
      <td>0.302520</td>
      <td>brain</td>
      <td>mus musculus</td>
      <td>C57BL/6J</td>
      <td>male</td>
      <td>W8</td>
      <td>Sham</td>
      <td>1</td>
      <td>2d sham; hemisphere; CD45hi + MG + EC</td>
      <td>BAM2</td>
      <td>BAM</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BRS02R1TAGCGCGAGACC</td>
      <td>GR180716</td>
      <td>3953.069361</td>
      <td>1494</td>
      <td>1.176273</td>
      <td>brain</td>
      <td>mus musculus</td>
      <td>C57BL/6J</td>
      <td>male</td>
      <td>W8</td>
      <td>Sham</td>
      <td>1</td>
      <td>2d sham; hemisphere; CD45hi + MG + EC</td>
      <td>Epi</td>
      <td>Epi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BRS02R1GCTGATTATTTG</td>
      <td>GR180716</td>
      <td>2841.509538</td>
      <td>1526</td>
      <td>0.655353</td>
      <td>brain</td>
      <td>mus musculus</td>
      <td>C57BL/6J</td>
      <td>male</td>
      <td>W8</td>
      <td>Sham</td>
      <td>1</td>
      <td>2d sham; hemisphere; CD45hi + MG + EC</td>
      <td>MdC3</td>
      <td>MdC</td>
    </tr>
  </tbody>
</table>
</div>


    
    ================================================================================
    GSM7060816_Brain_GR181128_counts.csv.gz
    Preview shape: (5, 4715)



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
      <th>Unnamed: 0</th>
      <th>BRS02R2TCACGGTTCGTA</th>
      <th>BRS02R2ATAAGGCGGCAA</th>
      <th>BRS02R2TACAAAACCGAA</th>
      <th>BRS02R2ACCGGATCGGAC</th>
      <th>BRS02R2AGCAAATTCTCT</th>
      <th>BRS02R2ACCCCCAGAGTT</th>
      <th>BRS02R2TCTAATCTCATC</th>
      <th>BRS02R2CCGACCCGAGAA</th>
      <th>BRS02R2GTTCAGCAATTA</th>
      <th>...</th>
      <th>BRS02R2ATACGTGGCCCC</th>
      <th>BRS02R2CGGGGTGAGGGG</th>
      <th>BRS02R2GGAAGGCTACAG</th>
      <th>BRS02R2GTGTGTTTTATC</th>
      <th>BRS02R2ATCGACAGGGGG</th>
      <th>BRS02R2GGGTGGGGTAGA</th>
      <th>BRS02R2AATATTGGCTCT</th>
      <th>BRS02R2AGTCGCGTCTGA</th>
      <th>BRS02R2ACTGCTTCATAA</th>
      <th>BRS02R2ACTGCTTACTGA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0610009B22Rik</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0610009E02Rik</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0610009L18Rik</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0610010F05Rik</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0610010K14Rik</td>
      <td>0.749681</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.986626</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 4715 columns</p>
</div>


    
    ================================================================================
    GSM7060816_Brain_GR181128_metadata.csv.gz
    Preview shape: (5, 15)



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
      <th>Unnamed: 0</th>
      <th>study.ident</th>
      <th>nCount_RNA</th>
      <th>nFeature_RNA</th>
      <th>percent.mt</th>
      <th>tissue</th>
      <th>organism</th>
      <th>strain</th>
      <th>sex</th>
      <th>age</th>
      <th>treatment</th>
      <th>Replicate</th>
      <th>Sample_description</th>
      <th>sub.celltype</th>
      <th>parent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BRS02R2TCACGGTTCGTA</td>
      <td>GR181128</td>
      <td>3018.552883</td>
      <td>2123</td>
      <td>3.494441</td>
      <td>brain</td>
      <td>mus musculus</td>
      <td>C57BL/6J</td>
      <td>male</td>
      <td>W8</td>
      <td>Sham</td>
      <td>2</td>
      <td>2d sham; hemisphere; CD45hi + MG + EC</td>
      <td>Mg2</td>
      <td>Mg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BRS02R2ATAAGGCGGCAA</td>
      <td>GR181128</td>
      <td>2808.690684</td>
      <td>1834</td>
      <td>3.624112</td>
      <td>brain</td>
      <td>mus musculus</td>
      <td>C57BL/6J</td>
      <td>male</td>
      <td>W8</td>
      <td>Sham</td>
      <td>2</td>
      <td>2d sham; hemisphere; CD45hi + MG + EC</td>
      <td>Mg2</td>
      <td>Mg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BRS02R2TACAAAACCGAA</td>
      <td>GR181128</td>
      <td>2513.732804</td>
      <td>1577</td>
      <td>2.662910</td>
      <td>brain</td>
      <td>mus musculus</td>
      <td>C57BL/6J</td>
      <td>male</td>
      <td>W8</td>
      <td>Sham</td>
      <td>2</td>
      <td>2d sham; hemisphere; CD45hi + MG + EC</td>
      <td>Mg2</td>
      <td>Mg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BRS02R2ACCGGATCGGAC</td>
      <td>GR181128</td>
      <td>2822.869808</td>
      <td>1553</td>
      <td>3.014790</td>
      <td>brain</td>
      <td>mus musculus</td>
      <td>C57BL/6J</td>
      <td>male</td>
      <td>W8</td>
      <td>Sham</td>
      <td>2</td>
      <td>2d sham; hemisphere; CD45hi + MG + EC</td>
      <td>Mg2</td>
      <td>Mg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BRS02R2AGCAAATTCTCT</td>
      <td>GR181128</td>
      <td>2896.671892</td>
      <td>805</td>
      <td>0.946141</td>
      <td>brain</td>
      <td>mus musculus</td>
      <td>C57BL/6J</td>
      <td>male</td>
      <td>W8</td>
      <td>Sham</td>
      <td>2</td>
      <td>2d sham; hemisphere; CD45hi + MG + EC</td>
      <td>Gran5</td>
      <td>Gran</td>
    </tr>
  </tbody>
</table>
</div>


    
    ================================================================================
    GSM7060817_Brain_GR181212_counts.csv.gz
    Preview shape: (5, 3049)



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
      <th>Unnamed: 0</th>
      <th>BRS02R3TTTATTTCGGTC</th>
      <th>BRS02R3CAAACTTTGCTG</th>
      <th>BRS02R3GCGAAGATCACA</th>
      <th>BRS02R3CGATCACGATCG</th>
      <th>BRS02R3CTATGTTGTTTT</th>
      <th>BRS02R3CAGTTGCTCACT</th>
      <th>BRS02R3TGTACTTGAATC</th>
      <th>BRS02R3ACAATGCACGCC</th>
      <th>BRS02R3TATCATGGCCCG</th>
      <th>...</th>
      <th>BRS02R3CCGCGGCGGTGC</th>
      <th>BRS02R3TGGGCCGGTATC</th>
      <th>BRS02R3GTCTCCCTCTAT</th>
      <th>BRS02R3CGCCTGGTAGTG</th>
      <th>BRS02R3ATGAATTTCTTT</th>
      <th>BRS02R3ATACACAGGTAC</th>
      <th>BRS02R3TTCCCCTGCTTG</th>
      <th>BRS02R3AGATAGTCACCG</th>
      <th>BRS02R3GTGGGCGGCCTG</th>
      <th>BRS02R3CTCCCCTGCTTA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0610009B22Rik</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0610009E02Rik</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.981743</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0610009L18Rik</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0610010F05Rik</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.80012</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0610010K14Rik</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 3049 columns</p>
</div>


## 2. Combine metadata across sequencing samples

Each downloaded sample includes a metadata table describing cell-level annotations such as tissue, treatment, sample description, parent cell type, and sub-cell type. These metadata files are concatenated into one table so that endothelial cells and stroke-related experimental groups can be selected consistently across samples.


```python
# -------------------------
# 2.1 Load and concatenate metadata files
# -------------------------

metadata_files = sorted(data_dir.glob("*metadata.csv.gz"))

print(f"Metadata files found: {len(metadata_files)}")
for f in metadata_files:
    print("-", f.name)

all_meta = []

for f in metadata_files:
    meta = pd.read_csv(f)
    meta["source_file"] = f.name
    all_meta.append(meta)

meta_all = pd.concat(all_meta, ignore_index=True)

print("Combined metadata shape:", meta_all.shape)
display(meta_all.head())
```

    Metadata files found: 29
    - GSM7060815_Brain_GR180716_metadata.csv.gz
    - GSM7060816_Brain_GR181128_metadata.csv.gz
    - GSM7060817_Brain_GR181212_metadata.csv.gz
    - GSM7060818_Brain_GR190110_metadata.csv.gz
    - GSM7060819_Brain_GR180426_metadata.csv.gz
    - GSM7060820_Brain_GR180614_metadata.csv.gz
    - GSM7060821_Brain_GR180919_metadata.csv.gz
    - GSM7060822_Brain_GR181024_metadata.csv.gz
    - GSM7060823_Brain_GR180125_metadata.csv.gz
    - GSM7060824_Brain_GR180613_metadata.csv.gz
    - GSM7060825_Brain_GR180905_metadata.csv.gz
    - GSM7060826_Brain_GR181114_metadata.csv.gz
    - GSM7060827_Brain_aged_GR210708_metadata.csv.gz
    - GSM7060828_Brain_aged_GR200728_metadata.csv.gz
    - GSM7060829_Brain_aged_GR200723_metadata.csv.gz
    - GSM7060830_Brain_aged_GR200716_metadata.csv.gz
    - GSM7060831_Brain_aged_GR210225_metadata.csv.gz
    - GSM7060832_Brain_aged_GR200812_metadata.csv.gz
    - GSM7060833_Blood_GR18110701_metadata.csv.gz
    - GSM7060834_Blood_GR18121901_metadata.csv.gz
    - GSM7060835_Blood_GR190417_metadata.csv.gz
    - GSM7060836_Blood_GR190522_metadata.csv.gz
    - GSM7060837_Blood_GR18110702_metadata.csv.gz
    - GSM7060838_Blood_GR18121902_metadata.csv.gz
    - GSM7060839_Blood_GR190116_metadata.csv.gz
    - GSM7060840_Blood_GR190410_metadata.csv.gz
    - GSM7060841_Blood_GR181120_metadata.csv.gz
    - GSM7060842_Blood_GR190415_metadata.csv.gz
    - GSM7060843_Blood_GR190529_metadata.csv.gz
    Combined metadata shape: (106977, 16)



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
      <th>Unnamed: 0</th>
      <th>study.ident</th>
      <th>nCount_RNA</th>
      <th>nFeature_RNA</th>
      <th>percent.mt</th>
      <th>tissue</th>
      <th>organism</th>
      <th>strain</th>
      <th>sex</th>
      <th>age</th>
      <th>treatment</th>
      <th>Replicate</th>
      <th>Sample_description</th>
      <th>sub.celltype</th>
      <th>parent</th>
      <th>source_file</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BRS02R1GGGATCGATATG</td>
      <td>GR180716</td>
      <td>2988.583761</td>
      <td>973</td>
      <td>5.347042</td>
      <td>brain</td>
      <td>mus musculus</td>
      <td>C57BL/6J</td>
      <td>male</td>
      <td>W8</td>
      <td>Sham</td>
      <td>1</td>
      <td>2d sham; hemisphere; CD45hi + MG + EC</td>
      <td>Epi</td>
      <td>Epi</td>
      <td>GSM7060815_Brain_GR180716_metadata.csv.gz</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BRS02R1ATTCAATATCAC</td>
      <td>GR180716</td>
      <td>3825.015376</td>
      <td>1138</td>
      <td>0.340638</td>
      <td>brain</td>
      <td>mus musculus</td>
      <td>C57BL/6J</td>
      <td>male</td>
      <td>W8</td>
      <td>Sham</td>
      <td>1</td>
      <td>2d sham; hemisphere; CD45hi + MG + EC</td>
      <td>Gran5</td>
      <td>Gran</td>
      <td>GSM7060815_Brain_GR180716_metadata.csv.gz</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BRS02R1ATTTCCGTCCGC</td>
      <td>GR180716</td>
      <td>3426.051680</td>
      <td>1915</td>
      <td>0.302520</td>
      <td>brain</td>
      <td>mus musculus</td>
      <td>C57BL/6J</td>
      <td>male</td>
      <td>W8</td>
      <td>Sham</td>
      <td>1</td>
      <td>2d sham; hemisphere; CD45hi + MG + EC</td>
      <td>BAM2</td>
      <td>BAM</td>
      <td>GSM7060815_Brain_GR180716_metadata.csv.gz</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BRS02R1TAGCGCGAGACC</td>
      <td>GR180716</td>
      <td>3953.069361</td>
      <td>1494</td>
      <td>1.176273</td>
      <td>brain</td>
      <td>mus musculus</td>
      <td>C57BL/6J</td>
      <td>male</td>
      <td>W8</td>
      <td>Sham</td>
      <td>1</td>
      <td>2d sham; hemisphere; CD45hi + MG + EC</td>
      <td>Epi</td>
      <td>Epi</td>
      <td>GSM7060815_Brain_GR180716_metadata.csv.gz</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BRS02R1GCTGATTATTTG</td>
      <td>GR180716</td>
      <td>2841.509538</td>
      <td>1526</td>
      <td>0.655353</td>
      <td>brain</td>
      <td>mus musculus</td>
      <td>C57BL/6J</td>
      <td>male</td>
      <td>W8</td>
      <td>Sham</td>
      <td>1</td>
      <td>2d sham; hemisphere; CD45hi + MG + EC</td>
      <td>MdC3</td>
      <td>MdC</td>
      <td>GSM7060815_Brain_GR180716_metadata.csv.gz</td>
    </tr>
  </tbody>
</table>
</div>



```python
# -------------------------
# 2.2 Inspect metadata columns
# -------------------------

print("Metadata columns:")
for col in meta_all.columns:
    print("-", col)
```

    Metadata columns:
    - Unnamed: 0
    - study.ident
    - nCount_RNA
    - nFeature_RNA
    - percent.mt
    - tissue
    - organism
    - strain
    - sex
    - age
    - treatment
    - Replicate
    - Sample_description
    - sub.celltype
    - parent
    - source_file



```python
# -------------------------
# 2.3 Inspect tissue and treatment labels
# -------------------------

print("Tissues:")
display(meta_all["tissue"].value_counts().to_frame("n_cells"))

print("Treatments:")
display(meta_all["treatment"].value_counts().to_frame("n_cells"))

print("Sample descriptions:")
display(meta_all["Sample_description"].value_counts().to_frame("n_cells"))
```

    Tissues:



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
      <th>n_cells</th>
    </tr>
    <tr>
      <th>tissue</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>brain</th>
      <td>55268</td>
    </tr>
    <tr>
      <th>PB</th>
      <td>51709</td>
    </tr>
  </tbody>
</table>
</div>


    Treatments:



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
      <th>n_cells</th>
    </tr>
    <tr>
      <th>treatment</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>D02</th>
      <td>38311</td>
    </tr>
    <tr>
      <th>Sham</th>
      <td>36970</td>
    </tr>
    <tr>
      <th>D14</th>
      <td>31696</td>
    </tr>
  </tbody>
</table>
</div>


    Sample descriptions:



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
      <th>n_cells</th>
    </tr>
    <tr>
      <th>Sample_description</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Peripheral blood leukocytes (240K)</th>
      <td>51709</td>
    </tr>
    <tr>
      <th>2d sham; hemisphere; CD45hi + MG + EC</th>
      <td>15029</td>
    </tr>
    <tr>
      <th>2d stroke; ipsilateral hemisphere; CD45hi + MG + EC</th>
      <td>14823</td>
    </tr>
    <tr>
      <th>14d stroke; ipsilateral hemisphere; CD45hi + MG + EC</th>
      <td>13417</td>
    </tr>
    <tr>
      <th>2d stroke; ipsilateral hemisphere; CD45hi MG + EC</th>
      <td>3682</td>
    </tr>
    <tr>
      <th>2d sham; ipsilateral hemisphere; CD45hi MG + EC</th>
      <td>3069</td>
    </tr>
    <tr>
      <th>14d stroke; ipsilateral hemisphere; CD45hi MG + EC</th>
      <td>2636</td>
    </tr>
    <tr>
      <th>EC (20K); MG(20K); CD45hi(120k);</th>
      <td>1469</td>
    </tr>
    <tr>
      <th>EC (54K); MG(52K); CD45hi(27k);</th>
      <td>1143</td>
    </tr>
  </tbody>
</table>
</div>



```python
# -------------------------
# 2.4 Inspect cell-type annotations
# -------------------------

print("Parent cell types:")
display(meta_all["parent"].value_counts().to_frame("n_cells"))

print("Top sub-cell types:")
display(meta_all["sub.celltype"].value_counts().head(50).to_frame("n_cells"))
```

    Parent cell types:



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
      <th>n_cells</th>
    </tr>
    <tr>
      <th>parent</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Mg</th>
      <td>24707</td>
    </tr>
    <tr>
      <th>Bc</th>
      <td>18881</td>
    </tr>
    <tr>
      <th>Neu</th>
      <td>17186</td>
    </tr>
    <tr>
      <th>MdC</th>
      <td>12553</td>
    </tr>
    <tr>
      <th>Tc</th>
      <td>9047</td>
    </tr>
    <tr>
      <th>EC</th>
      <td>7627</td>
    </tr>
    <tr>
      <th>Mo</th>
      <td>6558</td>
    </tr>
    <tr>
      <th>DC</th>
      <td>4256</td>
    </tr>
    <tr>
      <th>NK</th>
      <td>2077</td>
    </tr>
    <tr>
      <th>Gran</th>
      <td>1531</td>
    </tr>
    <tr>
      <th>BAM</th>
      <td>1453</td>
    </tr>
    <tr>
      <th>Eos.Bas</th>
      <td>317</td>
    </tr>
    <tr>
      <th>pre</th>
      <td>263</td>
    </tr>
    <tr>
      <th>MC</th>
      <td>228</td>
    </tr>
    <tr>
      <th>Epi</th>
      <td>95</td>
    </tr>
    <tr>
      <th>OD</th>
      <td>87</td>
    </tr>
    <tr>
      <th>UC</th>
      <td>68</td>
    </tr>
    <tr>
      <th>MaC</th>
      <td>43</td>
    </tr>
  </tbody>
</table>
</div>


    Top sub-cell types:



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
      <th>n_cells</th>
    </tr>
    <tr>
      <th>sub.celltype</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bc1</th>
      <td>8910</td>
    </tr>
    <tr>
      <th>Mg1</th>
      <td>8519</td>
    </tr>
    <tr>
      <th>Neu1</th>
      <td>5918</td>
    </tr>
    <tr>
      <th>Mg2</th>
      <td>5567</td>
    </tr>
    <tr>
      <th>Mg3</th>
      <td>5118</td>
    </tr>
    <tr>
      <th>Neu2</th>
      <td>4698</td>
    </tr>
    <tr>
      <th>Bc2</th>
      <td>3589</td>
    </tr>
    <tr>
      <th>Tc1</th>
      <td>3257</td>
    </tr>
    <tr>
      <th>Tc2</th>
      <td>3203</td>
    </tr>
    <tr>
      <th>EC1</th>
      <td>3126</td>
    </tr>
    <tr>
      <th>MdC1</th>
      <td>2947</td>
    </tr>
    <tr>
      <th>Neu3</th>
      <td>2898</td>
    </tr>
    <tr>
      <th>MdC2</th>
      <td>2805</td>
    </tr>
    <tr>
      <th>MdC3</th>
      <td>2078</td>
    </tr>
    <tr>
      <th>MdC4</th>
      <td>2039</td>
    </tr>
    <tr>
      <th>Mo1</th>
      <td>1929</td>
    </tr>
    <tr>
      <th>Mo2</th>
      <td>1898</td>
    </tr>
    <tr>
      <th>Bc3</th>
      <td>1866</td>
    </tr>
    <tr>
      <th>Mg4</th>
      <td>1822</td>
    </tr>
    <tr>
      <th>Mo3</th>
      <td>1758</td>
    </tr>
    <tr>
      <th>NK1</th>
      <td>1675</td>
    </tr>
    <tr>
      <th>Bc4</th>
      <td>1666</td>
    </tr>
    <tr>
      <th>EC2</th>
      <td>1621</td>
    </tr>
    <tr>
      <th>Neu4</th>
      <td>1617</td>
    </tr>
    <tr>
      <th>MdC5</th>
      <td>1467</td>
    </tr>
    <tr>
      <th>Mg5</th>
      <td>1420</td>
    </tr>
    <tr>
      <th>Tc3</th>
      <td>1419</td>
    </tr>
    <tr>
      <th>Mg6</th>
      <td>1394</td>
    </tr>
    <tr>
      <th>Bc5</th>
      <td>1361</td>
    </tr>
    <tr>
      <th>Neu5</th>
      <td>1352</td>
    </tr>
    <tr>
      <th>Bc6</th>
      <td>1271</td>
    </tr>
    <tr>
      <th>MdC6</th>
      <td>1217</td>
    </tr>
    <tr>
      <th>DC1</th>
      <td>1091</td>
    </tr>
    <tr>
      <th>EC3</th>
      <td>950</td>
    </tr>
    <tr>
      <th>DC2</th>
      <td>887</td>
    </tr>
    <tr>
      <th>DC3</th>
      <td>776</td>
    </tr>
    <tr>
      <th>EC4</th>
      <td>768</td>
    </tr>
    <tr>
      <th>Gran1</th>
      <td>712</td>
    </tr>
    <tr>
      <th>Neu6</th>
      <td>703</td>
    </tr>
    <tr>
      <th>BAM1</th>
      <td>548</td>
    </tr>
    <tr>
      <th>Mo4</th>
      <td>545</td>
    </tr>
    <tr>
      <th>Gran2</th>
      <td>529</td>
    </tr>
    <tr>
      <th>EC5</th>
      <td>526</td>
    </tr>
    <tr>
      <th>Mg8</th>
      <td>523</td>
    </tr>
    <tr>
      <th>BAM2</th>
      <td>470</td>
    </tr>
    <tr>
      <th>DC4</th>
      <td>461</td>
    </tr>
    <tr>
      <th>Tc4</th>
      <td>452</td>
    </tr>
    <tr>
      <th>DC5</th>
      <td>438</td>
    </tr>
    <tr>
      <th>Mo5</th>
      <td>428</td>
    </tr>
    <tr>
      <th>Tc5</th>
      <td>377</td>
    </tr>
  </tbody>
</table>
</div>


## 3. Select brain endothelial cells

The receptor-mining workflow focuses on endothelial cells from brain tissue because these cells represent the blood-brain barrier and vascular interface relevant to peptide-shuttle design. This section filters the combined metadata to brain endothelial cells and checks whether enough cells are available across treatments, subclusters, and source files.


```python
# -------------------------
# 3.1 Filter brain endothelial cells
# -------------------------

ec_meta = meta_all[
    (meta_all["tissue"] == "brain")
    & (meta_all["parent"] == "EC")
].copy()

print("Brain endothelial metadata shape:", ec_meta.shape)
display(ec_meta.head())
```

    Brain endothelial metadata shape: (7627, 16)



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
      <th>Unnamed: 0</th>
      <th>study.ident</th>
      <th>nCount_RNA</th>
      <th>nFeature_RNA</th>
      <th>percent.mt</th>
      <th>tissue</th>
      <th>organism</th>
      <th>strain</th>
      <th>sex</th>
      <th>age</th>
      <th>treatment</th>
      <th>Replicate</th>
      <th>Sample_description</th>
      <th>sub.celltype</th>
      <th>parent</th>
      <th>source_file</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>BRS02R1CACAGCCGGTTC</td>
      <td>GR180716</td>
      <td>2994.936921</td>
      <td>1594</td>
      <td>1.055879</td>
      <td>brain</td>
      <td>mus musculus</td>
      <td>C57BL/6J</td>
      <td>male</td>
      <td>W8</td>
      <td>Sham</td>
      <td>1</td>
      <td>2d sham; hemisphere; CD45hi + MG + EC</td>
      <td>EC3</td>
      <td>EC</td>
      <td>GSM7060815_Brain_GR180716_metadata.csv.gz</td>
    </tr>
    <tr>
      <th>23</th>
      <td>BRS02R1TCTCCGGGCTCG</td>
      <td>GR180716</td>
      <td>2973.772621</td>
      <td>1563</td>
      <td>1.198053</td>
      <td>brain</td>
      <td>mus musculus</td>
      <td>C57BL/6J</td>
      <td>male</td>
      <td>W8</td>
      <td>Sham</td>
      <td>1</td>
      <td>2d sham; hemisphere; CD45hi + MG + EC</td>
      <td>EC3</td>
      <td>EC</td>
      <td>GSM7060815_Brain_GR180716_metadata.csv.gz</td>
    </tr>
    <tr>
      <th>37</th>
      <td>BRS02R1GCCTTCTCTGAC</td>
      <td>GR180716</td>
      <td>2167.838957</td>
      <td>1154</td>
      <td>0.560680</td>
      <td>brain</td>
      <td>mus musculus</td>
      <td>C57BL/6J</td>
      <td>male</td>
      <td>W8</td>
      <td>Sham</td>
      <td>1</td>
      <td>2d sham; hemisphere; CD45hi + MG + EC</td>
      <td>EC1</td>
      <td>EC</td>
      <td>GSM7060815_Brain_GR180716_metadata.csv.gz</td>
    </tr>
    <tr>
      <th>46</th>
      <td>BRS02R1GACTACGTTTTT</td>
      <td>GR180716</td>
      <td>1476.342542</td>
      <td>1027</td>
      <td>0.560441</td>
      <td>brain</td>
      <td>mus musculus</td>
      <td>C57BL/6J</td>
      <td>male</td>
      <td>W8</td>
      <td>Sham</td>
      <td>1</td>
      <td>2d sham; hemisphere; CD45hi + MG + EC</td>
      <td>EC4</td>
      <td>EC</td>
      <td>GSM7060815_Brain_GR180716_metadata.csv.gz</td>
    </tr>
    <tr>
      <th>58</th>
      <td>BRS02R1TAGGCGAGGATT</td>
      <td>GR180716</td>
      <td>1813.946297</td>
      <td>1253</td>
      <td>0.635394</td>
      <td>brain</td>
      <td>mus musculus</td>
      <td>C57BL/6J</td>
      <td>male</td>
      <td>W8</td>
      <td>Sham</td>
      <td>1</td>
      <td>2d sham; hemisphere; CD45hi + MG + EC</td>
      <td>EC1</td>
      <td>EC</td>
      <td>GSM7060815_Brain_GR180716_metadata.csv.gz</td>
    </tr>
  </tbody>
</table>
</div>



```python
# -------------------------
# 3.2 Inspect endothelial cells by treatment
# -------------------------

display(
    ec_meta["treatment"]
    .value_counts()
    .to_frame("n_cells")
)
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
      <th>n_cells</th>
    </tr>
    <tr>
      <th>treatment</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>D02</th>
      <td>3239</td>
    </tr>
    <tr>
      <th>Sham</th>
      <td>2602</td>
    </tr>
    <tr>
      <th>D14</th>
      <td>1786</td>
    </tr>
  </tbody>
</table>
</div>



```python
# -------------------------
# 3.3 Inspect endothelial subclusters
# -------------------------

display(
    ec_meta["sub.celltype"]
    .value_counts()
    .to_frame("n_cells")
)
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
      <th>n_cells</th>
    </tr>
    <tr>
      <th>sub.celltype</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>EC1</th>
      <td>3126</td>
    </tr>
    <tr>
      <th>EC2</th>
      <td>1621</td>
    </tr>
    <tr>
      <th>EC3</th>
      <td>950</td>
    </tr>
    <tr>
      <th>EC4</th>
      <td>768</td>
    </tr>
    <tr>
      <th>EC5</th>
      <td>526</td>
    </tr>
    <tr>
      <th>EC6</th>
      <td>350</td>
    </tr>
    <tr>
      <th>EC7</th>
      <td>151</td>
    </tr>
    <tr>
      <th>EC8</th>
      <td>105</td>
    </tr>
    <tr>
      <th>EC9</th>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>



```python
# -------------------------
# 3.4 Cross-tabulate endothelial subclusters by treatment
# -------------------------

ec_subcluster_treatment_counts = pd.crosstab(
    ec_meta["sub.celltype"],
    ec_meta["treatment"],
)

display(ec_subcluster_treatment_counts)
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
      <th>treatment</th>
      <th>D02</th>
      <th>D14</th>
      <th>Sham</th>
    </tr>
    <tr>
      <th>sub.celltype</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>EC1</th>
      <td>774</td>
      <td>788</td>
      <td>1564</td>
    </tr>
    <tr>
      <th>EC2</th>
      <td>1486</td>
      <td>57</td>
      <td>78</td>
    </tr>
    <tr>
      <th>EC3</th>
      <td>269</td>
      <td>452</td>
      <td>229</td>
    </tr>
    <tr>
      <th>EC4</th>
      <td>244</td>
      <td>123</td>
      <td>401</td>
    </tr>
    <tr>
      <th>EC5</th>
      <td>202</td>
      <td>233</td>
      <td>91</td>
    </tr>
    <tr>
      <th>EC6</th>
      <td>154</td>
      <td>74</td>
      <td>122</td>
    </tr>
    <tr>
      <th>EC7</th>
      <td>61</td>
      <td>30</td>
      <td>60</td>
    </tr>
    <tr>
      <th>EC8</th>
      <td>29</td>
      <td>28</td>
      <td>48</td>
    </tr>
    <tr>
      <th>EC9</th>
      <td>20</td>
      <td>1</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



```python
# -------------------------
# 3.5 Inspect endothelial cells by treatment and source file
# -------------------------

ec_by_source = (
    ec_meta
    .groupby(["treatment", "source_file"])
    .size()
    .reset_index(name="n_cells")
    .sort_values(["treatment", "source_file"])
)

display(ec_by_source)
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
      <th>treatment</th>
      <th>source_file</th>
      <th>n_cells</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>D02</td>
      <td>GSM7060819_Brain_GR180426_metadata.csv.gz</td>
      <td>360</td>
    </tr>
    <tr>
      <th>1</th>
      <td>D02</td>
      <td>GSM7060820_Brain_GR180614_metadata.csv.gz</td>
      <td>458</td>
    </tr>
    <tr>
      <th>2</th>
      <td>D02</td>
      <td>GSM7060821_Brain_GR180919_metadata.csv.gz</td>
      <td>502</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D02</td>
      <td>GSM7060822_Brain_GR181024_metadata.csv.gz</td>
      <td>1171</td>
    </tr>
    <tr>
      <th>4</th>
      <td>D02</td>
      <td>GSM7060829_Brain_aged_GR200723_metadata.csv.gz</td>
      <td>325</td>
    </tr>
    <tr>
      <th>5</th>
      <td>D02</td>
      <td>GSM7060830_Brain_aged_GR200716_metadata.csv.gz</td>
      <td>219</td>
    </tr>
    <tr>
      <th>6</th>
      <td>D02</td>
      <td>GSM7060831_Brain_aged_GR210225_metadata.csv.gz</td>
      <td>204</td>
    </tr>
    <tr>
      <th>7</th>
      <td>D14</td>
      <td>GSM7060823_Brain_GR180125_metadata.csv.gz</td>
      <td>674</td>
    </tr>
    <tr>
      <th>8</th>
      <td>D14</td>
      <td>GSM7060824_Brain_GR180613_metadata.csv.gz</td>
      <td>319</td>
    </tr>
    <tr>
      <th>9</th>
      <td>D14</td>
      <td>GSM7060825_Brain_GR180905_metadata.csv.gz</td>
      <td>300</td>
    </tr>
    <tr>
      <th>10</th>
      <td>D14</td>
      <td>GSM7060826_Brain_GR181114_metadata.csv.gz</td>
      <td>317</td>
    </tr>
    <tr>
      <th>11</th>
      <td>D14</td>
      <td>GSM7060832_Brain_aged_GR200812_metadata.csv.gz</td>
      <td>176</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Sham</td>
      <td>GSM7060815_Brain_GR180716_metadata.csv.gz</td>
      <td>546</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Sham</td>
      <td>GSM7060816_Brain_GR181128_metadata.csv.gz</td>
      <td>617</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Sham</td>
      <td>GSM7060817_Brain_GR181212_metadata.csv.gz</td>
      <td>321</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Sham</td>
      <td>GSM7060818_Brain_GR190110_metadata.csv.gz</td>
      <td>295</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Sham</td>
      <td>GSM7060827_Brain_aged_GR210708_metadata.csv.gz</td>
      <td>585</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Sham</td>
      <td>GSM7060828_Brain_aged_GR200728_metadata.csv.gz</td>
      <td>238</td>
    </tr>
  </tbody>
</table>
</div>


## 4. Validate endothelial identity using marker genes

Before prioritising receptor candidates, the endothelial-cell annotation is checked using known endothelial and blood-brain-barrier markers. The expectation is that brain EC-labelled cells should express endothelial markers such as `Cldn5`, `Pecam1`, `Cdh5`, `Kdr`, `Flt1`, `Tek`, `Slc2a1`, and `Mfsd2a`, while non-endothelial markers such as immune, pericyte, mural, and astrocyte markers should remain comparatively low.

This step is a sanity check to support that the EC subset used for downstream receptor mining is biologically reasonable.


```python
# -------------------------
# 4.1 Define endothelial and negative-control marker genes
# -------------------------

endothelial_markers = [
    "Cldn5", "Pecam1", "Cdh5", "Kdr", "Flt1", "Tek",
    "Slc2a1", "Mfsd2a", "Abcb1a", "Abcg2", "Tfrc", "Slc7a5",
    "Vwf", "Cav1", "Plvap",
]

negative_markers = [
    "Ptprc",   # immune cells
    "Aif1",   # microglia/macrophages
    "Pdgfrb", # pericytes
    "Rgs5",   # mural/pericyte
    "Gfap",   # astrocytes
    "Aqp4",   # astrocyte/endfoot-associated
]

marker_genes = endothelial_markers + negative_markers

print("Endothelial markers:", len(endothelial_markers))
print("Negative-control markers:", len(negative_markers))
print("Total marker genes:", len(marker_genes))
```

    Endothelial markers: 15
    Negative-control markers: 6
    Total marker genes: 21



```python
# -------------------------
# 4.2 Helper function to calculate EC marker expression per sample
# -------------------------

def load_gene_panel_for_ec(meta_path, genes):
    counts_path = Path(str(meta_path).replace("_metadata.csv.gz", "_counts.csv.gz"))

    if not counts_path.exists():
        print(f"Counts file missing for: {meta_path.name}")
        return None

    meta = pd.read_csv(meta_path)
    meta["Unnamed: 0"] = meta["Unnamed: 0"].astype(str)

    ec_meta_sample = meta[
        (meta["tissue"] == "brain")
        & (meta["parent"] == "EC")
    ].copy()

    if ec_meta_sample.empty:
        return None

    ec_cells = ec_meta_sample["Unnamed: 0"].tolist()

    counts = pd.read_csv(counts_path, index_col=0)
    counts.index = counts.index.astype(str)
    counts.columns = counts.columns.astype(str)

    available_genes = [g for g in genes if g in counts.index]
    available_cells = [c for c in ec_cells if c in counts.columns]

    print(
        meta_path.name,
        "| EC cells:", len(ec_cells),
        "| matched cells:", len(available_cells),
        "| matched genes:", len(available_genes),
    )

    if len(available_genes) == 0 or len(available_cells) == 0:
        return None

    expr = counts.loc[available_genes, available_cells]

    out = expr.mean(axis=1).reset_index()
    out.columns = ["gene", "mean_expression"]

    out["treatment"] = ec_meta_sample["treatment"].iloc[0]
    out["source_file"] = meta_path.name
    out["n_ec_cells"] = len(available_cells)

    return out
```


```python
# -------------------------
# 4.3 Calculate marker expression across EC samples
# -------------------------

all_marker_expr = []

for meta_path in sorted(data_dir.glob("*metadata.csv.gz")):
    result = load_gene_panel_for_ec(meta_path, marker_genes)
    if result is not None:
        all_marker_expr.append(result)

if len(all_marker_expr) == 0:
    raise ValueError("No marker expression results found. Check data_dir and filenames.")

marker_expr = pd.concat(all_marker_expr, ignore_index=True)

print("Marker expression table shape:", marker_expr.shape)
display(marker_expr.head())
```

    GSM7060815_Brain_GR180716_metadata.csv.gz | EC cells: 546 | matched cells: 546 | matched genes: 20
    GSM7060816_Brain_GR181128_metadata.csv.gz | EC cells: 617 | matched cells: 617 | matched genes: 20
    GSM7060817_Brain_GR181212_metadata.csv.gz | EC cells: 321 | matched cells: 321 | matched genes: 20
    GSM7060818_Brain_GR190110_metadata.csv.gz | EC cells: 295 | matched cells: 295 | matched genes: 20
    GSM7060819_Brain_GR180426_metadata.csv.gz | EC cells: 360 | matched cells: 360 | matched genes: 20
    GSM7060820_Brain_GR180614_metadata.csv.gz | EC cells: 458 | matched cells: 458 | matched genes: 20
    GSM7060821_Brain_GR180919_metadata.csv.gz | EC cells: 502 | matched cells: 502 | matched genes: 20
    GSM7060822_Brain_GR181024_metadata.csv.gz | EC cells: 1171 | matched cells: 1171 | matched genes: 20
    GSM7060823_Brain_GR180125_metadata.csv.gz | EC cells: 674 | matched cells: 674 | matched genes: 20
    GSM7060824_Brain_GR180613_metadata.csv.gz | EC cells: 319 | matched cells: 319 | matched genes: 20
    GSM7060825_Brain_GR180905_metadata.csv.gz | EC cells: 300 | matched cells: 300 | matched genes: 20
    GSM7060826_Brain_GR181114_metadata.csv.gz | EC cells: 317 | matched cells: 317 | matched genes: 20
    GSM7060827_Brain_aged_GR210708_metadata.csv.gz | EC cells: 585 | matched cells: 585 | matched genes: 21
    GSM7060828_Brain_aged_GR200728_metadata.csv.gz | EC cells: 238 | matched cells: 238 | matched genes: 21
    GSM7060829_Brain_aged_GR200723_metadata.csv.gz | EC cells: 325 | matched cells: 325 | matched genes: 21
    GSM7060830_Brain_aged_GR200716_metadata.csv.gz | EC cells: 219 | matched cells: 219 | matched genes: 21
    GSM7060831_Brain_aged_GR210225_metadata.csv.gz | EC cells: 204 | matched cells: 204 | matched genes: 21
    GSM7060832_Brain_aged_GR200812_metadata.csv.gz | EC cells: 176 | matched cells: 176 | matched genes: 21
    Marker expression table shape: (366, 5)



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
      <th>gene</th>
      <th>mean_expression</th>
      <th>treatment</th>
      <th>source_file</th>
      <th>n_ec_cells</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cldn5</td>
      <td>4.046015</td>
      <td>Sham</td>
      <td>GSM7060815_Brain_GR180716_metadata.csv.gz</td>
      <td>546</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pecam1</td>
      <td>0.668399</td>
      <td>Sham</td>
      <td>GSM7060815_Brain_GR180716_metadata.csv.gz</td>
      <td>546</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cdh5</td>
      <td>0.363003</td>
      <td>Sham</td>
      <td>GSM7060815_Brain_GR180716_metadata.csv.gz</td>
      <td>546</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kdr</td>
      <td>0.250754</td>
      <td>Sham</td>
      <td>GSM7060815_Brain_GR180716_metadata.csv.gz</td>
      <td>546</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Flt1</td>
      <td>3.538528</td>
      <td>Sham</td>
      <td>GSM7060815_Brain_GR180716_metadata.csv.gz</td>
      <td>546</td>
    </tr>
  </tbody>
</table>
</div>



```python
# -------------------------
# 4.4 Summarise marker expression by treatment
# -------------------------

marker_summary = (
    marker_expr
    .groupby(["gene", "treatment"])["mean_expression"]
    .mean()
    .reset_index()
    .pivot(index="gene", columns="treatment", values="mean_expression")
)

display(marker_summary)
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
      <th>treatment</th>
      <th>D02</th>
      <th>D14</th>
      <th>Sham</th>
    </tr>
    <tr>
      <th>gene</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Abcb1a</th>
      <td>7.003937e-01</td>
      <td>1.268677e+00</td>
      <td>1.054301</td>
    </tr>
    <tr>
      <th>Abcg2</th>
      <td>5.175889e-01</td>
      <td>5.456159e-01</td>
      <td>0.431363</td>
    </tr>
    <tr>
      <th>Aif1</th>
      <td>1.219461e-15</td>
      <td>4.575544e-11</td>
      <td>0.003254</td>
    </tr>
    <tr>
      <th>Aqp4</th>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Cav1</th>
      <td>2.438081e-01</td>
      <td>2.362453e-01</td>
      <td>0.174210</td>
    </tr>
    <tr>
      <th>Cdh5</th>
      <td>2.364959e-01</td>
      <td>1.750940e-01</td>
      <td>0.129798</td>
    </tr>
    <tr>
      <th>Cldn5</th>
      <td>2.277336e+00</td>
      <td>2.905112e+00</td>
      <td>1.813087</td>
    </tr>
    <tr>
      <th>Flt1</th>
      <td>1.591764e+00</td>
      <td>2.360281e+00</td>
      <td>2.061095</td>
    </tr>
    <tr>
      <th>Gfap</th>
      <td>4.220719e-04</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Kdr</th>
      <td>3.307039e-01</td>
      <td>3.346536e-01</td>
      <td>0.201088</td>
    </tr>
    <tr>
      <th>Mfsd2a</th>
      <td>8.065566e-02</td>
      <td>1.586472e-01</td>
      <td>0.099872</td>
    </tr>
    <tr>
      <th>Pdgfrb</th>
      <td>1.132484e-03</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Pecam1</th>
      <td>7.779820e-01</td>
      <td>4.715836e-01</td>
      <td>0.377730</td>
    </tr>
    <tr>
      <th>Plvap</th>
      <td>5.198196e-02</td>
      <td>5.313738e-02</td>
      <td>0.038875</td>
    </tr>
    <tr>
      <th>Ptprc</th>
      <td>5.713150e-17</td>
      <td>2.185921e-03</td>
      <td>0.002197</td>
    </tr>
    <tr>
      <th>Rgs5</th>
      <td>9.811499e-02</td>
      <td>1.657289e-01</td>
      <td>0.143743</td>
    </tr>
    <tr>
      <th>Slc2a1</th>
      <td>9.418846e-01</td>
      <td>6.113081e-01</td>
      <td>0.622390</td>
    </tr>
    <tr>
      <th>Slc7a5</th>
      <td>7.360032e-02</td>
      <td>1.361557e-01</td>
      <td>0.092941</td>
    </tr>
    <tr>
      <th>Tek</th>
      <td>2.611584e-01</td>
      <td>3.557953e-01</td>
      <td>0.234254</td>
    </tr>
    <tr>
      <th>Tfrc</th>
      <td>1.472513e-01</td>
      <td>3.443653e-01</td>
      <td>0.222496</td>
    </tr>
    <tr>
      <th>Vwf</th>
      <td>2.198314e-01</td>
      <td>7.246424e-02</td>
      <td>0.065979</td>
    </tr>
  </tbody>
</table>
</div>



```python
# -------------------------
# 4.5 Save EC marker validation outputs
# -------------------------

processed_dir = Path("../data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

marker_summary_path = processed_dir / "ec_marker_summary.csv"
marker_expr_path = processed_dir / "ec_marker_expression_by_sample.csv"

marker_summary.to_csv(marker_summary_path)
marker_expr.to_csv(marker_expr_path, index=False)

print("Saved marker summary:", marker_summary_path)
print("Saved marker expression by sample:", marker_expr_path)
```

    Saved marker summary: ../data/processed/ec_marker_summary.csv
    Saved marker expression by sample: ../data/processed/ec_marker_expression_by_sample.csv


## 5. Build EC pseudobulk expression and compare stroke time points

To prioritise stroke-induced endothelial receptor candidates, a sample-level EC pseudobulk matrix is created. For each sample, the expression of each gene is averaged across brain endothelial cells. This converts the cell-level count matrices into one expression profile per biological sample.

The pseudobulk matrix is then used to compare early and subacute stroke time points against sham controls:

- `D02` versus `Sham`
- `D14` versus `Sham`

The resulting tables provide simple effect-size, log2 fold-change proxy, and Welch t-test statistics for each gene. These outputs are used later for receptor prioritisation rather than formal differential-expression claims.


```python
# -------------------------
# 5.1 Build EC pseudobulk expression matrix
# -------------------------

from scipy.stats import ttest_ind

out_dir = Path("../data/processed/")
out_dir.mkdir(parents=True, exist_ok=True)

sample_expr = []
sample_info = []

for meta_path in sorted(data_dir.glob("*metadata.csv.gz")):
    counts_path = Path(str(meta_path).replace("_metadata.csv.gz", "_counts.csv.gz"))

    if not counts_path.exists():
        continue

    meta = pd.read_csv(meta_path)
    meta["Unnamed: 0"] = meta["Unnamed: 0"].astype(str)

    # Keep only brain endothelial cells
    ec_meta_sample = meta[
        (meta["tissue"] == "brain")
        & (meta["parent"] == "EC")
    ].copy()

    if ec_meta_sample.empty:
        continue

    ec_cells = ec_meta_sample["Unnamed: 0"].tolist()

    counts = pd.read_csv(counts_path, index_col=0)
    counts.index = counts.index.astype(str)
    counts.columns = counts.columns.astype(str)

    available_cells = [c for c in ec_cells if c in counts.columns]

    if len(available_cells) == 0:
        continue

    # Pseudobulk: average expression of each gene across EC cells in this sample
    mean_expr = counts[available_cells].mean(axis=1)

    sample_id = meta_path.name.replace("_metadata.csv.gz", "")
    mean_expr.name = sample_id

    sample_expr.append(mean_expr)

    sample_info.append({
        "sample_id": sample_id,
        "source_file": meta_path.name,
        "treatment": ec_meta_sample["treatment"].iloc[0],
        "age": ec_meta_sample["age"].iloc[0],
        "sex": ec_meta_sample["sex"].iloc[0],
        "n_ec_cells": len(available_cells),
    })

ec_pseudobulk = pd.concat(sample_expr, axis=1)
ec_sample_info = pd.DataFrame(sample_info)

print("Pseudobulk matrix shape:", ec_pseudobulk.shape)
display(ec_sample_info)
```

    Pseudobulk matrix shape: (27641, 18)



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
      <th>sample_id</th>
      <th>source_file</th>
      <th>treatment</th>
      <th>age</th>
      <th>sex</th>
      <th>n_ec_cells</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GSM7060815_Brain_GR180716</td>
      <td>GSM7060815_Brain_GR180716_metadata.csv.gz</td>
      <td>Sham</td>
      <td>W8</td>
      <td>male</td>
      <td>546</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GSM7060816_Brain_GR181128</td>
      <td>GSM7060816_Brain_GR181128_metadata.csv.gz</td>
      <td>Sham</td>
      <td>W8</td>
      <td>male</td>
      <td>617</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GSM7060817_Brain_GR181212</td>
      <td>GSM7060817_Brain_GR181212_metadata.csv.gz</td>
      <td>Sham</td>
      <td>W10</td>
      <td>male</td>
      <td>321</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GSM7060818_Brain_GR190110</td>
      <td>GSM7060818_Brain_GR190110_metadata.csv.gz</td>
      <td>Sham</td>
      <td>W8</td>
      <td>male</td>
      <td>295</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GSM7060819_Brain_GR180426</td>
      <td>GSM7060819_Brain_GR180426_metadata.csv.gz</td>
      <td>D02</td>
      <td>W8</td>
      <td>male</td>
      <td>360</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GSM7060820_Brain_GR180614</td>
      <td>GSM7060820_Brain_GR180614_metadata.csv.gz</td>
      <td>D02</td>
      <td>W8</td>
      <td>male</td>
      <td>458</td>
    </tr>
    <tr>
      <th>6</th>
      <td>GSM7060821_Brain_GR180919</td>
      <td>GSM7060821_Brain_GR180919_metadata.csv.gz</td>
      <td>D02</td>
      <td>W8</td>
      <td>male</td>
      <td>502</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GSM7060822_Brain_GR181024</td>
      <td>GSM7060822_Brain_GR181024_metadata.csv.gz</td>
      <td>D02</td>
      <td>W8</td>
      <td>male</td>
      <td>1171</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GSM7060823_Brain_GR180125</td>
      <td>GSM7060823_Brain_GR180125_metadata.csv.gz</td>
      <td>D14</td>
      <td>W10</td>
      <td>male</td>
      <td>674</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GSM7060824_Brain_GR180613</td>
      <td>GSM7060824_Brain_GR180613_metadata.csv.gz</td>
      <td>D14</td>
      <td>W8</td>
      <td>male</td>
      <td>319</td>
    </tr>
    <tr>
      <th>10</th>
      <td>GSM7060825_Brain_GR180905</td>
      <td>GSM7060825_Brain_GR180905_metadata.csv.gz</td>
      <td>D14</td>
      <td>W8</td>
      <td>male</td>
      <td>300</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GSM7060826_Brain_GR181114</td>
      <td>GSM7060826_Brain_GR181114_metadata.csv.gz</td>
      <td>D14</td>
      <td>W8</td>
      <td>male</td>
      <td>317</td>
    </tr>
    <tr>
      <th>12</th>
      <td>GSM7060827_Brain_aged_GR210708</td>
      <td>GSM7060827_Brain_aged_GR210708_metadata.csv.gz</td>
      <td>Sham</td>
      <td>M20</td>
      <td>male</td>
      <td>585</td>
    </tr>
    <tr>
      <th>13</th>
      <td>GSM7060828_Brain_aged_GR200728</td>
      <td>GSM7060828_Brain_aged_GR200728_metadata.csv.gz</td>
      <td>Sham</td>
      <td>M18</td>
      <td>female</td>
      <td>238</td>
    </tr>
    <tr>
      <th>14</th>
      <td>GSM7060829_Brain_aged_GR200723</td>
      <td>GSM7060829_Brain_aged_GR200723_metadata.csv.gz</td>
      <td>D02</td>
      <td>M17</td>
      <td>male</td>
      <td>325</td>
    </tr>
    <tr>
      <th>15</th>
      <td>GSM7060830_Brain_aged_GR200716</td>
      <td>GSM7060830_Brain_aged_GR200716_metadata.csv.gz</td>
      <td>D02</td>
      <td>M18</td>
      <td>female</td>
      <td>219</td>
    </tr>
    <tr>
      <th>16</th>
      <td>GSM7060831_Brain_aged_GR210225</td>
      <td>GSM7060831_Brain_aged_GR210225_metadata.csv.gz</td>
      <td>D02</td>
      <td>M18</td>
      <td>female</td>
      <td>204</td>
    </tr>
    <tr>
      <th>17</th>
      <td>GSM7060832_Brain_aged_GR200812</td>
      <td>GSM7060832_Brain_aged_GR200812_metadata.csv.gz</td>
      <td>D14</td>
      <td>M18</td>
      <td>female</td>
      <td>176</td>
    </tr>
  </tbody>
</table>
</div>



```python
# -------------------------
# 5.2 Save EC pseudobulk outputs
# -------------------------

ec_pseudobulk_path = out_dir / "ec_pseudobulk_mean_expression.csv"
ec_sample_info_path = out_dir / "ec_sample_info.csv"

ec_pseudobulk.to_csv(ec_pseudobulk_path)
ec_sample_info.to_csv(ec_sample_info_path, index=False)

print("Saved EC pseudobulk matrix:", ec_pseudobulk_path)
print("Saved EC sample information:", ec_sample_info_path)
```

    Saved EC pseudobulk matrix: ../data/processed/ec_pseudobulk_mean_expression.csv
    Saved EC sample information: ../data/processed/ec_sample_info.csv



```python
# -------------------------
# 5.3 Define treatment comparison helper
# -------------------------

def compare_treatments(expr, info, case, control="Sham"):
    case_samples = info.loc[info["treatment"] == case, "sample_id"].tolist()
    control_samples = info.loc[info["treatment"] == control, "sample_id"].tolist()

    print(f"{case} samples:", len(case_samples))
    print(f"{control} samples:", len(control_samples))

    if len(case_samples) == 0 or len(control_samples) == 0:
        raise ValueError(f"Missing samples for comparison: {case} vs {control}")

    x_case = expr[case_samples]
    x_control = expr[control_samples]

    mean_case = x_case.mean(axis=1)
    mean_control = x_control.mean(axis=1)

    effect = mean_case - mean_control

    eps = 1e-6
    log2fc_proxy = np.log2((mean_case + eps) / (mean_control + eps))

    pvals = ttest_ind(
        x_case.T,
        x_control.T,
        axis=0,
        equal_var=False,
        nan_policy="omit",
    ).pvalue

    out = pd.DataFrame({
        "gene": expr.index,
        f"mean_{case}": mean_case.values,
        f"mean_{control}": mean_control.values,
        f"effect_{case}_vs_{control}": effect.values,
        f"log2fc_proxy_{case}_vs_{control}": log2fc_proxy.values,
        f"pvalue_{case}_vs_{control}": pvals,
    })

    return out.sort_values(
        f"effect_{case}_vs_{control}",
        ascending=False,
    ).reset_index(drop=True)
```


```python
# -------------------------
# 5.4 Compare D02 and D14 versus Sham
# -------------------------

de_d02 = compare_treatments(
    ec_pseudobulk,
    ec_sample_info,
    case="D02",
)

de_d14 = compare_treatments(
    ec_pseudobulk,
    ec_sample_info,
    case="D14",
)

de_d02_path = out_dir / "ec_DE_D02_vs_Sham.csv"
de_d14_path = out_dir / "ec_DE_D14_vs_Sham.csv"

de_d02.to_csv(de_d02_path, index=False)
de_d14.to_csv(de_d14_path, index=False)

print("Saved D02 comparison:", de_d02_path)
print("Saved D14 comparison:", de_d14_path)

print("Top D02-induced EC genes:")
display(de_d02.head(20))

print("Top D14-induced EC genes:")
display(de_d14.head(20))
```

    D02 samples: 7
    Sham samples: 6
    D14 samples: 5
    Sham samples: 6
    Saved D02 comparison: ../data/processed/ec_DE_D02_vs_Sham.csv
    Saved D14 comparison: ../data/processed/ec_DE_D14_vs_Sham.csv
    Top D02-induced EC genes:



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
      <th>gene</th>
      <th>mean_D02</th>
      <th>mean_Sham</th>
      <th>effect_D02_vs_Sham</th>
      <th>log2fc_proxy_D02_vs_Sham</th>
      <th>pvalue_D02_vs_Sham</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tmsb4x</td>
      <td>8.151992</td>
      <td>2.224929</td>
      <td>5.927064</td>
      <td>1.873393</td>
      <td>0.001179</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Igfbp7</td>
      <td>6.496007</td>
      <td>1.453066</td>
      <td>5.042942</td>
      <td>2.160453</td>
      <td>0.008618</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ly6a</td>
      <td>6.753810</td>
      <td>2.183473</td>
      <td>4.570337</td>
      <td>1.629077</td>
      <td>0.008333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ctla2a</td>
      <td>4.136150</td>
      <td>0.477916</td>
      <td>3.658234</td>
      <td>3.113456</td>
      <td>0.005249</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sparc</td>
      <td>4.324855</td>
      <td>0.831155</td>
      <td>3.493701</td>
      <td>2.379462</td>
      <td>0.021555</td>
    </tr>
    <tr>
      <th>5</th>
      <td>mt-Rnr2</td>
      <td>22.573062</td>
      <td>19.394774</td>
      <td>3.178288</td>
      <td>0.218934</td>
      <td>0.656604</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Actb</td>
      <td>4.920596</td>
      <td>1.761817</td>
      <td>3.158780</td>
      <td>1.481769</td>
      <td>0.030027</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Rplp1</td>
      <td>3.151766</td>
      <td>0.759192</td>
      <td>2.392574</td>
      <td>2.053622</td>
      <td>0.006898</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Fth1</td>
      <td>3.664283</td>
      <td>1.307899</td>
      <td>2.356383</td>
      <td>1.486278</td>
      <td>0.002023</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Malat1</td>
      <td>5.356480</td>
      <td>3.006139</td>
      <td>2.350341</td>
      <td>0.833373</td>
      <td>0.136055</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Ly6c1</td>
      <td>5.853538</td>
      <td>3.717894</td>
      <td>2.135644</td>
      <td>0.654823</td>
      <td>0.123421</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Rps2</td>
      <td>2.213400</td>
      <td>0.269103</td>
      <td>1.944297</td>
      <td>3.040031</td>
      <td>0.022293</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Ifitm3</td>
      <td>2.267544</td>
      <td>0.545412</td>
      <td>1.722131</td>
      <td>2.055709</td>
      <td>0.004427</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Hsp90ab1</td>
      <td>2.911258</td>
      <td>1.262513</td>
      <td>1.648745</td>
      <td>1.205344</td>
      <td>0.068693</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Rpl41</td>
      <td>2.390194</td>
      <td>0.783314</td>
      <td>1.606880</td>
      <td>1.609464</td>
      <td>0.001745</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Rpl32</td>
      <td>1.939308</td>
      <td>0.416076</td>
      <td>1.523232</td>
      <td>2.220620</td>
      <td>0.001146</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Vim</td>
      <td>1.873383</td>
      <td>0.365958</td>
      <td>1.507425</td>
      <td>2.355893</td>
      <td>0.012264</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Calm1</td>
      <td>3.666962</td>
      <td>2.188762</td>
      <td>1.478201</td>
      <td>0.744470</td>
      <td>0.078412</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Rpl13</td>
      <td>2.010565</td>
      <td>0.641047</td>
      <td>1.369518</td>
      <td>1.649097</td>
      <td>0.014789</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Rps14</td>
      <td>2.465033</td>
      <td>1.108772</td>
      <td>1.356261</td>
      <td>1.152644</td>
      <td>0.005326</td>
    </tr>
  </tbody>
</table>
</div>


    Top D14-induced EC genes:



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
      <th>gene</th>
      <th>mean_D14</th>
      <th>mean_Sham</th>
      <th>effect_D14_vs_Sham</th>
      <th>log2fc_proxy_D14_vs_Sham</th>
      <th>pvalue_D14_vs_Sham</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bsg</td>
      <td>8.237078</td>
      <td>5.933134</td>
      <td>2.303945</td>
      <td>0.473338</td>
      <td>0.319651</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Malat1</td>
      <td>4.893720</td>
      <td>3.006139</td>
      <td>1.887581</td>
      <td>0.703020</td>
      <td>0.217132</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tmsb4x</td>
      <td>3.805056</td>
      <td>2.224929</td>
      <td>1.580128</td>
      <td>0.774158</td>
      <td>0.151303</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hbb-bs</td>
      <td>1.440309</td>
      <td>0.176782</td>
      <td>1.263527</td>
      <td>3.026329</td>
      <td>0.130952</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Itm2a</td>
      <td>2.889887</td>
      <td>1.651506</td>
      <td>1.238381</td>
      <td>0.807231</td>
      <td>0.235502</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Pltp</td>
      <td>2.359225</td>
      <td>1.130960</td>
      <td>1.228264</td>
      <td>1.060764</td>
      <td>0.141996</td>
    </tr>
    <tr>
      <th>6</th>
      <td>B2m</td>
      <td>2.226214</td>
      <td>1.035883</td>
      <td>1.190331</td>
      <td>1.103731</td>
      <td>0.090017</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cldn5</td>
      <td>2.905112</td>
      <td>1.813087</td>
      <td>1.092024</td>
      <td>0.680145</td>
      <td>0.279994</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Actb</td>
      <td>2.770267</td>
      <td>1.761817</td>
      <td>1.008450</td>
      <td>0.652961</td>
      <td>0.185234</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Hsp90ab1</td>
      <td>2.101411</td>
      <td>1.262513</td>
      <td>0.838899</td>
      <td>0.735060</td>
      <td>0.176462</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Jun</td>
      <td>1.489664</td>
      <td>0.671416</td>
      <td>0.818247</td>
      <td>1.149706</td>
      <td>0.078283</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Sparcl1</td>
      <td>1.935904</td>
      <td>1.127995</td>
      <td>0.807909</td>
      <td>0.779246</td>
      <td>0.084158</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Slco1a4</td>
      <td>1.997028</td>
      <td>1.372543</td>
      <td>0.624485</td>
      <td>0.541003</td>
      <td>0.217050</td>
    </tr>
    <tr>
      <th>13</th>
      <td>H2-D1</td>
      <td>1.292630</td>
      <td>0.674647</td>
      <td>0.617983</td>
      <td>0.938104</td>
      <td>0.058116</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Ifitm3</td>
      <td>1.133212</td>
      <td>0.545412</td>
      <td>0.587799</td>
      <td>1.054997</td>
      <td>0.070530</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Sparc</td>
      <td>1.402183</td>
      <td>0.831155</td>
      <td>0.571028</td>
      <td>0.754485</td>
      <td>0.207293</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Hba-a1</td>
      <td>0.643265</td>
      <td>0.075853</td>
      <td>0.567413</td>
      <td>3.084123</td>
      <td>0.130816</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Ly6e</td>
      <td>1.902943</td>
      <td>1.352209</td>
      <td>0.550733</td>
      <td>0.492913</td>
      <td>0.389992</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Fth1</td>
      <td>1.843739</td>
      <td>1.307899</td>
      <td>0.535839</td>
      <td>0.495382</td>
      <td>0.249927</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Nfkbia</td>
      <td>1.289718</td>
      <td>0.779835</td>
      <td>0.509883</td>
      <td>0.725814</td>
      <td>0.163031</td>
    </tr>
  </tbody>
</table>
</div>


## 6. Inspect known BBB receptor, transporter, and endothelial surface candidates

Before mining the full transcriptome for novel receptor candidates, a curated target panel is inspected. This panel includes known or plausible BBB transport receptors, endothelial receptors, barrier identity markers, efflux transporters, inflammatory adhesion molecules, and stroke-induced surface candidates.

This step serves as a biological sanity check. It asks whether known BBB/endothelial genes are detected in the EC pseudobulk table and whether they are preserved, induced, or suppressed after stroke. The resulting table is used for interpretation and context, while the final receptor shortlist is generated in later sections using broader surface-accessibility and induction criteria.


```python
# -------------------------
# 6.1 Define curated BBB/receptor target panel
# -------------------------

target_panel = {
    # known / plausible BBB shuttle or transport-relevant candidates
    "Tfrc": "known BBB receptor-mediated uptake candidate",
    "Lrp1": "endocytic BBB transport receptor",
    "Insr": "receptor-mediated transcytosis precedent",
    "Igf1r": "receptor-mediated transcytosis precedent",
    "Bsg": "membrane glycoprotein, vascular/inflammatory relevance",
    "Cd36": "scavenger receptor, vascular/endothelial relevance",
    "Scarb1": "scavenger receptor, lipid transport relevance",

    # BBB transporters / barrier identity
    "Slc2a1": "BBB glucose transporter and EC identity marker",
    "Slc7a5": "BBB amino-acid transporter",
    "Mfsd2a": "BBB lipid transporter and barrier identity marker",
    "Abcb1a": "BBB efflux transporter",
    "Abcg2": "BBB efflux transporter",
    "Slco1a4": "brain endothelial solute carrier candidate",

    # vascular / endothelial receptors
    "Kdr": "VEGF receptor, endothelial receptor",
    "Flt1": "VEGF receptor, endothelial receptor",
    "Tek": "Tie2 endothelial receptor",
    "Eng": "endothelial accessory receptor",

    # stroke/inflammatory endothelial activation
    "Icam1": "activated endothelial adhesion receptor",
    "Vcam1": "activated endothelial adhesion receptor",
    "Sele": "activated endothelial adhesion receptor",
    "Plvap": "vascular permeability-associated membrane protein",
    "Ly6a": "stroke-induced endothelial surface marker candidate",
    "Ly6c1": "stroke-associated vascular/immune interface marker",

    # membrane/cell-surface candidates from top results
    "Igfbp7": "secreted/perivascular EC-associated factor, target-context marker",
    "Itm2a": "membrane protein candidate",
}

print("Number of curated targets:", len(target_panel))
```

    Number of curated targets: 25



```python
# -------------------------
# 6.2 Define curated target-table helper
# -------------------------

def make_target_table(de_d02, de_d14, target_panel):
    d02_cols = [
        "gene",
        "mean_D02",
        "mean_Sham",
        "effect_D02_vs_Sham",
        "log2fc_proxy_D02_vs_Sham",
        "pvalue_D02_vs_Sham",
    ]

    d14_cols = [
        "gene",
        "mean_D14",
        "mean_Sham",
        "effect_D14_vs_Sham",
        "log2fc_proxy_D14_vs_Sham",
        "pvalue_D14_vs_Sham",
    ]

    d02 = de_d02[d02_cols].copy()
    d14 = de_d14[d14_cols].copy()

    merged = d02.merge(
        d14,
        on="gene",
        how="outer",
        suffixes=("_D02table", "_D14table"),
    )

    panel = pd.DataFrame({
        "gene": list(target_panel.keys()),
        "BBB_relevance": list(target_panel.values()),
    })

    out = panel.merge(merged, on="gene", how="left")

    def classify_response(row):
        d02_effect = row.get("effect_D02_vs_Sham", 0)
        d14_effect = row.get("effect_D14_vs_Sham", 0)

        if pd.isna(d02_effect):
            d02_effect = 0
        if pd.isna(d14_effect):
            d14_effect = 0

        if d02_effect > 0.5 and d14_effect > 0.5:
            return "persistent stroke-induced"
        if d02_effect > 0.5 and d14_effect <= 0.5:
            return "early D02-induced"
        if d02_effect <= 0.5 and d14_effect > 0.5:
            return "late D14-induced"
        if d02_effect < -0.5 or d14_effect < -0.5:
            return "stroke-suppressed"
        return "preserved / weakly changed"

    out["stroke_response"] = out.apply(classify_response, axis=1)

    out["cell_source_evidence"] = (
        "brain EC pseudobulk; EC identity verified by canonical endothelial and BBB markers"
    )

    target_constraints = {
        "Tfrc": "bind extracellular receptor region; avoid disrupting iron transport",
        "Lrp1": "exploit endocytic uptake; avoid broad off-target uptake",
        "Insr": "avoid metabolic agonism while preserving transport",
        "Igf1r": "avoid growth-factor-like signalling activation",
        "Bsg": "target extracellular domain; check inflammatory/off-target expression",
        "Cd36": "avoid broad scavenger-receptor off-target effects",
        "Scarb1": "balance lipid-transport biology and endothelial specificity",
        "Slc2a1": "use mainly as BBB identity marker, not primary peptide target",
        "Slc7a5": "transporter biology relevant; peptide binding may be difficult",
        "Mfsd2a": "BBB identity marker; targetability requires caution",
        "Abcb1a": "efflux transporter; not ideal as shuttle target",
        "Abcg2": "efflux transporter; not ideal as shuttle target",
        "Slco1a4": "evaluate extracellular accessibility and species translation",
        "Kdr": "avoid angiogenic signalling activation",
        "Flt1": "avoid VEGF-pathway perturbation",
        "Tek": "avoid Tie2 pathway activation unless intended",
        "Eng": "evaluate vascular specificity and signalling risk",
        "Icam1": "stroke-activated targeting; avoid systemic immune adhesion effects",
        "Vcam1": "stroke-activated targeting; avoid broad inflammatory endothelium",
        "Sele": "stroke-activated targeting; likely inflammation-state specific",
        "Plvap": "vascular permeability marker; assess BBB specificity",
        "Ly6a": "strong mouse stroke EC signal; check human translational relevance",
        "Ly6c1": "stroke-associated but immune/vascular specificity must be checked",
        "Igfbp7": "use as EC/stroke context marker more than direct shuttle receptor",
        "Itm2a": "evaluate membrane topology and extracellular accessibility",
    }

    out["target_constraint"] = out["gene"].map(target_constraints)

    response_order = {
        "early D02-induced": 0,
        "persistent stroke-induced": 1,
        "late D14-induced": 2,
        "preserved / weakly changed": 3,
        "stroke-suppressed": 4,
    }

    out["response_order"] = out["stroke_response"].map(response_order)

    out = out.sort_values(
        ["response_order", "effect_D02_vs_Sham"],
        ascending=[True, False],
    ).drop(columns="response_order")

    return out.reset_index(drop=True)
```


```python
# -------------------------
# 6.3 Build curated target table
# -------------------------

target_table = make_target_table(
    de_d02=de_d02,
    de_d14=de_d14,
    target_panel=target_panel,
)

display(target_table)
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
      <th>gene</th>
      <th>BBB_relevance</th>
      <th>mean_D02</th>
      <th>mean_Sham_D02table</th>
      <th>effect_D02_vs_Sham</th>
      <th>log2fc_proxy_D02_vs_Sham</th>
      <th>pvalue_D02_vs_Sham</th>
      <th>mean_D14</th>
      <th>mean_Sham_D14table</th>
      <th>effect_D14_vs_Sham</th>
      <th>log2fc_proxy_D14_vs_Sham</th>
      <th>pvalue_D14_vs_Sham</th>
      <th>stroke_response</th>
      <th>cell_source_evidence</th>
      <th>target_constraint</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Igfbp7</td>
      <td>secreted/perivascular EC-associated factor, ta...</td>
      <td>6.496007</td>
      <td>1.453066e+00</td>
      <td>5.042942</td>
      <td>2.160453</td>
      <td>0.008618</td>
      <td>1.609349</td>
      <td>1.453066e+00</td>
      <td>0.156283</td>
      <td>0.147377</td>
      <td>0.745711</td>
      <td>early D02-induced</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>use as EC/stroke context marker more than dire...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ly6a</td>
      <td>stroke-induced endothelial surface marker cand...</td>
      <td>6.753810</td>
      <td>2.183473e+00</td>
      <td>4.570337</td>
      <td>1.629077</td>
      <td>0.008333</td>
      <td>2.595743</td>
      <td>2.183473e+00</td>
      <td>0.412271</td>
      <td>0.249523</td>
      <td>0.602446</td>
      <td>early D02-induced</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>strong mouse stroke EC signal; check human tra...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ly6c1</td>
      <td>stroke-associated vascular/immune interface ma...</td>
      <td>5.853538</td>
      <td>3.717894e+00</td>
      <td>2.135644</td>
      <td>0.654823</td>
      <td>0.123421</td>
      <td>3.744299</td>
      <td>3.717894e+00</td>
      <td>0.026405</td>
      <td>0.010210</td>
      <td>0.981860</td>
      <td>early D02-induced</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>stroke-associated but immune/vascular specific...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Itm2a</td>
      <td>membrane protein candidate</td>
      <td>1.907739</td>
      <td>1.651506e+00</td>
      <td>0.256233</td>
      <td>0.208081</td>
      <td>0.641626</td>
      <td>2.889887</td>
      <td>1.651506e+00</td>
      <td>1.238381</td>
      <td>0.807231</td>
      <td>0.235502</td>
      <td>late D14-induced</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>evaluate membrane topology and extracellular a...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Slco1a4</td>
      <td>brain endothelial solute carrier candidate</td>
      <td>1.008533</td>
      <td>1.372543e+00</td>
      <td>-0.364010</td>
      <td>-0.444593</td>
      <td>0.150921</td>
      <td>1.997028</td>
      <td>1.372543e+00</td>
      <td>0.624485</td>
      <td>0.541003</td>
      <td>0.217050</td>
      <td>late D14-induced</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>evaluate extracellular accessibility and speci...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bsg</td>
      <td>membrane glycoprotein, vascular/inflammatory r...</td>
      <td>3.999826</td>
      <td>5.933134e+00</td>
      <td>-1.933308</td>
      <td>-0.568857</td>
      <td>0.159024</td>
      <td>8.237078</td>
      <td>5.933134e+00</td>
      <td>2.303945</td>
      <td>0.473338</td>
      <td>0.319651</td>
      <td>late D14-induced</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>target extracellular domain; check inflammator...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Eng</td>
      <td>endothelial accessory receptor</td>
      <td>0.588987</td>
      <td>2.291584e-01</td>
      <td>0.359828</td>
      <td>1.361886</td>
      <td>0.020719</td>
      <td>0.317102</td>
      <td>2.291584e-01</td>
      <td>0.087944</td>
      <td>0.468600</td>
      <td>0.349628</td>
      <td>preserved / weakly changed</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>evaluate vascular specificity and signalling risk</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Slc2a1</td>
      <td>BBB glucose transporter and EC identity marker</td>
      <td>0.941885</td>
      <td>6.223895e-01</td>
      <td>0.319495</td>
      <td>0.597732</td>
      <td>0.207402</td>
      <td>0.611308</td>
      <td>6.223895e-01</td>
      <td>-0.011081</td>
      <td>-0.025918</td>
      <td>0.956956</td>
      <td>preserved / weakly changed</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>use mainly as BBB identity marker, not primary...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Kdr</td>
      <td>VEGF receptor, endothelial receptor</td>
      <td>0.330704</td>
      <td>2.010880e-01</td>
      <td>0.129616</td>
      <td>0.717710</td>
      <td>0.183229</td>
      <td>0.334654</td>
      <td>2.010880e-01</td>
      <td>0.133566</td>
      <td>0.734838</td>
      <td>0.143415</td>
      <td>preserved / weakly changed</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>avoid angiogenic signalling activation</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Abcg2</td>
      <td>BBB efflux transporter</td>
      <td>0.517589</td>
      <td>4.313633e-01</td>
      <td>0.086226</td>
      <td>0.262903</td>
      <td>0.230239</td>
      <td>0.545616</td>
      <td>4.313633e-01</td>
      <td>0.114253</td>
      <td>0.338981</td>
      <td>0.379466</td>
      <td>preserved / weakly changed</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>efflux transporter; not ideal as shuttle target</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Scarb1</td>
      <td>scavenger receptor, lipid transport relevance</td>
      <td>0.135296</td>
      <td>7.833059e-02</td>
      <td>0.056965</td>
      <td>0.788461</td>
      <td>0.133127</td>
      <td>0.139267</td>
      <td>7.833059e-02</td>
      <td>0.060936</td>
      <td>0.830195</td>
      <td>0.170679</td>
      <td>preserved / weakly changed</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>balance lipid-transport biology and endothelia...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Icam1</td>
      <td>activated endothelial adhesion receptor</td>
      <td>0.068126</td>
      <td>2.520701e-02</td>
      <td>0.042919</td>
      <td>1.434335</td>
      <td>0.001767</td>
      <td>0.059311</td>
      <td>2.520701e-02</td>
      <td>0.034104</td>
      <td>1.234446</td>
      <td>0.171641</td>
      <td>preserved / weakly changed</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>stroke-activated targeting; avoid systemic imm...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Tek</td>
      <td>Tie2 endothelial receptor</td>
      <td>0.261158</td>
      <td>2.342537e-01</td>
      <td>0.026905</td>
      <td>0.156853</td>
      <td>0.680414</td>
      <td>0.355795</td>
      <td>2.342537e-01</td>
      <td>0.121542</td>
      <td>0.602973</td>
      <td>0.169949</td>
      <td>preserved / weakly changed</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>avoid Tie2 pathway activation unless intended</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Sele</td>
      <td>activated endothelial adhesion receptor</td>
      <td>0.032491</td>
      <td>8.453521e-03</td>
      <td>0.024038</td>
      <td>1.942297</td>
      <td>0.055451</td>
      <td>0.010223</td>
      <td>8.453521e-03</td>
      <td>0.001770</td>
      <td>0.274224</td>
      <td>0.858942</td>
      <td>preserved / weakly changed</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>stroke-activated targeting; likely inflammatio...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Insr</td>
      <td>receptor-mediated transcytosis precedent</td>
      <td>0.055316</td>
      <td>3.734711e-02</td>
      <td>0.017968</td>
      <td>0.566675</td>
      <td>0.186223</td>
      <td>0.036849</td>
      <td>3.734711e-02</td>
      <td>-0.000498</td>
      <td>-0.019382</td>
      <td>0.972820</td>
      <td>preserved / weakly changed</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>avoid metabolic agonism while preserving trans...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Plvap</td>
      <td>vascular permeability-associated membrane protein</td>
      <td>0.051982</td>
      <td>3.887528e-02</td>
      <td>0.013107</td>
      <td>0.419149</td>
      <td>0.561476</td>
      <td>0.053137</td>
      <td>3.887528e-02</td>
      <td>0.014262</td>
      <td>0.450864</td>
      <td>0.717907</td>
      <td>preserved / weakly changed</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>vascular permeability marker; assess BBB speci...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Lrp1</td>
      <td>endocytic BBB transport receptor</td>
      <td>0.000904</td>
      <td>8.086688e-08</td>
      <td>0.000904</td>
      <td>9.709795</td>
      <td>0.355958</td>
      <td>0.001360</td>
      <td>8.086688e-08</td>
      <td>0.001360</td>
      <td>10.297906</td>
      <td>0.373927</td>
      <td>preserved / weakly changed</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>exploit endocytic uptake; avoid broad off-targ...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Cd36</td>
      <td>scavenger receptor, vascular/endothelial relev...</td>
      <td>0.000922</td>
      <td>1.106649e-03</td>
      <td>-0.000184</td>
      <td>-0.262687</td>
      <td>0.889688</td>
      <td>0.001211</td>
      <td>1.106649e-03</td>
      <td>0.000105</td>
      <td>0.130406</td>
      <td>0.939166</td>
      <td>preserved / weakly changed</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>avoid broad scavenger-receptor off-target effects</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Vcam1</td>
      <td>activated endothelial adhesion receptor</td>
      <td>0.084779</td>
      <td>8.979383e-02</td>
      <td>-0.005015</td>
      <td>-0.082908</td>
      <td>0.883378</td>
      <td>0.113664</td>
      <td>8.979383e-02</td>
      <td>0.023871</td>
      <td>0.340088</td>
      <td>0.612119</td>
      <td>preserved / weakly changed</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>stroke-activated targeting; avoid broad inflam...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Mfsd2a</td>
      <td>BBB lipid transporter and barrier identity marker</td>
      <td>0.080656</td>
      <td>9.987187e-02</td>
      <td>-0.019216</td>
      <td>-0.308299</td>
      <td>0.555729</td>
      <td>0.158647</td>
      <td>9.987187e-02</td>
      <td>0.058775</td>
      <td>0.667666</td>
      <td>0.229024</td>
      <td>preserved / weakly changed</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>BBB identity marker; targetability requires ca...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Slc7a5</td>
      <td>BBB amino-acid transporter</td>
      <td>0.073600</td>
      <td>9.294074e-02</td>
      <td>-0.019340</td>
      <td>-0.336595</td>
      <td>0.602395</td>
      <td>0.136156</td>
      <td>9.294074e-02</td>
      <td>0.043215</td>
      <td>0.550869</td>
      <td>0.320153</td>
      <td>preserved / weakly changed</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>transporter biology relevant; peptide binding ...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Tfrc</td>
      <td>known BBB receptor-mediated uptake candidate</td>
      <td>0.147251</td>
      <td>2.224964e-01</td>
      <td>-0.075245</td>
      <td>-0.595499</td>
      <td>0.097876</td>
      <td>0.344365</td>
      <td>2.224964e-01</td>
      <td>0.121869</td>
      <td>0.630156</td>
      <td>0.240256</td>
      <td>preserved / weakly changed</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>bind extracellular receptor region; avoid disr...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Igf1r</td>
      <td>receptor-mediated transcytosis precedent</td>
      <td>0.287879</td>
      <td>4.940599e-01</td>
      <td>-0.206181</td>
      <td>-0.779224</td>
      <td>0.030673</td>
      <td>0.476834</td>
      <td>4.940599e-01</td>
      <td>-0.017226</td>
      <td>-0.051199</td>
      <td>0.870940</td>
      <td>preserved / weakly changed</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>avoid growth-factor-like signalling activation</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Abcb1a</td>
      <td>BBB efflux transporter</td>
      <td>0.700394</td>
      <td>1.054301e+00</td>
      <td>-0.353907</td>
      <td>-0.590048</td>
      <td>0.010133</td>
      <td>1.268677</td>
      <td>1.054301e+00</td>
      <td>0.214376</td>
      <td>0.267038</td>
      <td>0.417386</td>
      <td>preserved / weakly changed</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>efflux transporter; not ideal as shuttle target</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Flt1</td>
      <td>VEGF receptor, endothelial receptor</td>
      <td>1.591764</td>
      <td>2.061095e+00</td>
      <td>-0.469331</td>
      <td>-0.372784</td>
      <td>0.243733</td>
      <td>2.360281</td>
      <td>2.061095e+00</td>
      <td>0.299187</td>
      <td>0.195548</td>
      <td>0.558748</td>
      <td>preserved / weakly changed</td>
      <td>brain EC pseudobulk; EC identity verified by c...</td>
      <td>avoid VEGF-pathway perturbation</td>
    </tr>
  </tbody>
</table>
</div>



```python
# -------------------------
# 6.4 Save curated target table
# -------------------------

target_table_path = out_dir / "ec_receptor_transporter_target_table.csv"

target_table.to_csv(target_table_path, index=False)

print("Saved curated receptor/transporter target table:", target_table_path)
```

    Saved curated receptor/transporter target table: ../data/processed/ec_receptor_transporter_target_table.csv


## 7. Identify data-driven stroke-responsive EC candidates

After inspecting a curated BBB/receptor panel, a broader data-driven screen is performed across the EC pseudobulk comparison results. The aim is to identify genes that are strongly induced after stroke and expressed in endothelial cells, before applying surface-accessibility and UniProt annotation filters in later sections.

This step removes obvious housekeeping or non-target-like genes, such as ribosomal, mitochondrial, haemoglobin, predicted, and microRNA genes. The remaining genes are filtered by minimum case expression and stroke-induced effect size.


```python
# -------------------------
# 7.1 Define data-driven candidate filter
# -------------------------

import re

def filter_data_driven_candidates(
    de,
    case,
    min_mean_case=0.05,
    min_effect=0.25,
):
    effect_col = f"effect_{case}_vs_Sham"
    mean_col = f"mean_{case}"
    logfc_col = f"log2fc_proxy_{case}_vs_Sham"

    df = de.copy()

    # Remove obvious non-target / housekeeping-style genes
    bad_patterns = [
        r"^Rpl",   # ribosomal large subunit
        r"^Rps",   # ribosomal small subunit
        r"^mt-",   # mitochondrial
        r"^Hba",   # haemoglobin alpha
        r"^Hbb",   # haemoglobin beta
        r"^Gm\d+", # predicted genes
        r"^Mir",   # microRNAs
    ]

    bad_exact = {
        "Malat1",
        "Actb",
        "Gapdh",
        "Tmsb4x",
    }

    pattern = "|".join(bad_patterns)

    df = df[
        ~df["gene"].str.contains(pattern, regex=True, na=False)
        & ~df["gene"].isin(bad_exact)
    ].copy()

    # Keep genes with meaningful expression and stroke induction
    df = df[
        (df[mean_col] >= min_mean_case)
        & (df[effect_col] >= min_effect)
    ].copy()

    # Sort by absolute effect first, then fold-change proxy
    df = df.sort_values(
        [effect_col, logfc_col],
        ascending=False,
    ).reset_index(drop=True)

    return df
```


```python
# -------------------------
# 7.2 Apply data-driven filtering to D02 and D14
# -------------------------

d02_candidates = filter_data_driven_candidates(
    de=de_d02,
    case="D02",
)

d14_candidates = filter_data_driven_candidates(
    de=de_d14,
    case="D14",
)

print("D02 candidates:", d02_candidates.shape)
print("D14 candidates:", d14_candidates.shape)

print("Top data-driven D02 EC candidates:")
display(d02_candidates.head(50))

print("Top data-driven D14 EC candidates:")
display(d14_candidates.head(50))
```

    D02 candidates: (150, 6)
    D14 candidates: (43, 6)
    Top data-driven D02 EC candidates:



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
      <th>gene</th>
      <th>mean_D02</th>
      <th>mean_Sham</th>
      <th>effect_D02_vs_Sham</th>
      <th>log2fc_proxy_D02_vs_Sham</th>
      <th>pvalue_D02_vs_Sham</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Igfbp7</td>
      <td>6.496007</td>
      <td>1.453066</td>
      <td>5.042942</td>
      <td>2.160453</td>
      <td>0.008618</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ly6a</td>
      <td>6.753810</td>
      <td>2.183473</td>
      <td>4.570337</td>
      <td>1.629077</td>
      <td>0.008333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ctla2a</td>
      <td>4.136150</td>
      <td>0.477916</td>
      <td>3.658234</td>
      <td>3.113456</td>
      <td>0.005249</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sparc</td>
      <td>4.324855</td>
      <td>0.831155</td>
      <td>3.493701</td>
      <td>2.379462</td>
      <td>0.021555</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fth1</td>
      <td>3.664283</td>
      <td>1.307899</td>
      <td>2.356383</td>
      <td>1.486278</td>
      <td>0.002023</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ly6c1</td>
      <td>5.853538</td>
      <td>3.717894</td>
      <td>2.135644</td>
      <td>0.654823</td>
      <td>0.123421</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ifitm3</td>
      <td>2.267544</td>
      <td>0.545412</td>
      <td>1.722131</td>
      <td>2.055709</td>
      <td>0.004427</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Hsp90ab1</td>
      <td>2.911258</td>
      <td>1.262513</td>
      <td>1.648745</td>
      <td>1.205344</td>
      <td>0.068693</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Vim</td>
      <td>1.873383</td>
      <td>0.365958</td>
      <td>1.507425</td>
      <td>2.355893</td>
      <td>0.012264</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Calm1</td>
      <td>3.666962</td>
      <td>2.188762</td>
      <td>1.478201</td>
      <td>0.744470</td>
      <td>0.078412</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Ybx1</td>
      <td>1.875088</td>
      <td>0.663966</td>
      <td>1.211122</td>
      <td>1.497776</td>
      <td>0.025729</td>
    </tr>
    <tr>
      <th>11</th>
      <td>B2m</td>
      <td>2.225297</td>
      <td>1.035883</td>
      <td>1.189415</td>
      <td>1.103137</td>
      <td>0.021049</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Tmem252</td>
      <td>1.471563</td>
      <td>0.291582</td>
      <td>1.179982</td>
      <td>2.335374</td>
      <td>0.000901</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Eef1a1</td>
      <td>1.717794</td>
      <td>0.617760</td>
      <td>1.100034</td>
      <td>1.475437</td>
      <td>0.022715</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Tm4sf1</td>
      <td>1.421943</td>
      <td>0.355311</td>
      <td>1.066632</td>
      <td>2.000707</td>
      <td>0.003731</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Mgp</td>
      <td>1.570391</td>
      <td>0.505085</td>
      <td>1.065305</td>
      <td>1.636522</td>
      <td>0.002117</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Ly6e</td>
      <td>2.406169</td>
      <td>1.352209</td>
      <td>1.053960</td>
      <td>0.831419</td>
      <td>0.108692</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Ftl1</td>
      <td>1.462249</td>
      <td>0.450939</td>
      <td>1.011310</td>
      <td>1.697183</td>
      <td>0.011467</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Hsp90b1</td>
      <td>1.335427</td>
      <td>0.439727</td>
      <td>0.895700</td>
      <td>1.602618</td>
      <td>0.055815</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Egfl7</td>
      <td>1.513354</td>
      <td>0.673415</td>
      <td>0.839940</td>
      <td>1.168182</td>
      <td>0.016951</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Ppia</td>
      <td>1.232202</td>
      <td>0.392335</td>
      <td>0.839867</td>
      <td>1.651077</td>
      <td>0.038470</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Fkbp1a</td>
      <td>0.977528</td>
      <td>0.140090</td>
      <td>0.837438</td>
      <td>2.802778</td>
      <td>0.001469</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Pabpc1</td>
      <td>1.100891</td>
      <td>0.299823</td>
      <td>0.801068</td>
      <td>1.876486</td>
      <td>0.139083</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Tpm3</td>
      <td>1.239211</td>
      <td>0.491503</td>
      <td>0.747708</td>
      <td>1.334148</td>
      <td>0.017033</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Anxa2</td>
      <td>0.952120</td>
      <td>0.223352</td>
      <td>0.728768</td>
      <td>2.091821</td>
      <td>0.004561</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Cd200</td>
      <td>1.035134</td>
      <td>0.317887</td>
      <td>0.717247</td>
      <td>1.703227</td>
      <td>0.000703</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Hspa5</td>
      <td>0.918031</td>
      <td>0.220700</td>
      <td>0.697331</td>
      <td>2.056454</td>
      <td>0.012144</td>
    </tr>
    <tr>
      <th>27</th>
      <td>S100a11</td>
      <td>0.825515</td>
      <td>0.137775</td>
      <td>0.687740</td>
      <td>2.582976</td>
      <td>0.006845</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Glul</td>
      <td>1.146988</td>
      <td>0.462845</td>
      <td>0.684143</td>
      <td>1.309247</td>
      <td>0.034632</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Edn1</td>
      <td>0.906098</td>
      <td>0.224776</td>
      <td>0.681322</td>
      <td>2.011173</td>
      <td>0.001177</td>
    </tr>
    <tr>
      <th>30</th>
      <td>H2-D1</td>
      <td>1.326389</td>
      <td>0.674647</td>
      <td>0.651742</td>
      <td>0.975298</td>
      <td>0.025642</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Lrg1</td>
      <td>0.704301</td>
      <td>0.065713</td>
      <td>0.638588</td>
      <td>3.421930</td>
      <td>0.001453</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Ece1</td>
      <td>0.981473</td>
      <td>0.370838</td>
      <td>0.610635</td>
      <td>1.404157</td>
      <td>0.005822</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Calr</td>
      <td>0.833457</td>
      <td>0.237697</td>
      <td>0.595760</td>
      <td>1.809977</td>
      <td>0.059921</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Itgb1</td>
      <td>0.876678</td>
      <td>0.295314</td>
      <td>0.581364</td>
      <td>1.569795</td>
      <td>0.016298</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Gpx1</td>
      <td>0.882360</td>
      <td>0.313602</td>
      <td>0.568759</td>
      <td>1.492432</td>
      <td>0.006150</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Esam</td>
      <td>0.978323</td>
      <td>0.419158</td>
      <td>0.559165</td>
      <td>1.222813</td>
      <td>0.029184</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Ecscr</td>
      <td>0.639574</td>
      <td>0.083505</td>
      <td>0.556069</td>
      <td>2.937155</td>
      <td>0.000125</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Tmsb10</td>
      <td>0.991232</td>
      <td>0.435698</td>
      <td>0.555534</td>
      <td>1.185892</td>
      <td>0.005459</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Ubb</td>
      <td>1.287912</td>
      <td>0.739424</td>
      <td>0.548489</td>
      <td>0.800561</td>
      <td>0.043463</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Myl12a</td>
      <td>1.021412</td>
      <td>0.477085</td>
      <td>0.544328</td>
      <td>1.098247</td>
      <td>0.010147</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Hspa8</td>
      <td>0.975532</td>
      <td>0.439680</td>
      <td>0.535852</td>
      <td>1.149733</td>
      <td>0.017574</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Mt2</td>
      <td>0.702129</td>
      <td>0.169450</td>
      <td>0.532679</td>
      <td>2.050874</td>
      <td>0.130675</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Rack1</td>
      <td>0.753514</td>
      <td>0.221606</td>
      <td>0.531909</td>
      <td>1.765634</td>
      <td>0.020701</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Ndufa4</td>
      <td>0.780018</td>
      <td>0.256117</td>
      <td>0.523901</td>
      <td>1.606699</td>
      <td>0.003174</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Msn</td>
      <td>0.719441</td>
      <td>0.195917</td>
      <td>0.523524</td>
      <td>1.876627</td>
      <td>0.011529</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Prdx1</td>
      <td>0.754655</td>
      <td>0.240419</td>
      <td>0.514237</td>
      <td>1.650264</td>
      <td>0.012344</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Tpm4</td>
      <td>0.858688</td>
      <td>0.362326</td>
      <td>0.496362</td>
      <td>1.244843</td>
      <td>0.003986</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Ncl</td>
      <td>0.698063</td>
      <td>0.206711</td>
      <td>0.491352</td>
      <td>1.755735</td>
      <td>0.036913</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Arpc2</td>
      <td>0.915514</td>
      <td>0.425859</td>
      <td>0.489655</td>
      <td>1.104205</td>
      <td>0.020514</td>
    </tr>
  </tbody>
</table>
</div>


    Top data-driven D14 EC candidates:



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
      <th>gene</th>
      <th>mean_D14</th>
      <th>mean_Sham</th>
      <th>effect_D14_vs_Sham</th>
      <th>log2fc_proxy_D14_vs_Sham</th>
      <th>pvalue_D14_vs_Sham</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bsg</td>
      <td>8.237078</td>
      <td>5.933134</td>
      <td>2.303945</td>
      <td>0.473338</td>
      <td>0.319651</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Itm2a</td>
      <td>2.889887</td>
      <td>1.651506</td>
      <td>1.238381</td>
      <td>0.807231</td>
      <td>0.235502</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pltp</td>
      <td>2.359225</td>
      <td>1.130960</td>
      <td>1.228264</td>
      <td>1.060764</td>
      <td>0.141996</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B2m</td>
      <td>2.226214</td>
      <td>1.035883</td>
      <td>1.190331</td>
      <td>1.103731</td>
      <td>0.090017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cldn5</td>
      <td>2.905112</td>
      <td>1.813087</td>
      <td>1.092024</td>
      <td>0.680145</td>
      <td>0.279994</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Hsp90ab1</td>
      <td>2.101411</td>
      <td>1.262513</td>
      <td>0.838899</td>
      <td>0.735060</td>
      <td>0.176462</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Jun</td>
      <td>1.489664</td>
      <td>0.671416</td>
      <td>0.818247</td>
      <td>1.149706</td>
      <td>0.078283</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sparcl1</td>
      <td>1.935904</td>
      <td>1.127995</td>
      <td>0.807909</td>
      <td>0.779246</td>
      <td>0.084158</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Slco1a4</td>
      <td>1.997028</td>
      <td>1.372543</td>
      <td>0.624485</td>
      <td>0.541003</td>
      <td>0.217050</td>
    </tr>
    <tr>
      <th>9</th>
      <td>H2-D1</td>
      <td>1.292630</td>
      <td>0.674647</td>
      <td>0.617983</td>
      <td>0.938104</td>
      <td>0.058116</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Ifitm3</td>
      <td>1.133212</td>
      <td>0.545412</td>
      <td>0.587799</td>
      <td>1.054997</td>
      <td>0.070530</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Sparc</td>
      <td>1.402183</td>
      <td>0.831155</td>
      <td>0.571028</td>
      <td>0.754485</td>
      <td>0.207293</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Ly6e</td>
      <td>1.902943</td>
      <td>1.352209</td>
      <td>0.550733</td>
      <td>0.492913</td>
      <td>0.389992</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Fth1</td>
      <td>1.843739</td>
      <td>1.307899</td>
      <td>0.535839</td>
      <td>0.495382</td>
      <td>0.249927</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Nfkbia</td>
      <td>1.289718</td>
      <td>0.779835</td>
      <td>0.509883</td>
      <td>0.725814</td>
      <td>0.163031</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Hspa1a</td>
      <td>0.692744</td>
      <td>0.213477</td>
      <td>0.479267</td>
      <td>1.698235</td>
      <td>0.102154</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Fos</td>
      <td>0.836271</td>
      <td>0.386049</td>
      <td>0.450222</td>
      <td>1.115186</td>
      <td>0.128877</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Tsc22d1</td>
      <td>1.296371</td>
      <td>0.867824</td>
      <td>0.428547</td>
      <td>0.579003</td>
      <td>0.212901</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Egfl7</td>
      <td>1.088847</td>
      <td>0.673415</td>
      <td>0.415433</td>
      <td>0.693234</td>
      <td>0.164772</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Ly6a</td>
      <td>2.595743</td>
      <td>2.183473</td>
      <td>0.412271</td>
      <td>0.249523</td>
      <td>0.602446</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Ptn</td>
      <td>1.160549</td>
      <td>0.753595</td>
      <td>0.406954</td>
      <td>0.622946</td>
      <td>0.219380</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Itm2b</td>
      <td>1.228511</td>
      <td>0.823283</td>
      <td>0.405228</td>
      <td>0.577450</td>
      <td>0.241448</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Spock2</td>
      <td>1.751931</td>
      <td>1.353703</td>
      <td>0.398228</td>
      <td>0.372035</td>
      <td>0.350452</td>
    </tr>
    <tr>
      <th>23</th>
      <td>H2-K1</td>
      <td>0.763320</td>
      <td>0.383286</td>
      <td>0.380033</td>
      <td>0.993863</td>
      <td>0.048221</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Hspb1</td>
      <td>0.867264</td>
      <td>0.493256</td>
      <td>0.374008</td>
      <td>0.814133</td>
      <td>0.302167</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Klf2</td>
      <td>1.134736</td>
      <td>0.763502</td>
      <td>0.371233</td>
      <td>0.571651</td>
      <td>0.297269</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Cst3</td>
      <td>0.932105</td>
      <td>0.561985</td>
      <td>0.370120</td>
      <td>0.729960</td>
      <td>0.243445</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Gnas</td>
      <td>1.083172</td>
      <td>0.717016</td>
      <td>0.366156</td>
      <td>0.595184</td>
      <td>0.316541</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Calm1</td>
      <td>2.532381</td>
      <td>2.188762</td>
      <td>0.343620</td>
      <td>0.210380</td>
      <td>0.434480</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Car4</td>
      <td>0.765598</td>
      <td>0.439015</td>
      <td>0.326583</td>
      <td>0.802315</td>
      <td>0.157735</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Dusp1</td>
      <td>0.542749</td>
      <td>0.232273</td>
      <td>0.310475</td>
      <td>1.224456</td>
      <td>0.181954</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Slco1c1</td>
      <td>0.965487</td>
      <td>0.664650</td>
      <td>0.300837</td>
      <td>0.538662</td>
      <td>0.305896</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Flt1</td>
      <td>2.360281</td>
      <td>2.061095</td>
      <td>0.299187</td>
      <td>0.195548</td>
      <td>0.558748</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Junb</td>
      <td>0.496032</td>
      <td>0.199388</td>
      <td>0.296644</td>
      <td>1.314852</td>
      <td>0.129439</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Esam</td>
      <td>0.708916</td>
      <td>0.419158</td>
      <td>0.289758</td>
      <td>0.758119</td>
      <td>0.164360</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Zfp36</td>
      <td>0.419832</td>
      <td>0.136515</td>
      <td>0.283317</td>
      <td>1.620743</td>
      <td>0.150368</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Arl4a</td>
      <td>0.677190</td>
      <td>0.401541</td>
      <td>0.275649</td>
      <td>0.754011</td>
      <td>0.122734</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Ablim1</td>
      <td>0.816497</td>
      <td>0.543430</td>
      <td>0.273068</td>
      <td>0.587354</td>
      <td>0.185764</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Hspa8</td>
      <td>0.704441</td>
      <td>0.439680</td>
      <td>0.264761</td>
      <td>0.680023</td>
      <td>0.244612</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Apod</td>
      <td>0.376634</td>
      <td>0.113396</td>
      <td>0.263238</td>
      <td>1.731786</td>
      <td>0.023250</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Hnrnpa2b1</td>
      <td>0.688884</td>
      <td>0.430384</td>
      <td>0.258500</td>
      <td>0.678635</td>
      <td>0.237022</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Ctla2a</td>
      <td>0.730297</td>
      <td>0.477916</td>
      <td>0.252381</td>
      <td>0.611724</td>
      <td>0.185777</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Ubb</td>
      <td>0.989485</td>
      <td>0.739424</td>
      <td>0.250061</td>
      <td>0.420276</td>
      <td>0.410225</td>
    </tr>
  </tbody>
</table>
</div>



```python
# -------------------------
# 7.3 Save data-driven candidate tables
# -------------------------

d02_candidates_path = out_dir / "data_driven_D02_EC_candidates.csv"
d14_candidates_path = out_dir / "data_driven_D14_EC_candidates.csv"

d02_candidates.head(200).to_csv(d02_candidates_path, index=False)
d14_candidates.head(200).to_csv(d14_candidates_path, index=False)

print("Saved D02 data-driven candidates:", d02_candidates_path)
print("Saved D14 data-driven candidates:", d14_candidates_path)
```

    Saved D02 data-driven candidates: ../data/processed/data_driven_D02_EC_candidates.csv
    Saved D14 data-driven candidates: ../data/processed/data_driven_D14_EC_candidates.csv


## 8. Annotate data-driven candidates with UniProt surface-accessibility information

The data-driven D02 and D14 candidate lists include many genes that are stroke-responsive in endothelial cells but are not necessarily suitable peptide-shuttle targets. For peptide targeting, the protein should ideally be cell-surface accessible, membrane-associated with an extracellular domain, or otherwise biologically relevant to the extracellular endothelial environment.

This section queries UniProt for the top data-driven candidates and extracts canonical protein annotations, subcellular localisation, and keywords. These annotations are then converted into broad accessibility classes:

- `surface_accessible`: strongest class for direct peptide targeting
- `membrane_unknown_location`: possible target, requires topology/domain inspection
- `secreted_or_extracellular`: useful biology or context marker, but usually not a direct receptor
- `intracellular_or_organelle`: rejected for extracellular peptide targeting
- `unknown`: requires manual inspection


```python
# -------------------------
# 8.1 Define UniProt query and annotation helpers
# -------------------------

import time
import requests

def safe_text(x):
    if x is None or pd.isna(x):
        return ""
    return str(x)


def query_uniprot_mouse_gene(gene):
    url = "https://rest.uniprot.org/uniprotkb/search"
    fields = "accession,gene_names,protein_name,cc_subcellular_location,keyword"

    queries = [
        f"gene_exact:{gene} AND organism_id:10090 AND reviewed:true",
        f"gene_exact:{gene} AND organism_id:10090",
    ]

    for query in queries:
        r = requests.get(
            url,
            params={
                "query": query,
                "format": "json",
                "size": 1,
                "fields": fields,
            },
            timeout=30,
        )
        r.raise_for_status()
        results = r.json().get("results", [])

        if results:
            entry = results[0]
            break
    else:
        return {
            "gene": gene,
            "uniprot_accession": None,
            "protein_name": None,
            "subcellular_location_text": None,
            "keywords_text": None,
            "uniprot_status": "not_found",
        }

    protein_desc = entry.get("proteinDescription", {})
    protein_name = (
        protein_desc
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value")
    )

    if protein_name is None and protein_desc.get("submissionNames"):
        protein_name = (
            protein_desc["submissionNames"][0]
            .get("fullName", {})
            .get("value")
        )

    locations = []

    for comment in entry.get("comments", []):
        if comment.get("commentType") == "SUBCELLULAR LOCATION":
            for loc in comment.get("subcellularLocations", []):
                pieces = [
                    loc.get("location", {}).get("value"),
                    loc.get("topology", {}).get("value"),
                    loc.get("orientation", {}).get("value"),
                ]
                pieces = [p for p in pieces if p]

                if pieces:
                    locations.append("; ".join(pieces))

    keywords = [
        kw["name"]
        for kw in entry.get("keywords", [])
        if kw.get("name")
    ]

    return {
        "gene": gene,
        "uniprot_accession": entry.get("primaryAccession"),
        "protein_name": protein_name,
        "subcellular_location_text": " | ".join(locations) if locations else None,
        "keywords_text": " | ".join(keywords) if keywords else None,
        "uniprot_status": "found",
    }
```


```python
# -------------------------
# 8.2 Define surface-accessibility classifier
# -------------------------

def classify_surface_accessibility(location_text, keywords_text):
    text = f"{safe_text(location_text)} {safe_text(keywords_text)}".lower()

    surface_terms = [
        "cell membrane",
        "plasma membrane",
        "cell surface",
        "external side of plasma membrane",
        "extracellular side",
        "gpi-anchor",
        "glycosylphosphatidylinositol",
        "lipid-anchor",
    ]

    extracellular_terms = [
        "secreted",
        "extracellular space",
        "extracellular matrix",
        "extracellular region",
    ]

    intracellular_terms = [
        "mitochondrion",
        "mitochondrial",
        "endoplasmic reticulum",
        "sarcoplasmic reticulum",
        "golgi",
        "lysosome",
        "endosome",
        "cytoplasmic vesicle",
        "nucleus",
        "nuclear",
        "cytoplasm",
        "cytosol",
        "ribosome",
        "peroxisome",
    ]

    has_surface = any(t in text for t in surface_terms)
    has_extra = any(t in text for t in extracellular_terms)
    has_intra = any(t in text for t in intracellular_terms)
    has_membrane = "membrane" in text or "transmembrane" in text

    if has_surface:
        return "surface_accessible"
    if has_extra and has_intra:
        return "secreted_or_extracellular_mixed"
    if has_extra:
        return "secreted_or_extracellular"
    if has_intra:
        return "intracellular_or_organelle"
    if has_membrane:
        return "membrane_unknown_location"

    return "unknown"


def targetability_comment(surface_class):
    comments = {
        "surface_accessible": "best class for direct peptide-shuttle targeting",
        "membrane_unknown_location": "possible target, but needs topology/extracellular-domain check",
        "secreted_or_extracellular": "useful EC/stroke context marker, usually not a direct receptor",
        "secreted_or_extracellular_mixed": "possible extracellular context marker, but not a clean direct receptor",
        "unknown": "manual annotation required",
        "intracellular_or_organelle": "reject for extracellular peptide targeting",
    }

    return comments.get(surface_class, "manual annotation required")


def targetability_score(surface_class):
    scores = {
        "surface_accessible": 4,
        "membrane_unknown_location": 2,
        "secreted_or_extracellular": 1,
        "secreted_or_extracellular_mixed": 0.5,
        "unknown": 0,
        "intracellular_or_organelle": -1,
    }

    return scores.get(surface_class, 0)
```


```python
# -------------------------
# 8.3 Annotate candidates with UniProt
# -------------------------

def annotate_with_uniprot(df, gene_col="gene", sleep=0.2):
    rows = []

    for gene in df[gene_col].dropna().astype(str).unique():
        try:
            row = query_uniprot_mouse_gene(gene)
        except Exception as e:
            row = {
                "gene": gene,
                "uniprot_accession": None,
                "protein_name": None,
                "subcellular_location_text": None,
                "keywords_text": None,
                "uniprot_status": "error",
                "uniprot_error": str(e),
            }

        rows.append(row)
        time.sleep(sleep)

    ann = pd.DataFrame(rows)

    ann["surface_accessibility"] = ann.apply(
        lambda r: classify_surface_accessibility(
            r.get("subcellular_location_text"),
            r.get("keywords_text"),
        ),
        axis=1,
    )

    ann["targetability_score"] = ann["surface_accessibility"].apply(
        targetability_score
    )

    ann["targetability_comment"] = ann["surface_accessibility"].apply(
        targetability_comment
    )

    return df.merge(ann, on="gene", how="left")


d02_annotated = annotate_with_uniprot(d02_candidates.head(100))
d14_annotated = annotate_with_uniprot(d14_candidates.head(100))

d02_annotated_path = out_dir / "D02_candidates_uniprot_annotated.csv"
d14_annotated_path = out_dir / "D14_candidates_uniprot_annotated.csv"

d02_annotated.to_csv(d02_annotated_path, index=False)
d14_annotated.to_csv(d14_annotated_path, index=False)

print("Saved D02 UniProt annotations:", d02_annotated_path)
print("Saved D14 UniProt annotations:", d14_annotated_path)
```

    Saved D02 UniProt annotations: ../data/processed/D02_candidates_uniprot_annotated.csv
    Saved D14 UniProt annotations: ../data/processed/D14_candidates_uniprot_annotated.csv



```python
# -------------------------
# 8.4 Inspect surface-accessibility classes
# -------------------------

print("D02 surface-accessibility classes:")
display(
    d02_annotated["surface_accessibility"]
    .value_counts(dropna=False)
    .to_frame("n_genes")
)

print("D14 surface-accessibility classes:")
display(
    d14_annotated["surface_accessibility"]
    .value_counts(dropna=False)
    .to_frame("n_genes")
)
```

    D02 surface-accessibility classes:



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
      <th>n_genes</th>
    </tr>
    <tr>
      <th>surface_accessibility</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>intracellular_or_organelle</th>
      <td>41</td>
    </tr>
    <tr>
      <th>surface_accessible</th>
      <td>31</td>
    </tr>
    <tr>
      <th>secreted_or_extracellular</th>
      <td>10</td>
    </tr>
    <tr>
      <th>unknown</th>
      <td>9</td>
    </tr>
    <tr>
      <th>secreted_or_extracellular_mixed</th>
      <td>6</td>
    </tr>
    <tr>
      <th>membrane_unknown_location</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


    D14 surface-accessibility classes:



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
      <th>n_genes</th>
    </tr>
    <tr>
      <th>surface_accessibility</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>surface_accessible</th>
      <td>16</td>
    </tr>
    <tr>
      <th>intracellular_or_organelle</th>
      <td>12</td>
    </tr>
    <tr>
      <th>secreted_or_extracellular</th>
      <td>9</td>
    </tr>
    <tr>
      <th>membrane_unknown_location</th>
      <td>3</td>
    </tr>
    <tr>
      <th>secreted_or_extracellular_mixed</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


## 9. Prioritise peptide-shuttle receptor candidates

The D02 response is used as the primary prioritisation window because the aim is to identify early stroke-induced endothelial surface candidates. Candidate genes are first filtered to keep proteins with plausible extracellular or membrane relevance based on UniProt annotation. Manual biological prioritisation is then applied to separate likely peptide-target candidates from secondary candidates, extracellular context markers, and false-positive surface annotations.

The final ranking combines:

- EC pseudobulk expression after D02 stroke,
- D02 induction effect size,
- log2 fold-change proxy,
- UniProt surface-accessibility class,
- and manual biological interpretation.


```python
# -------------------------
# 9.1 Build D02 candidate pool from UniProt-annotated genes
# -------------------------

candidate_classes = [
    "surface_accessible",
    "membrane_unknown_location",
    "secreted_or_extracellular",
    "secreted_or_extracellular_mixed",
    "unknown",
]

candidate_pool = d02_annotated[
    d02_annotated["surface_accessibility"].isin(candidate_classes)
].copy()

print("Candidate pool shape:", candidate_pool.shape)

display(
    candidate_pool[
        [
            "gene",
            "protein_name",
            "mean_D02",
            "effect_D02_vs_Sham",
            "log2fc_proxy_D02_vs_Sham",
            "surface_accessibility",
            "targetability_comment",
        ]
    ].head(20)
)
```

    Candidate pool shape: (59, 14)



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
      <th>gene</th>
      <th>protein_name</th>
      <th>mean_D02</th>
      <th>effect_D02_vs_Sham</th>
      <th>log2fc_proxy_D02_vs_Sham</th>
      <th>surface_accessibility</th>
      <th>targetability_comment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Igfbp7</td>
      <td>Insulin-like growth factor-binding protein 7</td>
      <td>6.496007</td>
      <td>5.042942</td>
      <td>2.160453</td>
      <td>secreted_or_extracellular</td>
      <td>useful EC/stroke context marker, usually not a...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ly6a</td>
      <td>Lymphocyte antigen 6A-2/6E-1</td>
      <td>6.753810</td>
      <td>4.570337</td>
      <td>1.629077</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ctla2a</td>
      <td>Protein CTLA-2-alpha</td>
      <td>4.136150</td>
      <td>3.658234</td>
      <td>3.113456</td>
      <td>secreted_or_extracellular</td>
      <td>useful EC/stroke context marker, usually not a...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sparc</td>
      <td>SPARC</td>
      <td>4.324855</td>
      <td>3.493701</td>
      <td>2.379462</td>
      <td>secreted_or_extracellular</td>
      <td>useful EC/stroke context marker, usually not a...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ly6c1</td>
      <td>Lymphocyte antigen 6C1</td>
      <td>5.853538</td>
      <td>2.135644</td>
      <td>0.654823</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ifitm3</td>
      <td>Interferon-induced transmembrane protein 3</td>
      <td>2.267544</td>
      <td>1.722131</td>
      <td>2.055709</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Hsp90ab1</td>
      <td>Heat shock protein HSP 90-beta</td>
      <td>2.911258</td>
      <td>1.648745</td>
      <td>1.205344</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Vim</td>
      <td>Vimentin</td>
      <td>1.873383</td>
      <td>1.507425</td>
      <td>2.355893</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Ybx1</td>
      <td>Y-box-binding protein 1</td>
      <td>1.875088</td>
      <td>1.211122</td>
      <td>1.497776</td>
      <td>secreted_or_extracellular_mixed</td>
      <td>possible extracellular context marker, but not...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>B2m</td>
      <td>Beta-2-microglobulin</td>
      <td>2.225297</td>
      <td>1.189415</td>
      <td>1.103137</td>
      <td>secreted_or_extracellular</td>
      <td>useful EC/stroke context marker, usually not a...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Tmem252</td>
      <td>Transmembrane protein 252</td>
      <td>1.471563</td>
      <td>1.179982</td>
      <td>2.335374</td>
      <td>membrane_unknown_location</td>
      <td>possible target, but needs topology/extracellu...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Eef1a1</td>
      <td>Elongation factor 1-alpha 1</td>
      <td>1.717794</td>
      <td>1.100034</td>
      <td>1.475437</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Tm4sf1</td>
      <td>Transmembrane 4 L6 family member 1</td>
      <td>1.421943</td>
      <td>1.066632</td>
      <td>2.000707</td>
      <td>membrane_unknown_location</td>
      <td>possible target, but needs topology/extracellu...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Mgp</td>
      <td>Matrix Gla protein</td>
      <td>1.570391</td>
      <td>1.065305</td>
      <td>1.636522</td>
      <td>secreted_or_extracellular</td>
      <td>useful EC/stroke context marker, usually not a...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Ly6e</td>
      <td>Lymphocyte antigen 6E</td>
      <td>2.406169</td>
      <td>1.053960</td>
      <td>0.831419</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Egfl7</td>
      <td>Epidermal growth factor-like protein 7</td>
      <td>1.513354</td>
      <td>0.839940</td>
      <td>1.168182</td>
      <td>secreted_or_extracellular</td>
      <td>useful EC/stroke context marker, usually not a...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Ppia</td>
      <td>Peptidyl-prolyl cis-trans isomerase A</td>
      <td>1.232202</td>
      <td>0.839867</td>
      <td>1.651077</td>
      <td>secreted_or_extracellular_mixed</td>
      <td>possible extracellular context marker, but not...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Anxa2</td>
      <td>Annexin A2</td>
      <td>0.952120</td>
      <td>0.728768</td>
      <td>2.091821</td>
      <td>secreted_or_extracellular_mixed</td>
      <td>possible extracellular context marker, but not...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Cd200</td>
      <td>OX-2 membrane glycoprotein</td>
      <td>1.035134</td>
      <td>0.717247</td>
      <td>1.703227</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Hspa5</td>
      <td>Endoplasmic reticulum chaperone BiP</td>
      <td>0.918031</td>
      <td>0.697331</td>
      <td>2.056454</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
    </tr>
  </tbody>
</table>
</div>



```python
# -------------------------
# 9.2 Classify D02 stroke response
# -------------------------

def classify_stroke_response(row):
    effect = row["effect_D02_vs_Sham"]
    logfc = row["log2fc_proxy_D02_vs_Sham"]
    mean_d02 = row["mean_D02"]

    if effect >= 1.0 and logfc >= 1.0:
        return "strong early D02-induced"
    if effect >= 0.5 and logfc >= 0.5:
        return "moderate early D02-induced"
    if mean_d02 >= 1.0 and abs(effect) < 0.5:
        return "highly expressed / preserved"
    if effect > 0:
        return "weakly D02-induced"

    return "not D02-induced"


candidate_pool["stroke_response"] = candidate_pool.apply(
    classify_stroke_response,
    axis=1,
)

display(
    candidate_pool["stroke_response"]
    .value_counts()
    .to_frame("n_genes")
)
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
      <th>n_genes</th>
    </tr>
    <tr>
      <th>stroke_response</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>moderate early D02-induced</th>
      <td>21</td>
    </tr>
    <tr>
      <th>weakly D02-induced</th>
      <td>20</td>
    </tr>
    <tr>
      <th>strong early D02-induced</th>
      <td>13</td>
    </tr>
    <tr>
      <th>highly expressed / preserved</th>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



```python
# -------------------------
# 9.3 Manual biological prioritisation
# -------------------------

primary = {
    "Ly6a", "Ly6e", "Tm4sf1", "Esam",
    "Cd200", "Itgb1", "Fxyd5", "Ecscr",
}

secondary = {
    "Ly6c1", "Ifitm3", "Ifitm2", "Itm2a",
    "Tmem252", "Ramp2", "Cd81", "Pecam1",
}

context = {
    "Igfbp7", "Sparc", "Egfl7", "Edn1", "Col4a1", "Col4a2",
}

reject_surface = {
    "Hsp90ab1", "Vim", "Eef1a1", "Glul", "Hspa5", "Calr",
    "Gnas", "Hspa8", "Rack1", "Msn", "Cdc42", "Cfl1", "Rab11a",
}


def manual_priority(gene):
    if gene in primary:
        return "primary peptide-target candidate"
    if gene in secondary:
        return "secondary candidate / needs validation"
    if gene in context:
        return "stroke EC context marker, not direct receptor"
    if gene in reject_surface:
        return "reject despite surface annotation"

    return "not prioritised"


manual_notes = {
    "Ly6a": "Strongest D02-induced GPI-anchored surface signal; attractive mouse stroke-BBB target but human translation must be checked.",
    "Ly6e": "GPI-anchored surface protein with early induction; possible Ly6-family target.",
    "Tm4sf1": "Transmembrane protein with strong D02 induction; extracellular loops/topology need validation.",
    "Esam": "Endothelial-selective adhesion molecule; surface accessible and EC-relevant.",
    "Cd200": "Single-pass membrane glycoprotein with significant early induction; extracellular domain likely accessible.",
    "Itgb1": "Surface integrin receptor; targetable but broad expression/off-target risk.",
    "Fxyd5": "Single-pass membrane protein with early induction; needs BBB/endothelial specificity check.",
    "Ecscr": "Endothelial cell-specific surface regulator with strong logFC and significant D02 response.",
    "Ly6c1": "Strong surface signal but immune/vascular specificity and translational relevance need caution.",
    "Ifitm3": "Induced membrane protein but more interferon/stress-associated than classic shuttle receptor.",
    "Ifitm2": "Induced membrane protein; similar caution as Ifitm3.",
    "Itm2a": "Membrane protein candidate; extracellular topology and BBB specificity need validation.",
    "Tmem252": "Strong D02 induction; membrane topology and function unclear.",
    "Ramp2": "Surface receptor-modifying protein; relevant but signalling biology may complicate targeting.",
    "Cd81": "Surface tetraspanin; targetable but broad expression risk.",
    "Pecam1": "Endothelial surface marker; useful control/reference but broad vascular expression.",
    "Igfbp7": "Strong EC/stroke marker but secreted, therefore not a clean direct receptor.",
    "Sparc": "Secreted ECM/remodelling marker; useful context marker.",
    "Egfl7": "Secreted endothelial/angiogenic context marker, not direct receptor.",
    "Edn1": "Secreted vasoactive peptide; context marker, not shuttle receptor.",
    "Col4a1": "Extracellular matrix marker, not direct shuttle receptor.",
    "Col4a2": "Extracellular matrix marker, not direct shuttle receptor.",
}


def final_rationale(priority):
    if priority == "primary peptide-target candidate":
        return (
            "prioritised for peptide design; inspect extracellular domain/topology, "
            "internalisation, BBB specificity, and human orthology"
        )

    if priority == "secondary candidate / needs validation":
        return (
            "possible peptide-design target; requires validation of specificity, "
            "topology, or translational relevance"
        )

    if priority == "stroke EC context marker, not direct receptor":
        return (
            "useful for defining stroke endothelial state, but not prioritised "
            "as direct shuttle receptor"
        )

    if priority == "reject despite surface annotation":
        return (
            "not prioritised for peptide design because primary biology is "
            "intracellular/stress/cytoskeletal/signalling"
        )

    return "not prioritised in this toy analysis"


candidate_pool["manual_priority"] = candidate_pool["gene"].apply(
    manual_priority
)

candidate_pool["manual_interpretation"] = (
    candidate_pool["gene"]
    .map(manual_notes)
    .fillna("")
)

candidate_pool["final_peptide_design_rationale"] = (
    candidate_pool["manual_priority"]
    .apply(final_rationale)
)
```


```python
# -------------------------
# 9.4 Compute omics priority score and final manual ranking
# -------------------------

priority_order = {
    "primary peptide-target candidate": 1,
    "secondary candidate / needs validation": 2,
    "stroke EC context marker, not direct receptor": 3,
    "reject despite surface annotation": 4,
    "not prioritised": 5,
}

candidate_pool["priority_order"] = candidate_pool["manual_priority"].map(
    priority_order
)

candidate_pool["omics_priority_score"] = (
    candidate_pool["mean_D02"].rank(pct=True)
    + candidate_pool["effect_D02_vs_Sham"].rank(pct=True)
    + candidate_pool["log2fc_proxy_D02_vs_Sham"].rank(pct=True)
)

final_manual_table = candidate_pool.sort_values(
    ["priority_order", "omics_priority_score"],
    ascending=[True, False],
).copy()

final_columns = [
    "gene",
    "protein_name",
    "mean_D02",
    "mean_Sham",
    "effect_D02_vs_Sham",
    "log2fc_proxy_D02_vs_Sham",
    "pvalue_D02_vs_Sham",
    "omics_priority_score",
    "stroke_response",
    "subcellular_location_text",
    "surface_accessibility",
    "targetability_comment",
    "manual_priority",
    "manual_interpretation",
    "final_peptide_design_rationale",
    "uniprot_accession",
]

display(final_manual_table[final_columns].head(50))
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
      <th>gene</th>
      <th>protein_name</th>
      <th>mean_D02</th>
      <th>mean_Sham</th>
      <th>effect_D02_vs_Sham</th>
      <th>log2fc_proxy_D02_vs_Sham</th>
      <th>pvalue_D02_vs_Sham</th>
      <th>omics_priority_score</th>
      <th>stroke_response</th>
      <th>subcellular_location_text</th>
      <th>surface_accessibility</th>
      <th>targetability_comment</th>
      <th>manual_priority</th>
      <th>manual_interpretation</th>
      <th>final_peptide_design_rationale</th>
      <th>uniprot_accession</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Ly6a</td>
      <td>Lymphocyte antigen 6A-2/6E-1</td>
      <td>6.753810</td>
      <td>2.183473</td>
      <td>4.570337</td>
      <td>1.629077</td>
      <td>0.008333</td>
      <td>2.593220</td>
      <td>strong early D02-induced</td>
      <td>Cell membrane; Lipid-anchor, GPI-anchor</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>primary peptide-target candidate</td>
      <td>Strongest D02-induced GPI-anchored surface sig...</td>
      <td>prioritised for peptide design; inspect extrac...</td>
      <td>P05533</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Tm4sf1</td>
      <td>Transmembrane 4 L6 family member 1</td>
      <td>1.421943</td>
      <td>0.355311</td>
      <td>1.066632</td>
      <td>2.000707</td>
      <td>0.003731</td>
      <td>2.288136</td>
      <td>strong early D02-induced</td>
      <td>Membrane; Multi-pass membrane protein</td>
      <td>membrane_unknown_location</td>
      <td>possible target, but needs topology/extracellu...</td>
      <td>primary peptide-target candidate</td>
      <td>Transmembrane protein with strong D02 inductio...</td>
      <td>prioritised for peptide design; inspect extrac...</td>
      <td>Q64302</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Cd200</td>
      <td>OX-2 membrane glycoprotein</td>
      <td>1.035134</td>
      <td>0.317887</td>
      <td>0.717247</td>
      <td>1.703227</td>
      <td>0.000703</td>
      <td>2.000000</td>
      <td>moderate early D02-induced</td>
      <td>Cell membrane; Single-pass type I membrane pro...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>primary peptide-target candidate</td>
      <td>Single-pass membrane glycoprotein with signifi...</td>
      <td>prioritised for peptide design; inspect extrac...</td>
      <td>O54901</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Ly6e</td>
      <td>Lymphocyte antigen 6E</td>
      <td>2.406169</td>
      <td>1.352209</td>
      <td>1.053960</td>
      <td>0.831419</td>
      <td>0.108692</td>
      <td>1.779661</td>
      <td>moderate early D02-induced</td>
      <td>Cell membrane; Lipid-anchor, GPI-anchor</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>primary peptide-target candidate</td>
      <td>GPI-anchored surface protein with early induct...</td>
      <td>prioritised for peptide design; inspect extrac...</td>
      <td>Q64253</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Ecscr</td>
      <td>Endothelial cell-specific chemotaxis regulator</td>
      <td>0.639574</td>
      <td>0.083505</td>
      <td>0.556069</td>
      <td>2.937155</td>
      <td>0.000125</td>
      <td>1.677966</td>
      <td>moderate early D02-induced</td>
      <td>Cell membrane; Single-pass type I membrane pro...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>primary peptide-target candidate</td>
      <td>Endothelial cell-specific surface regulator wi...</td>
      <td>prioritised for peptide design; inspect extrac...</td>
      <td>Q3TZW0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Itgb1</td>
      <td>Integrin beta-1</td>
      <td>0.876678</td>
      <td>0.295314</td>
      <td>0.581364</td>
      <td>1.569795</td>
      <td>0.016298</td>
      <td>1.559322</td>
      <td>moderate early D02-induced</td>
      <td>Cell membrane; Single-pass type I membrane pro...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>primary peptide-target candidate</td>
      <td>Surface integrin receptor; targetable but broa...</td>
      <td>prioritised for peptide design; inspect extrac...</td>
      <td>P09055</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Esam</td>
      <td>Endothelial cell-selective adhesion molecule</td>
      <td>0.978323</td>
      <td>0.419158</td>
      <td>0.559165</td>
      <td>1.222813</td>
      <td>0.029184</td>
      <td>1.440678</td>
      <td>moderate early D02-induced</td>
      <td>Cell junction, adherens junction | Cell juncti...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>primary peptide-target candidate</td>
      <td>Endothelial-selective adhesion molecule; surfa...</td>
      <td>prioritised for peptide design; inspect extrac...</td>
      <td>Q925F2</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Fxyd5</td>
      <td>FXYD domain-containing ion transport regulator 5</td>
      <td>0.920225</td>
      <td>0.445696</td>
      <td>0.474529</td>
      <td>1.045925</td>
      <td>0.030093</td>
      <td>1.118644</td>
      <td>weakly D02-induced</td>
      <td>Cell membrane; Single-pass type I membrane pro...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>primary peptide-target candidate</td>
      <td>Single-pass membrane protein with early induct...</td>
      <td>prioritised for peptide design; inspect extrac...</td>
      <td>P97808</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ifitm3</td>
      <td>Interferon-induced transmembrane protein 3</td>
      <td>2.267544</td>
      <td>0.545412</td>
      <td>1.722131</td>
      <td>2.055709</td>
      <td>0.004427</td>
      <td>2.593220</td>
      <td>strong early D02-induced</td>
      <td>Cell membrane; Single-pass type II membrane pr...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>secondary candidate / needs validation</td>
      <td>Induced membrane protein but more interferon/s...</td>
      <td>possible peptide-design target; requires valid...</td>
      <td>Q9CQW9</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Tmem252</td>
      <td>Transmembrane protein 252</td>
      <td>1.471563</td>
      <td>0.291582</td>
      <td>1.179982</td>
      <td>2.335374</td>
      <td>0.000901</td>
      <td>2.457627</td>
      <td>strong early D02-induced</td>
      <td>Membrane; Multi-pass membrane protein</td>
      <td>membrane_unknown_location</td>
      <td>possible target, but needs topology/extracellu...</td>
      <td>secondary candidate / needs validation</td>
      <td>Strong D02 induction; membrane topology and fu...</td>
      <td>possible peptide-design target; requires valid...</td>
      <td>Q8C353</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ly6c1</td>
      <td>Lymphocyte antigen 6C1</td>
      <td>5.853538</td>
      <td>3.717894</td>
      <td>2.135644</td>
      <td>0.654823</td>
      <td>0.123421</td>
      <td>1.932203</td>
      <td>moderate early D02-induced</td>
      <td>Cell membrane; Lipid-anchor, GPI-anchor</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>secondary candidate / needs validation</td>
      <td>Strong surface signal but immune/vascular spec...</td>
      <td>possible peptide-design target; requires valid...</td>
      <td>P0CW02</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Ifitm2</td>
      <td>Interferon-induced transmembrane protein 2</td>
      <td>0.621725</td>
      <td>0.196543</td>
      <td>0.425182</td>
      <td>1.661429</td>
      <td>0.009214</td>
      <td>1.067797</td>
      <td>weakly D02-induced</td>
      <td>Cell membrane; Single-pass type II membrane pr...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>secondary candidate / needs validation</td>
      <td>Induced membrane protein; similar caution as I...</td>
      <td>possible peptide-design target; requires valid...</td>
      <td>Q99J93</td>
    </tr>
    <tr>
      <th>81</th>
      <td>Cd81</td>
      <td>CD81 antigen</td>
      <td>0.619079</td>
      <td>0.231242</td>
      <td>0.387838</td>
      <td>1.420719</td>
      <td>0.061443</td>
      <td>0.779661</td>
      <td>weakly D02-induced</td>
      <td>Cell membrane; Multi-pass membrane protein | B...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>secondary candidate / needs validation</td>
      <td>Surface tetraspanin; targetable but broad expr...</td>
      <td>possible peptide-design target; requires valid...</td>
      <td>P35762</td>
    </tr>
    <tr>
      <th>78</th>
      <td>Pecam1</td>
      <td>Platelet endothelial cell adhesion molecule</td>
      <td>0.777982</td>
      <td>0.377730</td>
      <td>0.400252</td>
      <td>1.042380</td>
      <td>0.025715</td>
      <td>0.711864</td>
      <td>weakly D02-induced</td>
      <td>Cell membrane; Single-pass type I membrane pro...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>secondary candidate / needs validation</td>
      <td>Endothelial surface marker; useful control/ref...</td>
      <td>possible peptide-design target; requires valid...</td>
      <td>Q08481</td>
    </tr>
    <tr>
      <th>93</th>
      <td>Ramp2</td>
      <td>Receptor activity-modifying protein 2</td>
      <td>0.902703</td>
      <td>0.558342</td>
      <td>0.344361</td>
      <td>0.693100</td>
      <td>0.209434</td>
      <td>0.559322</td>
      <td>weakly D02-induced</td>
      <td>Cell membrane; Single-pass type I membrane pro...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>secondary candidate / needs validation</td>
      <td>Surface receptor-modifying protein; relevant b...</td>
      <td>possible peptide-design target; requires valid...</td>
      <td>Q9WUP0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Igfbp7</td>
      <td>Insulin-like growth factor-binding protein 7</td>
      <td>6.496007</td>
      <td>1.453066</td>
      <td>5.042942</td>
      <td>2.160453</td>
      <td>0.008618</td>
      <td>2.847458</td>
      <td>strong early D02-induced</td>
      <td>Secreted</td>
      <td>secreted_or_extracellular</td>
      <td>useful EC/stroke context marker, usually not a...</td>
      <td>stroke EC context marker, not direct receptor</td>
      <td>Strong EC/stroke marker but secreted, therefor...</td>
      <td>useful for defining stroke endothelial state, ...</td>
      <td>Q61581</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sparc</td>
      <td>SPARC</td>
      <td>4.324855</td>
      <td>0.831155</td>
      <td>3.493701</td>
      <td>2.379462</td>
      <td>0.021555</td>
      <td>2.830508</td>
      <td>strong early D02-induced</td>
      <td>Secreted, extracellular space, extracellular m...</td>
      <td>secreted_or_extracellular</td>
      <td>useful EC/stroke context marker, usually not a...</td>
      <td>stroke EC context marker, not direct receptor</td>
      <td>Secreted ECM/remodelling marker; useful contex...</td>
      <td>useful for defining stroke endothelial state, ...</td>
      <td>P07214</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Edn1</td>
      <td>Endothelin-1</td>
      <td>0.906098</td>
      <td>0.224776</td>
      <td>0.681322</td>
      <td>2.011173</td>
      <td>0.001177</td>
      <td>1.881356</td>
      <td>moderate early D02-induced</td>
      <td>Secreted</td>
      <td>secreted_or_extracellular</td>
      <td>useful EC/stroke context marker, usually not a...</td>
      <td>stroke EC context marker, not direct receptor</td>
      <td>Secreted vasoactive peptide; context marker, n...</td>
      <td>useful for defining stroke endothelial state, ...</td>
      <td>P22387</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Egfl7</td>
      <td>Epidermal growth factor-like protein 7</td>
      <td>1.513354</td>
      <td>0.673415</td>
      <td>0.839940</td>
      <td>1.168182</td>
      <td>0.016951</td>
      <td>1.830508</td>
      <td>moderate early D02-induced</td>
      <td>Secreted, extracellular space</td>
      <td>secreted_or_extracellular</td>
      <td>useful EC/stroke context marker, usually not a...</td>
      <td>stroke EC context marker, not direct receptor</td>
      <td>Secreted endothelial/angiogenic context marker...</td>
      <td>useful for defining stroke endothelial state, ...</td>
      <td>Q9QXT5</td>
    </tr>
    <tr>
      <th>70</th>
      <td>Col4a2</td>
      <td>Collagen alpha-2(IV) chain</td>
      <td>0.502416</td>
      <td>0.075367</td>
      <td>0.427049</td>
      <td>2.736857</td>
      <td>0.052475</td>
      <td>1.220339</td>
      <td>weakly D02-induced</td>
      <td>Secreted, extracellular space, extracellular m...</td>
      <td>secreted_or_extracellular</td>
      <td>useful EC/stroke context marker, usually not a...</td>
      <td>stroke EC context marker, not direct receptor</td>
      <td>Extracellular matrix marker, not direct shuttl...</td>
      <td>useful for defining stroke endothelial state, ...</td>
      <td>P08122</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Col4a1</td>
      <td>Collagen alpha-1(IV) chain</td>
      <td>0.513086</td>
      <td>0.098889</td>
      <td>0.414197</td>
      <td>2.375311</td>
      <td>0.035601</td>
      <td>1.169492</td>
      <td>weakly D02-induced</td>
      <td>Secreted, extracellular space, extracellular m...</td>
      <td>secreted_or_extracellular</td>
      <td>useful EC/stroke context marker, usually not a...</td>
      <td>stroke EC context marker, not direct receptor</td>
      <td>Extracellular matrix marker, not direct shuttl...</td>
      <td>useful for defining stroke endothelial state, ...</td>
      <td>P02463</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Vim</td>
      <td>Vimentin</td>
      <td>1.873383</td>
      <td>0.365958</td>
      <td>1.507425</td>
      <td>2.355893</td>
      <td>0.012264</td>
      <td>2.593220</td>
      <td>strong early D02-induced</td>
      <td>Cytoplasm | Cytoplasm, cytoskeleton | Nucleus ...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>P20152</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Hsp90ab1</td>
      <td>Heat shock protein HSP 90-beta</td>
      <td>2.911258</td>
      <td>1.262513</td>
      <td>1.648745</td>
      <td>1.205344</td>
      <td>0.068693</td>
      <td>2.152542</td>
      <td>strong early D02-induced</td>
      <td>Cytoplasm | Melanosome | Nucleus | Secreted | ...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>P11499</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Eef1a1</td>
      <td>Elongation factor 1-alpha 1</td>
      <td>1.717794</td>
      <td>0.617760</td>
      <td>1.100034</td>
      <td>1.475437</td>
      <td>0.022715</td>
      <td>2.135593</td>
      <td>strong early D02-induced</td>
      <td>Cytoplasm | Nucleus | Nucleus, nucleolus | Cel...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>P10126</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Hspa5</td>
      <td>Endoplasmic reticulum chaperone BiP</td>
      <td>0.918031</td>
      <td>0.220700</td>
      <td>0.697331</td>
      <td>2.056454</td>
      <td>0.012144</td>
      <td>1.983051</td>
      <td>moderate early D02-induced</td>
      <td>Endoplasmic reticulum lumen | Melanosome | Cyt...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>P20029</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Glul</td>
      <td>Glutamine synthetase</td>
      <td>1.146988</td>
      <td>0.462845</td>
      <td>0.684143</td>
      <td>1.309247</td>
      <td>0.034632</td>
      <td>1.728814</td>
      <td>moderate early D02-induced</td>
      <td>Cytoplasm, cytosol | Microsome | Mitochondrion...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>P15105</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Calr</td>
      <td>Calreticulin</td>
      <td>0.833457</td>
      <td>0.237697</td>
      <td>0.595760</td>
      <td>1.809977</td>
      <td>0.059921</td>
      <td>1.661017</td>
      <td>moderate early D02-induced</td>
      <td>Endoplasmic reticulum lumen | Cytoplasm, cytos...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>P14211</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Rack1</td>
      <td>Small ribosomal subunit protein RACK1</td>
      <td>0.753514</td>
      <td>0.221606</td>
      <td>0.531909</td>
      <td>1.765634</td>
      <td>0.020701</td>
      <td>1.457627</td>
      <td>moderate early D02-induced</td>
      <td>Cell membrane; Peripheral membrane protein | C...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>P68040</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Msn</td>
      <td>Moesin</td>
      <td>0.719441</td>
      <td>0.195917</td>
      <td>0.523524</td>
      <td>1.876627</td>
      <td>0.011529</td>
      <td>1.423729</td>
      <td>moderate early D02-induced</td>
      <td>Cell membrane; Peripheral membrane protein; Cy...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>P26041</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Hspa8</td>
      <td>Heat shock cognate 71 kDa protein</td>
      <td>0.975532</td>
      <td>0.439680</td>
      <td>0.535852</td>
      <td>1.149733</td>
      <td>0.017574</td>
      <td>1.305085</td>
      <td>moderate early D02-induced</td>
      <td>Cytoplasm | Melanosome | Nucleus, nucleolus | ...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>P63017</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Gnas</td>
      <td>Guanine nucleotide-binding protein G(s) subuni...</td>
      <td>1.202829</td>
      <td>0.717016</td>
      <td>0.485813</td>
      <td>0.746353</td>
      <td>0.231448</td>
      <td>1.152542</td>
      <td>highly expressed / preserved</td>
      <td>Cell membrane; Lipid-anchor | Cytoplasm</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>P63094</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Cdc42</td>
      <td>Cell division control protein 42 homolog</td>
      <td>0.732141</td>
      <td>0.286725</td>
      <td>0.445416</td>
      <td>1.352451</td>
      <td>0.006836</td>
      <td>1.033898</td>
      <td>weakly D02-induced</td>
      <td>Cell membrane; Lipid-anchor; Cytoplasmic side ...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>P60766</td>
    </tr>
    <tr>
      <th>91</th>
      <td>Rab11a</td>
      <td>Ras-related protein Rab-11A</td>
      <td>0.571084</td>
      <td>0.212752</td>
      <td>0.358332</td>
      <td>1.424528</td>
      <td>0.004831</td>
      <td>0.677966</td>
      <td>weakly D02-induced</td>
      <td>Cell membrane; Lipid-anchor | Endosome membran...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>P62492</td>
    </tr>
    <tr>
      <th>94</th>
      <td>Cfl1</td>
      <td>Cofilin-1</td>
      <td>0.573371</td>
      <td>0.233459</td>
      <td>0.339912</td>
      <td>1.296297</td>
      <td>0.052836</td>
      <td>0.542373</td>
      <td>weakly D02-induced</td>
      <td>Nucleus matrix | Cytoplasm, cytoskeleton | Cel...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>P18760</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ctla2a</td>
      <td>Protein CTLA-2-alpha</td>
      <td>4.136150</td>
      <td>0.477916</td>
      <td>3.658234</td>
      <td>3.113456</td>
      <td>0.005249</td>
      <td>2.881356</td>
      <td>strong early D02-induced</td>
      <td>Secreted</td>
      <td>secreted_or_extracellular</td>
      <td>useful EC/stroke context marker, usually not a...</td>
      <td>not prioritised</td>
      <td></td>
      <td>not prioritised in this toy analysis</td>
      <td>P12399</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Ybx1</td>
      <td>Y-box-binding protein 1</td>
      <td>1.875088</td>
      <td>0.663966</td>
      <td>1.211122</td>
      <td>1.497776</td>
      <td>0.025729</td>
      <td>2.254237</td>
      <td>strong early D02-induced</td>
      <td>Cytoplasm | Nucleus | Cytoplasmic granule | Se...</td>
      <td>secreted_or_extracellular_mixed</td>
      <td>possible extracellular context marker, but not...</td>
      <td>not prioritised</td>
      <td></td>
      <td>not prioritised in this toy analysis</td>
      <td>P62960</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Mgp</td>
      <td>Matrix Gla protein</td>
      <td>1.570391</td>
      <td>0.505085</td>
      <td>1.065305</td>
      <td>1.636522</td>
      <td>0.002117</td>
      <td>2.186441</td>
      <td>strong early D02-induced</td>
      <td>Secreted</td>
      <td>secreted_or_extracellular</td>
      <td>useful EC/stroke context marker, usually not a...</td>
      <td>not prioritised</td>
      <td></td>
      <td>not prioritised in this toy analysis</td>
      <td>P19788</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Anxa2</td>
      <td>Annexin A2</td>
      <td>0.952120</td>
      <td>0.223352</td>
      <td>0.728768</td>
      <td>2.091821</td>
      <td>0.004561</td>
      <td>2.067797</td>
      <td>moderate early D02-induced</td>
      <td>Secreted, extracellular space, extracellular m...</td>
      <td>secreted_or_extracellular_mixed</td>
      <td>possible extracellular context marker, but not...</td>
      <td>not prioritised</td>
      <td></td>
      <td>not prioritised in this toy analysis</td>
      <td>P07356</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Ppia</td>
      <td>Peptidyl-prolyl cis-trans isomerase A</td>
      <td>1.232202</td>
      <td>0.392335</td>
      <td>0.839867</td>
      <td>1.651077</td>
      <td>0.038470</td>
      <td>2.050847</td>
      <td>moderate early D02-induced</td>
      <td>Cytoplasm | Secreted | Nucleus</td>
      <td>secreted_or_extracellular_mixed</td>
      <td>possible extracellular context marker, but not...</td>
      <td>not prioritised</td>
      <td></td>
      <td>not prioritised in this toy analysis</td>
      <td>P17742</td>
    </tr>
    <tr>
      <th>11</th>
      <td>B2m</td>
      <td>Beta-2-microglobulin</td>
      <td>2.225297</td>
      <td>1.035883</td>
      <td>1.189415</td>
      <td>1.103137</td>
      <td>0.021049</td>
      <td>1.966102</td>
      <td>strong early D02-induced</td>
      <td>Secreted</td>
      <td>secreted_or_extracellular</td>
      <td>useful EC/stroke context marker, usually not a...</td>
      <td>not prioritised</td>
      <td></td>
      <td>not prioritised in this toy analysis</td>
      <td>P01887</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Lrg1</td>
      <td>Leucine-rich HEV glycoprotein</td>
      <td>0.704301</td>
      <td>0.065713</td>
      <td>0.638588</td>
      <td>3.421930</td>
      <td>0.001453</td>
      <td>1.847458</td>
      <td>moderate early D02-induced</td>
      <td>NaN</td>
      <td>unknown</td>
      <td>manual annotation required</td>
      <td>not prioritised</td>
      <td></td>
      <td>not prioritised in this toy analysis</td>
      <td>Q91XL1</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Ece1</td>
      <td>Endothelin-converting enzyme 1</td>
      <td>0.981473</td>
      <td>0.370838</td>
      <td>0.610635</td>
      <td>1.404157</td>
      <td>0.005822</td>
      <td>1.627119</td>
      <td>moderate early D02-induced</td>
      <td>Cell membrane; Single-pass type II membrane pr...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>not prioritised</td>
      <td></td>
      <td>not prioritised in this toy analysis</td>
      <td>Q4PZA2</td>
    </tr>
    <tr>
      <th>30</th>
      <td>H2-D1</td>
      <td>H-2 class I histocompatibility antigen, D-B al...</td>
      <td>1.326389</td>
      <td>0.674647</td>
      <td>0.651742</td>
      <td>0.975298</td>
      <td>0.025642</td>
      <td>1.508475</td>
      <td>moderate early D02-induced</td>
      <td>Membrane; Single-pass type I membrane protein</td>
      <td>membrane_unknown_location</td>
      <td>possible target, but needs topology/extracellu...</td>
      <td>not prioritised</td>
      <td></td>
      <td>not prioritised in this toy analysis</td>
      <td>P01899</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Mt2</td>
      <td>Metallothionein-2</td>
      <td>0.702129</td>
      <td>0.169450</td>
      <td>0.532679</td>
      <td>2.050874</td>
      <td>0.130675</td>
      <td>1.491525</td>
      <td>moderate early D02-induced</td>
      <td>NaN</td>
      <td>unknown</td>
      <td>manual annotation required</td>
      <td>not prioritised</td>
      <td></td>
      <td>not prioritised in this toy analysis</td>
      <td>P02798</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Myl12a</td>
      <td>Myosin, light chain 12A, regulatory, non-sarco...</td>
      <td>1.021412</td>
      <td>0.477085</td>
      <td>0.544328</td>
      <td>1.098247</td>
      <td>0.010147</td>
      <td>1.338983</td>
      <td>moderate early D02-induced</td>
      <td>NaN</td>
      <td>unknown</td>
      <td>manual annotation required</td>
      <td>not prioritised</td>
      <td></td>
      <td>not prioritised in this toy analysis</td>
      <td>D3Z249</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Cldn5</td>
      <td>Claudin-5</td>
      <td>2.277336</td>
      <td>1.813087</td>
      <td>0.464248</td>
      <td>0.328899</td>
      <td>0.481601</td>
      <td>1.237288</td>
      <td>highly expressed / preserved</td>
      <td>Cell junction, tight junction | Cell membrane;...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>not prioritised</td>
      <td></td>
      <td>not prioritised in this toy analysis</td>
      <td>O54942</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Txn1</td>
      <td>Thioredoxin</td>
      <td>0.599075</td>
      <td>0.156001</td>
      <td>0.443074</td>
      <td>1.941171</td>
      <td>0.008162</td>
      <td>1.186441</td>
      <td>weakly D02-induced</td>
      <td>Nucleus | Cytoplasm | Secreted</td>
      <td>secreted_or_extracellular_mixed</td>
      <td>possible extracellular context marker, but not...</td>
      <td>not prioritised</td>
      <td></td>
      <td>not prioritised in this toy analysis</td>
      <td>P10639</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Itm2b</td>
      <td>Integral membrane protein 2B</td>
      <td>1.309536</td>
      <td>0.823283</td>
      <td>0.486253</td>
      <td>0.669595</td>
      <td>0.059706</td>
      <td>1.169492</td>
      <td>highly expressed / preserved</td>
      <td>Golgi apparatus membrane; Single-pass type II ...</td>
      <td>surface_accessible</td>
      <td>best class for direct peptide-shuttle targeting</td>
      <td>not prioritised</td>
      <td></td>
      <td>not prioritised in this toy analysis</td>
      <td>O89051</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Myl6</td>
      <td>Myosin light polypeptide 6</td>
      <td>1.029494</td>
      <td>0.556270</td>
      <td>0.473224</td>
      <td>0.888077</td>
      <td>0.058037</td>
      <td>1.135593</td>
      <td>highly expressed / preserved</td>
      <td>NaN</td>
      <td>unknown</td>
      <td>manual annotation required</td>
      <td>not prioritised</td>
      <td></td>
      <td>not prioritised in this toy analysis</td>
      <td>Q60605</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Atox1</td>
      <td>Copper transport protein ATOX1</td>
      <td>1.028519</td>
      <td>0.556801</td>
      <td>0.471719</td>
      <td>0.885335</td>
      <td>0.023495</td>
      <td>1.084746</td>
      <td>highly expressed / preserved</td>
      <td>NaN</td>
      <td>unknown</td>
      <td>manual annotation required</td>
      <td>not prioritised</td>
      <td></td>
      <td>not prioritised in this toy analysis</td>
      <td>O08997</td>
    </tr>
  </tbody>
</table>
</div>



```python
# -------------------------
# 9.5 Save prioritised candidate table
# -------------------------

final_manual_path = out_dir / "FINAL_manual_prioritised_EC_peptide_target_candidates.csv"

final_manual_table[final_columns].to_csv(
    final_manual_path,
    index=False,
)

print("Saved prioritised EC peptide-target candidates:", final_manual_path)
```

    Saved prioritised EC peptide-target candidates: ../data/processed/FINAL_manual_prioritised_EC_peptide_target_candidates.csv


## 10. Export final receptor-candidate report table

The final report table reformats the prioritised candidates into a compact, human-readable output for downstream peptide-design notebooks. This table includes the gene candidate, UniProt protein annotation, EC pseudobulk effect statistics, surface-accessibility class, biological interpretation, peptide-design rationale, and caveats.

The output remains a gene-level receptor-candidate shortlist. It does not yet validate isoform-specific topology, extracellular-domain structure, internalisation, human translation, or binding suitability.


```python
# -------------------------
# 10.1 Build final report table
# -------------------------

final_report_table = final_manual_table.copy()

final_report_table = final_report_table.rename(columns={
    "gene": "gene_candidate",
    "protein_name": "uniprot_canonical_protein_annotation",
    "subcellular_location_text": "uniprot_subcellular_location",
    "surface_accessibility": "putative_accessibility_class",
    "manual_priority": "target_prioritisation_class",
    "manual_interpretation": "biological_interpretation",
})

final_report_table["evidence_level"] = (
    "gene-level transcriptomic candidate with UniProt canonical protein annotation; "
    "requires isoform, extracellular-domain, receptor-complex, mouse-human orthology, "
    "and internalisation validation"
)

final_report_columns = [
    "gene_candidate",
    "uniprot_canonical_protein_annotation",
    "mean_D02",
    "mean_Sham",
    "effect_D02_vs_Sham",
    "log2fc_proxy_D02_vs_Sham",
    "pvalue_D02_vs_Sham",
    "omics_priority_score",
    "stroke_response",
    "uniprot_subcellular_location",
    "putative_accessibility_class",
    "target_prioritisation_class",
    "biological_interpretation",
    "final_peptide_design_rationale",
    "evidence_level",
    "uniprot_accession",
]

display(final_report_table[final_report_columns].head(30))
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
      <th>gene_candidate</th>
      <th>uniprot_canonical_protein_annotation</th>
      <th>mean_D02</th>
      <th>mean_Sham</th>
      <th>effect_D02_vs_Sham</th>
      <th>log2fc_proxy_D02_vs_Sham</th>
      <th>pvalue_D02_vs_Sham</th>
      <th>omics_priority_score</th>
      <th>stroke_response</th>
      <th>uniprot_subcellular_location</th>
      <th>putative_accessibility_class</th>
      <th>target_prioritisation_class</th>
      <th>biological_interpretation</th>
      <th>final_peptide_design_rationale</th>
      <th>evidence_level</th>
      <th>uniprot_accession</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Ly6a</td>
      <td>Lymphocyte antigen 6A-2/6E-1</td>
      <td>6.753810</td>
      <td>2.183473</td>
      <td>4.570337</td>
      <td>1.629077</td>
      <td>0.008333</td>
      <td>2.593220</td>
      <td>strong early D02-induced</td>
      <td>Cell membrane; Lipid-anchor, GPI-anchor</td>
      <td>surface_accessible</td>
      <td>primary peptide-target candidate</td>
      <td>Strongest D02-induced GPI-anchored surface sig...</td>
      <td>prioritised for peptide design; inspect extrac...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>P05533</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Tm4sf1</td>
      <td>Transmembrane 4 L6 family member 1</td>
      <td>1.421943</td>
      <td>0.355311</td>
      <td>1.066632</td>
      <td>2.000707</td>
      <td>0.003731</td>
      <td>2.288136</td>
      <td>strong early D02-induced</td>
      <td>Membrane; Multi-pass membrane protein</td>
      <td>membrane_unknown_location</td>
      <td>primary peptide-target candidate</td>
      <td>Transmembrane protein with strong D02 inductio...</td>
      <td>prioritised for peptide design; inspect extrac...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>Q64302</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Cd200</td>
      <td>OX-2 membrane glycoprotein</td>
      <td>1.035134</td>
      <td>0.317887</td>
      <td>0.717247</td>
      <td>1.703227</td>
      <td>0.000703</td>
      <td>2.000000</td>
      <td>moderate early D02-induced</td>
      <td>Cell membrane; Single-pass type I membrane pro...</td>
      <td>surface_accessible</td>
      <td>primary peptide-target candidate</td>
      <td>Single-pass membrane glycoprotein with signifi...</td>
      <td>prioritised for peptide design; inspect extrac...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>O54901</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Ly6e</td>
      <td>Lymphocyte antigen 6E</td>
      <td>2.406169</td>
      <td>1.352209</td>
      <td>1.053960</td>
      <td>0.831419</td>
      <td>0.108692</td>
      <td>1.779661</td>
      <td>moderate early D02-induced</td>
      <td>Cell membrane; Lipid-anchor, GPI-anchor</td>
      <td>surface_accessible</td>
      <td>primary peptide-target candidate</td>
      <td>GPI-anchored surface protein with early induct...</td>
      <td>prioritised for peptide design; inspect extrac...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>Q64253</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Ecscr</td>
      <td>Endothelial cell-specific chemotaxis regulator</td>
      <td>0.639574</td>
      <td>0.083505</td>
      <td>0.556069</td>
      <td>2.937155</td>
      <td>0.000125</td>
      <td>1.677966</td>
      <td>moderate early D02-induced</td>
      <td>Cell membrane; Single-pass type I membrane pro...</td>
      <td>surface_accessible</td>
      <td>primary peptide-target candidate</td>
      <td>Endothelial cell-specific surface regulator wi...</td>
      <td>prioritised for peptide design; inspect extrac...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>Q3TZW0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Itgb1</td>
      <td>Integrin beta-1</td>
      <td>0.876678</td>
      <td>0.295314</td>
      <td>0.581364</td>
      <td>1.569795</td>
      <td>0.016298</td>
      <td>1.559322</td>
      <td>moderate early D02-induced</td>
      <td>Cell membrane; Single-pass type I membrane pro...</td>
      <td>surface_accessible</td>
      <td>primary peptide-target candidate</td>
      <td>Surface integrin receptor; targetable but broa...</td>
      <td>prioritised for peptide design; inspect extrac...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>P09055</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Esam</td>
      <td>Endothelial cell-selective adhesion molecule</td>
      <td>0.978323</td>
      <td>0.419158</td>
      <td>0.559165</td>
      <td>1.222813</td>
      <td>0.029184</td>
      <td>1.440678</td>
      <td>moderate early D02-induced</td>
      <td>Cell junction, adherens junction | Cell juncti...</td>
      <td>surface_accessible</td>
      <td>primary peptide-target candidate</td>
      <td>Endothelial-selective adhesion molecule; surfa...</td>
      <td>prioritised for peptide design; inspect extrac...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>Q925F2</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Fxyd5</td>
      <td>FXYD domain-containing ion transport regulator 5</td>
      <td>0.920225</td>
      <td>0.445696</td>
      <td>0.474529</td>
      <td>1.045925</td>
      <td>0.030093</td>
      <td>1.118644</td>
      <td>weakly D02-induced</td>
      <td>Cell membrane; Single-pass type I membrane pro...</td>
      <td>surface_accessible</td>
      <td>primary peptide-target candidate</td>
      <td>Single-pass membrane protein with early induct...</td>
      <td>prioritised for peptide design; inspect extrac...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>P97808</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ifitm3</td>
      <td>Interferon-induced transmembrane protein 3</td>
      <td>2.267544</td>
      <td>0.545412</td>
      <td>1.722131</td>
      <td>2.055709</td>
      <td>0.004427</td>
      <td>2.593220</td>
      <td>strong early D02-induced</td>
      <td>Cell membrane; Single-pass type II membrane pr...</td>
      <td>surface_accessible</td>
      <td>secondary candidate / needs validation</td>
      <td>Induced membrane protein but more interferon/s...</td>
      <td>possible peptide-design target; requires valid...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>Q9CQW9</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Tmem252</td>
      <td>Transmembrane protein 252</td>
      <td>1.471563</td>
      <td>0.291582</td>
      <td>1.179982</td>
      <td>2.335374</td>
      <td>0.000901</td>
      <td>2.457627</td>
      <td>strong early D02-induced</td>
      <td>Membrane; Multi-pass membrane protein</td>
      <td>membrane_unknown_location</td>
      <td>secondary candidate / needs validation</td>
      <td>Strong D02 induction; membrane topology and fu...</td>
      <td>possible peptide-design target; requires valid...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>Q8C353</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ly6c1</td>
      <td>Lymphocyte antigen 6C1</td>
      <td>5.853538</td>
      <td>3.717894</td>
      <td>2.135644</td>
      <td>0.654823</td>
      <td>0.123421</td>
      <td>1.932203</td>
      <td>moderate early D02-induced</td>
      <td>Cell membrane; Lipid-anchor, GPI-anchor</td>
      <td>surface_accessible</td>
      <td>secondary candidate / needs validation</td>
      <td>Strong surface signal but immune/vascular spec...</td>
      <td>possible peptide-design target; requires valid...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>P0CW02</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Ifitm2</td>
      <td>Interferon-induced transmembrane protein 2</td>
      <td>0.621725</td>
      <td>0.196543</td>
      <td>0.425182</td>
      <td>1.661429</td>
      <td>0.009214</td>
      <td>1.067797</td>
      <td>weakly D02-induced</td>
      <td>Cell membrane; Single-pass type II membrane pr...</td>
      <td>surface_accessible</td>
      <td>secondary candidate / needs validation</td>
      <td>Induced membrane protein; similar caution as I...</td>
      <td>possible peptide-design target; requires valid...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>Q99J93</td>
    </tr>
    <tr>
      <th>81</th>
      <td>Cd81</td>
      <td>CD81 antigen</td>
      <td>0.619079</td>
      <td>0.231242</td>
      <td>0.387838</td>
      <td>1.420719</td>
      <td>0.061443</td>
      <td>0.779661</td>
      <td>weakly D02-induced</td>
      <td>Cell membrane; Multi-pass membrane protein | B...</td>
      <td>surface_accessible</td>
      <td>secondary candidate / needs validation</td>
      <td>Surface tetraspanin; targetable but broad expr...</td>
      <td>possible peptide-design target; requires valid...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>P35762</td>
    </tr>
    <tr>
      <th>78</th>
      <td>Pecam1</td>
      <td>Platelet endothelial cell adhesion molecule</td>
      <td>0.777982</td>
      <td>0.377730</td>
      <td>0.400252</td>
      <td>1.042380</td>
      <td>0.025715</td>
      <td>0.711864</td>
      <td>weakly D02-induced</td>
      <td>Cell membrane; Single-pass type I membrane pro...</td>
      <td>surface_accessible</td>
      <td>secondary candidate / needs validation</td>
      <td>Endothelial surface marker; useful control/ref...</td>
      <td>possible peptide-design target; requires valid...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>Q08481</td>
    </tr>
    <tr>
      <th>93</th>
      <td>Ramp2</td>
      <td>Receptor activity-modifying protein 2</td>
      <td>0.902703</td>
      <td>0.558342</td>
      <td>0.344361</td>
      <td>0.693100</td>
      <td>0.209434</td>
      <td>0.559322</td>
      <td>weakly D02-induced</td>
      <td>Cell membrane; Single-pass type I membrane pro...</td>
      <td>surface_accessible</td>
      <td>secondary candidate / needs validation</td>
      <td>Surface receptor-modifying protein; relevant b...</td>
      <td>possible peptide-design target; requires valid...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>Q9WUP0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Igfbp7</td>
      <td>Insulin-like growth factor-binding protein 7</td>
      <td>6.496007</td>
      <td>1.453066</td>
      <td>5.042942</td>
      <td>2.160453</td>
      <td>0.008618</td>
      <td>2.847458</td>
      <td>strong early D02-induced</td>
      <td>Secreted</td>
      <td>secreted_or_extracellular</td>
      <td>stroke EC context marker, not direct receptor</td>
      <td>Strong EC/stroke marker but secreted, therefor...</td>
      <td>useful for defining stroke endothelial state, ...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>Q61581</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sparc</td>
      <td>SPARC</td>
      <td>4.324855</td>
      <td>0.831155</td>
      <td>3.493701</td>
      <td>2.379462</td>
      <td>0.021555</td>
      <td>2.830508</td>
      <td>strong early D02-induced</td>
      <td>Secreted, extracellular space, extracellular m...</td>
      <td>secreted_or_extracellular</td>
      <td>stroke EC context marker, not direct receptor</td>
      <td>Secreted ECM/remodelling marker; useful contex...</td>
      <td>useful for defining stroke endothelial state, ...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>P07214</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Edn1</td>
      <td>Endothelin-1</td>
      <td>0.906098</td>
      <td>0.224776</td>
      <td>0.681322</td>
      <td>2.011173</td>
      <td>0.001177</td>
      <td>1.881356</td>
      <td>moderate early D02-induced</td>
      <td>Secreted</td>
      <td>secreted_or_extracellular</td>
      <td>stroke EC context marker, not direct receptor</td>
      <td>Secreted vasoactive peptide; context marker, n...</td>
      <td>useful for defining stroke endothelial state, ...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>P22387</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Egfl7</td>
      <td>Epidermal growth factor-like protein 7</td>
      <td>1.513354</td>
      <td>0.673415</td>
      <td>0.839940</td>
      <td>1.168182</td>
      <td>0.016951</td>
      <td>1.830508</td>
      <td>moderate early D02-induced</td>
      <td>Secreted, extracellular space</td>
      <td>secreted_or_extracellular</td>
      <td>stroke EC context marker, not direct receptor</td>
      <td>Secreted endothelial/angiogenic context marker...</td>
      <td>useful for defining stroke endothelial state, ...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>Q9QXT5</td>
    </tr>
    <tr>
      <th>70</th>
      <td>Col4a2</td>
      <td>Collagen alpha-2(IV) chain</td>
      <td>0.502416</td>
      <td>0.075367</td>
      <td>0.427049</td>
      <td>2.736857</td>
      <td>0.052475</td>
      <td>1.220339</td>
      <td>weakly D02-induced</td>
      <td>Secreted, extracellular space, extracellular m...</td>
      <td>secreted_or_extracellular</td>
      <td>stroke EC context marker, not direct receptor</td>
      <td>Extracellular matrix marker, not direct shuttl...</td>
      <td>useful for defining stroke endothelial state, ...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>P08122</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Col4a1</td>
      <td>Collagen alpha-1(IV) chain</td>
      <td>0.513086</td>
      <td>0.098889</td>
      <td>0.414197</td>
      <td>2.375311</td>
      <td>0.035601</td>
      <td>1.169492</td>
      <td>weakly D02-induced</td>
      <td>Secreted, extracellular space, extracellular m...</td>
      <td>secreted_or_extracellular</td>
      <td>stroke EC context marker, not direct receptor</td>
      <td>Extracellular matrix marker, not direct shuttl...</td>
      <td>useful for defining stroke endothelial state, ...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>P02463</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Vim</td>
      <td>Vimentin</td>
      <td>1.873383</td>
      <td>0.365958</td>
      <td>1.507425</td>
      <td>2.355893</td>
      <td>0.012264</td>
      <td>2.593220</td>
      <td>strong early D02-induced</td>
      <td>Cytoplasm | Cytoplasm, cytoskeleton | Nucleus ...</td>
      <td>surface_accessible</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>P20152</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Hsp90ab1</td>
      <td>Heat shock protein HSP 90-beta</td>
      <td>2.911258</td>
      <td>1.262513</td>
      <td>1.648745</td>
      <td>1.205344</td>
      <td>0.068693</td>
      <td>2.152542</td>
      <td>strong early D02-induced</td>
      <td>Cytoplasm | Melanosome | Nucleus | Secreted | ...</td>
      <td>surface_accessible</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>P11499</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Eef1a1</td>
      <td>Elongation factor 1-alpha 1</td>
      <td>1.717794</td>
      <td>0.617760</td>
      <td>1.100034</td>
      <td>1.475437</td>
      <td>0.022715</td>
      <td>2.135593</td>
      <td>strong early D02-induced</td>
      <td>Cytoplasm | Nucleus | Nucleus, nucleolus | Cel...</td>
      <td>surface_accessible</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>P10126</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Hspa5</td>
      <td>Endoplasmic reticulum chaperone BiP</td>
      <td>0.918031</td>
      <td>0.220700</td>
      <td>0.697331</td>
      <td>2.056454</td>
      <td>0.012144</td>
      <td>1.983051</td>
      <td>moderate early D02-induced</td>
      <td>Endoplasmic reticulum lumen | Melanosome | Cyt...</td>
      <td>surface_accessible</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>P20029</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Glul</td>
      <td>Glutamine synthetase</td>
      <td>1.146988</td>
      <td>0.462845</td>
      <td>0.684143</td>
      <td>1.309247</td>
      <td>0.034632</td>
      <td>1.728814</td>
      <td>moderate early D02-induced</td>
      <td>Cytoplasm, cytosol | Microsome | Mitochondrion...</td>
      <td>surface_accessible</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>P15105</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Calr</td>
      <td>Calreticulin</td>
      <td>0.833457</td>
      <td>0.237697</td>
      <td>0.595760</td>
      <td>1.809977</td>
      <td>0.059921</td>
      <td>1.661017</td>
      <td>moderate early D02-induced</td>
      <td>Endoplasmic reticulum lumen | Cytoplasm, cytos...</td>
      <td>surface_accessible</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>P14211</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Rack1</td>
      <td>Small ribosomal subunit protein RACK1</td>
      <td>0.753514</td>
      <td>0.221606</td>
      <td>0.531909</td>
      <td>1.765634</td>
      <td>0.020701</td>
      <td>1.457627</td>
      <td>moderate early D02-induced</td>
      <td>Cell membrane; Peripheral membrane protein | C...</td>
      <td>surface_accessible</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>P68040</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Msn</td>
      <td>Moesin</td>
      <td>0.719441</td>
      <td>0.195917</td>
      <td>0.523524</td>
      <td>1.876627</td>
      <td>0.011529</td>
      <td>1.423729</td>
      <td>moderate early D02-induced</td>
      <td>Cell membrane; Peripheral membrane protein; Cy...</td>
      <td>surface_accessible</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>P26041</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Hspa8</td>
      <td>Heat shock cognate 71 kDa protein</td>
      <td>0.975532</td>
      <td>0.439680</td>
      <td>0.535852</td>
      <td>1.149733</td>
      <td>0.017574</td>
      <td>1.305085</td>
      <td>moderate early D02-induced</td>
      <td>Cytoplasm | Melanosome | Nucleus, nucleolus | ...</td>
      <td>surface_accessible</td>
      <td>reject despite surface annotation</td>
      <td></td>
      <td>not prioritised for peptide design because pri...</td>
      <td>gene-level transcriptomic candidate with UniPr...</td>
      <td>P63017</td>
    </tr>
  </tbody>
</table>
</div>



```python
# -------------------------
# 10.2 Save final report table
# -------------------------

final_report_path = out_dir / "FINAL_gene_level_EC_peptide_target_candidates_with_caveats.csv"

final_report_table[final_report_columns].to_csv(
    final_report_path,
    index=False,
)

print("Saved final gene-level EC peptide-target candidate report:", final_report_path)
```

    Saved final gene-level EC peptide-target candidate report: ../data/processed/FINAL_gene_level_EC_peptide_target_candidates_with_caveats.csv


Primary candidates identified in this run included `Ly6a`, `Tm4sf1`, `Ly6e`, `Ecscr`, `Cd200`, `Itgb1`, `Esam`, and `Fxyd5`. Secondary and context candidates were retained for interpretation and future expansion.

## Final note

This notebook exports the receptor-candidate shortlist used by the downstream peptide-design notebooks. The prioritised candidates are gene-level hypotheses, not validated shuttle receptors.

The final report table is saved as:

`../data/processed/FINAL_gene_level_EC_peptide_target_candidates_with_caveats.csv`
