# DMIS at BioCreative IX MedHopQA Track

This repository contains the code implementation of our methodology for generating answers to multi-step reasoning questions in the healthcare and biomedical domains, as introduced in the BioCreative IX MedhopQA Challenge.

## Overview & Quick Links

* **Task description**: The MedHopQA task requires the generation of answers to multi-step reasoning questions in the healthcare and biomedical domains. For detailed information, please visit to the official BC9 website (**[link](https://www.ncbi.nlm.nih.gov/research/bionlp/medhopqa)**) or see the overview paper (**[PDF](https://zenodo.org/records/16992154)**).
* **System description**: Please check out our paper listed below.
  * **[This paper](https://zenodo.org/records/16875789)** provides a brief description of our system.

## Requirements
This repository has been tested on `Python==3.10`, `PyTorch==2.10`, and `Transformers==4.57.6`. See the `requirements.txt` file and instructions below.

```bash
# Download this project
git clone https://github.com/dmis-lab/bc9-medhop-qa.git
cd bc9-medhop-qa

# Create a conda environment
conda create -n bc9 python=3.10
conda activate bc9

# Install all requirements
pip install -r requirements.txt
```

## File Structure

```bash
bc9-medhop-qa
├── core
│   ├── llm
│   │   ├── gemini.py
│   │   ├── gpt_4o.py
│   │   ├── gpt_5_2.py
│   │   └── o3.py
│   ├── medhop_dataset.py
│   ├── reranker
│   │   ├── medcpt_reranker.py
│   │   ├── qwen3_reranker.py
│   │   └── reranker_base.py
│   └── retriever
│       ├── dense_wikipedia.py
│       ├── elastic_wikipedia.py
│       └── retriever_base.py
├── data
│   └── MedHopQA_Test_Dataset.csv
├── decision_making.py
├── output
│   ├── result_decision_making.csv
│   ├── result_q2d.csv
│   ├── result_rag.csv
│   ├── result_rationale.csv
│   └── result_web_search.csv
├── prompt
│   ├── decision_making.py
│   ├── query2doc.py
│   ├── rationale.py
│   └── web_search.py
├── query2doc.py
├── rationale.py
├── README.md
├── utils
│   ├── merge_csvs.py
│   └── normalize_long_answer.py
└── web_search.py
```

## Excecution

```bash
# 1. Run Query2Doc
python query2doc.py

# 2. Run Rationale
python ratinoale.py

# 3. Run Web Search
python web_search.py

# 4. Merge outputs
python utils/merge_csvs.py

# 5. Run decision-making
python decision-making.py

```

* Note: The Elasticsearch server and Wikipedia embeddings required for retrieval must be configured independently.

## References

Please cite the papers below if you use our code, method, or if your work is inspired by ours.

```bash
@inproceedings{jung2025dmis,
  author    = {Jung, J. and Hwang, H. and Park, Y. and Song, M. and Yoon, J. and Hwang, H. and Lee, S. and Sohn, J. and Kang, J.},
  title     = {DMIS Lab at MedHopQA-2025: Ensemble Multi-Retrieval Methodologies with Reasoning Language Model Decision},
  booktitle = {Proceedings of the BioCreative IX Challenge and Workshop (BC9): Large Language Models for Clinical and Biomedical NLP at the International Joint Conference on Artificial Intelligence (IJCAI)},
  year      = {2025},
  doi       = {10.5281/zenodo.16875789},
  url       = {https://doi.org/10.5281/zenodo.16875789}
}
```

Also, it should be noted that appropriate references must be cited when using the MedHopQA dataset or citing BC9 challenge results, etc.
