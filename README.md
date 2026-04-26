# Evaluating the Utility and Privacy of Synthetic Data for Mental Health Clinical NLP Models: A Comparative Study of Prompting Strategies and Model Architecture
## Abstract

**Background:** The limited availability of high-quality, de-identified patient data and inadequate anonymisation techniques restricts the development and distribution of advanced Natural Language Processing (NLP) models in mental health research, posing challenges for integrating Large Language Models (LLMs).
 
**Aims:** This study assessed the utility-privacy trade-off of using LLM-generated synthetic social media posts as a privacy-preserving alternative to real-world data for training clinical NLP classifiers to predict mental health severity. Specifically focusing on how the prompting strategy and model architecture influence data utility, realism, and privacy risks.

**Methods:** Using a “plug-and-play” pipeline, two models (Falcon3-7B-Instruct and MentaLLaMA-chat-7B) generated 40,000 synthetic social media posts using zero-shot and few-shot prompting. Utility was evaluated by training two downstream classifiers (DistilBERT and MentalBERT) against a real-world baseline. Linguistic realism and privacy risk were assessed using an automated LLM-as-a-judge framework (Gemini 3 Flash).

**Results:** Few-shot prompting significantly improved data utility, doubling F1-scores compared to zero-shot generation. Under few-shot conditions, the domain-specific MentaLLaMA model achieved the highest performance, though results remained below the real-world benchmark. The domain-specific classifier (MentalBERT) consistently outperformed the general classifier (DistilBERT). Privacy evaluation identified only one instance of personally identifiable information across 2,000 sampled posts, solely in the MentaLLaMA model, indicating effective data leakage mitigation. However, under few-shot conditions, MentaLLaMA frequently produced incomplete sentences (49.6%), whereas Falcon maintained high structural completeness (99.8%).

**Conclusion:** Synthetic data offers a viable, privacy-preserving approach for mental health NLP by minimising exposure to sensitive information. An optimal strategy involves a hybrid pipeline: few-shot prompting with a general-purpose model for structural realism, followed by a domain-specific classifier to maximise semantic utility. Future research should expand few-shot demonstrations to narrow the real-world performance gap and confirm stability through k-fold cross-validation.

## Data & Project Overview:
This project utilises the Depression Severity Levels (DSL) Dataset, compiled by Priyadarshana, Y. H. P. P et al. (2023) (https://github.com/KUAS-ubicomp-lab/Depression_Severity_Levels_Dataset), as the primary real-world baseline.

Synthetic data will be generated using two distinct Large Language Models (LLMs): Falcon (a general-purpose model) and MentaLLaMA (a domain-specific model fine-tuned for mental health). 

These datasets would be evaluated using 2 different classification models: DistillBERT (a general-purpose model) and MentalBERT (a domain-specific model fine-tuned for mental health). 

Realism would be evaluated on the ability of the synthetic data to create complete sentences.

Privacy would be evaluated on the presence of Personal Identifiable Information (PII) in the synthetic data.

Finally, the performance of the models would be compared to determine the effectiveness of synthetic data in identifying mental health trends.

### Synthetic Datasets
The different models were successful in creating synthetic data for the different depression severity levels.

**Synthetic data created at temperature = 0.9:**
Depression Severity Level | Falcon FS | Falcon ZS | MentaLLaMA FS | MentaLLaMA ZS
--- | --- | --- | --- | ---
Minimum | 9,458 | 9,812 | 9,975 | 10,000
Mild | 9,399 | 9,810 | 9,880 | 10,000
Moderate | 9,467 | 9,796 | 9,992 | 10,000
Severe | 9,309 | 9,868 | 9,905 | 10,000

**Synthetic data results for ZS performance using the best model combination at each temperature:**
Generation Model | Temperature | Classifier | Macro F1-Score | ROC AUC
--- | --- | --- | --- | ---
MentaLLaMA | 0.9 | MentalBERT | 0.2820 | 0.6166
MentaLLaMA | 1.0 | MentalBERT | 0.2506 | 0.5339
MentaLLaMA | 0.8 | MentalBERT | 0.2361 | 0.5584
MentaLLaMA | 0.7 | MentalBERT | 0.2189 | 0.5695
MentaLLaMA | 0.5 | MentalBERT | 0.2155 | 0.5290
MentaLLaMA | 0.6 | MentalBERT | 0.2092 | 0.5730

**Synthetic data results for FS performance using the best model combination at each temperature:**
Generation Model | Temperature | Classifier | Macro F1-Score | ROC AUC
--- | --- | --- | --- | ---
MentaLLaMA | 0.9 | MentalBERT | 0.6449 | 0.8842
MentaLLaMA | 0.8 | MentalBERT | 0.6287 | 0.8810
MentaLLaMA | 1.0 | MentalBERT | 0.6277 | 0.8805
MentaLLaMA | 0.5 | MentalBERT | 0.6257 | 0.8742
MentaLLaMA | 0.6 | MentalBERT | 0.6212 | 0.8772
MentaLLaMA | 0.7 | MentalBERT | 0.6161 | 0.8780

### Realism and Privacy Evaluation
An automated LLM-as-a-Judge evaluation architecture was used (using gemini-flash-3-preview) for 500 random samples from the threshold 0.9 datasets.

Generation Model | Prompting Method | Complete Sentences | Incomplete Sentences | PII Detected
--- | --- | --- | --- | ---
Falcon | ZS | 432 | 68 | 0
Falcon | FS | 499 | 1 | 0
MentaLLaMA | ZS | 493 | 7 | 0
MentaLLaMA | FS | 252 | 248 | 1

# References
* Priyadarshana, Y. H. P. P, Liang, Z., & Piumarta, I. (2023). HelaDepDet: A novel multi-class classification model for detecting the severity of human depression (H. Takada, M. D. Moritz, C. Alvarez, T. Inoue, Y. Hayashi, & D. Hernandez-Leo, Eds.; pp. 3–18). Springer Nature Switzerland. https://link.springer.com/chapter/10.1007/978-3-031-42141-9_1