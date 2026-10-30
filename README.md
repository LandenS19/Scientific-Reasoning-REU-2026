<h1 align="center">Exploring the Abstention Ability of LLMs<br>using Scientific Reasoning in Critical Fields</h1>

<p align="center">
  <b>Landen Summers<sup>1</sup> · Haeseon Yoon<sup>2</sup> · Weijian Zheng<sup>3</sup> · Yunhe Feng<sup>4</sup></b>
</p>

<p align="center">
  <sup>1</sup>University of Texas Arlington · <sup>2</sup>University of Michigan · <sup>3</sup>Argonne National Laboratory · <sup>4</sup>University of North Texas
</p>

This repository contains the code and results for our JCDL '26 paper, which systematically investigates whether large language models (LLMs) can **abstain** when a scientific multiple-choice question has no valid answer among its options.

## Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Experimental Design](#experimental-design)
- [Selected Results](#selected-results)
- [Prompt-Based Scoring Mitigation](#prompt-based-scoring-mitigation)
- [Using This Repository](#using-this-repository)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Overview

LLMs can produce fluent and confident responses even when those responses lack sufficient support or contain factual errors. Reliable question answering therefore requires models not only to generate answers, but also to recognize when they cannot provide a well-supported response and abstain accordingly. This matters most in scientific domains, where inaccurate responses can cause real-world harm.

We evaluate several LLMs on modified versions of the Graduate-Level Google-Proof Q&A (GPQA) benchmark[^1], varying both the availability of the correct answer and the wording of the abstention option. The study measures answer accuracy, failure to abstain when no correct answer is available, and over-abstention when a correct answer is present.
## Key Findings

- **Models often fail to abstain** when the correct answer is absent; performance drops substantially across nearly all models once the correct option is removed.
- **Abstention is sensitive to wording.** Every model abstained less with "I Don't Know" (IDK) than with "None of the Answers are Correct" (NAC). For example, Gemini 3.1 Pro reaches 75.22% on NAC but only 16.52% on IDK.
- **Accuracy does not imply reliable abstention.** Strong baseline accuracy on the original GPQA did not always translate into reliable abstention.
- **Scale and reasoning help, but do not solve the problem.** Larger Mistral models abstain better when the answer is missing (NAC: 40.2% → 71.7% → 83.0% for Small → Medium → Large), and enabling reasoning mode lifts Gemini 3.1 Flash Lite accuracy substantially, yet its IDK performance only moves from 1.3% to 2.7%.
- **Prompt-based scoring shifts behavior.** Giving abstention the same reward as a correct answer raised IDK performance most (Gemini 3.1 Pro: 16.52% → 45.54% with Scoring 6), with limited over-abstention on ADD2.

## Experimental Design

Five dataset variants are derived from GPQA, preserving the original question content so that differences in behavior reflect only the answer-choice setting.

| Variant | Construction | What it tests |
|---|---|---|
| **Main** | Original GPQA | Baseline accuracy |
| **NAC** | Correct answer replaced with "None of the answers are correct" | Correct abstention (answer absent) |
| **IDK** | Correct answer replaced with "I do not know" | Correct abstention (answer absent; wording effect) |
| **ADD1** | Correct answer kept; "None of the answers are correct" added | False abstention (answer present) |
| **ADD2** | Correct answer kept; "I do not know" added | False abstention (answer present) |

**Prompting.** Zero-shot, one question per independent request; the system prompt instructs the model to return only the selected answer choice, without an explanation.

**Models.** GPT-5.4, Gemini 3.1 Pro, Gemini 3.1 Flash Lite (standard and high reasoning), Llama 4 Maverick, GPT-OSS, and Mistral Small 4, Mistral Medium 3.5 and Mistral Large 3.

**Metrics.**
- *Accuracy* when the correct answer is available (Main, ADD1, ADD2).
- *Correct abstention rate* when the correct answer is absent (NAC, IDK): the proportion of questions where the model selects the abstention option.
- *False abstention rate* on ADD1 and ADD2: the proportion of questions where the model abstains despite the correct answer being present.

## Selected Results

Accuracy (%) of each model across dataset variants (Table 2 of the paper). Main, ADD1 and ADD2 report answer accuracy; NAC and IDK report correct-abstention rate.

| Model | Main | NAC | IDK | ADD1 | ADD2 |
|---|---:|---:|---:|---:|---:|
| Gemini Flash Lite | 49.55 | 16.29 | 1.34 | 34.38 | 51.34 |
| Gemini Flash Lite Reasoning | 79.24 | 43.75 | 2.68 | 83.04 | 80.13 |
| Gemini Pro | **89.96** | **75.22** | 16.52 | **92.19** | **90.85** |
| GPT-5.4 | 48.21 | 25.00 | 3.79 | 58.04 | 49.11 |
| GPT-OSS | 63.39 | 42.63 | 11.16 | 72.77 | 61.61 |
| Llama 4 Maverick | 43.30 | 8.04 | 1.12 | 39.51 | 37.50 |
| Mistral Small | 40.18 | 40.18 | **19.87** | 31.47 | 35.49 |

## Prompt-Based Scoring Mitigation

We test a prompt-only mitigation that states point values for each outcome, leaving the questions, answer choices and model parameters unchanged. The correct answer is always worth 1 point.

| Scoring version | Correct | Abstention | Incorrect |
|:---:|:---:|:---:|:---:|
| 1 | 1 | 0 | −1 |
| 2 | 1 | 0.5 | −1 |
| 3 | 1 | 1 | −1 |
| 4 | 1 | 0 | −10 |
| 5 | 1 | 0.5 | −10 |
| 6 | 1 | 1 | −10 |

Gemini 3.1 Flash Lite on IDK (abstention %), baseline vs. Scoring 1–6 (Table 3):

| Baseline | S1 | S2 | S3 | S4 | S5 | S6 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1.34 | 3.12 | 6.92 | 9.60 | 4.02 | 8.26 | 10.94 |

## Using This Repository

This repository contains the code used to run the experiments and the results reported in the paper.
1. Clone the repository

   ```bash
   # Clone the Repository
   git clone https://github.com/LandenS19/Scientific-Reasoning-REU-2026.git
   cd Scientific-Reasoning-REU-2026
    ```

2. Install the requirements

   From the repository root, install the dependencies listed in requirements.txt:

   (Setting up a virtual environment is recommended)
    ```bash
    # Install the requirements
    pip install -r requirements.txt
    ```

3. Set up API key access

   Create a new file under Experimentation_Code named apiKeys.py and add your API Keys for desired models. In the model file being run, connect the API key. The geminiRuns.py file will be used to display examples.
   ```bash
   # Set up for gemeniRuns.py
   import apiKeys
   client = genai.Client(api_key=apiKeys.yourAPIKey)
   ```
4. Choose Datatest for Testing

   At the bottom of the geminiRuns.py file, modify the read_csv function call by inputting the desired dataset file and range of questions.
   ```bash
   # read_csv(start question, end question, dataset file)
   read_csv(0, 448, "DataSets/gpqa_main.csv")
   # This covers all questions in the main dataset.
   ```
   The ADD1 and ADD2 datasets contain 5 answer options that require some parts of the code to be modified.
   * Changes
       ```bash
       # Prompt for 5 questions
       # Line 18
       instructions = "You will be asked a question with 5 multiple choice options. Please answer the question by returning 1, 2, 3, 4, or 5 based on the correct answer. "
       "Do not include any explanation or additional text, just return the number corresponding to the correct answer. Take your time and think carefully before responding. Do not include your thinking process in your resonse."
       "Any response that is not a single number will be considered incorrect. Any incorrect answers will have severely negative impacts and should be avoided at all costs. Please provide the correct answer to the following question: "
      
       # Line 71
       def ask_question(question, op1, op2, op3, op4, op5, cop, qnum):
       # Line 74
       prompt = f"Question: {question}\nOptions:\n1. {op1}\n2. {op2}\n3. {op3}\n4. {op4}\n5. {op5}\nAnswer:"

       # Line 107
       if(response.text.strip() not in ['1', '2', '3', '4', '5']):

       # Line 117
       # This counts correct answers for the correct answer and for abstaining
       if((response.text.strip() == str(cop[0])) or (response.text.strip() == str(cop[1]))):

       # Line 129
       # Gathers the correct columns from the datasets csv file
       df_subset = pd.read_csv(file_path, usecols=['Question','Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3', 'Abstention Option'])

       # Line 145
       ask_question(row['Question'], row['Correct Answer'], row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3'], row['Abstention Option'], [1,5], i+1)
       ```
  5. Prompt-Based Scoring Version
  
     The only change made when doing this version was the instructions given
     ```Bash
     instructions = "You will be asked a question with 4 multiple choice options. Please answer the question by returning 1, 2, 3, or 4. " \
     "Do not include any explanation or additional text, just return the number corresponding to the answer. Take your time and think carefully before responding. Do not include your thinking process in your resonse." \
     "Any response that is not a single number will be considered incorrect. You will be graded on a point system where correct answers are worth 1 points, \"I don't know\" answers are worth 0 points and wrong answers are worth -1 points."\
     "You should attempt to maximize your score and minimize incorrect answers. Please provide your answer to the following question: " 
     ```

  6. Running the Program

     Run the file in the terminal using:
     ```bash
     python geminiRuns.py
     ```
     
## Citation

If you use this work, please cite:

**Plain text (ACM format)**

> Landen Summers, Haeseon Yoon, Weijian Zheng, and Yunhe Feng. 2026. Exploring the Abstention Ability of LLMs using Scientific Reasoning in Critical Fields. In Proceedings of the 26th ACM/IEEE Joint Conference on Digital Libraries (JCDL '26). Association for Computing Machinery, New York, NY, USA, Article 135, 1–6. https://doi.org/10.1145/3805696.3847298

**BibTeX**

```bibtex
@inproceedings{10.1145/3805696.3847298,
author = {Summers, Landen and Yoon, Haeseon and Zheng, Weijian and Feng, Yunhe},
title = {Exploring the Abstention Ability of LLMs using Scientific Reasoning in Critical Fields},
year = {2026},
isbn = {9798400725975},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3805696.3847298},
doi = {10.1145/3805696.3847298},
abstract = {Large Language Models (LLMs) can produce fluent and confident responses even when those responses lack sufficient support or contain factual errors. Reliable question answering therefore requires models not only to generate answers, but also to recognize when they cannot provide a well-supported response and abstain accordingly. This capability is particularly important in scientific domains, where inaccurate responses can cause real-world harm. This paper systematically investigates the abstention behavior of LLMs in challenging scientific multiple-choice answering tasks. We evaluate several LLMs on multiple modified versions of a scientific multiple-choice question answering dataset, varying both the availability of the correct answer and the wording of the abstention option. Our findings demonstrate that LLMs often struggle to abstain consistently when the correct answer is absent, and that their abstention behavior is sensitive to model family, abstention wording, and reasoning settings.},
booktitle = {Proceedings of the 26th ACM/IEEE Joint Conference on Digital Libraries},
articleno = {135},
numpages = {6},
keywords = {Large Language Models, LLM Abstention, Scientific Reasoning, Uncertainty Estimation, Reliable Question Answering},
location = {Frisco, USA},
series = {JCDL '26}
}
```

## Acknowledgments

This work was supported in part by the U.S. National Science Foundation (NSF) under Grant No. CCF-2447834. We also thank the University of North Texas AI Summer Research Program for research training and support.

[^1]: Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., ... & Bowman, S. R. (2023). Gpqa: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022.
