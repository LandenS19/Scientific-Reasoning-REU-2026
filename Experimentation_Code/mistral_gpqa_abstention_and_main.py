import sys
import time
import random
import pandas as pd
from pathlib import Path
from mistralai import Mistral

sys.path.append(str(Path(__file__).resolve().parents[1]))

import mistralKeys


MODEL_NAME = "mistral-large-latest"


def build_prompt(question, options):
    prompt = f"""You are answering a difficult multiple-choice science question.

Choose the single best answer from the options below.

Return only the number of the correct option: 1, 2, 3, 4, or 5.

Question:
{question}

Options:
1. {options[0]}
2. {options[1]}
3. {options[2]}
4. {options[3]}
5. {options[4]}
"""
    return prompt


def extract_answer_number(response_text):
    response_text = str(response_text).strip()

    for char in response_text:
        if char in ["1", "2", "3", "4", "5"]:
            return char

    return ""


def call_mistral(client, prompt, max_retries=6):
    for attempt in range(max_retries):
        try:
            response = client.chat.complete(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            return response.choices[0].message.content

        except Exception as e:
            wait_time = min(60, 2 ** attempt)
            print(f"Mistral API error: {e}")
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    return "ERROR"


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 mistral_gpqa_main_plus_abstention.py <input_csv> <start_index> <end_index>")
        sys.exit(1)

    input_file = sys.argv[1]
    start_index = int(sys.argv[2])
    end_index = int(sys.argv[3])

    df = pd.read_csv(input_file)

    client = Mistral(api_key=mistralKeys.key)

    results = []

    for i in range(start_index, min(end_index, len(df))):
        row = df.iloc[i]

        question = row["Question"]
        correct_answer = row["Correct Answer"]
        abstention_option = row["Abstention Option"]

        original_options = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
            row["Abstention Option"]
        ]

        # Shuffle options so correct/abstention positions are not predictable
        shuffled_options = original_options.copy()
        random.seed(i)
        random.shuffle(shuffled_options)

        correct_option_number = shuffled_options.index(correct_answer) + 1
        abstention_option_number = shuffled_options.index(abstention_option) + 1

        prompt = build_prompt(question, shuffled_options)

        print(f"Running row {i}...")
        raw_model_answer = call_mistral(client, prompt)
        model_answer = extract_answer_number(raw_model_answer)

        is_correct = model_answer == str(correct_option_number)
        model_chose_abstention = model_answer == str(abstention_option_number)

        results.append({
            "row_index": i,
            "question": question,
            "option_1": shuffled_options[0],
            "option_2": shuffled_options[1],
            "option_3": shuffled_options[2],
            "option_4": shuffled_options[3],
            "option_5": shuffled_options[4],
            "correct_answer": correct_answer,
            "correct_option": correct_option_number,
            "abstention_option": abstention_option,
            "abstention_option_number": abstention_option_number,
            "model_answer": model_answer,
            "raw_model_answer": raw_model_answer,
            "is_correct": is_correct,
            "model_chose_abstention": model_chose_abstention,
            "source_file": Path(input_file).name
        })

        time.sleep(5)

    input_stem = Path(input_file).stem
    output_file = f"Results/mistral_results_{input_stem}_{start_index}_{end_index - 1}.csv"

    Path("Results").mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(output_file, index=False)

    print(f"Saved results to {output_file}")

    if results:
        correct_count = sum(r["is_correct"] for r in results)
        abstention_count = sum(r["model_chose_abstention"] for r in results)
        total = len(results)

        print(f"Accuracy: {correct_count}/{total} = {correct_count / total:.2%}")
        print(f"Over-abstention rate: {abstention_count}/{total} = {abstention_count / total:.2%}")


if __name__ == "__main__":
    main()