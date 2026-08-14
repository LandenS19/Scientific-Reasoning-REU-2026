from mistralai import Mistral
import pandas as pd
import random
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mistralKeys

print("mistral_gpqa.py started")

client = Mistral(api_key=mistralKeys.key)

model = "mistral-large-latest"

instructions = (
    "You will be asked a difficult science multiple choice question with 4 options. "
    "Return only the number 1, 2, 3, or 4 corresponding to the correct answer. "
    "Do not include explanation or extra text."
)

num_correct = 0
total_q = 0


def ask_question(question, choices, correct_option):
    global num_correct, total_q

    user_prompt = (
        f"Question: {question}\n"
        f"Options:\n"
        f"1. {choices[0]}\n"
        f"2. {choices[1]}\n"
        f"3. {choices[2]}\n"
        f"4. {choices[3]}\n"
        f"Answer:"
    )

    while True:
        try:
            response = client.chat.complete(
                model=model,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user_prompt}
                ],
            )
            break

        except Exception as e:
            error_message = str(e)

            if "429" in error_message or "rate" in error_message.lower():
                print()
                print("Rate limit hit. Waiting 60 seconds, then retrying the same question...")
                time.sleep(60)
                print("Retrying now...")
                print()
            elif "503" in error_message or "unavailable" in error_message.lower():
                print()
                print("Mistral service unavailable. Waiting 30 seconds, then retrying the same question...")
                time.sleep(30)
                print("Retrying now...")
                print()
            else:
                raise e

    model_answer = response.choices[0].message.content.strip()

    print("Question:", question)
    print("1.", choices[0])
    print("2.", choices[1])
    print("3.", choices[2])
    print("4.", choices[3])
    print("Correct option:", correct_option)
    print("Mistral response:", model_answer)

    is_correct = model_answer == str(correct_option)

    if is_correct:
        num_correct += 1

    total_q += 1

    print("Result:", "Correct" if is_correct else "Wrong")
    print("-" * 80)

    return model_answer, is_correct


def test_dataset(filename, start_index, num_questions):
    global num_correct, total_q

    num_correct = 0
    total_q = 0

    print(f"Loading file: {filename}")
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} rows")

    results = []

    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_folder = "Results"
    os.makedirs(output_folder, exist_ok=True)

    output_name = os.path.join(
        output_folder,
        f"mistral_results_{base_name}_{start_index}_{start_index + num_questions - 1}.csv"
    )

    for i in range(start_index, start_index + num_questions):
        row = df.iloc[i]

        question = row["Question"]

        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"]
        ]

        random.shuffle(choices)

        correct_option = choices.index(row["Correct Answer"]) + 1

        model_answer, is_correct = ask_question(question, choices, correct_option)

        results.append({
            "row_index": i,
            "question": question,
            "option_1": choices[0],
            "option_2": choices[1],
            "option_3": choices[2],
            "option_4": choices[3],
            "correct_answer": row["Correct Answer"],
            "correct_option": correct_option,
            "model_answer": model_answer,
            "is_correct": is_correct
        })

        # Save progress after every question
        pd.DataFrame(results).to_csv(output_name, index=False)

        # Small pause to reduce rate-limit issues
        time.sleep(5)

    print()
    print(f"Performance: {num_correct}/{total_q} correct = {num_correct / total_q * 100:.2f}%")
    print(f"Saved results to {output_name}")


random.seed(42)

filename = "gpqa_main.csv"
start_index = 0
num_questions = 10

if len(sys.argv) >= 2:
    filename = sys.argv[1]

if len(sys.argv) >= 3:
    start_index = int(sys.argv[2])

if len(sys.argv) >= 4:
    num_questions = int(sys.argv[3])

test_dataset(filename, start_index, num_questions)