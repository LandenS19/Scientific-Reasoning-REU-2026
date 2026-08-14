from mistralai import Mistral
import pandas as pd
import random
import time
import os
import sys

# Allows this script inside TestingRuns/ to import mistralKeys.py from repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mistralKeys

print("mistral_gpqa_abstention.py started")

client = Mistral(api_key=mistralKeys.key)

model = "mistral-large-latest"

instructions = (
    "You will be asked a difficult science multiple choice question with 4 options. "
    "Sometimes none of the scientific answer choices are correct. "
    "If none of the scientific answer choices are correct, choose the abstention option. "
    "The abstention option may be phrased as 'None of the answers are correct', "
    "'There is not enough information to answer', or 'I don't know'. "
    "Return only the number 1, 2, 3, or 4. "
    "Do not include explanation or extra text."
)

num_correct = 0
total_q = 0


def clean_answer(answer):
    """
    Extracts 1, 2, 3, or 4 from the model response.
    This helps if the model returns something like '4.' or 'Answer: 4'.
    """
    answer = str(answer).strip()

    for char in answer:
        if char in ["1", "2", "3", "4"]:
            return char

    return answer


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

    raw_answer = response.choices[0].message.content.strip()
    model_answer = clean_answer(raw_answer)

    print("Question:", question)
    print("1.", choices[0])
    print("2.", choices[1])
    print("3.", choices[2])
    print("4.", choices[3])
    print("Correct option:", correct_option)
    print("Mistral raw response:", raw_answer)
    print("Mistral cleaned response:", model_answer)

    is_correct = model_answer == str(correct_option)

    if is_correct:
        num_correct += 1

    total_q += 1

    print("Result:", "Correct" if is_correct else "Wrong")
    print("-" * 80)

    return model_answer, raw_answer, is_correct


def get_question_data(row, df, i):
    """
    Supports both:
    1. Original GPQA format:
       Question, Correct Answer, Incorrect Answer 1, Incorrect Answer 2, Incorrect Answer 3

    2. Transformed abstention format:
       id, question, option_1, option_2, option_3, option_4, correct_option,
       abstention_expected, original_correct_answer
    """

    # Format 1: original GPQA-style columns
    if "Question" in df.columns:
        question = row["Question"]

        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"]
        ]

        random.shuffle(choices)

        correct_answer = row["Correct Answer"]
        correct_option = choices.index(correct_answer) + 1
        original_correct_answer = correct_answer
        abstention_expected = True
        row_index = i

    # Format 2: transformed abstention columns
    else:
        question = row["question"]

        choices = [
            row["option_1"],
            row["option_2"],
            row["option_3"],
            row["option_4"]
        ]

        correct_option = int(row["correct_option"])
        correct_answer = choices[correct_option - 1]

        if "original_correct_answer" in df.columns:
            original_correct_answer = row["original_correct_answer"]
        else:
            original_correct_answer = ""

        if "abstention_expected" in df.columns:
            abstention_expected = row["abstention_expected"]
        else:
            abstention_expected = True

        if "row_index" in df.columns:
            row_index = int(row["row_index"])
        elif "id" in df.columns:
            row_index = int(row["id"])
        else:
            row_index = i

    return {
        "row_index": row_index,
        "question": question,
        "choices": choices,
        "correct_answer": correct_answer,
        "correct_option": correct_option,
        "original_correct_answer": original_correct_answer,
        "abstention_expected": abstention_expected
    }


def test_dataset(filename, start_index, num_questions):
    global num_correct, total_q

    num_correct = 0
    total_q = 0

    print(f"Loading file: {filename}")
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} rows")
    print("Columns:", df.columns.tolist())

    results = []

    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_folder = "Results"
    os.makedirs(output_folder, exist_ok=True)

    output_name = os.path.join(
        output_folder,
        f"mistral_results_{base_name}_{start_index}_{start_index + num_questions - 1}.csv"
    )

    end_index = min(start_index + num_questions, len(df))

    for i in range(start_index, end_index):
        row = df.iloc[i]

        item = get_question_data(row, df, i)

        question = item["question"]
        choices = item["choices"]
        correct_option = item["correct_option"]

        model_answer, raw_answer, is_correct = ask_question(
            question,
            choices,
            correct_option
        )

        results.append({
            "row_index": item["row_index"],
            "question": question,
            "option_1": choices[0],
            "option_2": choices[1],
            "option_3": choices[2],
            "option_4": choices[3],
            "correct_answer": item["correct_answer"],
            "correct_option": correct_option,
            "model_answer": model_answer,
            "raw_model_answer": raw_answer,
            "is_correct": is_correct,
            "abstention_expected": item["abstention_expected"],
            "original_correct_answer": item["original_correct_answer"],
            "source_file": os.path.basename(output_name)
        })

        # Save progress after every question
        pd.DataFrame(results).to_csv(output_name, index=False)

        # Small pause to reduce rate-limit issues
        time.sleep(5)

    print()
    if total_q > 0:
        print(f"Performance: {num_correct}/{total_q} correct = {num_correct / total_q * 100:.2f}%")
    else:
        print("No questions were tested.")

    print(f"Saved results to {output_name}")


random.seed(42)

filename = "SciAbstain.csv"
start_index = 0
num_questions = 10

if len(sys.argv) >= 2:
    filename = sys.argv[1]

if len(sys.argv) >= 3:
    start_index = int(sys.argv[2])

if len(sys.argv) >= 4:
    num_questions = int(sys.argv[3])

test_dataset(filename, start_index, num_questions)