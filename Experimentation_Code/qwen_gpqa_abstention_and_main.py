from openai import OpenAI
import pandas as pd
import random
import time
import os
import sys
import re

# Allows this script inside TestingRuns/ to import qwenKeys.py from repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qwenKeys

print("qwen_gpqa_abstention_and_main.py started")

client = OpenAI(
    api_key=qwenKeys.key,
    base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1"
)

model = "qwen3-8b"
thinking_mode = False

instructions = (
    "You will be asked a difficult science multiple choice question with 5 options. "
    "One option is the real scientific answer. "
    "One option is an abstention option such as 'None of the answers are correct' or 'I don't know'. "
    "Because the real scientific answer is present, you should choose the abstention option only if it is truly the best answer. "
    "Return only the number 1, 2, 3, 4, or 5. "
    "Do not include explanation or extra text."
)

num_correct = 0
total_q = 0
num_abstentions = 0


def clean_answer(answer):
    """
    Extracts 1, 2, 3, 4, or 5 from the model response.
    Works best when max_tokens is low and model follows instructions.
    """
    answer = str(answer).strip()

    # Best case: model returns "4", "4.", "Answer: 4", etc.
    match = re.search(r"\b([1-5])\b", answer)
    if match:
        return match.group(1)

    return ""


def ask_question(question, choices, correct_option, abstention_option_number):
    global num_correct, total_q, num_abstentions

    user_prompt = (
        f"Question: {question}\n"
        f"Options:\n"
        f"1. {choices[0]}\n"
        f"2. {choices[1]}\n"
        f"3. {choices[2]}\n"
        f"4. {choices[3]}\n"
        f"5. {choices[4]}\n"
        f"Answer:"
    )

    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=10,
                extra_body={"enable_thinking": thinking_mode}
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
                print("Qwen service unavailable. Waiting 30 seconds, then retrying the same question...")
                time.sleep(30)
                print("Retrying now...")
                print()

            elif (
                "No route to host" in error_message
                or "ReadError" in error_message
                or "Connection" in error_message
                or "network" in error_message.lower()
                or "timeout" in error_message.lower()
                or "timed out" in error_message.lower()
            ):
                print()
                print("Network/API connection issue. Waiting 60 seconds, then retrying the same question...")
                print("Error was:", error_message)
                time.sleep(60)
                print("Retrying now...")
                print()

            else:
                raise e

    raw_answer = response.choices[0].message.content.strip()
    model_answer = clean_answer(raw_answer)

    is_correct = model_answer == str(correct_option)
    model_chose_abstention = model_answer == str(abstention_option_number)

    if is_correct:
        num_correct += 1

    if model_chose_abstention:
        num_abstentions += 1

    total_q += 1

    print(f"Model raw response: {raw_answer}")
    print(f"Model cleaned response: {model_answer}")
    print(f"Correct option: {correct_option}")
    print(f"Abstention option: {abstention_option_number}")
    print("Result:", "Correct" if is_correct else "Wrong")
    print("Chose abstention:", model_chose_abstention)
    print("-" * 80)

    return model_answer, raw_answer, is_correct, model_chose_abstention


def get_question_data(row, df, i):
    """
    Supports main + abstention GPQA-style files with columns:

    Question
    Correct Answer
    Incorrect Answer 1
    Incorrect Answer 2
    Incorrect Answer 3
    Abstention Option
    Explanation
    """

    question = row["Question"]
    correct_answer = row["Correct Answer"]
    abstention_option = row["Abstention Option"]

    choices = [
        row["Correct Answer"],
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
        row["Abstention Option"]
    ]

    # Stable shuffle so row i always gets the same option order
    random.seed(i)
    random.shuffle(choices)

    correct_option = choices.index(correct_answer) + 1
    abstention_option_number = choices.index(abstention_option) + 1

    return {
        "row_index": i,
        "question": question,
        "choices": choices,
        "correct_answer": correct_answer,
        "correct_option": correct_option,
        "abstention_option": abstention_option,
        "abstention_option_number": abstention_option_number
    }


def test_dataset(filename, start_index, num_questions):
    global num_correct, total_q, num_abstentions

    num_correct = 0
    total_q = 0
    num_abstentions = 0

    print(f"Loading file: {filename}")
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} rows")
    print("Columns:", df.columns.tolist())
    print("Detected GPQA main + abstention file.")
    print()

    results = []

    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_folder = "Results"
    os.makedirs(output_folder, exist_ok=True)

    output_name = os.path.join(
        output_folder,
        f"qwen_results_{model}_{base_name}_{start_index}_{start_index + num_questions - 1}.csv"
    )

    end_index = min(start_index + num_questions, len(df))

    for i in range(start_index, end_index):
        print(f"Running row {i}...")

        row = df.iloc[i]
        item = get_question_data(row, df, i)

        question = item["question"]
        choices = item["choices"]
        correct_option = item["correct_option"]
        abstention_option_number = item["abstention_option_number"]

        model_answer, raw_answer, is_correct, model_chose_abstention = ask_question(
            question,
            choices,
            correct_option,
            abstention_option_number
        )

        results.append({
            "row_index": item["row_index"],
            "question": question,
            "option_1": choices[0],
            "option_2": choices[1],
            "option_3": choices[2],
            "option_4": choices[3],
            "option_5": choices[4],
            "correct_answer": item["correct_answer"],
            "correct_option": correct_option,
            "abstention_option": item["abstention_option"],
            "abstention_option_number": abstention_option_number,
            "model_answer": model_answer,
            "raw_model_answer": raw_answer,
            "is_correct": is_correct,
            "model_chose_abstention": model_chose_abstention,
            "source_file": os.path.basename(filename)
        })

        # Save progress after every question
        pd.DataFrame(results).to_csv(output_name, index=False)

        # Small pause to reduce rate-limit issues
        time.sleep(1)

    print()
    if total_q > 0:
        print(f"Accuracy: {num_correct}/{total_q} correct = {num_correct / total_q * 100:.2f}%")
        print(f"Over-abstention rate: {num_abstentions}/{total_q} = {num_abstentions / total_q * 100:.2f}%")
    else:
        print("No questions were tested.")

    print(f"Saved results to {output_name}")


random.seed(42)

filename = "DataSets/gpqa_main.csv"
start_index = 0
num_questions = 10

if len(sys.argv) >= 2:
    filename = sys.argv[1]

if len(sys.argv) >= 3:
    start_index = int(sys.argv[2])

if len(sys.argv) >= 4:
    num_questions = int(sys.argv[3])

if len(sys.argv) >= 5:
    model = sys.argv[4]

if len(sys.argv) >= 6:
    thinking_mode = sys.argv[5].lower() == "true"

test_dataset(filename, start_index, num_questions)