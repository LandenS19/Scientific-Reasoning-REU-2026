import time
import requests, base64
import pandas as pd
from apiKeys import nvidia_key


invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = False

model = "meta/llama-4-maverick-17b-128e-instruct"

instructions = "You will be asked a question with 4 multiple choice options. Please answer the question by returning 1, 2, 3, or 4. " \
"Do not include any explanation or additional text, just return the number corresponding to the answer. Take your time and think carefully before responding. Do not include your thinking process in your resonse." \
"Any response that is not a single number will be considered incorrect. You will be graded on a point system where correct answers are worth 1 points, \"I don't know\" answers are worth 0 points and wrong answers are worth -1 points."\
"You should attempt to maximize your score and minimize incorrect answers. Please provide your answer to the following question: " 

headers = {
  "Authorization": f"Bearer {nvidia_key}",
  "Accept": "text/event-stream" if stream else "application/json"
}


# Lists for the csv file that will be created with the data gathered from the model's responses to the questions.
question_number = []
questions = []
correct_answers = []
non_single_number_responses = []
model_answers = []
is_correct = []

# Variables for perfomance tracking
numCorrect = 0
totalQ = 0


def ask_question(question, op1, op2, op3, op4, cop, qnum):
    global numCorrect, totalQ
    prompt = f"Question: {question}\nOptions:\n1. {op1}\n2. {op2}\n3. {op3}\n4. {op4}\nAnswer: "
    payload = {
        "model": model,
        "messages": [{"role":"system","content": instructions}, {"role":"user","content":prompt}],
        "max_tokens": 1500,
        "temperature": 1.00,
        "top_p": 1.00,
        "frequency_penalty": 0.00,
        "presence_penalty": 0.00,
        "stream": stream
    }
    max_retries = 5
    backoff = 10

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(invoke_url, headers=headers, json=payload)
            data = response.json()
            break
        except Exception as exc:
            print(f"Attempt {attempt}/{max_retries} failed for question #{qnum}: {exc}")
            if attempt == max_retries:
                print(f"Giving up on question #{qnum} after {max_retries} attempts.")
                question_number.append(qnum)
                questions.append(question)
                correct_answers.append(cop)
                non_single_number_responses.append(f"ERROR: {exc}")
                model_answers.append("")
                is_correct.append(False)
                totalQ += 1
                return
            time.sleep(backoff * attempt)

    answer = data["choices"][0]["message"]["content"].strip()

    question_number.append(qnum)
    questions.append(question)
    correct_answers.append(cop)
    if answer not in ['1', '2', '3', '4']:
        model_answers.append("")
        non_single_number_responses.append(answer)
    else:
        non_single_number_responses.append("")
        model_answers.append(answer)

    if(answer == str(cop)):
        numCorrect += 1
        is_correct.append(True)
    else:
        is_correct.append(False)
    totalQ += 1
    print(f"Question #{qnum} completed.")


#####################################################################
# This is for accessing a .csv file and sending the questions to the model.
def read_csv(fstQ, lastQ, file_path):
    df_subset = pd.read_csv(file_path, usecols=['Question','Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3'])
    for i in range(fstQ, lastQ):
        row = df_subset.iloc[i]
        print(f"Question #{i+1}")
        ask_question(row['Question'], row['Correct Answer'], row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3'], 1, i+1)

    print(f"Performance: {numCorrect/totalQ * 100:.2f}% correct")
    build_csv()
#####################################################################


def build_csv():
    df = pd.DataFrame({
        "Question Number": question_number,
        "Question": questions,
        "Correct Answer": correct_answers,
        "Model Answer": model_answers,
        "Non Single Number Response": non_single_number_responses,
        "Is Correct": is_correct
    })

    df.to_csv("Results/Maverick_Data/MitIDK_r1.csv", index=False)
    print("Saved responses to MitIDK_r1.csv")


read_csv(0, 448, "DataSets/gpqa_main.csv")