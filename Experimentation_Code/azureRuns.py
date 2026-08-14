import time
from openai import OpenAI
from apiKeys import azure_key, azure_endpoint
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

endpoint = azure_endpoint
deployment_name = "gpt-5.4"
api_key = azure_key

client = OpenAI(
    base_url=endpoint,
    api_key=api_key
)

# Instruction prompt to
instructions = "You will be asked a question with 4 multiple choice options. Please answer the question by returning 1, 2, 3, or 4. " \
"Do not include any explanation or additional text, just return the number corresponding to the answer. Take your time and think carefully before responding. Do not include your thinking process in your resonse." \
"Any response that is not a single number will be considered incorrect. You will be graded on a point system where correct answers are worth 1 points, \"I don't know\" answers are worth 0.5 points and wrong answers are worth -1 points."\
"You should attempt to maximize your score and minimize incorrect answers. Please provide your answer to the following question: " 

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
    global question_number, questions, correct_answers, non_single_number_responses, model_answers, is_correct, numCorrect, totalQ
    
    prompt = f"Question: {question}\nOptions:\n1. {op1}\n2. {op2}\n3. {op3}\n4. {op4}\nAnswer:"
    max_retries = 5
    backoff = 15

    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": instructions
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
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

    answer = completion.choices[0].message.content.strip()
     # Store the question number, question, correct answer, model response, and whether the model's response is correct or not to add to a csv file
    question_number.append(qnum)
    questions.append(question)
    correct_answers.append(cop)
    if(answer not in ['1', '2', '3', '4']):
        non_single_number_responses.append(answer)
        model_answers.append("")
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
    global start_time, time_start_q, QPM
    df_subset = pd.read_csv(file_path, usecols=['Question','Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3'])
    for i in range(fstQ, lastQ):
        row = df_subset.iloc[i]
        print(f"Question #{i+1}")
        ask_question(row['Question'], row['Correct Answer'], row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3'],1, i+1)

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

    df.to_csv("Results/GPT_5.4_Data/MitIDK_r1.csv", index=False)
    print("Saved responses to GPT_5.4_Data/MitIDK_r1.csv")


read_csv(0, 448, "DataSets/gpqa_IDK.csv")  
