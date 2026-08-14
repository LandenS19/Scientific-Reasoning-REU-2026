from google import genai
from google.genai import types
import apiKeys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time
import threading
import keyboard

client = genai.Client(api_key=apiKeys.gemini_pro_key)

model = "gemini-3.1-pro-preview"

QPM = 448 # Number of questions per minute that can be sent to avoid rate limits.

# Instruction prompt to send
instructions = "You will be asked a question with 4 multiple choice options. Please answer the question by returning 1, 2, 3, 4, or 5 based on the correct answer. "
"Do not include any explanation or additional text, just return the number corresponding to the correct answer. Take your time and think carefully before responding. Do not include your thinking process in your resonse."
"Any response that is not a single number will be considered incorrect. Any incorrect answers will have severely negative impacts and should be avoided at all costs. Please provide the correct answer to the following question: "

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

token_input_total = 0
token_total = 0

# Variables for rate limit tracking
start_time = 0
time_start_q = 0

# Variable for pause state
is_paused = False

# Function to handle pause key press
def _pause_handler():
    global is_paused
    while True:
        if keyboard.is_pressed('p'):
            is_paused = not is_paused
            if is_paused:
                print("\n[PAUSED] Press 'p' again to resume...")
            else:
                print("[RESUMED] Continuing...\n")
            time.sleep(0.5)  # Debounce to prevent multiple toggles
        time.sleep(0.1)

# Start pause handler thread
threading.Thread(target=_pause_handler, daemon=True).start()

# Timer to avoid rate limits.
def _start_time_counter():
    global start_time
    while True:
        time.sleep(1)
        start_time += 1

# start background thread to increment start_time every second
threading.Thread(target=_start_time_counter, daemon=True).start()

# Function to send the question to the model and store the response
def ask_question(question, op1, op2, op3, op4, op5, cop, qnum):
    global question_number, questions, correct_answers, non_single_number_responses, model_answers, is_correct, numCorrect, totalQ, token_input_total, token_total
    
    prompt = f"Question: {question}\nOptions:\n1. {op1}\n2. {op2}\n3. {op3}\n4. {op4}\n5. {op5}\nAnswer:"
    max_retries = 5
    backoff = 15

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=instructions,
                    # thinking_config=types.ThinkingConfig(thinking_level="high")
                ),
                contents=prompt
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

    # Store the question number, question, correct answer, model response, and whether the model's response is correct or not to add to a csv file
    question_number.append(qnum)
    questions.append(question)
    correct_answers.append(cop)
    if(response.text.strip() not in ['1', '2', '3', '4', '5']):
        non_single_number_responses.append(response.text.strip())
        model_answers.append("")
    else:
        non_single_number_responses.append("")
        model_answers.append(response.text.strip()) 

    token_input_total += response.usage_metadata.prompt_token_count
    token_total += response.usage_metadata.total_token_count

    if((response.text.strip() == str(cop[0])) or (response.text.strip() == str(cop[1]))):
        numCorrect += 1
        is_correct.append(True)
    else:
        is_correct.append(False)
    totalQ += 1
    print(f"Question #{qnum} completed.")

#####################################################################
# This is for accessing a .csv file and sending the questions to the model.
def read_csv(fstQ, lastQ, file_path):
    global start_time, time_start_q, QPM, is_paused
    df_subset = pd.read_csv(file_path, usecols=['Question','Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3', 'Abstention Option'])
    for i in range(fstQ, lastQ):
        # Check if paused
        while is_paused:
            time.sleep(0.1)
        
        if(i - time_start_q == QPM and i != fstQ and start_time < 60): # Checks to make sure we don't call >  QPM questions in a minute.
            print(f"Completed {i-fstQ} questions, sleeping for {60 - start_time} seconds to avoid rate limits...")
            time.sleep(60 - start_time) # Sleep for the remaining time in the minute to avoid rate limits
            start_time = 0 # Reset the timer
            time_start_q = i # Reset the timer for the next question
        elif(start_time >= 60):
            start_time = 0
            time_start_q = i
        row = df_subset.iloc[i]
        print(f"Question #{i+1}")
        ask_question(row['Question'], row['Correct Answer'], row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3'], row['Abstention Option'], [1,5], i+1)

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
        "Is Correct": is_correct,
    })

    df.to_csv("Results/GeminiPro_Data/NAC+Main.csv", index=False)
    print("Saved responses to NAC+Main.csv")


read_csv(0, 448, "DataSets/gpqa_main.csv")


print(f"Token Input Total: {token_input_total}")
print(f"Token Output Total: {token_total - token_input_total}")
print(f"Token Total: {token_total}")
