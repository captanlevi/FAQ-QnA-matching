import pandas as pd
import pickle
import os
import csv
import gdown


def check_position(corpus_matches, original):
    # Return 1,2,3,4,5 for their respective position of the query_original_q in corpus_matches
    # Return -1 if not found
    for ind, match in enumerate(corpus_matches):
        if match.lower() == original.lower():
            return ind + 1
    return -1


def create_directory_if_missing(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"{folder_name} created")
    else:
        print(f"Directory {folder_name} already exists.")


def download_t5_model_if_not_present(model_path):
    create_directory_if_missing(model_path)
    config_file = 'config.json'
    pytorch_file = 'pytorch_model.bin'

    config_path = os.path.join(model_path, config_file)
    pytorch_path = os.path.join(model_path, pytorch_file)

    if os.path.isfile(config_path) and os.path.isfile(pytorch_path):
        print(f"Model found in {model_path}. No downloads needed.")

    else:
        config_location = "https://drive.google.com/uc?id=1x-Y__pyV6pOLn2UOZNNwYXWS0KtL-5WZ"

        pytorch_location = "https://drive.google.com/uc?id=1xvH9-BWqRweJ4Sr1c8oakOMDxI0iTYLd"

        gdown.download(config_location, config_path, False)
        gdown.download(pytorch_location, pytorch_path, False)


def extract_qa_from_csv(path):
    """
    the csv is assumed to have questions on column 1 and answers in column 2
    WITH NO HEADER
    """

    df = pd.read_csv(path, header=None)
    questions = df.iloc[:, 0]
    answers = df.iloc[:, 1]

    return questions, answers


def save_dict(obj, path):
    # Save in .pickle format
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict(path):
    # Load .pickle format as dictionary
    with open(path, 'rb') as f:
        return pickle.load(f)


def serialize_to_csv(file_path, input_file, output_file=None):
    """ Convert .pkl file to .csv file & delete legacy .pkl file
    NOTE: input_file should points to a .pkl file. i.e. input_file = "babyBonus" points to "babyBonus.pkl"
    bani_work's generation pipeline always stores the augmented FAQ in .pkl
    To adapt to bani_work's design, I converted the .pkl back to .csv for users' easy viewing
    """

    if output_file is None:
        output_file = input_file
    output_file = output_file + ".csv"
    input_file = input_file + ".pkl"
    input_path = os.path.join(file_path, input_file)
    faq_obj = load_dict(input_path)
    with open(output_file, mode='w') as new_csv:
        csv_writer = csv.writer(new_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # csv_writer.writerow(["Question", "Answer", "Label"])
        for faq_unit in faq_obj.FAQ:
            label = faq_unit.question.label
            answer = faq_unit.answer.text
            question = faq_unit.question.text
            csv_writer.writerow([question, answer, label])

    output_path = os.path.join(file_path, output_file)
    # Delete .pkl file
    if os.path.exists(input_path):
        os.remove(input_path)
    os.rename(output_file, output_path)



