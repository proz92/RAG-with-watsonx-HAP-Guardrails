# <hap_utilities.py>

#Necessary imports
import nltk
from nltk.tokenize import sent_tokenize, regexp_tokenize
# Ensure you have the required NLTK data
nltk.download('punkt')


#Helper functions
def split_text_preserve_newlines(text):
    """
    Split the given text into sentences while preserving newlines.
    1. Splits the text into paragraphs based on double newlines (\n\n).
    2. Tokenizes each paragraph separately into sentences.
    3. Reassembles the sentences while inserting double newlines between paragraphs.
    
    Args:
    text (str): The input text to split.
    
    Returns:
    List[str]: A list of sentences with preserved newlines.
    """
    # Split text by newline characters first
    paragraphs = text.split('\n\n')
    
    # Tokenize each paragraph separately
    sentences_with_newlines = []
    for para in paragraphs:
        sentences = sent_tokenize(para)
        if sentences:
            # Add newline between paragraphs
            sentences_with_newlines.extend(sentences)
            sentences_with_newlines.append('')  # This will add a '\n\n' in the final text
    
    # Remove the last extra newline added
    if sentences_with_newlines and sentences_with_newlines[-1] == '':
        sentences_with_newlines.pop()
    
    return sentences_with_newlines

def rebuild_text_with_newlines(sentences):
    """
    Rebuild the text from a list of sentences with preserved newlines. Joins sentences back into text, inserting double newlines where appropriate.
    
    Args:
    sentences (List[str]): A list of sentences to join.
    
    Returns:
    str: The rebuilt text with preserved newlines.
    """
    rebuilt_text = ''
    for sentence in sentences:
        if sentence == '':
            rebuilt_text += '\n\n'
        else:
            rebuilt_text += sentence + ' '
    
    return rebuilt_text.strip()


def find_substring_indices(full_strings, substrings):
    """
    Find the indices of substrings within a list of full strings.

    Parameters:
    full_strings (list[str]): A list of full strings to search for substrings.
    substrings (list[str]): A list of substrings to search for within the full strings.

    Returns:
    list[int]: A list of indices corresponding to the full strings that contain at least one substring.
    """
    indices = []
    
    # Iterate through each full string
    for i, full_string in enumerate(full_strings):
        # Check if any of the substrings is found in the current full string
        if any(sub in full_string for sub in substrings):
            indices.append(i)
    
    return indices

from typing import TypedDict
class Credentials(TypedDict):
    url_LLM: str
    url_LANGCHAIN: str
    apikey: str


from langchain_core.documents.base import Document 
def clean_hap_content(docs: list[Document], credentials: Credentials, project_id: str):
    # Each document of the list is a very long text, so we split each document into smaller chuncks
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        # If chunk size is too big, the model does not recognize the hap content within the chunk
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
        # separators=[
        #     '.',
        # ]
    )

    # Two-dimensional list where each sublist represent a document splitted in chunks
    docs_as_list_of_chuncks = []
    for doc in docs:
        # Split text in senteces, from . to .
        tmp_list = split_text_preserve_newlines(doc.page_content)
        # Remove '' from the list, added by the previous function to preserve newlines
        tmp_list = [s for s in tmp_list if s != '']
        # Split each sentence into smaller chunks if necessary
        final_list = []
        for s in tmp_list:
            final_list += text_splitter.split_text(s)
        docs_as_list_of_chuncks.append(final_list)

    # Load ibm-granite/granite-guardian-hap-38m model
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_name_or_path = 'ibm-granite/granite-guardian-hap-38m'
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Calculate HAP probability for every chunk. Save predictions and probabilities for every chunk in a two-dimensional list
    prediction_results = []
    probability_results = []
    for chunks in docs_as_list_of_chuncks:
        input = tokenizer(chunks, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**input).logits
            prediction_list = torch.argmax(logits, dim=1).detach().numpy().tolist() # Binary prediction where label 1 indicates toxicity.
            prediction_results.append(prediction_list)
            probability_list = torch.softmax(logits, dim=1).detach().numpy()[:,1].tolist() # Probability of toxicity.
            probability_results.append(probability_list)

    # Find the indices of the chuncks with HAP prediction of 1
    indices = [(i, j) for i, row in enumerate(prediction_results) for j, value in enumerate(row) if value == 1]
    # Save the chunks with HAP probability >90% in a key-value dictionary where:
    # - key: is the index of the original document
    # - value: is a list containg the chunks with HAP content
    matrix_of_chunks_with_hap = {}
    for tup in indices:
        i=tup[0]; j=tup[1];
        if(probability_results[i][j] >= 0.90):
            print(tup)
            if(i not in matrix_of_chunks_with_hap):
                matrix_of_chunks_with_hap[i] = []
            matrix_of_chunks_with_hap[i].append(docs_as_list_of_chuncks[i][j])
            print(f"Probability: {probability_results[i][j]}, Sentence: {docs_as_list_of_chuncks[i][j]}")
    print(f"There are {len(matrix_of_chunks_with_hap)} documents with at east one sentence with HAP probability > 90%. {list(matrix_of_chunks_with_hap.keys())}.")

    import requests
    # Generate Bearer token
    url = 'https://iam.cloud.ibm.com/identity/token'
    myobj = {'grant_type': 'urn:ibm:params:oauth:grant-type:apikey', 'apikey':credentials["apikey"]}
    #use the 'headers' parameter to set the HTTP headers:
    x = requests.post(url, data = myobj, headers = {"Content-Type": "application/x-www-form-urlencoded"})
    bearer_token = x.json()['access_token']


    for key in list(matrix_of_chunks_with_hap.keys()):
        old_doc = docs[key].page_content
        list_of_hap_strings = matrix_of_chunks_with_hap[key]

        # --- Retrieve only the sentences with HAP content ---
        # Split the text into sentences
        full_strings = split_text_preserve_newlines(old_doc)

        substrings = list_of_hap_strings

        # Find the indices of strings containing any of the substrings
        matching_indices = find_substring_indices(full_strings, substrings)

        print("Indices of matching strings:", matching_indices)
        # -----------------------------------------------------

        # -----------------------   GRANITE   ------------------------
        # Build the prompt and ensure that the revised sentences no longer have harmful or offensive language but retain the original meaning or intention in a more respectful and appropriate way.
        for i in range(len(matching_indices)):
            prompt_granite = """Rewrite the following sentence to remove any harmful, abusive, or inappropriate content while maintaining the original meaning where possible. Ensure the revised sentence is respectful, neutral, and appropriate for all audiences. The output should contain only the revised sentence itself, without any additional words, explanations, or punctuation marks.

Input: That was a stupid decision and you’re an idiot for making it.
Output: That was a decision that didn’t work out as planned.

Input: Why are you so incompetent? Can’t you do anything right?
Output: Why are you so ineffective? Can’t you do anything right?

Input: It’s absolutely pathetic how you can’t even get the simplest things right, and it’s clear that you’re just a complete f**king failure in every aspect.
Output: It’s disappointing how you can’t even get the simplest things right, and it’s clear that you need some improvements.


Input: """ + full_strings[matching_indices[i]] + """
Output: """

            body_granite = {
                "input": prompt_granite,
                "parameters": {
                    "decoding_method": "greedy",
                    "max_new_tokens": 200,
                    "repetition_penalty": 1.1
                },
                "model_id": "ibm/granite-13b-instruct-v2",
                "project_id": project_id
            }

            # Make the API call to the LLM model
            authorization = "Bearer " + bearer_token
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": authorization
            }

            response = requests.post(
                credentials["url_LLM"],
                headers=headers,
                json=body_granite
            )

            if response.status_code != 200:
                raise Exception("Non-200 response: " + str(response.text))

            result_granite = response.json()["results"][0]["generated_text"]
            # Remove the \n\n at the end of a string if it exists
            if result_granite.endswith('\n\n'):
                result_granite = result_granite[:-2]
            if result_granite.endswith('  '):
                result_granite = result_granite[:-2]
            
            print("Doc:", key, "- chunk:", matching_indices[i], "original :", full_strings[matching_indices[i]])
            print("Doc:", key, "- chunk:", matching_indices[i], "rewritten:", result_granite)
            # Change the original sentence with the revised sentence
            full_strings[matching_indices[i]] = result_granite
            
        docs[key].page_content = rebuild_text_with_newlines(full_strings)
        docs[key].metadata['rewritten'] = True