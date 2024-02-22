#!/usr/bin/env python
# coding: utf-8

# Get the Libraries 

# In[45]:


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from textblob import TextBlob
import string


# Download all the dependencies, 'punkt' is used for pre trained tokenization and stopwords for pre trained stopwords.

# In[3]:


nltk.download('punkt')
nltk.download('stopwords')


# Loading the files from the given text files in the drive, Converted those text file to UTF-8 Encoded files as UTF-8 encoded files have various advantages, <br> 
# UTF-8 is backward compatible with ASCII, which means that files containing only ASCII characters are also valid UTF-8 files. <br> 
# UTF-8 is a variable-width encoding, allowing it to represent an extensive range of characters.<br> 
# 
# load_words_from_file function:<br> 
# This function takes a file path as an argument then it opens the file in read mode using UTF-8 encoding.<br> 
# It reads the contents of the file, splits them into lines (using splitlines()), and stores them in a list called words, Finally, it returns a set containing the unique words from the file.
# <br> <br> 
# load_stopwords() Function:<br> 
# This function is responsible for loading stopwords from multiple text files and combining them into a single set.<br> 
# It initializes an empty set called stop_words.<br> 
# It iterates through a list of stopwords files (stopwords_files), and for each file, it updates the stop_words set by adding the words from that file.<br> 
# The function then returns the consolidated set of stopwords.

# In[46]:


# loading words from the text files provided in the drive
def load_words_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        words = file.read().splitlines()
    return set(words)

# loading positive and negative words from text files
positive_words = load_words_from_file("positive-words.txt")
negative_words = load_words_from_file("negative-words.txt")

# function to load various stopword files
def load_stopwords():
    stop_words = set()
    stopwords_files = ["StopWords_Auditor.txt", "StopWords_Currencies.txt", "StopWords_DatesandNumbers.txt",
                        "StopWords_Generic.txt", "StopWords_GenericLong.txt", "StopWords_Geographic.txt",
                        "StopWords_Names.txt"]
    for file_name in stopwords_files:
        stop_words.update(load_words_from_file(file_name))
    return stop_words


# stop_words = load_stopwords()<br>
# This line of code is used to load the stopwords into a varioable, will later be used for filtering out common words in the text during text analysis.<br>
# <br>
# count_syllables(word) Function:<br>
# It uses a simple algorithm to count syllables:<br>
# If the word starts with a vowel, it is counted as one syllable.<br>
# Additional syllables are counted based on consecutive vowels.<br>
# Silent 'e' at the end is excluded unless the word ends with two vowels (e.g., "lee").<br>
# The function is made to handle common cases for English words and is used later in the code for syllable counting in text analysis tasks.

# In[47]:


# loading stopwords to a variable for future use
stop_words = load_stopwords()
# function to count syllables in a word
def count_syllables(word):
    vowels = "aeiouy"
    count = 0
    if word[0] in vowels: # if the word starts with a vowel, count it as one syllable
        count += 1
    for index in range(1, len(word)): # counting additional syllables based on consecutive vowels
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1 
    if word.endswith("e") and word[-2] not in vowels:  # excluding silent 'e' at the end
        count -= 1
    count = max(1, count) # ensuring at least one syllable for short words
    return count


# Tokenization:<br>
# words = word_tokenize(text): This line uses the word_tokenize function from the nltk.tokenize module to split the input text into a list of words. Tokenization is the process of breaking down a text into individual words.<br>
# 
# Calculating Total Syllables:<br>
# total_syllables = sum(count_syllables(word) for word in words): This line calculates the total number of syllables in the text by summing the syllable count for each word. The count_syllables function, defined earlier, is used here to count syllables for each word.<br>
# 
# Calculating Total Words:<br>
# total_words = len(words): This line calculates the total number of words in the text.<br>
# 
# Handling Division by Zero:<br>
# total_words = max(1, total_words): This line ensures that there is at least one word to avoid division by zero. If the text is empty or has no words, it sets total_words to 1.<br>
# 
# Calculating Average Syllables per Word:<br>
# avg_syllables_per_word = total_syllables / total_words: This line computes the average number of syllables per word by dividing the total syllables by the total words.<br>
# 
# Returning the Result:<br>
# return avg_syllables_per_word: The function returns the calculated average syllables per word.

# In[48]:


# function for calculating the average number of syllables per word in a text
def calculate_avg_syllables_per_word(text):
    words = word_tokenize(text)
    total_syllables = sum(count_syllables(word) for word in words)
    total_words = len(words)
    total_words = max(1, total_words) # avoids division with zero
    avg_syllables_per_word = total_syllables / total_words
    return avg_syllables_per_word


# Identifying Complex Words:<br>
# complex_words = [word for word in words if count_syllables(word) > 2]: This line creates a list of words from the tokenized text, considering only those words with more than 2 syllables. The count_syllables function, defined earlier, is used to count syllables for each word.<br>
# 
# Calculating Percentage of Complex Words:<br>
# percentage_complex_words = (len(complex_words) / len(words)) * 100: This line calculates the percentage of complex words by dividing the count of complex words by the total number of words and then multiplying by 100 to express it as a percentage.

# In[49]:


# function for calculating the percentage of complex words
def calculate_complexity(text):
    words = word_tokenize(text)
    complex_words = [word for word in words if count_syllables(word) > 2]  
    percentage_complex_words = (len(complex_words) / len(words)) * 100
    return percentage_complex_words


# In[50]:


# function for calculating Fog Index
def calculate_fog_index(avg_sentence_length, percentage_complex_words):
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    return fog_index


# Personal Pronouns:<br>
# personal_pronouns: A list containing specific personal pronouns such as 'I', 'we', 'my', 'ours', and 'us'.
# Regex Counting:<br>
# 
# The function uses a regular expression (regex) to find and count the occurrences of each personal pronoun in the given text. The re.findall function is employed for this purpose.

# In[51]:


# function for counting personal pronouns
def count_personal_pronouns(text):
    personal_pronouns = ['i', 'we', 'my', 'ours', 'us']
    counts = {pronoun: len(re.findall(r'\b' + pronoun + r'\b', text.lower())) for pronoun in personal_pronouns}
    counts = {pronoun: int(count) for pronoun, count in counts.items()}
    total_count = sum(counts.values())
    return total_count


# Stopwords and Punctuation Removal:<br>
# The function now removes stopwords and punctuation from the words list, creating a words_without_stopwords list.<br>
# 
# Updated Score Calculation:<br>
# Positive and negative scores are now calculated based on the words_without_stopwords list.<br>
# 
# Subjectivity Score Update:<br>
# The subjectivity score is adjusted to consider the length of words_without_stopwords.<br>
# 
# Average Word Per Sentence Update:<br>
# The avg_words_per_sentence calculation now uses the length of words_without_stopwords.

# In[52]:


# function to analyze text and compute variables
def analyze_text(text, positive_words, negative_words, stop_words):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    words = word_tokenize(text.lower())  
    # removing stopwords and punctuation
    words_without_stopwords = [word for word in words if word not in stop_words and word not in string.punctuation]
    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(words) + 0.000001)
    sentences = sent_tokenize(text)
    words = word_tokenize(text) # removing punctuation
    # removing stopwords
    words_without_stopwords = [word for word in words if word.lower() not in stop_words and word not in string.punctuation]
    avg_sentence_length = len(words_without_stopwords) / len(sentences)
    avg_words_per_sentence = len(words) / len(sentences)  
    percentage_complex_words = calculate_complexity(text)
    fog_index = calculate_fog_index(avg_sentence_length, percentage_complex_words)
    
    complex_word_count = len([word for word in words_without_stopwords if len(word) > 6])  

    avg_word_length = sum(len(word) for word in words_without_stopwords) / len(words_without_stopwords)

    avg_syllables_per_word = calculate_avg_syllables_per_word(text)

    personal_pronoun_count = count_personal_pronouns(text)

    return positive_score, negative_score, polarity_score, subjectivity_score, avg_sentence_length, percentage_complex_words, fog_index, avg_words_per_sentence, complex_word_count, len(words_without_stopwords), avg_syllables_per_word, avg_word_length, personal_pronoun_count


# This part of the code processes all text files from "blackassign0001.txt" to "blackassign0100.txt" and analyzes each file using the analyze_text function. <br>

# In[53]:


# processing all text files from blackassign0001.txt to blackassign0100.txt
for i in range(1, 101):
    file_name = f"blackassign{str(i).zfill(4)}.txt"
    file_path = f"{file_name}"  # Replace with the actual path to your files

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            article_text = file.read()
        results = analyze_text(article_text, positive_words, negative_words, stop_words)
        columns = ['Positive_Score', 'Negative_Score', 'Polarity_Score', 'Subjectivity_Score', 'Avg_Sentence_Length',
                   'Percentage_of_Complex_Words', 'Fog_Index', 'Avg_Number_of_Words_Per_Sentence', 'Complex_Word_Count',
                   'Word_Count', 'Avg_Syllables_Per_Word', 'Avg_Word_Length', 'Personal_Pronoun_Count']

        print(f"\nResults for {file_name}:")
        for col, result in zip(columns, results):
            print(f"{col}: {result}")

    except FileNotFoundError:
        print(f"\nFile {file_name} not found. Skipping...")
        continue


# Pandas DataFrame:<br>
# The existing Excel file is read into a Pandas DataFrame using pd.read_excel(input_file).<br>
# 
# DataFrame Update:<br>
# The DataFrame is then updated with the computed values for each URL_ID. The URL_ID is used to locate the corresponding row in the DataFrame.<br>
# 
# Data Writing:<br>
# The updated DataFrame is written back to the Excel file using df.to_excel(output_file, index=False).<br>
# 
# Exception Handling:<br>
# If the input file is not found, an exception is caught, and a message is printed indicating that the file path should be checked.

# In[54]:


import pandas as pd
def update_excel_file(input_file, output_file, results_dict):
    try:
        df = pd.read_excel(input_file)
        for url_id, results in results_dict.items():
            # checking if the URL_ID exists in the DataFrame
            if url_id in df['URL_ID'].values:
                index = df.index[df['URL_ID'] == url_id].tolist()[0]

                df.at[index, 'POSITIVE SCORE'] = results[0]
                df.at[index, 'NEGATIVE SCORE'] = results[1]
                df.at[index, 'POLARITY SCORE'] = results[2]
                df.at[index, 'SUBJECTIVITY SCORE'] = results[3]
                df.at[index, 'AVG SENTENCE LENGTH'] = results[4]
                df.at[index, 'PERCENTAGE OF COMPLEX WORDS'] = results[5]
                df.at[index, 'FOG INDEX'] = results[6]
                df.at[index, 'AVG NUMBER OF WORDS PER SENTENCE'] = results[7]
                df.at[index, 'COMPLEX WORD COUNT'] = results[8]
                df.at[index, 'WORD COUNT'] = results[9]
                df.at[index, 'SYLLABLE PER WORD'] = results[10]
                df.at[index, 'PERSONAL PRONOUNS'] = results[11]
                df.at[index, 'AVG WORD LENGTH'] = results[12]

        df.to_excel(output_file, index=False) # writing the updated data back to the Excel file
        print("Data updated and written to the Excel file successfully.")

    except FileNotFoundError:
        print(f"Input file {input_file} not found. Please check the file path.")

results_dict = {} # dictionary to store results for each URL_ID

for i in range(1, 101):
    file_name = f"blackassign{str(i).zfill(4)}.txt"
    file_path = f"{file_name}"

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            article_text = file.read()
        results = analyze_text(article_text, positive_words, negative_words, stop_words)
        url_id = f"blackassign{str(i).zfill(4)}"
        results_dict[url_id] = results
    except FileNotFoundError:
        print(f"\nFile {file_name} not found. Skipping...")
update_excel_file("Output Data Structure.xlsx", "Output Data Structure.xlsx", results_dict)


# In[ ]:




