#!/usr/bin/env python
# coding: utf-8

# pandas is used for handling data frames and goose3 is a library for extracting information from web pages.<br>
# It reads a the CSV file named "input.csv" (coverted into csv form xlsx) using pandas and stores it.<br>
# A Goose instance (g) is created. This instance is used to extract information from web pages.<br>
# The function extract_article is defined that takes a URL as input, uses the goose instance to extract information (title and cleaned text) from the web page, and returns the title and text.<br>
# It iterates through each row in the DataFrame (df). It extracts the 'URL_ID' and 'URL' from each row.<br>
# Inside the loop, it tries to extract the article's title and text using the extract_article function.<br>
# If successful, it creates a text file named "{url_id}.txt" and writes the title and text into it.<br>
# If there is an exception (error), it prints an error message.

# In[1]:


import pandas as pd
from goose3 import Goose
csv_file = "input.csv"
df = pd.read_csv(csv_file)
g = Goose() # Create a Goose instance
def extract_article(url):
    article = g.extract(url)
    title = article.title
    text = article.cleaned_text
    return title, text
for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    try:
        title, text = extract_article(url)
        output_file = f"{url_id}.txt"
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(f"Title: {title}\n\n{text}")
        print(f"Article extracted and saved for URL_ID: {url_id}")
    except Exception as e:
        print(f"Error processing URL_ID {url_id}: {str(e)}")


# In[ ]:




