This projected was designed to analyze large Financial reports in a quick and efficient manner so Financial Analysits can focus their effort and time towards other tasks. The Program uses pdfplumber to go through any text only pdfs to quickly extract the raw text. However, if the text is embedded in an image within the pdf then the program will then convert the pdf to an image and use tesseract, an OCR program, to extract the text. A personal OpenAI key is required but training data is not as the model being used is pretrained by OpenAI.

**Future work**
-integrate this script into a website using react and AWS frame work
-Update the LangChain code to work better with newer version as the load_qa_chain function will not be supported in future releases of the library

Sample Response:

This financial report includes information on revenues and other income, costs and other deductions, income (loss) before income taxes, income tax expense (benefit), net income (loss) including noncontrolling interests, net income (loss) attributable to ExxonMobil, and earnings (loss) per common share. It also mentions the inclusion of Notes to Consolidated Financial Statements as an important part of the statements. In terms of revenue, the report lists sales and other operating revenue at 18 million dollars, income from equity affiliates at 7 million dollars, and other income at 3.5 million dollars. In terms of costs and deductions, the report lists various expenses such as crude oil and product purchases, production and manufacturing expenses, and interest expense. The report also includes information on income before taxes, income tax expenses, and net income (attributable to both noncontrolling interests and ExxonMobil). The earnings per common share are listed as 13.26 dollars and 5.39 dollars, assuming dilution. Overall, the report shows a net income of 37.354 million dollars, with a portion attributable to noncontrolling interests and the majority attributable to ExxonMobil.

Below is the code for this project

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from PIL import Image
import pdfplumber
import os
import pytesseract
import pdf2image

os.environ["OPENAI_API_KEY"] = ''

# Data preprocessing
file = ''

def tesseract_open(file1):
    return(pytesseract.image_to_string(Image.open(file1)))

if file[len(file) - 3:len(file)] == 'pdf':
    PDF_read = pdfplumber.open(file)
    text1 = ''
    for page in PDF_read.pages:
        page_text = page.extract_text()
        if page_text:  
            text1 += page_text
    if text1 == '':
        pages = pdf2image.convert_from_path(file)
        for i, page in enumerate(pages):
            file_name = f'output_{i}.jpg'
            page.save(file_name, 'JPEG')
            text1 += tesseract_open(file_name) 
else:
    text1 = tesseract_open(file)

text_split = CharacterTextSplitter(
    separator= "/n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len
)
fulltext = text_split.split_text(text1)

#OpenAI query
embeddings = OpenAIEmbeddings()
doc_search = FAISS.from_texts(fulltext, embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")

query = "please give me a comprehensive summary of the data in this finnancial report" #Custumuizable this is just a place holder
docs = doc_search.similarity_search(query)
response = chain.run(input_documents=docs, question=query)
print(response)
