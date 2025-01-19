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

os.environ["OPENAI_API_KEY"] = 'insert key here'

# Data preprocessing
file = 'insert file path here'

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

query = "please give me a comprehensive summary of the data in this finnancial report"
docs = doc_search.similarity_search(query)
response = chain.run(input_documents=docs, question=query)
print(response)








