import openai
import numpy
import faiss
from tika import parser
from langchain.callbacks import get_openai_callback
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import (
    ConversationBufferMemory,
)
from borb.pdf import Document
from borb.pdf.page.page import Page
from borb.pdf import PDF
from borb.pdf import Paragraph
from borb.pdf import SingleColumnLayout
from borb.pdf import FixedColumnWidthTable
from borb.pdf import HSVColor
from borb.pdf import SmartArt
from decimal import Decimal

GPT_API_KEY = '###'


def get_embedding(text, mode="online"):
    EMBEDDING_MODEL = "text-embedding-ada-002"
    result = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=text,
        api_key=GPT_API_KEY
    )
    text_vector = result["data"][0]["embedding"]
    return text_vector


def get_embeddings(texts, mode='online'):
    embeddings = [get_embedding(text) for text in texts]
    embeddings = numpy.array(embeddings)
    return embeddings


def index_embeddings(embeddings):
    d = embeddings.shape[1]
    embeddings = embeddings.astype('float32')
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


# Retrieving the relevant paragraph
def retrieve_relevant_info(index, query_vector, k=3):
    D, I = index.search(query_vector, k)
    relevant_indexes = I.tolist()[0]
    return relevant_indexes


def get_query_vector(query):
    embeddings = numpy.array(get_embedding(query))
    embeddings = embeddings.astype('float32')
    query_vector = numpy.array(embeddings)
    query_vector = numpy.reshape(query_vector, (1, 1536))
    return query_vector


def get_openai_response(query, context):
    messages = [
        {"role": "system", "content": "you are helpful assitant."},
        {"role": "system", "content": get_enhanced_query(query, context)}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.1,
        max_tokens=2000,
        api_key=GPT_API_KEY,
        frequency_penalty=0,
        presence_penalty=0
    )
    output = response.get("choices")[0].get("message").get("content")
    return output

def get_enhanced_query(query, context):
    prompt = f"""
        you are a helpful assitant.
        Answer the query based on the provided context . If the query cannot be answered using the context return I don't know as the answer.
        ###
        Query:{query}
        ###
        context:{context}
        ###
        Answer:
    """
    return prompt

def clean_text(text):
    cleaned_string = text.replace("\n", "").replace("..", "")
    return cleaned_string

def count_token(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')
    return result

def get_action_items(context):
    messages = [
        {"role": "system", "content": "you are helpful assitant."},
        {"role": "system", "content": get_action_itme_query(context)}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=2000,
        api_key=GPT_API_KEY,
        frequency_penalty=0,
        presence_penalty=0
    )
    output = response.get("choices")[0].get("message").get("content")
    return output

def get_action_itme_query(context):
    prompt = f"""
        ###
        context:{context}
        ###
        query:list all action items from the above context.
    """
    return prompt

def get_data_from_doc(filename):
    parsed_pdf = parser.from_file(filename)
    # Extracting content
    data = parsed_pdf['content']
    paragraphs = re.split('\n\s*\n', data)
    cleaned_paragraphs = [clean_text(para) for para in paragraphs]
    finalData = ' '.join([str(elem) for elem in cleaned_paragraphs])
    return finalData


def create_doc(layout, data_list):
    document = Document()
    page = Page()
    page_layout = SingleColumnLayout(page)
    page_layout.add(Paragraph("Action Items"))
    if layout == 1:
        table = FixedColumnWidthTable(number_of_columns=3, number_of_rows=int(len(data_list) / 3) + 1)
        table.set_padding_on_all_cells(Decimal(5), Decimal(5), Decimal(5), Decimal(5))
        for data in data_list:
            if data != '':
                table.add(Paragraph(data))

        page_layout.add(table)
        # Append page
        document.add_page(page)

        # Persist PDF to file
        with open("output3.pdf", "wb") as pdf_file_handle:
            PDF.dumps(pdf_file_handle, document)
    elif layout == 2:
        # Layout

        table = FixedColumnWidthTable(number_of_columns=1, number_of_rows=len(data_list))
        table.set_padding_on_all_cells(Decimal(5), Decimal(5), Decimal(5), Decimal(5))

            # set padding on all (implicit) TableCell objects in the FixedColumnWidthTable

        # table.set_padding_on_all_cells(Decimal(5), Decimal(5), Decimal(5), Decimal(5))
        for data in data_list:
            if data != '':
                table.add(Paragraph(data))

        page_layout.add(table)
        # Append page
        document.add_page(page)

        # Persist PDF to file
        with open("output3.pdf", "wb") as pdf_file_handle:
            PDF.dumps(pdf_file_handle, document)
    else:
        # Layout
        colors = [
            HSVColor(Decimal(x / 360), Decimal(1), Decimal(1))
            for x in range(0, 360, int(360 / 20))
        ]
        colors_size=20
        i=0
        for data in data_list:
            if data != '':
                page_layout.add(Paragraph(data,font_color=colors[i%colors_size]))
                i=i+1
        # Append page
        document.add_page(page)

        # Persist PDF to file
        with open("output3.pdf", "wb") as pdf_file_handle:
            PDF.dumps(pdf_file_handle, document)

    print('document created')


# step-1 reading doc
finalData = get_data_from_doc("sample1.pdf")
# creating action item
result = get_action_items(finalData)
result = result.split("\n")
layoutType = int(input("select layout type (1,2,3):"))
create_doc(layoutType, result)
print("file generated")
# print(finalData)
# doc_loader = UnstructuredWordDocumentLoader("Employee_Handbook .pdf")
# doc = doc_loader.load()
# print(finalData)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = text_splitter.split_text(finalData)
#print(len(chunks))

# chunks = [chunk.page_content for chunk in chunks]
embeddings = get_embeddings(chunks, mode='online')
print(embeddings[0].shape)

# step -2 store embeddings
index = index_embeddings(embeddings)
print(index)

# langchain
llm = OpenAI(
    temperature=0,
    openai_api_key=GPT_API_KEY,
    model_name='gpt-3.5-turbo'
)
conversation_buf = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)
# step -3 query

query = input("enter your query: ")
while query != "exit":
    query_vector = get_query_vector(query)
    relevant_indexes = retrieve_relevant_info(index, query_vector, k=3)
    # print(relevant_indexes)
    relevant_para = []
    for i in relevant_indexes:
        relevant_para.append(chunks[i])
    context = ''
    for para in relevant_para:
        context += para + '###########'
    # response = get_openai_response(query, context)
    response = count_token(
        conversation_buf,
        get_enhanced_query(query, context)
    )
    print(response)
    query = input("enter another query: ")

# print(conversation_buf.memory)
# print(conversation_buf.prompt)

