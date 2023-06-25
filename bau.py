"""
Turmbauten
Copyright (C) 2023 Tobias Fankhänel

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import argparse
import configparser
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.memory import ConversationSummaryBufferMemory, VectorStoreRetrieverMemory, CombinedMemory
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def prepare_embedding_model(embeddings_config):
    print('Preparing embedding model...')
    model_name = embeddings_config['ModelName']
    model_kwargs = {'device': embeddings_config['ModelDevice']}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        cache_folder = embeddings_config['ModelPath']
    )
    print('Embedding model prepared.')

    return embeddings

def document_name_to_dir_path(document_path, document_name):
    return document_path + document_name + '/'

def document_name_to_file_path(document_path, document_name):
    return document_path + document_name + '/' + document_name + '.pdf'

def pdf_to_text(doc_path):
    print('Converting PDF to text...')
    loader = PyPDFLoader(doc_path)
    pages = loader.load_and_split()
    fulltext = " ".join(map(lambda page: page.page_content, pages)).replace("\n", " ")
    print('PDF converted to text.')

    return fulltext

def text_to_snippets(fulltext, snippet_config):
    print('Converting text to snippets...')
    snippet_size = int(snippet_config['SnippetSize'])
    snippet_overlap = int(snippet_config['SnippetOverlap'])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=snippet_size, chunk_overlap=snippet_overlap, separators=["\n\n", "\n", '  ', '\.', ", ", " ", ""], keep_separator=False)
    snippets = text_splitter.create_documents([fulltext])
    print('Text converted to snippets.')

    return snippets

def read_from_file(path):
    with open(path, 'r') as f:
        contents = f.read()

    return contents

def persist_for_debugging(dir_path, fulltext, snippets):
    print('Persisting text and snippets for debugging...')
    fulltext_file = os.path.dirname(dir_path) + '/fulltext'
    snippets_file = os.path.dirname(dir_path) + '/snippets'
    with open(fulltext_file, 'w') as f:
        f.write(fulltext)
    with open(snippets_file, 'w') as f:
        f.write("\n\n".join(map(lambda snippet: snippet.page_content, snippets)))
    print('Text and snippets persisted.')

def vectorize_snippets(dir_path, snippets, embeddings):
    print('Creating embeddings from snippets...')
    persist_directory = os.path.dirname(dir_path) + '/db'
    db = Chroma.from_documents(documents=snippets, embedding=embeddings,
                           persist_directory=persist_directory)
    db.persist()
    print('Embeddings created.')

    return db

def read_question(initial_prompt_file, prompts_path):
    prompt_file = prompts_path + initial_prompt_file
    with open(prompt_file, 'r') as promptfile:
        question = promptfile.read()

    return question

def setup_llm(conversation_config):
    model_configs_path = conversation_config['ModelConfigsPath']
    llm_name = conversation_config['LlmName']
    model_config = configparser.ConfigParser()
    model_config.read(model_configs_path + llm_name)
    model_directory = model_config['PATHS']['Directory']
    model_filename = model_config['PATHS']['Filename']
    model_fullpath = conversation_config['ModelsPath'] + model_directory + '/' + model_filename
    n_gpu_layers = model_config['SETTINGS']['NGpuLayers']
    summarize_model_template = read_from_file(conversation_config['TemplatesPath'] + model_config['PROPERTIES']['Template'] + '-summarize')
    summarize_prompt_template = PromptTemplate(template=summarize_model_template, input_variables=["summaries"])
    standalone_model_template = read_from_file(conversation_config['TemplatesPath'] + model_config['PROPERTIES']['Template'] + '-standalone')
    standalone_prompt_template = PromptTemplate(template=standalone_model_template, input_variables=["question", "history", "conversation_memory", "context"])
    context_length = 2048
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    n_batch = 512
    llm = LlamaCpp(
        model_path=model_fullpath,
        n_gpu_layers=n_gpu_layers, n_batch=n_batch,
        max_tokens=context_length / 5.5, n_ctx=context_length,
        callback_manager=callback_manager
    )
    llm.client.verbose = True

    return llm, summarize_prompt_template, standalone_prompt_template, context_length

def setup_short_term_memory(llm, context_length):
    summary_buffer_memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=context_length/8, input_key='question', output_key='answer')

    return summary_buffer_memory

def setup_long_term_memory(embeddings_config, conversation_config):
    model_name = embeddings_config['ModelName']
    model_kwargs = {'device': embeddings_config['ModelDevice']}
    long_term_memories_path = conversation_config['LongTermMemoriesPath']
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        cache_folder = embeddings_config['ModelPath']
    )
    conversational_memory_db = Chroma(
        persist_directory=long_term_memories_path,
        embedding_function=embeddings
    )
    retriever = conversational_memory_db.as_retriever(search_kwargs=dict(k=2))
    class AnswerVectorMemory(VectorStoreRetrieverMemory):
        def save_context(self, inputs: dict[str, any], outputs: dict[str, str]) -> None:
            return super().save_context({'question': inputs['question']},{'answer': outputs['answer']})
    vectorstore_memory = AnswerVectorMemory(retriever=retriever, memory_key='conversation_memory', input_key='question')

    return vectorstore_memory

def setup_combined_memory(llm, context_length, embeddings_config, conversation_config):
    short_term_memory = setup_short_term_memory(llm, context_length)
    long_term_memory = setup_long_term_memory(embeddings_config, conversation_config)
    combined_memory = CombinedMemory(memories=[short_term_memory, long_term_memory])

    return combined_memory

def setup_document_retriever(embeddings_config, document_name):
    document_db_directory = embeddings_config['DocumentsPath'] + document_name + '/db'
    embeddings = prepare_embedding_model(embeddings_config)
    document_db = Chroma(persist_directory=document_db_directory, embedding_function=embeddings)
    document_retriever = document_db.as_retriever()
    document_retriever.search_kwargs['distance_metric'] = 'cos'
    document_retriever.search_kwargs['k'] = 2

    return document_retriever

def setup_conversational_retrieval_chain(embeddings_config, conversation_config, document_name):
    llm, summarize_prompt_template, standalone_prompt_template, context_length = setup_llm(conversation_config)
    document_retriever = setup_document_retriever(embeddings_config, document_name)
    combined_memory = setup_combined_memory(llm, context_length, embeddings_config, conversation_config)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        condense_question_llm=llm,
        retriever=document_retriever,
        condense_question_prompt=summarize_prompt_template,
        chain_type="map_reduce",
        combine_docs_chain_kwargs={
            "question_prompt": standalone_prompt_template,
            "combine_prompt": summarize_prompt_template,
        },
        verbose=True,
        memory=combined_memory,
        return_source_documents=True,
        output_key='answer',
    )

    return chain

def print_result(result):
    print('\n\n\n')
    print('=== Conversation memory: ===\n\n', result['conversation_memory'])
    print('\n')
    print('=== History: ===\n\n', result['history'])
    print('\n')
    for document in result['source_documents']:
        print('=== Dokument: ===\n\n', document.page_content)
        print('\n')
    print('=== Question: ===\n\n', result['question'])
    print('\n')
    print('=== Answer: ===\n\n', result['answer'])

def ask_ai_interactive(llm_chain, initial_query):
    chat_history = []
    result = llm_chain({"question": initial_query, "chat_history": chat_history})
    print_result(result)

    while True:
        query = input("\nRequest: ")

        if query.lower() == "exit":
            break

        result = llm_chain({"question": query, "chat_history": chat_history})
        print_result(result)

def main(args=None):
    parser = argparse.ArgumentParser(description='Try out new LLMs more quickly')

    config = configparser.ConfigParser()
    config.read('config.ini')

    parser.add_argument('task', metavar='TASK', choices=['vectorize', 'converse'])
    parser.add_argument('--config', help='Path to the configuration file', default='config.ini')
    parser.add_argument('--document', help='name of document-specific subdirectory')
    parser.add_argument('--initial-prompt-file', help='name of file in prompt directory to use for prompt')
    parser.add_argument('--initial-prompt', help='initial prompt')
    embeddings_config = config['EMBEDDINGS']
    documents_path = embeddings_config['DocumentsPath']
    conversation_config = config['CONVERSATION']
    prompts_path = conversation_config['PromptsPath']
    default_prompt = conversation_config['DefaultPrompt']

    args = parser.parse_args(args)

    if args.task == 'vectorize':
        if args.document == None:
            print("Please supply --document that you'd like to vectorize")
            return {'arg': args}

        embeddings = prepare_embedding_model(embeddings_config)
        document = args.document
        file_path = document_name_to_file_path(documents_path, document)
        fulltext = pdf_to_text(file_path)
        snippets = text_to_snippets(fulltext, embeddings_config)
        dir_path = document_name_to_dir_path(documents_path, document)
        persist_for_debugging(dir_path, fulltext, snippets)
        vectorize_snippets(dir_path, snippets, embeddings)

        return {'arg': args}

    if args.task == 'converse':
        document = args.document
        if args.initial_prompt_file != None:
            initial_prompt_file = args.initial_prompt_file
            question = read_question(initial_prompt_file, prompts_path)
        elif args.initial_prompt != None:
            question = args.initial_prompt
        else:
            question = default_prompt
        chain = setup_conversational_retrieval_chain(embeddings_config, conversation_config, document)
        ask_ai_interactive(chain, question)

        return {'arg': args}

    return {'arg': args}

if __name__ == "__main__":
    main()
