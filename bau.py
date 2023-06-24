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
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

def prepare_embedding_model():
    print('Preparing embedding model...')
    model_name = "intfloat/e5-large-v2"
    model_kwargs = {'device': 'cuda'}
    embedding_model_cache_folder = '/ntfs/models/e5-large-v2/'
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        cache_folder = embedding_model_cache_folder
    )
    print('Embedding model prepared.')

    return embeddings

def document_name_to_dir_path(document_name):
    return '/ntfs/documents/' + document_name + '/'

def document_name_to_file_path(document_name):
    return '/ntfs/documents/' + document_name + '/' + document_name + '.pdf'

def pdf_to_text(doc_path):
    print('Converting PDF to text...')
    loader = PyPDFLoader(doc_path)
    pages = loader.load_and_split()
    fulltext = " ".join(map(lambda page: page.page_content, pages)).replace("\n", " ")
    print('PDF converted to text.')

    return fulltext

def text_to_snippets(fulltext):
    print('Converting text to snippets...')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", '  ', '\.', ", ", " ", ""], keep_separator=False)
    snippets = text_splitter.create_documents([fulltext])
    print('Text converted to snippets.')

    return snippets

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

def main(args=None):
    parser = argparse.ArgumentParser(description='Try out new LLMs more quickly')

    config = configparser.ConfigParser()
    config.read('config.ini')

    parser.add_argument('task', metavar='TASK', choices=['vectorize', 'converse'])
    parser.add_argument('-a','--arg', default=config.get('DEFAULT', 'ArgDefaultValue', fallback=None), help='Help for arg')
    parser.add_argument('--license', action='store_true', help='Display the license information')
    parser.add_argument('--config', help='Path to the configuration file', default='config.ini')
    parser.add_argument('--document', help='name of document-specific subdirectory')

    args = parser.parse_args(args)

    if args.task == 'vectorize':
        if args.document == None:
            print("Please supply --document that you'd like to vectorize")
            return {'arg': args.arg}

        embeddings = prepare_embedding_model()
        document = args.document
        file_path = document_name_to_file_path(document)
        fulltext = pdf_to_text(file_path)
        snippets = text_to_snippets(fulltext)
        dir_path = document_name_to_dir_path(document)
        persist_for_debugging(dir_path, fulltext, snippets)
        vectorize_snippets(dir_path, snippets, embeddings)

        return {'arg': args.arg}

    if args.task == 'converse':
        print('to be implemented')
        return {'arg': args.arg}

    if args.license:
        print_license()
        return {'arg': args.arg, 'license': True}

    return {'arg': args.arg, 'license': False}

def print_license():
    print("""
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
    """)

if __name__ == "__main__":
    main()
