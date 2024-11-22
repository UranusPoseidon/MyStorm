"""
This STORM Wiki pipeline powered by GPT-3.5/4 and local retrieval model that uses Qdrant.
You need to set up the following environment variables to run this script:
    - OPENAI_API_KEY: OpenAI API key
    - OPENAI_API_TYPE: OpenAI API type (e.g., 'openai' or 'azure')
    - QDRANT_API_KEY: Qdrant API key (needed ONLY if online vector store was used)

You will also need an existing Qdrant vector store either saved in a folder locally offline or in a server online.
If not, then you would need a CSV file with documents, and the script is going to create the vector store for you.
The CSV should be in the following format:
content  | title  |  url  |  description
I am a document. | Document 1 | docu-n-112 | A self-explanatory document.
I am another document. | Document 2 | docu-l-13 | Another self-explanatory document.

Notice that the URL will be a unique identifier for the document so ensure different documents have different urls.

Output will be structured as below
args.output_dir/
    topic_name/  # topic_name will follow convention of underscore-connected topic name w/o space and slash
        conversation_log.json           # Log of information-seeking conversation
        raw_search_results.json         # Raw search results from search engine
        direct_gen_outline.txt          # Outline directly generated with LLM's parametric knowledge
        storm_gen_outline.txt           # Outline refined with collected information
        url_to_info.json                # Sources that are used in the final article
        storm_gen_article.txt           # Final article generated
        storm_gen_article_polished.txt  # Polished final article (if args.do_polish_article is True)
"""
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

import os
from argparse import ArgumentParser

from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs, encoder
from knowledge_storm.rm import VectorRM
from knowledge_storm.lm import OpenAIModel, AzureOpenAIModel
from knowledge_storm.utils import load_api_key, QdrantVectorStoreManager

import csv
import openai
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Example usage
# qdrant_client = QdrantClient(path='path_to_vector_store')
# convert_markdown_to_corpus_and_add_to_qdrant('path_to_markdown.md', qdrant_client, 'my_collection', model)

# Convert the markdown file to the required format and save it in the specified directory

# def convert_markdown_to_csv(markdown_file_path, csv_file_path):
#     """
#     Convert a markdown document to a CSV format suitable for the project.

#     Args:
#         markdown_file_path (str): Path to the markdown file.
#         csv_file_path (str): Path where the CSV file will be saved.
#     """
#     import pandas as pd

#     # Read the markdown file
#     with open(markdown_file_path, 'r', encoding='utf-8',errors='replace') as file:
#         content = file.read()

#     # Split the content into sections based on headers
#     sections = content.split('\n# ')
#     documents = []

#     for section in sections:
#         lines = section.split('\n')
#         title = lines[0].strip() if lines else "Untitled"
#         body = "\n".join(lines[1:]).strip()
#         url = f"doc-{hash(title)}"  # Generate a unique URL based on the title
#         description = body[:100]  # Use the first 100 characters as a description

#         documents.append({
#             "content": body,
#             "title": title,
#             "url": url,
#             "description": description
#         })

#     # Create a DataFrame and save it as a CSV
#     df = pd.DataFrame(documents)
#     df.to_csv(csv_file_path, index=False, encoding='utf-8-sig',errors='replace')

# # Example usage
# convert_markdown_to_csv(
#     markdown_file_path='D:\STORM\storm-main\whzgd6350f90c1b2e4.md',
#     csv_file_path='D:/STORM/storm-main/examples/storm_examples/converted_document2.csv'
# )
# import pandas as pd

# def merge_csv_files(file_path1, file_path2, output_file_path):
#     """
#     Merge two CSV files with the same format into a single CSV file.

#     Args:
#         file_path1 (str): Path to the first CSV file.
#         file_path2 (str): Path to the second CSV file.
#         output_file_path (str): Path where the merged CSV file will be saved.
#     """
#     # Read the CSV files
#     df1 = pd.read_csv(file_path1, encoding='utf-8')
#     df2 = pd.read_csv(file_path2, encoding='utf-8')

#     # Concatenate the DataFrames
#     merged_df = pd.concat([df1, df2], ignore_index=True)

#     # Save the merged DataFrame to a new CSV file
#     merged_df.to_csv(output_file_path, index=False, encoding='utf-8-sig', errors='replace')

# # Example usage
# merge_csv_files(
#     file_path1='D:/STORM/storm-main/examples/storm_examples/converted_document.csv',
#     file_path2='D:/STORM/storm-main/examples/storm_examples/converted_document2.csv',
#     output_file_path='D:/STORM/storm-main/examples/storm_examples/merged_documents.csv'
# )

def main(args):
    args.vector_db_mode = 'offline'
    args.offline_vector_db_dir='D:/STORM/storm-main/examples/storm_examples'
    # Load API key from the specified toml file path
    load_api_key(toml_file_path='secrets.toml')

    # Initialize the language model configurations
    engine_lm_configs = STORMWikiLMConfigs()
    openai_kwargs = {
        'api_key': os.getenv("OPENAI_API_KEY"),
        'temperature': 1.0,
        'top_p': 0.9,
    }
    
    openai_kwargs['api_base'] = 'https://api.closeai-proxy.xyz/v1/'
    openai.api_base = 'https://api.closeai-proxy.xyz/v1/'

    ModelClass = OpenAIModel if os.getenv('OPENAI_API_TYPE') == 'openai' else AzureOpenAIModel
    # If you are using Azure service, make sure the model name matches your own deployed model name.
    # The default name here is only used for demonstration and may not match your case.
    gpt_35_model_name = 'gpt-4o-mini' if os.getenv('OPENAI_API_TYPE') == 'openai' else 'gpt-35-turbo'
    gpt_4_model_name = 'gpt-4o'
    if os.getenv('OPENAI_API_TYPE') == 'azure':
        openai_kwargs['api_base'] = os.getenv('AZURE_API_BASE')
        openai_kwargs['api_version'] = os.getenv('AZURE_API_VERSION')
    
    openai_kwargs['api_base'] = 'https://api.closeai-proxy.xyz/v1/'

    # STORM is a LM system so different components can be powered by different models.
    # For a good balance between cost and quality, you can choose a cheaper/faster model for conv_simulator_lm 
    # which is used to split queries, synthesize answers in the conversation. We recommend using stronger models
    # for outline_gen_lm which is responsible for organizing the collected information, and article_gen_lm
    # which is responsible for generating sections with citations.
    conv_simulator_lm = ModelClass(model=gpt_35_model_name, max_tokens=500, **openai_kwargs)
    question_asker_lm = ModelClass(model=gpt_35_model_name, max_tokens=500, **openai_kwargs)
    outline_gen_lm = ModelClass(model=gpt_4_model_name, max_tokens=400, **openai_kwargs)
    article_gen_lm = ModelClass(model=gpt_4_model_name, max_tokens=700, **openai_kwargs)
    article_polish_lm = ModelClass(model=gpt_4_model_name, max_tokens=4000, **openai_kwargs)
    conv_simulator_lm = OpenAIModel(model='gpt-4o-mini', max_tokens=500, **openai_kwargs)
    question_asker_lm = OpenAIModel(model='gpt-4o-mini', max_tokens=500, **openai_kwargs)
    outline_gen_lm = OpenAIModel(model='gpt-4-0125-preview', max_tokens=400, **openai_kwargs)
    article_gen_lm = OpenAIModel(model='gpt-4-0125-preview', max_tokens=700, **openai_kwargs)
    article_polish_lm = OpenAIModel(model='gpt-4-0125-preview', max_tokens=4000, **openai_kwargs)

    engine_lm_configs.set_conv_simulator_lm(conv_simulator_lm)
    engine_lm_configs.set_question_asker_lm(question_asker_lm)
    engine_lm_configs.set_outline_gen_lm(outline_gen_lm)
    engine_lm_configs.set_article_gen_lm(article_gen_lm)
    engine_lm_configs.set_article_polish_lm(article_polish_lm)

    # Initialize the engine arguments
    engine_args = STORMWikiRunnerArguments(
        output_dir=args.output_dir,
        max_conv_turn=args.max_conv_turn,
        max_perspective=args.max_perspective,
        search_top_k=args.search_top_k,
        max_thread_num=args.max_thread_num,
    )

    
#Create / update the vector store with the documents in the csv file
    if args.csv_file_path:
        kwargs = {
            'file_path': args.csv_file_path,
            'content_column': 'content',
            'title_column': 'title',
            'url_column': 'url',
            'desc_column': 'description',
            'batch_size': args.embed_batch_size,
            'vector_db_mode': args.vector_db_mode,
            'collection_name': args.collection_name,
            'embedding_model': args.embedding_model,
            'device': args.device,
        }
        if args.vector_db_mode == 'offline':
            QdrantVectorStoreManager.create_or_update_vector_store(
                vector_store_path=args.offline_vector_db_dir,
                **kwargs
            )
        elif args.vector_db_mode == 'online':
            QdrantVectorStoreManager.create_or_update_vector_store(
                url=args.online_vector_db_url,
                api_key=os.getenv('QDRANT_API_KEY'),
                **kwargs
            )
    
    # Setup VectorRM to retrieve information from your own data
    rm = VectorRM(collection_name=args.collection_name, embedding_model=args.embedding_model, device=args.device, k=engine_args.search_top_k)

    # initialize the vector store, either online (store the db on Qdrant server) or offline (store the db locally):
    if args.vector_db_mode == 'offline':
        rm.init_offline_vector_db(vector_store_path=args.offline_vector_db_dir)
    elif args.vector_db_mode == 'online':
        rm.init_online_vector_db(url=args.online_vector_db_url, api_key=os.getenv('QDRANT_API_KEY'))

    # Update the vector store with the documents in the csv file
    if args.update_vector_store:
        rm.update_vector_store(
            file_path=args.csv_file_path,
            content_column='content',
            title_column='title',
            url_column='url',
            desc_column='description',
            batch_size=args.embed_batch_size
        )

    # Initialize the STORM Wiki Runner
    runner = STORMWikiRunner(engine_args, engine_lm_configs, rm)

    # run the pipeline
    topic = input("Enter the topic: ")
    runner.run(
        topic=topic,
        do_research=args.do_research,
        do_generate_outline=args.do_generate_outline,
        do_generate_article=args.do_generate_article,
        do_polish_article=args.do_polish_article,
    )
    runner.post_run()
    runner.summary()

import argparse
if __name__ == "__main__":
    parser = ArgumentParser()
    # global arguments
    parser = argparse.ArgumentParser(description="Run STORM Wiki GPT with VectorRM")
    parser.add_argument('--output-dir', type=str, default='./results/gpt_retrieval',
                        help='Directory to store the outputs.')
    parser.add_argument('--max-thread-num', type=int, default=3,
                        help='Maximum number of threads to use. The information seeking part and the article generation'
                             'part can speed up by using multiple threads. Consider reducing it if keep getting '
                             '"Exceed rate limit" error when calling LM API.')
    # provide local corpus and set up vector db
    parser.add_argument('--collection-name', type=str, default="my_documents",
                        help='The collection name for vector store.')
    parser.add_argument('--embedding_model', type=str, default="BAAI/bge-m3",
                        help='The collection name for vector store.')
    parser.add_argument('--device', type=str, default="mps",
                        help='The device used to run the retrieval model (mps, cuda, cpu, etc).')
    parser.add_argument('--vector-db-mode', type=str, choices=['offline', 'online'],
                        help='The mode of the Qdrant vector store (offline or online).')
    parser.add_argument('--offline-vector-db-dir', type=str, default='./vector_store',
                        help='If use offline mode, please provide the directory to store the vector store.')
    parser.add_argument('--online-vector-db-url', type=str,
                        help='If use online mode, please provide the url of the Qdrant server.')
    parser.add_argument('--csv-file-path', type=str, default=None,
                        help='The path of the custom document corpus in CSV format. The CSV file should include '
                             'content, title, url, and description columns.')
    parser.add_argument('--embed-batch-size', type=int, default=64,
                        help='Batch size for embedding the documents in the csv file.')
    # stage of the pipeline
    parser.add_argument('--do-research', action='store_true',
                        help='If True, simulate conversation to research the topic; otherwise, load the results.')
    parser.add_argument('--do-generate-outline', action='store_true',
                        help='If True, generate an outline for the topic; otherwise, load the results.')
    parser.add_argument('--do-generate-article', action='store_true',
                        help='If True, generate an article for the topic; otherwise, load the results.')
    parser.add_argument('--do-polish-article', action='store_true',
                        help='If True, polish the article by adding a summarization section and (optionally) removing '
                             'duplicate content.')
    # hyperparameters for the pre-writing stage
    parser.add_argument('--max-conv-turn', type=int, default=3,
                        help='Maximum number of questions in conversational question asking.')
    parser.add_argument('--max-perspective', type=int, default=3,
                        help='Maximum number of perspectives to consider in perspective-guided question asking.')
    parser.add_argument('--search-top-k', type=int, default=3,
                        help='Top k search results to consider for each search query.')
    # hyperparameters for the writing stage
    parser.add_argument('--retrieve-top-k', type=int, default=3,
                        help='Top k collected references for each section title.')
    parser.add_argument('--remove-duplicate', action='store_true',
                        help='If True, remove duplicate content from the article.')
    parser.add_argument('--update-vector-store', action='store_true', help='Update the vector store if set')
    main(parser.parse_args())