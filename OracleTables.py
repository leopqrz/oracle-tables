import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, SoupStrainer
from urls import urls
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm

only_summary_columns = SoupStrainer(summary="Columns")


def fetch_data_from_url(url):
    try:
        response = requests.get(url)
        # Raise an exception for non-200 status codes
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print("Error accessing URL:", str(e))
        return None


def process_table_data(topic, table_type, table_name, html_text):
    only_summary_columns = SoupStrainer(summary="Columns")
    table = BeautifulSoup(html_text, 'lxml', parse_only=only_summary_columns)
    df = pd.read_html(str(table), flavor='bs4', header=0)[0]

    if table_type == 'Tables':
        df = df[['Name', 'Comments']]
    else:
        series = df['Name'].str.split(expand=True).stack()
        series.name = df.columns[0]
        df = series.to_frame().reset_index(drop=True)
        df['Comments'] = np.nan

    df['Topic'] = topic
    df['Type'] = table_type
    df['Table/View Name'] = table_name

    return df


def oracle_tables(topics: list, update: bool = False):
    file = 'oracle_tables.parquet'
    if update or (not os.path.isfile(file)):
        data_frames = []

        def process_url(topic, table_type, table_name, url):
            html_text = fetch_data_from_url(url)
            if html_text is not None:
                df = process_table_data(topic, table_type,
                                        table_name, html_text)
                return df

        with ThreadPoolExecutor() as executor:
            futures = []
            for topic, tables in tqdm(urls.items()):
                if topic in topics or topics == 'all':
                    for table_type, urls_list in tables.items():
                        for table_name, url in urls_list.items():
                            future = executor.submit(partial(
                                process_url,
                                topic, table_type, table_name, url))
                            futures.append(future)

            for future in tqdm(futures):
                df = future.result()
                if df is not None:
                    data_frames.append(df)

        df = pd.concat(data_frames, ignore_index=True)
        df.to_parquet(file)
    return pd.read_parquet(file)
