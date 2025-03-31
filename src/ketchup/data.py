from itertools import chain
import sqlite3

import polars as pl
import sqlite_vec
from tqdm import tqdm

from .paper import Paper
from .utils import flush, get_embeddings, sentencize_batch


def initialize_data(*, dummy=False, batch_size=1000, db_name: str):
    '''
    Initialize data for the application.
    Args:
        dummy (bool): If True, use a tiny portion of real data as dummy data.
            Default to False.
        batch_size (int, optional): The number of elements to load in a batch.
            Default to 1000.
        db_name (str): The name of the database file.
    '''
    print('Creating database...')
    _create_database(db_name=db_name)
    print('Loading data...')
    df = _load_polars_data(dummy=dummy)
    print('Retrieving data about people...')
    people, person2id = _retrieve_people(df)
    print('Retrieving data about categories...')
    categories, category2id = _retrieve_categories(df)
    print('Standardizing data...')
    df = _standardize_polars_data(df, person2id, category2id)
    print('Inserting people into database...')
    _insert_people(list(people.items()), batch_size=batch_size, db_name=db_name)
    print('Inserting categories into database...')
    _insert_categories(list(categories.items()), batch_size=batch_size, db_name=db_name)
    print('Inserting papers into database...')
    _insert_papers(df, batch_size=batch_size, db_name=db_name)
    print('Inserting authorships into database...')
    _insert_authorships(df, batch_size=batch_size, db_name=db_name)
    print('Inserting paper\'s categories into database...')
    _insert_paper_categories(df, batch_size=batch_size, db_name=db_name)
    print('Inserting embeddings into database...')
    _insert_embeddings(df, batch_size=batch_size, db_name=db_name)
    print('✅ Complete initializing data')
    flush(verbose=False)

    
def get_papers(paper_ids: list[int], db_name: str) -> list[Paper]:
    '''
    Get paper data from a list of paper IDs.
    Args:
        paper_ids (list[int]): List of paper IDs to retrieve.
        db_name (str): The name of the database file.
    Returns:
        (list[Paper]): Paper data.
    '''
    query = f'''
        SELECT 
            p.paper_id,
            s.full_name AS submitter,
            GROUP_CONCAT(a.full_name, ', ' ORDER BY ap.ordering) AS authors,
            p.title,
            p.journal,
            p.doi,
            p.abstract,
            p.year,
            GROUP_CONCAT(c.category, ', ') AS categories
        FROM paper p
        LEFT JOIN person s ON p.submitter_id = s.person_id
        LEFT JOIN authorship ap ON p.paper_id = ap.paper_id
        LEFT JOIN person a ON ap.author_id = a.person_id
        LEFT JOIN paper_category pc ON p.paper_id = pc.paper_id
        LEFT JOIN category c ON pc.category_id = c.category_id
        WHERE p.paper_id IN ({', '.join(list(map(str, paper_ids)))})
        GROUP BY p.paper_id
    '''
    with sqlite3.connect(db_name) as conn:
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            papers = [Paper(**dict(zip(cursor.column_names, row))) for row in rows]
            return papers
        except sqlite3.Error as e:
            print(f'Error fetching papers: {e}')
            return []


def _create_database(db_name):
    '''
    Create necessary tables for the database.
    Args:
        db_name (str): Name of the database file.
    '''
    queries = (
        '''
        CREATE TABLE IF NOT EXISTS person (
            person_id INTEGER PRIMARY KEY,
            full_name TEXT NOT NULL,
            CONSTRAINT uq__person__full_name
                UNIQUE (full_name) ON CONFLICT ROLLBACK
        );
        CREATE TABLE IF NOT EXISTS category (
            category_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            CONSTRAINT uq__category__name UNIQUE (name) ON CONFLICT ROLLBACK
        );
        CREATE TABLE IF NOT EXISTS paper (
            paper_id INTEGER PRIMARY KEY,
            submitter_id INTEGER,
            title TEXT NOT NULL,
            journal TEXT,
            doi TEXT,
            abstract TEXT NOT NULL,
            year INTEGER NOT NULL,
            CONSTRAINT fk__paper__submitter_id
                FOREIGN KEY (submitter_id) REFERENCES person (person_id)
        );
        CREATE TABLE IF NOT EXISTS authorship (
            authorship_id INTEGER PRIMARY KEY,
            paper_id INTEGER,
            author_id INTEGER,
            ordering INTEGER NOT NULL,
            CONSTRAINT fk__authorship__paper_id
                FOREIGN KEY (paper_id) REFERENCES paper (paper_id),
            CONSTRAINT fk__authorship__author_id
                FOREIGN KEY (author_id) REFERENCES person (person_id),
            CONSTRAINT uq__authorship__paper_author
                UNIQUE (paper_id, author_id) ON CONFLICT ROLLBACK
        );
        CREATE TABLE IF NOT EXISTS paper_category (
            paper_category_id INTEGER PRIMARY KEY,
            paper_id INTEGER,
            category_id INTEGER,
            CONSTRAINT fk__paper_category__paper_id
                FOREIGN KEY (paper_id) REFERENCES paper (paper_id)
            CONSTRAINT fk__paper_category__category_id
                FOREIGN KEY (category_id) REFERENCES category (category_id)
        );
        CREATE VIRTUAL TABLE embedding using vec0(
            embedding float[768],
            +paper_id INTEGER
        )
        '''
    ).split(';')
    with sqlite3.connect(db_name) as conn:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        cursor = conn.cursor()
        for query in queries:
            cursor.execute(query)
        conn.commit()


def _load_polars_data(*, dummy=False):
    '''
    Load data as Polars DataFrame.
    Args:
        dummy (bool): If True, use a tiny portion real data as dummy data.
            Default to False.
    Returns:
        (pl.DataFrame): Polars DataFrame containing the data.
    '''
    df = (
        pl.scan_ndjson(
            'hf://datasets/UniverseTBD/arxiv-abstracts-large/'
            'arxiv-metadata-oai-snapshot.json'
        )
        .rename({ 'id': 'paper_id', 'journal-ref': 'journal' })
        .with_columns(
            pl.col('update_date')
            .str.slice(0, 4)
            .cast(pl.Int64)
            .alias('year'),
        )
        .select(
            'paper_id', 'submitter', 'authors', 'title', 'journal', 'doi',
            'categories', 'abstract', 'year',
        )
        .collect()
    )
    if dummy:
        df = df.sample(1000, seed=2025)
    data_size = len(df)
    with tqdm(total=data_size * 2) as pbar:
        def fn(x):
            pbar.update(1)
            return list(map(lambda s: s.strip(), x))
        return df.with_columns(
            pl.col('authors')
            .str.split(',')
            .map_elements(fn, pl.List(pl.String)),
            pl.col('categories')
            .str.split(' ')
            .map_elements(fn, pl.List(pl.String)),
        )


def _standardize_polars_data(
        df: pl.DataFrame,
        person2id: dict[str, int],
        category2id: dict[str, int],
):
    '''
    Standardize data for the application.
    Args:
        df (pl.DataFrame): Polars DataFrame containing the data.
        person2id (dict[str, int]): Mapping of person names to person IDs.
        category2id (dict[str, int]): Mapping of category names to category IDs.
    Returns:
        (pl.DataFrame): Standardized Polars DataFrame.
    '''
    return (
        df.lazy()
        .with_columns(
            pl.col('submitter')
            .map_elements(person2id.__getitem__, pl.Int64)
            .alias('submitter_id'),
            pl.col('authors')
            .map_elements(
                lambda x: list(map(person2id.__getitem__, x)),
                pl.List(pl.Int64),
            )
            .alias('author_ids'),
            pl.col('categories')
            .map_elements(
                lambda x: list(map(category2id.__getitem__, x)),
                pl.List(pl.Int64),
            )
            .alias('category_ids'),
        )
        .collect()
    )


def _retrieve_people(df: pl.DataFrame):
    '''
    Retrieve unique list of people's names from orginal data.
    Args:
        df (pl.DataFrame): Original Polars DataFrame.
    Returns:
        (tuple[dict[int, str], dict[str, int]]): Tuple of two dicts, one maps
            person IDs to their names, and the other maps the other direction.
    '''
    people = set(
        chain.from_iterable(df.select('authors').to_series().to_list()),
    )
    people.update(df.select('submitter').to_series().to_list())
    people = { i: person for i, person in enumerate(people, 1000) }
    person2id = { person: i for i, person in people.items() }
    return people, person2id


def _retrieve_categories(df: pl.DataFrame):
    '''
    Retrieve unique list of categories from orginal data.
    Args:
        df (pl.DataFrame): Original Polars DataFrame.
    Returns:
        (tuple[dict[int, str], dict[str, int]]): Tuple of two dicts, one maps
            category IDs to their names, and the other maps the other direction.
    '''
    categories = set(
        chain.from_iterable(df.select('categories').to_series().to_list()),
    )
    categories = { i: category for i, category in enumerate(categories, 1000) }
    category2id = { category: i for i, category in categories.items() }
    return categories, category2id


def _insert_papers(
        df: pl.DataFrame,
        *,
        batch_size=1000,
        db_name,
):
    '''
    Insert paper data into the database.
    Args:
        df (pl.DataFrame): Polars DataFrame.
        batch_size (int): Size of batch for insertion.
        db_name (str): Name of the database file.
    '''
    _insert_batch(
        '''
        INSERT INTO paper (
            paper_id, submitter_id, title, journal, doi, abstract, year
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''',
        df.select(
            'paper_id',
            'submitter_id',
            'title',
            'journal',
            'doi',
            'abstract',
            'year',
        ),
        transform=lambda x: x.rows(),
        batch_size=batch_size,
        db_name=db_name,
    )


def _insert_categories(
        categories: list[tuple[int, str]],
        *,
        batch_size=1000,
        db_name,
):
    '''
    Insert category data into the database.
    Args:
        df (pl.DataFrame): Polars DataFrame.
        batch_size (int): Size of batch for insertion.
        db_name (str): Name of the database file.
    '''
    _insert_batch(
        'INSERT INTO category (category_id, name) VALUES (?, ?)',
        categories,
        batch_size=batch_size,
        db_name=db_name,
    )


def _insert_people(
        people: list[tuple[int, str]],
        *,
        batch_size=1000,
        db_name,
):
    '''
    Insert person data into the database.
    Args:
        df (pl.DataFrame): Polars DataFrame.
        batch_size (int): Size of batch for insertion.
        db_name (str): Name of the database file.
    '''
    _insert_batch(
        'INSERT INTO person (person_id, full_name) VALUES (?, ?)',
        people,
        batch_size=batch_size,
        db_name=db_name,
    )


def _insert_authorships(
        df: pl.DataFrame,
        *,
        batch_size=1000,
        db_name,
):
    '''
    Insert authorship data (author-paper relation) into the database.
    Args:
        df (pl.DataFrame): Polars DataFrame.
        batch_size (int): Size of batch for insertion.
        db_name (str): Name of the database file.
    '''
    authorships = (
        df.lazy()
        .select('paper_id', 'author_ids')
        .with_columns(
            pl.col('author_ids')
            .map_elements(
                lambda x: list(enumerate(x, 1)),
                pl.List(
                    pl.Struct({ 'ordering': pl.Int64, 'author_id': pl.Int64 }),
                ),
            ),
        )
        .explode('author_ids')
        .unnest('author_ids')
        .with_row_index('authorship_id', 1000)
        .select('authorship_id', 'paper_id', 'author_id', 'ordering')
        .collect()
    )
    _insert_batch(
        '''
        INSERT INTO authorship (authorship_id, paper_id, author_id, ordering)
        VALUES (?,?,?,?)
        ''',
        authorships,
        transform=lambda x: x.rows(),
        batch_size=batch_size,
        db_name=db_name,
    )


def _insert_paper_categories(
        df: pl.DataFrame,
        *,
        batch_size=1000,
        db_name,
):
    '''
    Insert paper-category relation data into the database.
    Args:
        df (pl.DataFrame): Polars DataFrame.
        batch_size (int): Size of batch for insertion.
        db_name (str): Name of the database file.
    '''
    paper_categories = (
        df.lazy()
        .select('paper_id', 'category_ids')
        .explode('category_ids')
        .with_row_index('paper_category_id', 1000)
        .select('paper_category_id', 'paper_id', 'category_ids')
        .collect()
        .rows()
    )
    _insert_batch(
        '''
        INSERT INTO paper_category (paper_category_id, paper_id, category_id)
        VALUES (?,?,?)
        ''',
        paper_categories,
        batch_size=batch_size,
        db_name=db_name,
    )


def _insert_embeddings(
        df: pl.DataFrame,
        *,
        batch_size=1000,
        db_name,
):
    '''
    Encode and insert embedding data into the database.
    Args:
        df (pl.DataFrame): Polars DataFrame.
        batch_size (int): Size of batch for insertion.
        db_name (str): Name of the database file.
    '''
    def transform(df_: pl.DataFrame):
        embeddings = get_embeddings(
            df_.select('sentence').to_series().to_list(),
        )
        df_ = (
            df_.with_columns(
                pl.Series('embedding', embeddings, pl.List(pl.Float32))
            )
            .select('embedding', 'paper_id')
        )
        return list(map(
            lambda row: (sqlite_vec.serialize_float32(row[0]), row[1]),
            df_.rows(),
        ))

    _insert_batch(
        'INSERT INTO embedding (embedding, paper_id) VALUES (?,?)',
        (
            df.select('paper_id', 'abstract')
            .with_columns(
                pl.Series(
                    'sentence',
                    sentencize_batch(
                        df.select('abstract').to_series().to_list(),
                    ),
                )
            )
            .drop('abstract')
            .explode('sentence')
            .select('paper_id', 'sentence')
        ),
        transform=transform,
        batch_size=batch_size,
        db_name=db_name,
    )


def _insert_batch(
        sql: str,
        parameters,
        *,
        transform=None,
        transform_each=None,
        batch_size=1000,
        db_name,
):
    '''
    Perform INSERT prepared statements on a batch of data.
    Args:
        sql (str): SQL query with placeholders.
        parameters (list): List of parameters to be used in the SQL query.
        transform (function, optional): Function to transform each batch.
        transform_each (function, optional): Function to transform each item in the batch.
        batch_size (int, optional): Size of batch for insertion.
        db_name (str, optional): Name of the database file.
    '''
    assert not (transform and transform_each), \
        'transform and transform_each cannot be used together'
    with sqlite3.connect(db_name) as conn:
        try:
            conn.execute('PRAGMA foreign_keys = 1')
            cursor = conn.cursor()
            for i in range(0, len(parameters), batch_size):
                cursor.execute('BEGIN TRANSACTION')
                batch = parameters[i:i + batch_size]
                if transform_each:
                    batch = list(map(transform_each, batch))
                if transform:
                    batch = transform(batch)
                cursor.executemany(sql, batch)
                conn.commit()
        except sqlite3.Error as e:
            print(f'Error inserting batch: {e}')
            print('Last SQL:')
            print(sql.strip())
            if conn:
                print('Rolling back...')
                conn.rollback()
    flush(verbose=False)

