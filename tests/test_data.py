import sqlite3

from ketchup.data import insert_embeddings

from utils import db_name


def test_initialize_data():
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()
        _test_table_authorship(cursor)
        _test_table_category(cursor)
        _test_table_paper(cursor)
        _test_table_paper_category(cursor)
        _test_table_person(cursor)


def test_insert_embeddings():
    insert_embeddings(
        batch_size=3, 
        preprocess_dataframe=lambda df: df.head(12), 
        db_name=db_name,
    )
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT paper_id, embedding FROM embedding LIMIT 12')
        rows = cursor.fetchall()
        assert len(rows) == 12, 'Not all embeddings retrieved'
        for row in rows:
            assert isinstance(row[0], str), 'Invalid paper ID'
            assert isinstance(row[1], bytes), 'Invalid embedding'


def _test_table_paper(cursor: sqlite3.Cursor):
    cursor.execute('''
SELECT paper_id, submitter_id, title, journal, doi, abstract, year
FROM paper
LIMIT 10
''')
    rows = cursor.fetchall()
    assert len(rows) == 10, 'Not all papers retrieved'
    for row in rows:
        assert isinstance(row[0], str), 'Invalid paper ID'
        assert isinstance(row[1], int) or row[1] is None, \
            'Invalid submitter ID'
        assert isinstance(row[2], str), 'Invalid title'
        assert isinstance(row[3], str) or row[3] is None, 'Invalid journal'
        assert isinstance(row[4], str) or row[4] is None, 'Invalid DOI'
        assert isinstance(row[5], str), 'Invalid abstract'
        assert isinstance(row[6], int) or row[6] is None, 'Invalid year'


def _test_table_paper_category(cursor: sqlite3.Cursor):
    cursor.execute('''
SELECT paper_category_id, paper_id, category_id
FROM paper_category
LIMIT 10
''')
    rows = cursor.fetchall()
    assert len(rows) == 10, 'Not all paper-category associations retrieved'
    for row in rows:
        assert isinstance(row[0], int), 'Invalid paper-category ID'
        assert isinstance(row[1], str), 'Invalid paper ID'
        assert isinstance(row[2], int), 'Invalid category ID'


def _test_table_authorship(cursor: sqlite3.Cursor):
    cursor.execute('''
SELECT authorship_id, paper_id, author_id, ordering
FROM authorship
LIMIT 10
''')
    rows = cursor.fetchall()
    assert len(rows) == 10, 'Not all authorships retrieved'
    for row in rows:
        assert isinstance(row[0], int), 'Invalid authorship ID'
        assert isinstance(row[1], str), 'Invalid paper ID'
        assert isinstance(row[2], int), 'Invalid author ID'
        assert isinstance(row[3], int) or row[3] is None, 'Invalid ordering'


def _test_table_category(cursor: sqlite3.Cursor):
    cursor.execute('SELECT category_id, name FROM category LIMIT 10')
    rows = cursor.fetchall()
    assert len(rows) == 10, 'Not all categories retrieved'
    for row in rows:
        assert isinstance(row[0], int), 'Invalid category ID'
        assert isinstance(row[1], str), 'Invalid category name'


def _test_table_person(cursor: sqlite3.Cursor):
    cursor.execute('SELECT person_id, full_name FROM person LIMIT 10')
    rows = cursor.fetchall()
    assert len(rows) == 10, 'Not all people retrieved'
    for row in rows:
        assert isinstance(row[0], int), 'Invalid person ID'
        assert isinstance(row[1], str), 'Invalid full name'

