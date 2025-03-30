import sqlite3
import sqlite_vec

from .data import db_name
from .utils import get_embeddings


def search_knn(query: str, *, k=5, db_name=db_name):
    '''
    Perform a k-nearest neighbors search using sqlite-vec extension.
    Args:
        query (str): The query to search.
        k (int, optional): The number of neighbors to return. Defaults to 5.
        db_name (str): The name of the database file.
    '''
    embedding = get_embeddings([query])[0]
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT paper_id, distance
            FROM embedding
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            ''',
            (sqlite_vec.serialize_float32(embedding), k),
        )
        results = cursor.fetchall()
        return results
