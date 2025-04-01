import sqlite3
import sqlite_vec

from .embed import get_embeddings


def search_knn(query: str, *, k=5, db_name):
    '''
    Perform a k-nearest neighbors search using sqlite-vec extension.
    Args:
        query (str): The query to search.
        k (int, optional): The number of neighbors to return. Defaults to 5.
        db_name (str): The name of the database file.
    '''
    embedding = get_embeddings([query])[0]
    with sqlite3.connect(db_name) as conn:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        cursor = conn.cursor()
        cursor.execute(
            '''
                SELECT paper_id, distance
                FROM embedding
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance
            ''',
            (sqlite_vec.serialize_float32(embedding), k),
        )
        results = cursor.fetchall()
        return results
