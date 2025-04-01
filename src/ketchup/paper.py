import sqlite3

import polars as pl
from pydantic import BaseModel, Field


class Paper(BaseModel):
    '''
    Data model for a Paper.
    '''
    paper_id: str = Field(description='ID of the paper')
    submitter: str | None = Field(description='Submitter\'s full name')
    authors: list[str] = Field(description='List of author\'s full names')
    title: str = Field(description='Paper title')
    journal: str | None = Field(description='Journal name')
    doi: str | None = Field(description='Digital Object Identifier')
    categories: list[str] = Field(description='List of categories')
    abstract: str = Field(description='Paper abstract')
    year: int = Field(description='Publication year')


def get_papers(paper_ids: list[str], db_name: str) -> list[Paper]:
    '''
    Get paper data from a list of paper IDs.
    Args:
        paper_ids (list[str]): List of paper IDs to retrieve.
        db_name (str): The name of the database file.
    Returns:
        (list[Paper]): Paper data.
    '''
    query = f'''
        SELECT 
            p.paper_id,
            s.full_name AS submitter,
            (
                SELECT GROUP_CONCAT(a2.full_name, ', ')
                FROM authorship ap2
                LEFT JOIN person a2 ON ap2.author_id = a2.person_id
                WHERE ap2.paper_id = p.paper_id
                ORDER BY ap2.ordering
            ) AS authors,
            p.title,
            p.journal,
            p.doi,
            p.abstract,
            p.year,
            GROUP_CONCAT(c.name, ', ') AS categories
        FROM paper p
        LEFT JOIN person s ON p.submitter_id = s.person_id
        LEFT JOIN authorship ap ON p.paper_id = ap.paper_id
        LEFT JOIN person a ON ap.author_id = a.person_id
        LEFT JOIN paper_category pc ON p.paper_id = pc.paper_id
        LEFT JOIN category c ON pc.category_id = c.category_id
        WHERE p.paper_id IN (%s)
        GROUP BY p.paper_id
    ''' % (', '.join(list(map(lambda s: "'%s'" % s, paper_ids))))
    with sqlite3.connect(db_name) as conn:
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            papers = [
                Paper(
                    paper_id=row[0],
                    submitter=row[1],
                    authors=row[2].split(', '),
                    title=row[3],
                    journal=row[4],
                    doi=row[5],
                    abstract=row[6],
                    year=row[7],
                    categories=row[8].split(', '),
                )
                for row in rows
            ]
            return papers
        except sqlite3.Error as e:
            print(f'Error fetching papers: {e}')
            return []
        

def get_abstracts(paper_ids = None, *, db_name: str) -> pl.DataFrame | None:
    '''
    Get abstracts for a given list of paper IDs, or get all abstracts.
    Args:
        paper_ids (list[str]): List of paper IDs to retrieve.
            If None, get all abstracts.
        db_name (str): The name of the database file.
    Returns:
        (pl.DataFrame | None): If success, returns a Polars DataFrame with two
            columns: paper IDs `paper_id` and abstract `abstract`. Otherwise,
            returns None.
    '''
    query = 'SELECT paper_id, abstract FROM paper'
    if paper_ids is not None:
        query += ' WHERE paper_id IN (%s)' % (
            ', '.join(list(map(lambda s: "'%s'" % s, paper_ids)))
        )
    with sqlite3.connect(db_name) as conn:
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            return pl.DataFrame(
                rows,
                { 'paper_id': pl.String, 'abstract': pl.String },
                orient='row',
            )
        except sqlite3.Error as e:
            print(f'Error fetching papers: {e}')
            return None
