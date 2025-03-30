from pydantic import BaseModel, Field


class Paper(BaseModel):
    '''
    Data model for a Paper.
    '''
    paper_id: int = Field(description='ID of the paper')
    submitter: str | None = Field(description='Submitter\'s full name')
    authors: list[str] = Field(description='List of author\'s full names')
    title: str = Field(description='Paper title')
    journal: str | None = Field(description='Journal name')
    doi: str | None = Field(description='Digital Object Identifier')
    categories: list[str] = Field(description='List of categories')
    abstract: str = Field(description='Paper abstract')
    update_time: int = Field(
        description='Update timestamp as seconds since the UNIX epoch.'
    )


