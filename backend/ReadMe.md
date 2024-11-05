# Update the database

### 1. Directory Structure Overview

```bash
# directory structure
├── chroma # vector database
├── data 
│     ├── papers # folder for storing .pdf files
│     └── texts # folder for storing text files (e.g., .md, .txt)
└── src
```
Place any .pdf files you want to process into the data/papers directory.

Place any text files (e.g., .md, .txt) into the data/texts directory.

### 2. Update the database
To update the vector database with new embeddings from the files in `data/papers` and `data/texts`, use the following command:
```    
python3 src/vectordb.py
```

This command will:
- Locate all files in data/papers and data/texts.
- Split (chunk) the documents into manageable parts.
- Request embeddings for each chunk from GPT, updating the database with these embeddings.

**Important Notes**:
- Add only the files you want to include in the database.
- Remove any old or redundant files from the data/papers and data/texts directories to avoid duplicate entries in the database.