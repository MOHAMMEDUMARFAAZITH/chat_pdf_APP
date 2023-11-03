# CHAT PDF APP

# Libraries 

1. import torch
2. import time
3. import streamlit as st
4. import numpy as np
5. from PyPDF2 import PdfReader
6. from sentence_transformers import util
7. from langchain.text_splitter import
8. RecursiveCharacterTextSplitter
9. import requests
10. from PIL import Image



## Procedure

1. The user provides a PDF file and to asks a question then the application will attempt to answer it based on the provided source.

2. We will extract the content using the PYPDF2 library.

3. we will split it into chunks using the RecursiveCharacterTextSplitter from the langchain library.

4. Calculate the corresponding word embedding vector using "all-MiniLM-L6-v2" model.

5. Embedding maps sentences & paragraphs to a 384 dimensional dense vector space.

6. word embedding is just a technique to represent word/sentence as a vector.

7. The same technique is applied to the user question.

8. The vectors are given as input to the semantic search function provided by sentence_transformers which is a  Python framework for state-of-the-art sentence, text and image embeddings.

9. This function will return the text chunk that may contain the answer.

10. the Question Answering model will generate the final answer based on the output of the semantic_search + user question.

## Deployment

To deploy this project run

```bash
  https://app1py-am8avyl54q5x4susmvthbe.streamlit.app/
```
```bash
https://drive.google.com/file/d/1Cqj_7t4sA7yxIwaqQojX4XYPxLrclBc1/view?usp=sharing
```

## Appendix

Any additional information goes here

https://huggingface.co/

https://www.langchain.com/

https://www.ai21.com/studio
## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/MOHAMMEDUMARFAAZITH)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mohammed-umar-faazith-k-176b621b3/)
