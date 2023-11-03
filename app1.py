#Importing Necessary libraries
import torch
import time
import streamlit as st
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import util
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from PIL import Image

#sample chatbot image
image = Image.open(r"chatbot.png")
st.image(image,width=675)


#sidebar contains some useful documentation link to develop a chatbot
with st.sidebar: 
    image = Image.open(r"chatbot.png")
    st.image(image)
    st.title("CHAT PDF APP")
    st.write("")
    st.write("")
    st.markdown('''
This is a Chat Pdf App using open source linraries, all without an OpenAI API key:

-  [Streamlit](https://huggingface.co/)

-  [LangChain](https://www.langchain.com/)

-  [AI21studio ](https://www.ai21.com/studio)
''')
    st.write("")
    st.write("")
    st.write("Made by [MOHAMMED UMAR FAAZITH K]")

#This is a streamlit main function for chatbot
def main():
	st.header('HELLO, *FRIENDS!* :sunglasses:')
	st.header('CHAT PDF APP')
	st.write("")
	st.write("")
	pdf = st.file_uploader("UPLOAD YOUR PDF",type='pdf')
	st.balloons()
	
	#pdf is none it will create some error
	if pdf is not None:
		#PdfReader - Used for to read the content in the PDF
		pdf_reader = PdfReader(pdf)
		text = ""
		for page in pdf_reader.pages:
			text += page.extract_text()
			
		#Split into Chunks
                #Each chunk will contain 1000 tokens, with 200 tokens overlapped to keep the chunks related and prevent separation
		text_splitter = RecursiveCharacterTextSplitter(
			chunk_size = 1000,
			chunk_overlap = 200,
			length_function=len)
		chunks = text_splitter.split_text(text=text)
		
		#Word Embedding - huggingface APIs
		model_id = "sentence-transformers/all-MiniLM-L6-v2"
		hf_token = "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"

		api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
		headers = {"Authorization": f"Bearer {hf_token}"}

		def query(texts):
  			response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
  			return response.json()
		with st.spinner('Wait for it...'):
    				time.sleep(5)
		st.success('This is a success message!', icon="✅")
		
		#Enter your question here
		user_question= st.text_input("Enter your question here ?")
		st.write(user_question)
		st.snow()
		
		
		try:
			question = query([user_question])
			query_embeddings = torch.FloatTensor(question)
    			output=query(chunks)
    			output=torch.from_numpy(np.array(output)).to(torch.float)
			
			#Semantic Search
			result=util.semantic_search(query_embeddings, output,top_k=2)
			final=[chunks[result[0][i]['corpus_id']] for i in range(len(result[0]))]
		except:
			st.write("Upload your valid PDF file")
			
    			#AI21studio Question Answer model
			AI21_api_key = 'q6QM3NWgec9PUO85V0shxwo0QueC9QXe'
			url = "https://api.ai21.com/studio/v1/answer"
		try:
			payload = {
                		"context":' '.join(final),
                		"question":user_question
          			}
  			headers = {
                		"accept": "application/json",
                		"content-type": "application/json",
                		"Authorization": f"Bearer {AI21_api_key}"
          			}
    			response = requests.post(url, json=payload, headers=headers)
		except:
			st.write("Upload your valid PDF file")
    
		
		try:	
			if(response.json()['answerInContext']):
				answer = response.json()['answer']
				st.write(answer)
				st.success('This is a success message!', icon="✅")
			else:
     				st.write('The answer is not found  in the document ⚠️,please reformulate your question.')
		except:
			st.write("Please enter your question first.......")

if __name__ == '__main__': 
	main()
