import torch
import streamlit as st
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import util
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests



with st.sidebar: 
    st.title("Chat App")
    #add_vertical_space(5)
    st.write("")
def main():
	st.header('HELLO, *FRIENDS!* :sunglasses:')
	st.header('CHAT PDF APP')
	
	pdf = st.file_uploader("UPLOAD YOUR PDF",type='pdf')
	#st.write(pdf.name)

	if pdf is not None:
		pdf_reader = PdfReader(pdf)
		text = ""
		for page in pdf_reader.pages:
			text += page.extract_text()
		text_splitter = RecursiveCharacterTextSplitter(
			chunk_size = 1000,
			chunk_overlap = 200,
			length_function=len)

		chunks = text_splitter.split_text(text=text)

		model_id = "sentence-transformers/all-MiniLM-L6-v2"
		hf_token = "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"

		api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
		headers = {"Authorization": f"Bearer {hf_token}"}

		def query(texts):
  			response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
  			return response.json()

		user_question= st.text_input("Enter your question here ?")
		st.write(user_question)

		question = query([user_question])
            
		query_embeddings = torch.FloatTensor(question)
    
		output=query(chunks)
    
		output=torch.from_numpy(np.array(output)).to(torch.float)

		result=util.semantic_search(query_embeddings, output,top_k=2)

		final=[chunks[result[0][i]['corpus_id']] for i in range(len(result[0]))]
	
		AI21_api_key = 'q6QM3NWgec9PUO85V0shxwo0QueC9QXe'

		url = "https://api.ai21.com/studio/v1/answer"
    
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
    
		
		try:	
			if(response.json()['answerInContext']):
				answer = response.json()['answer']
				st.write(answer)
			else:
     				st.write('The answer is not found  in the document ⚠️,please reformulate your question.')
		except:
			st.write("Please enter your question first.......")
		


if __name__ == '__main__': 
	main()
