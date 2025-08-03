import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key is None :
    st.error("🔴 GOOGLE_API_KEY environment variable not set. Please add it to your .env file.")
    st.stop()

st.set_page_config(page_title="YouTube Chatbot", page_icon="📺", layout="wide")

st.title("📺 YouTube Video Chatbot")
st.write("This chatbot can answer questions about a YouTube video. Just provide the video URL in the sidebar to get started.")

#fetching video id from the URL
def get_video_id(url):
    if "youtube.com/watch?v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("/")[-1]
    else:
        raise ValueError("Invalid YouTube URL")
    
def getTranscript(video_id) :

    ytt_api = YouTubeTranscriptApi()
    try :
        transcript_list = ytt_api.list(video_id)
        for t in transcript_list :
            if (t.language_code == "en") :
                temp = t.fetch()
                transcript = " ".join(chunk.text for chunk in temp)
                return transcript

        for t in transcript_list:
            temp = t.fetch()
            transcript = " ".join(chunk.text for chunk in temp)
            return transcript

    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None
    
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def format_chat_history(messages):
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#Initialize session state for messages, retriever, and video URL
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'video_url' not in st.session_state:
    st.session_state.video_url = ""

# --- Sidebar for video URL Input ---
with st.sidebar:
    st.header("Video Setup")
    video_url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=example")
    if st.button("Load Video", use_container_width=True):
        if video_url:
            st.session_state.video_url = video_url
            video_id = get_video_id(st.session_state.video_url)
            if video_id:
                with st.spinner("Fetching and processing transcript..."):
                    transcript = getTranscript(video_id)
                    if transcript:
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        docs = text_splitter.create_documents([transcript])
                        vector_store = FAISS.from_documents(docs, embeddings)
                        st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                        st.session_state.messages = [] # Clear previous chat
                        st.success("Video loaded successfully! You can now ask questions")
                
            else:
                st.error("Invalid YouTube URL. Please try again.")
        else:
            st.error("Please enter a valid YouTube video URL.")
                    
# ---Chat Interface ---
st.header("Chat with the Video")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message['role'], width ="stretch"):
        st.write(message['content'])



prompt = PromptTemplate(
    template = """
        You are an expert AI assistant. Your task is to answer a user's question as if you have watched the YouTube video they are asking about.

        **Previous Conversation:**
        {chat_history}

        I am providing you with the relevant spoken content from the video below. This is your primary source of information.

        **Instructions:**
        1.  Answer the user's question based on the **video's content** provided. Synthesize information from the different parts of the content to form a single, coherent answer.
        2.  **Frame your answer as if you are referencing the video directly.** For example, use phrases like "In the video, the creator explains..." or "The video shows..." Do NOT mention that you are using a "transcript" or "text".
        3.  The video's content might be in a language other than English. You must understand it, but your final answer **must be written in English**.
        4.  If the provided content does not contain enough information to fully answer the question, first use the information that is available from the video, and then supplement it with your own general knowledge to provide a complete and helpful response.

        ---

        **Content from the Video:**
        {context}

        ---

        **User's Question:**
        {question}

        ---

        **Answer (in English):**
    """,
    input_variables=["context", "question", "chat_history"]
)


# Handle user input
if question := st.chat_input("Ask a question about the video..."):
    #Add user question to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    #Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)
        
    if st.session_state.retriever:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_history_str = format_chat_history(st.session_state.messages[:-1])

                retriever = st.session_state.retriever

                rag_chain = (
                    {
                        "context": retriever | format_docs,
                        "question": RunnablePassthrough(),
                        "chat_history": lambda x: chat_history_str
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )

        
                response = st.write_stream(rag_chain.stream(question))

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

    else:
        st.warining("Please load a video first using the sidebar.")            