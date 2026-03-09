import streamlit as st
import requests

# Set the URL of your FastAPI backend
API_URL = "http://localhost:8000/ask"

# Configure the page layout
st.set_page_config(page_title="HR Policy Assistant", page_icon="🏢")
st.title("🏢 Company HR Assistant")
st.markdown("Ask me anything about company policies, benefits, or procedures!")

# --- NEW ADMIN SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Admin Panel")
    st.markdown("Upload new policy documents to the knowledge base.")
    
    # File uploader widget
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if st.button("Upload & Vectorize"):
        if uploaded_file is not None:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Prepare the file to send to FastAPI
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                
                try:
                    # Hit the new /upload endpoint
                    upload_response = requests.post("http://localhost:8000/upload", files=files)
                    upload_response.raise_for_status()
                    
                    st.success(f"✅ '{uploaded_file.name}' added to Pinecone successfully!")
                except Exception as e:
                    st.error(f"Failed to upload: {e}")
        else:
            st.warning("Please select a file first.")
# -------------------------

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Wait for user input
if prompt := st.chat_input("E.g., What is the policy for annual leave?"):
    
    # 1. Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call the FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Searching company policies..."):
            try:
                # Get all past messages EXCEPT the one the user just typed
                # We do this so the LLM doesn't get confused seeing the prompt twice
                history_to_send = st.session_state.messages[:-1] 

                # Send both the new question and the history to FastAPI
                payload = {
                    "question": prompt,
                    "chat_history": history_to_send
                }
                
                response = requests.post(API_URL, json=payload)
                response.raise_for_status() 
                
                data = response.json()
                
                # Format the response nicely (removing source text if no sources exist)
                answer = data.get("answer", "I couldn't process the answer.")
                sources = data.get("sources", [])
                
                if sources:
                    source_text = "\n\n**Sources:**\n"
                    for source in sources:
                        source_text += f"* {source}\n"
                    full_response = answer + source_text
                else:
                    full_response = answer
                
                # Display the response
                st.markdown(full_response)
                
                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except requests.exceptions.RequestException as e:
                error_msg = f"⚠️ Error connecting to the backend. ({e})"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})