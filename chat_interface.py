import streamlit as st
import os
from rag_system import CrustDataRAG
import json
from datetime import datetime
import pandas as pd
import plotly.express as px

class ChatInterface:
    def __init__(self):
        self.initialize_session_state()
        self.setup_page_config()
        
    def initialize_session_state(self):
        """Initialize all session state variables."""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'feedback_history' not in st.session_state:
            st.session_state.feedback_history = []
        if 'current_view' not in st.session_state:
            st.session_state.current_view = "chat"
        if 'api_key' not in st.session_state:
            st.session_state.api_key = None
        if 'rag_system' not in st.session_state and st.session_state.api_key:
            st.session_state.rag_system = CrustDataRAG(api_key=st.session_state.api_key)
            self.load_documents()

    def setup_page_config(self):
        """Configure page settings and load custom CSS."""
        st.set_page_config(
            page_title="CrustData API Assistant",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.markdown("""
        <style>
        .chat-message {
            padding: 1.5rem;
            border-radius: 1rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: flex-start;
        }
        
        .user-message {
            background: #E8F0FF;
            margin-left: 2rem;
        }
        
        .bot-message {
            background: #F0F2F6;
            margin-right: 2rem;
        }
        
        .stButton > button {
            border-radius: 20px;
            padding: 0.5rem 1rem;
        }
        
        .feedback-button {
            padding: 0.2rem 0.5rem;
            border-radius: 15px;
            border: none;
            background: #F0F2F6;
            margin-right: 0.5rem;
        }

        .api-key-container {
            padding: 2rem;
            background: #F8F9FA;
            border-radius: 1rem;
            margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)

    def render_api_key_input(self):
        """Render API key input form."""
        st.markdown("""
        <div class="api-key-container">
            <h2>🔑 OpenAI API Key Required</h2>
        </div>
        """, unsafe_allow_html=True)
        
        api_key = st.text_input(
            "Enter your OpenAI API key:",
            type="password",
            help="Your API key will be used only for this session and won't be stored permanently."
        )
        
        if st.button("Submit API Key"):
            if api_key:
                st.session_state.api_key = api_key
                st.session_state.rag_system = CrustDataRAG(api_key=api_key)
                self.load_documents()
                st.success("API key successfully set! You can now use the chat interface.")
                st.rerun()
            else:
                st.error("Please enter a valid API key.")

    def load_documents(self):
        """Load document data for the RAG system."""
        doc_paths = [
            'notion_content_20250112_212145.txt',
            'notion_content_20250112_213513.txt'
        ]
        st.session_state.rag_system.load_documents(doc_paths)

    def render_sidebar(self):
        """Render sidebar with navigation and settings."""
        with st.sidebar:
            st.title("🤖 CrustData Assistant")
            
            if st.session_state.api_key:
                # Navigation
                st.subheader("Navigation")
                if st.button("💭 Chat", key="nav_chat"):
                    st.session_state.current_view = "chat"
                if st.button("📊 Analytics", key="nav_analytics"):
                    st.session_state.current_view = "analytics"
                
                # Quick examples
                st.subheader("Quick Examples")
                examples = [
                    "How do I authenticate with the CrustData API?",
                    "How do i use the people endpoint? give example",
                ]
                
                for example in examples:
                    if st.button(example, key=f"example_{example}"):
                        self.process_user_input(example)
                
                if st.button("Export Chat"):
                    self.export_chat_history()

    def render_chat_interface(self):
        """Render the main chat interface."""
        if not st.session_state.api_key:
            self.render_api_key_input()
            return

        st.header("Chat with CrustData API Assistant")
        
        # Chat container
        chat_container = st.container()
        with chat_container:
            self.display_chat_history()
        
        # Input container
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input("Ask about CrustData APIs:", 
                                     key="user_input",
                                     placeholder="Type your question here...")
        with col2:
            if st.button("Send", key="send_button"):
                if user_input:
                    self.process_user_input(user_input)

    def render_analytics(self):
        """Render analytics dashboard."""
        if not st.session_state.api_key:
            self.render_api_key_input()
            return

        st.header("Chat Analytics")
        
        if st.session_state.chat_history:
            # Create analytics dataframe
            df = pd.DataFrame([
                {
                    'timestamp': datetime.now(),
                    'message_type': msg['role'],
                    'content_length': len(msg['content'])
                }
                for msg in st.session_state.chat_history
            ])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Message Distribution")
                fig = px.pie(df, names='message_type', title="Message Distribution")
                st.plotly_chart(fig)
            
            with col2:
                st.subheader("Message Lengths")
                fig = px.bar(df, x='message_type', y='content_length', 
                           title="Average Message Length by Type")
                st.plotly_chart(fig)

    def process_user_input(self, user_input: str):
        """Process user input and generate response."""
        if user_input:
            st.session_state.chat_history.append({"role": "user", 
                                                "content": user_input,
                                                "timestamp": datetime.now().isoformat()})
            
            with st.spinner("Thinking..."):
                response = st.session_state.rag_system.answer_query(
                    user_input,
                    st.session_state.chat_history
                )
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().isoformat()
                })
            
            st.rerun()

    def display_chat_history(self):
        """Display chat history with enhanced formatting."""
        for i, message in enumerate(st.session_state.chat_history):
            is_user = message["role"] == "user"
            
            if is_user:
                st.markdown(f"""
                    <div class="chat-message user-message">
                        <div>👤 You: {message["content"]}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div>🤖 Assistant: {message["content"]}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                cols = st.columns([0.1, 0.1, 0.8])
                with cols[0]:
                    if st.button("👍", key=f"thumbs_up_{i}"):
                        if "feedback" not in st.session_state:
                            st.session_state.feedback = {}
                        st.session_state.feedback[i] = "positive"
                with cols[1]:
                    if st.button("👎", key=f"thumbs_down_{i}"):
                        if "feedback" not in st.session_state:
                            st.session_state.feedback = {}
                        st.session_state.feedback[i] = "negative"

    def export_chat_history(self):
        """Export chat history to JSON file."""
        if st.session_state.chat_history:
            chat_data = {
                "timestamp": datetime.now().isoformat(),
                "messages": st.session_state.chat_history
            }
            
            json_str = json.dumps(chat_data, indent=2)
            
            st.download_button(
                label="Download Chat History",
                data=json_str,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    def main(self):
        """Main application loop."""
        self.render_sidebar()
        
        if st.session_state.current_view == "chat":
            self.render_chat_interface()
        elif st.session_state.current_view == "analytics":
            self.render_analytics()

if __name__ == "__main__":
    chat_app = ChatInterface()
    chat_app.main()