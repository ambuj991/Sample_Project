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
        self.MAX_INPUT_LENGTH = 1000
        
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
        if 'scroll_to_bottom' not in st.session_state:
            st.session_state.scroll_to_bottom = False
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
        /* Main chat container */
        .main-chat-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 1rem;
        }
        
        /* Chat message styling */
        .chat-message {
            padding: 1.5rem;
            border-radius: 1rem;
            margin-bottom: 1.5rem;
            display: flex;
            flex-direction: column;
            max-width: 85%;
            word-wrap: break-word;
            overflow-wrap: break-word;
            white-space: pre-wrap;
        }
        
        .user-message {
            background: #E8F0FF;
            margin-left: auto;
            border-bottom-right-radius: 0.2rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        .bot-message {
            background: #F0F2F6;
            margin-right: auto;
            border-bottom-left-radius: 0.2rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        /* Message content */
        .message-content {
            font-size: 1rem;
            line-height: 1.5;
            color: #1E1E1E;
        }
        
        .message-header {
            display: flex;
            align-items: center;
            margin-bottom: 0.5rem;
            font-weight: 500;
            color: #4A4A4A;
        }
        
        .message-icon {
            margin-right: 0.5rem;
            font-size: 1.2rem;
        }
        
        /* Code block styling */
        .chat-message pre {
            background: #2B2B2B;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 0.5rem 0;
            overflow-x: auto;
            font-family: 'Monaco', 'Menlo', monospace;
        }
        
        .chat-message code {
            color: #E0E0E0;
            font-size: 0.9rem;
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 20px;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Control panel */
        .control-panel {
            display: flex;
            gap: 1rem;
            margin: 1rem 0;
            padding: 1rem;
            background: #F8F9FA;
            border-radius: 0.5rem;
            align-items: center;
        }
        
        /* Character counter */
        .char-counter {
            color: #666;
            font-size: 0.9rem;
            text-align: right;
            margin-top: 0.2rem;
        }
        
        /* Floating action buttons */
        .floating-button {
            position: fixed;
            right: 2rem;
            padding: 0.8rem;
            border-radius: 50%;
            background: #ffffff;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            z-index: 1000;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .scroll-button {
            bottom: 2rem;
        }
        
        .clear-button {
            bottom: 6rem;
        }
        
        .floating-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 3px 8px rgba(0,0,0,0.3);
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: #F1F1F1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #C1C1C1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #A8A8A8;
        }
        
        /* Input container */
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: white;
            padding: 1rem;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        }
        
        /* Toast notification */
        .toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            padding: 1rem;
            background: #333;
            color: white;
            border-radius: 0.5rem;
            animation: fadeIn 0.3s, fadeOut 0.3s 2.7s;
            z-index: 1000;
        }
        
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        
        @keyframes fadeOut {
            from {opacity: 1;}
            to {opacity: 0;}
        }
        </style>
        """, unsafe_allow_html=True)

    def render_api_key_input(self):
        """Render API key input form."""
        st.markdown("""
        <div class="api-key-container">
            <h2>🔑 OpenAI API Key Required</h2>
            <p>Enter your API key to start using the chat interface.</p>
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
                st.experimental_rerun()
            else:
                st.error("Please enter a valid API key.")

    def load_documents(self):
        """Load document data for the RAG system."""
        doc_paths = [
            'notion_content_20250112_212145.txt',
            'notion_content_20250112_213513.txt'
        ]
        st.session_state.rag_system.load_documents(doc_paths)

    def add_scroll_button(self):
        """Add a floating scroll-to-bottom button."""
        if len(st.session_state.chat_history) > 5:
            st.markdown("""
                <div class="floating-button scroll-button" onclick="window.scrollTo(0, document.body.scrollHeight);">
                    ⬇️
                </div>
            """, unsafe_allow_html=True)

    def add_clear_chat_button(self):
        """Add a floating clear chat button."""
        if len(st.session_state.chat_history) > 0:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.experimental_rerun()

    def add_input_counter(self, input_text: str):
        """Add a character counter to the input field."""
        remaining = self.MAX_INPUT_LENGTH - len(input_text)
        color = "#666" if remaining > 100 else "#ff4b4b"
        st.markdown(
            f'<p class="char-counter" style="color: {color}">{remaining} characters remaining</p>',
            unsafe_allow_html=True
        )

    def render_control_panel(self):
        """Render the control panel with various options."""
        with st.container():
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                self.add_clear_chat_button()
            with col2:
                if st.button("📥 Export Chat"):
                    self.export_chat_history()
            with col3:
                st.empty()  # Placeholder for future controls

    def display_chat_history(self):
        """Display chat history with enhanced formatting."""
        st.markdown('<div class="main-chat-container">', unsafe_allow_html=True)
        
        for i, message in enumerate(st.session_state.chat_history):
            is_user = message["role"] == "user"
            content = message["content"]
            
            # Convert code blocks to proper formatting
            if "```" in content:
                parts = content.split("```")
                formatted_content = parts[0]
                for j in range(1, len(parts), 2):
                    code = parts[j].strip()
                    if j < len(parts) - 1:
                        after_code = parts[j + 1]
                    else:
                        after_code = ""
                    formatted_content += f"<pre><code>{code}</code></pre>{after_code}"
                content = formatted_content
            
            # Process markdown links
            content = content.replace("[", "<a href='").replace("]", "'></a>")
            
            message_class = "user-message" if is_user else "bot-message"
            icon = "👤" if is_user else "🤖"
            role = "You" if is_user else "Assistant"
            
            st.markdown(f"""
                <div class="chat-message {message_class}">
                    <div class="message-header">
                        <span class="message-icon">{icon}</span>
                        <span>{role}</span>
                    </div>
                    <div class="message-content">{content}</div>
                </div>
            """, unsafe_allow_html=True)
            
            if not is_user:
                with st.container():
                    cols = st.columns([0.1, 0.1, 0.8])
                    with cols[0]:
                        if st.button("👍", key=f"thumbs_up_{i}", help="This response was helpful"):
                            if "feedback" not in st.session_state:
                                st.session_state.feedback = {}
                            st.session_state.feedback[i] = "positive"
                            st.toast("Thanks for your feedback!")
                    with cols[1]:
                        if st.button("👎", key=f"thumbs_down_{i}", help="This response needs improvement"):
                            if "feedback" not in st.session_state:
                                st.session_state.feedback = {}
                            st.session_state.feedback[i] = "negative"
                            st.toast("Thanks for your feedback! We'll work on improving.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        self.add_scroll_button()

    def render_input_area(self):
        """Render the chat input area with character counter."""
        user_input = st.text_input(
            "Ask about CrustData APIs:",
            key="user_input",
            placeholder="Type your question here...",
            max_chars=self.MAX_INPUT_LENGTH
        )
        
        self.add_input_counter(user_input)
        
        if st.button("Send", key="send_button") or (user_input and len(user_input.strip()) > 0):
            self.process_user_input(user_input)

    def process_user_input(self, user_input: str):
        """Process user input and generate response."""
        if user_input and len(user_input.strip()) > 0:
            try:
                # Create a placeholder for the new messages
                message_placeholder = st.empty()
                
                # Add user message
                new_user_message = {
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.chat_history.append(new_user_message)
                
                # Show typing indicator
                with st.spinner("Assistant is typing..."):
                    try:
                        response = st.session_state.rag_system.answer_query(
                            user_input,
                            st.session_state.chat_history
                        )
                        
                        # Add assistant message
                        new_assistant_message = {
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.now().isoformat()
                        }
                        st.session_state.chat_history.append(new_assistant_message)
                        
                    except Exception as e:
                        error_msg = f"An error occurred: {str(e)}"
                        st.error(error_msg)
                        new_assistant_message = {
                            "role": "assistant",
                            "content": f"I apologize, but I encountered an error: {error_msg}",
                            "timestamp": datetime.now().isoformat()
                        }
                        st.session_state.chat_history.append(new_assistant_message)
                
                # Instead of directly modifying the input, we'll use a rerun to clear it
                if 'user_input' in st.session_state:
                    del st.session_state.user_input
                
                # Trigger scroll to bottom
                st.session_state.scroll_to_bottom = True
                
                # Rerun the app to refresh the chat
                st.experimental_rerun()
                
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                
            finally:
                # Clean up placeholder
                message_placeholder.empty()

    def export_chat_history(self):
        """Export chat history to JSON file."""
        if st.session_state.chat_history:
            chat_data = {
                "timestamp": datetime.now().isoformat(),
                "messages": st.session_state.chat_history
            }
            
            json_str = json.dumps(chat_data, indent=2)
            
            st.download_button(
                label="📥 Download Chat History",
                data=json_str,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    def render_chat_interface(self):
        """Render the main chat interface."""
        if not st.session_state.api_key:
            self.render_api_key_input()
            return

        # Create main containers
        header_container = st.container()
        chat_container = st.container()
        control_container = st.container()
        input_container = st.container()

        # Render header
        with header_container:
            st.header("Chat with CrustData API Assistant")
            self.render_control_panel()
        
        # Render chat history
        with chat_container:
            if st.session_state.chat_history:
                self.display_chat_history()
            else:
                st.markdown("""
                    <div style="text-align: center; padding: 2rem; color: #666;">
                        <h3>👋 Welcome to CrustData API Assistant!</h3>
                        <p>Start by asking a question about the CrustData API.</p>
                    </div>
                """, unsafe_allow_html=True)

        # Render input area with proper spacing
        with input_container:
            st.markdown("<div style='padding-bottom: 7rem;'></div>", unsafe_allow_html=True)
            with st.container():
                col1, col2 = st.columns([6, 1])
                with col1:
                    # Initialize the input field's state if not present
                    if 'user_input' not in st.session_state:
                        st.session_state.user_input = ""
                    
                    user_input = st.text_input(
                        "Ask about CrustData APIs:",
                        key="user_input",
                        placeholder="Type your question here...",
                        max_chars=self.MAX_INPUT_LENGTH,
                        label_visibility="collapsed",
                        value=st.session_state.user_input
                    )
                    self.add_input_counter(user_input)
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
                    if st.button("Send", key="send_button", use_container_width=True):
                        if user_input and len(user_input.strip()) > 0:
                            self.process_user_input(user_input)

        # Handle automatic scrolling
        if st.session_state.scroll_to_bottom:
            st.session_state.scroll_to_bottom = False
            st.markdown(
                """
                <script>
                    window.scrollTo(0, document.body.scrollHeight);
                </script>
                """,
                unsafe_allow_html=True
            )
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
                    'timestamp': datetime.strptime(msg['timestamp'], '%Y-%m-%dT%H:%M:%S.%f'),
                    'role': msg['role'],
                    'content_length': len(msg['content']),
                    'hour_of_day': datetime.strptime(msg['timestamp'], '%Y-%m-%dT%H:%M:%S.%f').hour
                }
                for msg in st.session_state.chat_history
            ])
            
            # Layout for analytics
            col1, col2 = st.columns(2)
            
            with col1:
                # Message distribution by role
                st.subheader("Message Distribution")
                fig_dist = px.pie(
                    df, 
                    names='role',
                    title="Distribution of Messages by Role",
                    color_discrete_sequence=['#E8F0FF', '#F0F2F6']
                )
                fig_dist.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Average message length
                st.subheader("Average Message Length")
                avg_lengths = df.groupby('role')['content_length'].mean().round(1)
                fig_avg = px.bar(
                    x=avg_lengths.index,
                    y=avg_lengths.values,
                    title="Average Message Length by Role",
                    labels={'x': 'Role', 'y': 'Average Characters'},
                    color_discrete_sequence=['#4A90E2']
                )
                st.plotly_chart(fig_avg, use_container_width=True)
            
            with col2:
                # Activity by hour
                st.subheader("Activity by Hour")
                hourly_counts = df.groupby('hour_of_day').size()
                fig_hourly = px.bar(
                    x=hourly_counts.index,
                    y=hourly_counts.values,
                    title="Message Activity by Hour of Day",
                    labels={'x': 'Hour of Day', 'y': 'Number of Messages'},
                    color_discrete_sequence=['#4A90E2']
                )
                fig_hourly.update_xaxes(tickmode='linear', tick0=0, dtick=1)
                st.plotly_chart(fig_hourly, use_container_width=True)
                
                # Feedback analysis if available
                if hasattr(st.session_state, 'feedback') and st.session_state.feedback:
                    st.subheader("Feedback Analysis")
                    feedback_df = pd.DataFrame([
                        {'feedback': feedback}
                        for feedback in st.session_state.feedback.values()
                    ])
                    feedback_counts = feedback_df['feedback'].value_counts()
                    fig_feedback = px.pie(
                        values=feedback_counts.values,
                        names=feedback_counts.index,
                        title="User Feedback Distribution",
                        color_discrete_sequence=['#28A745', '#DC3545']
                    )
                    st.plotly_chart(fig_feedback, use_container_width=True)

    def render_sidebar(self):
        """Render sidebar with navigation and settings."""
        with st.sidebar:
            st.title("🤖 CrustData Assistant")
            
            if st.session_state.api_key:
                # Navigation
                st.subheader("Navigation")
                nav_cols = st.columns(2)
                with nav_cols[0]:
                    if st.button("💭 Chat", key="nav_chat", use_container_width=True):
                        st.session_state.current_view = "chat"
                with nav_cols[1]:
                    if st.button("📊 Analytics", key="nav_analytics", use_container_width=True):
                        st.session_state.current_view = "analytics"
                
                # Quick examples
                st.subheader("Quick Examples")
                examples = [
                    "How do I authenticate with the CrustData API?",
                    "How do I use the people endpoint? Give example",
                    "What are the available API endpoints?",
                    "How can I filter results in the API?",
                    "What's the rate limit for API calls?"
                ]
                
                for example in examples:
                    if st.button(example, key=f"example_{example}", use_container_width=True):
                        self.process_user_input(example)
                
                # Settings
                st.subheader("Settings")
                max_length = st.slider(
                    "Max response length",
                    min_value=100,
                    max_value=2000,
                    value=500,
                    step=100,
                    help="Maximum number of characters in assistant's response"
                )
                
                temperature = st.slider(
                    "Response creativity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="Higher values make responses more creative but less focused"
                )
                
                # Save settings to session state
                st.session_state.max_length = max_length
                st.session_state.temperature = temperature
                
                # Export option
                st.subheader("Export")
                if st.button("📥 Export Chat History", use_container_width=True):
                    self.export_chat_history()
                
                # Footer
                st.markdown("---")
                st.markdown(
                    """
                    <div style='text-align: center; color: #666; padding: 1rem;'>
                        <small>CrustData API Assistant v1.0</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    def main(self):
        """Main application loop."""
        try:
            self.render_sidebar()
            
            main_container = st.container()
            with main_container:
                if st.session_state.current_view == "chat":
                    self.render_chat_interface()
                elif st.session_state.current_view == "analytics":
                    self.render_analytics()
                
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    chat_app = ChatInterface()
    chat_app.main()