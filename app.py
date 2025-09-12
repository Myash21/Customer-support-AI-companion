try:
    import pysqlite3 as sqlite3  # noqa: F401
    import sys as _sys
    _sys.modules["sqlite3"] = sqlite3
    _sys.modules["pysqlite3"] = sqlite3
except Exception:
    pass
import streamlit as st
import re
import json
import pandas as pd
from email_support.sender import send_email
from classification.classifier import TicketClassifier
from rag.rag_pipeline import rag_answer

# Load sample tickets
with open("data/sample_tickets.json", "r") as f:
    sample_tickets = json.load(f)

# Initialize classifier
classifier = TicketClassifier()

# Streamlit UI
st.set_page_config(page_title="Customer Support Copilot", layout="wide")
st.title("📩 Customer Support Copilot")

# API Key check
if not st.session_state.get('api_key_warning_shown', False):
    import os
    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("⚠️ Google API key not found. Please set GOOGLE_API_KEY environment variable for AI responses to work.")
        st.session_state['api_key_warning_shown'] = True

tabs = st.tabs(["Dashboard", "Test Individual Ticket", "Classification Stats"])

with tabs[0]:
    st.header("Bulk Ticket Classification Dashboard")
    
    if st.button("Classify All Tickets"):
        with st.spinner("Classifying tickets..."):
            results = []
            for ticket in sample_tickets:
                results.append(classifier.classify_ticket(ticket))
            
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="ticket_classifications.csv",
                mime="text/csv"
            )

            results_map = {r["id"]: r for r in results}
            st.divider()
            st.subheader("Tickets")
            for ticket in sample_tickets:
                header = f"{ticket['id']} — {ticket['subject']}"
                with st.expander(header):
                    col1, col2, col3 = st.columns(3)
                    r = results_map.get(ticket["id"]) if results_map else None
                    if r:
                        col1.metric("Topic", r["topic"])
                        col2.metric("Sentiment", r["sentiment"])
                        col3.metric("Priority", r["priority"])
                    else:
                        col1.metric("Topic", "-")
                        col2.metric("Sentiment", "-")
                        col3.metric("Priority", "-")
                    st.markdown("**Subject**")
                    st.write(ticket["subject"]) 
                    st.markdown("**Body**")
                    st.write(ticket["body"]) 

with tabs[1]:
    st.header("Test Individual Ticket")
    
    # Ticket input form
    with st.form("ticket_form"):
        ticket_id = st.text_input("Ticket ID", value="TEST-001")
        subject = st.text_input("Subject", value="How to connect Snowflake?")
        body = st.text_area("Body", value="I'm having trouble connecting our Snowflake database to Atlan. Can you help me with the required permissions?")
        
        submitted = st.form_submit_button("Classify Ticket")
        
        if submitted:
            ticket = {
                "id": ticket_id,
                "subject": subject,
                "body": body
            }
            
            with st.spinner("Classifying ticket..."):
                result = classifier.classify_ticket(ticket)
            
            # Store result in session state
            st.session_state['last_classification'] = {
                'result': result,
                'ticket': ticket
            }
    
    # Display classification results if available
    if 'last_classification' in st.session_state:
        result = st.session_state['last_classification']['result']
        ticket = st.session_state['last_classification']['ticket']
        ticket_id = ticket['id']
        subject = ticket['subject']
        body = ticket['body']
        
        st.success("Classification Complete!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Topic", result["topic"])
        with col2:
            st.metric("Sentiment", result["sentiment"])
        with col3:
            st.metric("Priority", result["priority"])
        
        # RAG Integration based on classification
        st.divider()
        st.subheader("🤖 AI Response")
        
        # Topics that should use RAG
        rag_topics = ["How-to", "Product", "Best practices", "API/SDK", "SSO"]
        
        if result["topic"] in rag_topics:
            with st.spinner("Generating AI response using knowledge base..."):
                try:
                    # Create a comprehensive query from the ticket
                    query = f"Subject: {subject}\n\nQuestion: {body}"
                    answer, sources = rag_answer(query)
                    
                    st.success("AI Response Generated!")
                    
                    # Display the answer
                    st.markdown("**Answer:**")
                    st.write(answer)
                    
                    # Display sources if available
                    if sources and sources != ["Unknown"]:
                        st.markdown("**Sources:**")
                        for i, source in enumerate(sources, 1):
                            st.write(f"{i}. {source}")
                    else:
                        st.warning("No specific sources found in knowledge base.")
                        
                except Exception as e:
                    st.error(f"Error generating AI response: {str(e)}")
                    st.info("Please check your Google API key configuration.")
        else:
            st.info(f"This ticket has been classified as a '{result['topic']}' issue")
            st.markdown("**Notify a domain expert by email**")
            expert_email = st.text_input("Domain expert email", key="expert_email_input")
            if st.button("Send email to expert", key="send_expert_email_btn"):
                email_pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
                if not expert_email or not re.match(email_pattern, expert_email):
                    st.error("Please enter a valid email address.")
                else:
                    try:
                        email_subject = f"Ticket {ticket_id}: {subject} — {result['topic']}"
                        email_body = (
                            f"A new ticket has been classified as '{result['topic']}'.\n\n"
                            f"Ticket ID: {ticket_id}\n"
                            f"Priority: {result['priority']}\n"
                            f"Sentiment: {result['sentiment']}\n\n"
                            f"Subject: {subject}\n\n"
                            f"Body:\n{body}\n"
                        )
                        send_email(expert_email, email_subject, email_body)
                        st.success("Email sent to domain expert.")
                    except Exception as e:
                        st.error(f"Failed to send email: {str(e)}")
        
        # Show full classification details
        with st.expander("View Full Classification Details"):
            st.json(result)

with tabs[2]:
    st.header("Classification Statistics")
    
    if st.button("Generate Statistics"):
        with st.spinner("Analyzing classifications..."):
            results = []
            for ticket in sample_tickets:
                results.append(classifier.classify_ticket(ticket))
            
            df = pd.DataFrame(results)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Topic Distribution")
                topic_counts = df['topic'].value_counts()
                st.bar_chart(topic_counts)
                
                st.subheader("Sentiment Distribution")
                sentiment_counts = df['sentiment'].value_counts()
                st.bar_chart(sentiment_counts)
            
            with col2:
                st.subheader("Priority Distribution")
                priority_counts = df['priority'].value_counts()
                st.bar_chart(priority_counts)
                
                st.subheader("Top Topics")
                st.write(topic_counts.head())
            
            st.subheader("Sample Classifications")
            st.dataframe(df.head(10), use_container_width=True)
