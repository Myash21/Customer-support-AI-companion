import streamlit as st
import json
import pandas as pd
from pipeline.classifier import TicketClassifier

# Load sample tickets
with open("data/sample_tickets.json", "r") as f:
    sample_tickets = json.load(f)

# Initialize classifier
classifier = TicketClassifier()

# Streamlit UI
st.set_page_config(page_title="Customer Support Copilot", layout="wide")
st.title("📩 Customer Support Copilot")

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
            
            st.success("Classification Complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Topic", result["topic"])
            with col2:
                st.metric("Sentiment", result["sentiment"])
            with col3:
                st.metric("Priority", result["priority"])
            
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
