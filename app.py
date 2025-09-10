import streamlit as st
import json
import pandas as pd
from pipeline.classifier import TicketClassifier
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

tabs = st.tabs(["Dashboard", "Test Individual Ticket", "Test Email", "Classification Stats"])

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
                # Non-RAG topics - show routing message
                st.info(f"📋 This ticket has been classified as a **'{result['topic']}'** issue and routed to the appropriate team.")
                st.write("The support team will handle this ticket based on the classification and priority level.")
            
            # Show full classification details
            with st.expander("View Full Classification Details"):
                st.json(result)

with tabs[2]:
    st.header("Test Email Processing")
    
    # Email input form
    with st.form("email_form"):
        st.subheader("📧 Email Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sender_name = st.text_input("Sender Name", value="John Doe")
            sender_email = st.text_input("Sender Email", value="john.doe@company.com")
        
        with col2:
            subject = st.text_input("Email Subject", value="Need help with database connection")
            priority_override = st.selectbox("Priority Override", ["Auto", "High", "Medium", "Low"])
        
        body = st.text_area(
            "Email Body",
            value="Hi Atlan Support team,\n\nI'm having trouble connecting our Snowflake database to Atlan. The connection keeps failing and I'm not sure what permissions are needed. Could you please help me with this?\n\nThanks,\nJohn",
            height=200
        )
        
        submitted = st.form_submit_button("Process Email")
        
        if submitted:
            # Create ticket from email
            ticket = {
                "id": f"EMAIL-{len(sample_tickets) + 1:03d}",
                "subject": subject,
                "body": f"From: {sender_name} <{sender_email}>\n\n{body}"
            }
            
            with st.spinner("Processing email..."):
                # Classify the email
                result = classifier.classify_ticket(ticket)
                
                # Add email-specific info
                result["sender_name"] = sender_name
                result["sender_email"] = sender_email
                result["channel"] = "Email"
                result["original_subject"] = subject
                
                # Override priority if specified
                if priority_override != "Auto":
                    result["priority"] = priority_override
            
            st.success("Email processed successfully!")
            
            # Display results similar to individual ticket
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
                        # Create a comprehensive query from the email
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
                # Non-RAG topics - show routing message
                st.info(f"📋 This email has been classified as a **'{result['topic']}'** issue and routed to the appropriate team.")
                st.write("The support team will handle this email based on the classification and priority level.")
            
            # Show full classification details
            with st.expander("View Full Classification Details"):
                st.json(result)

with tabs[3]:
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
