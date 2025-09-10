"""
Test script for the ticket classification pipeline
"""

import json
from classifier import TicketClassifier

def test_classification():
    """Test the ticket classification pipeline"""
    
    # Load sample tickets
    with open("../data/sample_tickets.json", "r") as f:
        sample_tickets = json.load(f)
    
    # Initialize classifier
    classifier = TicketClassifier()
    
    print("🧪 Testing Ticket Classification Pipeline")
    print("=" * 50)
    
    # Test a few sample tickets
    test_tickets = sample_tickets[:3]  # Test first 3 tickets
    
    for i, ticket in enumerate(test_tickets, 1):
        print(f"\n📋 Test Ticket {i}:")
        print(f"ID: {ticket['id']}")
        print(f"Subject: {ticket['subject']}")
        print(f"Body: {ticket['body'][:100]}...")
        
        # Classify the ticket
        result = classifier.classify_ticket(ticket)
        
        print(f"\n📊 Classification Results:")
        print(f"  Topic: {result['topic']}")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Priority: {result['priority']}")
        print("-" * 30)
    
    print("\n✅ Classification test completed!")

if __name__ == "__main__":
    test_classification()
