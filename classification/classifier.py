from transformers import pipeline
import torch
import concurrent.futures

class TicketClassifier:
    def __init__(self):
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Zero-shot classification for topics
        self.topic_classifier = pipeline("zero-shot-classification", 
                                         model="facebook/bart-large-mnli",
                                         device=self.device,
                                         torch_dtype=torch.float32)
        
        # Emotion classifier for sentiment
        self.sentiment_classifier = pipeline("text-classification", 
                                             model="j-hartmann/emotion-english-distilroberta-base",
                                             return_all_scores=False,
                                             device=self.device,
                                             torch_dtype=torch.float32)

        # Predefined topic labels
        self.topic_labels = [
            "How-to", "Product", "Connector", "Lineage", 
            "API/SDK", "SSO", "Glossary", "Best practices", "Sensitive data"
        ]

        # Mapping emotion labels → support categories
        self.emotion_to_sentiment = {
            "anger": "Angry",
            "disgust": "Frustrated",
            "sadness": "Frustrated",
            "fear": "Frustrated",    # fear usually signals urgency/blockers
            "joy": "Curious",
            "surprise": "Curious",
            "neutral": "Neutral"
        }

    def classify_topic(self, subject: str, body: str) -> str:
        """Use subject + body for topic classification"""
        text = f"{subject} {body}"
        result = self.topic_classifier(text, self.topic_labels)
        return result["labels"][0]  # best match

    def classify_sentiment(self, body: str) -> str:
        """Use body for sentiment classification"""
        result = self.sentiment_classifier(body)[0]  
        # Example: {'label': 'anger', 'score': 0.95}
        emotion = result["label"].lower()
        return self.emotion_to_sentiment.get(emotion, "Neutral")

    def classify_priority(self, body: str) -> str:
        text = body.lower()
        if any(word in text for word in [
            "urgent", "asap", "immediately", "critical", "critical failure", "outage",
            "blocked", "blocker", "blocking", "can't proceed", "cannot proceed", "can't move forward",
            "hard requirement", "deadline approaching", "approaching fast", "huge problem",
            "production down", "p0", "cannot continue", "can't continue", "this is blocking", "blocking our"
        ]):
            return "P0 (High)"
        elif any(word in text for word in [
            "soon", "important", "deadline", "required", "next week", "need to present",
            "need to provide", "not working", "failing", "failed", "confusing", "issue", "problem"
        ]):
            return "P1 (Medium)"
        else:
            return "P2 (Low)"

    def classify_ticket(self, ticket: dict) -> dict:
        """Full classification for one ticket"""
        subject, body = ticket["subject"], ticket["body"]
        return {
            "id": ticket["id"],
            "subject": subject,
            "topic": self.classify_topic(subject, body),
            "sentiment": self.classify_sentiment(body),
            "priority": self.classify_priority(body)
        }
    
    def classify_tickets_bulk(self, tickets: list, max_workers: int = 4) -> list:
        """Bulk classification with parallel processing"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticket = {
                executor.submit(self.classify_ticket, ticket): ticket 
                for ticket in tickets
            }
            
            results = []
            for future in concurrent.futures.as_completed(future_to_ticket):
                ticket = future_to_ticket[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error classifying ticket {ticket['id']}: {e}")
                    results.append({
                        "id": ticket["id"],
                        "subject": ticket["subject"],
                        "topic": "Error",
                        "sentiment": "Error", 
                        "priority": "Error"
                    })
        
        # Return results in original order
        return sorted(results, key=lambda x: tickets.index(next(t for t in tickets if t["id"] == x["id"])))
