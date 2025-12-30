import pandas as pd
from data_loader import DataLoader
from classifier import TicketClassifier
import json

def main():
    #  Load Data
    loader = DataLoader("data/customer_support_tickets.csv") 
    df = loader.load_data()

    #  Filter 
    if "Ticket Priority" in df.columns:
        df = df[df["Ticket Priority"] == "High"].copy()
    
    # Initialize Groq Classifier
    classifier = TicketClassifier()

    print(f"Starting classification for {len(df)} tickets...")
    
    results = []
    for i, text in enumerate(df['ticket_text']):
        print(f"Processing {i+1}/{len(df)}...")
        res = classifier.classify_ticket(text)
        results.append(res)

    #  Parse Results
    df['primary_tag'] = [r.get('primary_tag') for r in results]
    df['secondary_tag'] = [r.get('secondary_tag') for r in results]
    df['tertiary_tag'] = [r.get('tertiary_tag') for r in results]
    df['justification'] = [r.get('justification') for r in results]

    #  Save
    df.to_csv("data/tickets_classified.csv", index=False)
    print("Done! Results saved to data/tickets_classified.csv")

if __name__ == "__main__":
    main()