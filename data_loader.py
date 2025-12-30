import pandas as pd

class DataLoader:
    def __init__(self, path=None):
        self.path = path

    def load_data(self):
        if self.path:
            df = pd.read_csv(self.path)
        else:
            # Fallback for empty state
            df = pd.DataFrame(columns=["Ticket Subject", "Ticket Description", "Ticket Priority"])
        
        # Combine text fields for the AI to read
        df['ticket_text'] = (
            df['Ticket Subject'].fillna('') + 
            " | Context: " + 
            df['Ticket Description'].fillna('')
        )
        return df