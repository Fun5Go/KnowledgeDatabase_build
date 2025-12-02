from langchain_community.document_loaders.csv_loader import CSVLoader


def load_csv(file_path:str):
    """Load CSV file"""
    if file_path.lower().endswith(".csv"):        
        loader =  CSVLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Use CSV.")
    data =  loader.load()
    print(f"[INFO] Loaded from {file_path}")
    return data
    
def main():
    file_path = "../../../DATA/FMEA/FMEA6371240046R02.csv"
    data = load_csv(file_path)
    print(data)


if __name__ == "__main__":
    main()