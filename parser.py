import re
import pandas as pd

def parse_whatsapp_chat(file_path, sender_map):
    with open(file_path, encoding="utf-8") as file:
        chat = file.read()

    # Pattern to match messages with date, time, sender, and message
    # Example line: 20/11/2024, 18:22 - 수경 Colombia: I don't think so😅
    message_pattern = re.compile(
        r'(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}) - (.*?): (.*?)(?=\n\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} - |$)',
        re.DOTALL
    )

    messages = message_pattern.findall(chat)

    data = []
    for date, time, sender, message in messages:
        # Map sender if it matches file-specific sender name
        if sender in sender_map:
            sender_clean = sender_map[sender]
        else:
            sender_clean = sender
        
        datetime_str = f"{date} {time}"
        # Parse date and time in dd/mm/yyyy format
        try:
            datetime = pd.to_datetime(datetime_str, dayfirst=True)
        except:
            continue  # skip lines that don't parse

        # Clean message
        message = message.strip().replace('\n', ' ')

        data.append((datetime, sender_clean, message))

    df = pd.DataFrame(data, columns=["datetime", "sender", "message"])

    # Filter out 'null' and empty messages
    df = df[df['message'].notnull()]
    df = df[df['message'].str.lower() != 'null']
    df = df[df['message'].str.strip() != '']

    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.time
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.day_name()

    return df

def parse_multiple_chats(files):
    dfs = []
    for file_path in files:
        # Create sender mapping based on file name
        # Example: file "WhatsApp Chat with 수경 Colombia.txt" -> sender "수경 Colombia"
        sender_name = re.search(r"WhatsApp Chat with (.+)\.txt", file_path).group(1)
        sender_map = {sender_name: "Sukyoung"}

        df = parse_whatsapp_chat(file_path, sender_map)
        dfs.append(df)

    combined_df = pd.concat(dfs).sort_values(by="datetime").reset_index(drop=True)
    return combined_df

if __name__ == "__main__":
    files = [
        "WhatsApp Chat with 수경 Spain.txt",
        "WhatsApp Chat with 수경.txt",
        "WhatsApp Chat with 수경 Colombia.txt"
        
        
    ]
    df = parse_multiple_chats(files)

    print("✅ Chat parsed successfully!\n")
    print("📝 Sample messages:")
    print(df.head(10))
    print("\n📊 Message counts by sender:")
    print(df['sender'].value_counts())

    
    df.to_csv("parsed_whatsapp_chat.csv", index=False)
    print("\n💾 Saved parsed data to parsed_whatsapp_chat.csv")