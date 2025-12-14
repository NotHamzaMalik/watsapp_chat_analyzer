import re
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

sia = SentimentIntensityAnalyzer()
STOP_WORDS = stopwords.words('english')

try:
    with open('stop words.txt', 'r', encoding='utf-8') as f:
        custom_stopwords = set(line.strip().lower() for line in f)
    STOP_WORDS = list(set(STOP_WORDS) | custom_stopwords)
except:
    pass

def preprocess(data):
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}\s?[AP]M)\s-\s'
    
    parts = re.split(pattern, data)
    
    if not parts or len(parts) < 3:
        return pd.DataFrame()

    dates = [parts[i].strip() for i in range(1, len(parts), 2)]
    messages = [parts[i].strip() for i in range(2, len(parts), 2)]

    if len(dates) != len(messages):
        try:
            dates = re.findall(pattern, data)
            messages_split = re.split(r'\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}\s?[AP]M\s-\s', data)
            messages = [m.strip() for m in messages_split[1:] if m.strip()]
        except Exception:
            return (False, data, dates, messages)

    if len(dates) != len(messages):
        return (False, data, dates, messages)
        
    df = pd.DataFrame({'user_message': messages, 'date_string': dates})
    
    df['date'] = pd.to_datetime(df['date_string'], format='%m/%d/%y, %I:%M %p', errors='coerce')
    
    if df['date'].isnull().all():
         df['date'] = pd.to_datetime(df['date_string'], format='%d/%m/%Y, %H:%M', errors='coerce')

    users = []
    messages_text = []

    for message in df['user_message']:
        entry = re.split(r'^([^:]+?):\s', message, maxsplit=1)
        if len(entry) > 1:
            users.append(entry[1].strip())
            messages_text.append(entry[2].strip())
        else:
            users.append('group_notification')
            messages_text.append(entry[0].strip())

    df['user'] = users
    df['message'] = messages_text
    df.drop(columns=['user_message', 'date_string'], inplace=True)
    
    df = df[df['user'] != 'group_notification'].reset_index(drop=True)
    df.dropna(subset=['date'], inplace=True)

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['month_num'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['only_date'] = df['date'].dt.date
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df["period"] = df["hour"].apply(lambda h: f"{h:02d}-{(h+1)%24:02d}")
    
    df['sentiment'] = df['message'].astype(str).apply(lambda x: sia.polarity_scores(x)['compound'])
    df['sentiment_category'] = pd.cut(df['sentiment'], 
                                      bins=[-1, -0.5, 0, 0.5, 1], 
                                      labels=['Very Negative', 'Negative', 'Positive', 'Very Positive'])
    
    return df