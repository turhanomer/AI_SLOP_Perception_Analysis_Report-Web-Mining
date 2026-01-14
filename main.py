"""
YouTube Duygu ve Algı Analizi Projesi - Gelişmiş Versiyon
Video: AI Slop - Yapay Zeka tarafından üretilen kalitesiz içerikler
"""

import os
import re
import json
import html
from collections import Counter
from typing import List, Dict, Tuple, Optional

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# NLTK verilerini indir (ilk çalıştırmada)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Kriz ve Fırsat Kelimeleri
CRISIS_WORDS = ['risk', 'crisis', 'fake', 'dead', 'destroy', 'scary', 'bot', 
                'misinformation', 'dystopian', 'danger', 'threat', 'worry',
                'fear', 'concern', 'problem', 'bad', 'terrible', 'awful',
                'theft', 'scam', 'deception', 'lie', 'false', 'wrong']

OPPORTUNITY_WORDS = ['opportunity', 'growth', 'tool', 'help', 'future', 'benefit', 
                     'amazing', 'useful', 'great', 'good', 'positive', 'excited',
                     'wonderful', 'brilliant', 'innovative', 'progress', 'hope']


def load_comments_from_csv(csv_path: str = 'yorumlar_backup.csv') -> List[str]:
    """
    CSV dosyasından yorumları yükler.
    
    Args:
        csv_path: CSV dosya yolu
        
    Returns:
        Yorum metinlerinin listesi
    """
    print(f"CSV dosyasından yorumlar yükleniyor: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # 'comment' sütununu al
    if 'comment' in df.columns:
        comments = df['comment'].dropna().tolist()
    else:
        # İlk sütunu kullan
        comments = df.iloc[:, 0].dropna().tolist()
    
    print(f"Toplam {len(comments)} yorum yüklendi.\n")
    return comments


def advanced_preprocess_text(text: str) -> str:
    """
    Gelişmiş metin temizleme: HTML etiketleri, HTML entity'leri, URL'ler, emoji'ler.
    
    Args:
        text: Ham metin
        
    Returns:
        Temizlenmiş metin
    """
    if not isinstance(text, str):
        return ""
    
    # HTML etiketlerini temizle (BeautifulSoup kullan)
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    
    # HTML entity'lerini decode et (&quot; -> ", &#39; -> ', vb.)
    text = html.unescape(text)
    
    # URL'leri temizle
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # HTML etiketlerini regex ile de temizle (kalanlar için)
    text = re.sub(r'<[^>]+>', '', text)
    
    # HTML entity'leri temizle (kalanlar için)
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    text = re.sub(r'&#\d+;', '', text)
    
    # Emoji'leri temizle
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    
    # Başında ve sonundaki boşlukları temizle
    text = text.strip()
    
    return text


def tokenize_and_clean(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Metni tokenize eder, noktalama işaretlerini ve stopwords'leri kaldırır.
    
    Args:
        text: Temizlenmiş metin
        remove_stopwords: Stopwords kaldırılsın mı
        
    Returns:
        Token listesi
    """
    if not text:
        return []
    
    # Küçük harfe çevir
    text = text.lower()
    
    # Tokenize et
    tokens = word_tokenize(text)
    
    # Sadece harf içeren tokenları al (noktalama işaretlerini çıkar)
    tokens = [token for token in tokens if token.isalpha()]
    
    # Minimum uzunluk kontrolü (1 karakterden uzun)
    tokens = [token for token in tokens if len(token) > 1]
    
    # Stopwords kaldır
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    return tokens


def analyze_sentiment(comments: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    VADER Sentiment ile yorumları analiz eder.
    
    Args:
        comments: Yorum listesi
        
    Returns:
        (DataFrame, İstatistikler dict)
    """
    analyzer = SentimentIntensityAnalyzer()
    
    results = []
    
    print("Duygu analizi yapılıyor...")
    
    for comment in comments:
        # Gelişmiş ön işleme
        cleaned = advanced_preprocess_text(comment)
        
        if not cleaned or len(cleaned) < 3:
            continue
        
        # Sentiment analizi
        scores = analyzer.polarity_scores(cleaned)
        
        # Kategori belirleme
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'Positive'
        elif compound <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        results.append({
            'comment': cleaned,
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': compound,
            'sentiment': sentiment
        })
    
    df = pd.DataFrame(results)
    
    # İstatistikler
    total = len(df)
    positive_count = len(df[df['sentiment'] == 'Positive'])
    negative_count = len(df[df['sentiment'] == 'Negative'])
    neutral_count = len(df[df['sentiment'] == 'Neutral'])
    
    stats = {
        'total': total,
        'positive': positive_count,
        'negative': negative_count,
        'neutral': neutral_count,
        'positive_pct': (positive_count / total * 100) if total > 0 else 0,
        'negative_pct': (negative_count / total * 100) if total > 0 else 0,
        'neutral_pct': (neutral_count / total * 100) if total > 0 else 0
    }
    
    print(f"  Analiz edilen yorum: {stats['total']}")
    print(f"  Pozitif: {stats['positive']} (%{stats['positive_pct']:.2f})")
    print(f"  Negatif: {stats['negative']} (%{stats['negative_pct']:.2f})")
    print(f"  Nötr: {stats['neutral']} (%{stats['neutral_pct']:.2f})\n")
    
    return df, stats


def extract_keywords(comments: List[str], top_n: int = 20) -> Dict[str, List[Tuple[str, int]]]:
    """
    Pozitif ve negatif yorumlardan en sık geçen anahtar kelimeleri çıkarır.
    
    Args:
        comments: Yorum listesi (temizlenmiş)
        top_n: Her kategoriden kaç kelime
        
    Returns:
        {'positive': [(kelime, frekans), ...], 'negative': [...]}
    """
    print("Anahtar kelime analizi yapılıyor...")
    
    analyzer = SentimentIntensityAnalyzer()
    
    positive_tokens = []
    negative_tokens = []
    
    for comment in comments:
        cleaned = advanced_preprocess_text(comment)
        if not cleaned or len(cleaned) < 3:
            continue
        
        # Sentiment belirle
        scores = analyzer.polarity_scores(cleaned)
        compound = scores['compound']
        
        # Tokenize et
        tokens = tokenize_and_clean(cleaned, remove_stopwords=True)
        
        if compound >= 0.05:
            positive_tokens.extend(tokens)
        elif compound <= -0.05:
            negative_tokens.extend(tokens)
    
    # En sık geçen kelimeleri bul
    positive_counter = Counter(positive_tokens)
    negative_counter = Counter(negative_tokens)
    
    # "ai" ve "slop" kelimelerini filtrele (çok sık geçiyor, analizi bozuyor)
    filtered_positive = {k: v for k, v in positive_counter.items() 
                        if k not in ['ai', 'slop', 'video', 'youtube', 'channel']}
    filtered_negative = {k: v for k, v in negative_counter.items() 
                        if k not in ['ai', 'slop', 'video', 'youtube', 'channel']}
    
    positive_top = Counter(filtered_positive).most_common(top_n)
    negative_top = Counter(filtered_negative).most_common(top_n)
    
    print(f"  Pozitif kelimeler: {len(positive_top)}")
    print(f"  Negatif kelimeler: {len(negative_top)}\n")
    
    return {
        'positive': positive_top,
        'negative': negative_top
    }


def analyze_contextual_words(df: pd.DataFrame) -> Dict:
    """
    Kriz ve Fırsat kelimelerinin frekansını analiz eder.
    
    Args:
        df: Sentiment analizi sonuçları DataFrame'i
        
    Returns:
        Kelime frekansları ve karşılaştırma
    """
    print("Bağlam analizi yapılıyor (Kriz vs Fırsat)...")
    
    all_tokens = []
    crisis_words_found = []
    opportunity_words_found = []
    
    for comment in df['comment']:
        tokens = tokenize_and_clean(comment, remove_stopwords=True)
        all_tokens.extend(tokens)
        
        # Kriz ve fırsat kelimelerini tespit et
        for token in tokens:
            if token in CRISIS_WORDS:
                crisis_words_found.append(token)
            if token in OPPORTUNITY_WORDS:
                opportunity_words_found.append(token)
    
    # Kriz kelimelerini say
    crisis_count = len(crisis_words_found)
    crisis_counter = Counter(crisis_words_found)
    
    # Fırsat kelimelerini say
    opportunity_count = len(opportunity_words_found)
    opportunity_counter = Counter(opportunity_words_found)
    
    # Toplam kelime sayısı
    total_words = len(all_tokens)
    
    result = {
        'crisis_count': crisis_count,
        'opportunity_count': opportunity_count,
        'total_words': total_words,
        'crisis_ratio': (crisis_count / total_words * 100) if total_words > 0 else 0,
        'opportunity_ratio': (opportunity_count / total_words * 100) if total_words > 0 else 0,
        'crisis_words': dict(crisis_counter),
        'opportunity_words': dict(opportunity_counter)
    }
    
    print(f"  Kriz kelimeleri: {crisis_count}")
    print(f"  Fırsat kelimeleri: {opportunity_count}\n")
    
    return result


def lda_topic_modeling(negative_comments: List[str], n_topics: int = 5, n_words: int = 10) -> List[Dict]:
    """
    Negatif yorumlarda LDA topic modeling yapar.
    
    Args:
        negative_comments: Negatif yorum listesi
        n_topics: Tespit edilecek topic sayısı
        n_words: Her topic için gösterilecek kelime sayısı
        
    Returns:
        Topic listesi (her topic: {'topic_id': int, 'words': [(kelime, weight), ...]})
    """
    print(f"LDA Topic Modeling yapılıyor ({n_topics} topic)...")
    
    if not negative_comments or len(negative_comments) < n_topics:
        print("  Yeterli negatif yorum yok, topic modeling atlanıyor.\n")
        return []
    
    # Yorumları temizle ve tokenize et
    processed_comments = []
    for comment in negative_comments:
        cleaned = advanced_preprocess_text(comment)
        tokens = tokenize_and_clean(cleaned, remove_stopwords=True)
        # Token listesini string'e çevir (TF-IDF için)
        processed_comments.append(' '.join(tokens))
    
    # Boş yorumları filtrele
    processed_comments = [c for c in processed_comments if len(c) > 0]
    
    if len(processed_comments) < n_topics:
        print("  Yeterli işlenmiş yorum yok, topic modeling atlanıyor.\n")
        return []
    
    try:
        # TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.95)
        tfidf_matrix = vectorizer.fit_transform(processed_comments)
        
        # LDA Model
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
        lda.fit(tfidf_matrix)
        
        # Feature names (kelimeler)
        feature_names = vectorizer.get_feature_names_out()
        
        # Topic'leri çıkar
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            # En yüksek ağırlıklı kelimeleri al
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [(feature_names[i], float(topic[i])) for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx + 1,
                'words': top_words,
                'description': ', '.join([word for word, _ in top_words[:5]])
            })
        
        print(f"  {len(topics)} topic tespit edildi.\n")
        return topics
        
    except Exception as e:
        print(f"  Topic modeling hatası: {e}\n")
        return []


def generate_json_report(sentiment_stats: Dict,
                        keywords: Dict[str, List[Tuple[str, int]]],
                        contextual_stats: Dict,
                        topics: List[Dict]) -> Dict:
    """
    JSON formatında detaylı analiz raporu oluşturur.
    
    Args:
        sentiment_stats: Sentiment istatistikleri
        keywords: Anahtar kelimeler
        contextual_stats: Bağlam analizi istatistikleri
        topics: LDA topic modeling sonuçları
        
    Returns:
        JSON formatında rapor dict
    """
    report = {
        'general_sentiment': {
            'total_comments': sentiment_stats['total'],
            'positive': {
                'count': sentiment_stats['positive'],
                'percentage': round(sentiment_stats['positive_pct'], 2)
            },
            'negative': {
                'count': sentiment_stats['negative'],
                'percentage': round(sentiment_stats['negative_pct'], 2)
            },
            'neutral': {
                'count': sentiment_stats['neutral'],
                'percentage': round(sentiment_stats['neutral_pct'], 2)
            },
            'atmosphere': 'GÜVEN' if sentiment_stats['positive_pct'] > sentiment_stats['negative_pct'] else 'KORKU'
        },
        'top_keywords': {
            'positive': [{'word': word, 'frequency': freq} for word, freq in keywords['positive']],
            'negative': [{'word': word, 'frequency': freq} for word, freq in keywords['negative']]
        },
        'contextual_analysis': {
            'crisis': {
                'total_count': contextual_stats['crisis_count'],
                'percentage': round(contextual_stats['crisis_ratio'], 2),
                'words': contextual_stats['crisis_words']
            },
            'opportunity': {
                'total_count': contextual_stats['opportunity_count'],
                'percentage': round(contextual_stats['opportunity_ratio'], 2),
                'words': contextual_stats['opportunity_words']
            },
            'positioning': 'RİSK' if contextual_stats['crisis_count'] > contextual_stats['opportunity_count'] else 'FIRSAT'
        },
        'negative_comment_topics': topics
    }
    
    return report


def main():
    """Ana fonksiyon - tüm analiz sürecini yönetir."""
    print("=" * 60)
    print("YouTube Duygu ve Algı Analizi - Gelişmiş Versiyon")
    print("=" * 60)
    print()
    
    # 1. CSV'den yorumları yükle
    comments = load_comments_from_csv('yorumlar_backup.csv')
    
    if not comments:
        print("HATA: Yorum yüklenemedi!")
        return
    
    # 2. Duygu Analizi
    df, sentiment_stats = analyze_sentiment(comments)
    
    # 3. Anahtar Kelime Analizi
    keywords = extract_keywords(comments, top_n=20)
    
    # 4. Bağlam Analizi
    contextual_stats = analyze_contextual_words(df)
    
    # 5. LDA Topic Modeling (Negatif yorumlar)
    negative_comments = df[df['sentiment'] == 'Negative']['comment'].tolist()
    topics = lda_topic_modeling(negative_comments, n_topics=5, n_words=10)
    
    # 6. JSON Rapor Oluşturma
    print("JSON raporu oluşturuluyor...")
    json_report = generate_json_report(sentiment_stats, keywords, contextual_stats, topics)
    
    # JSON dosyasına kaydet
    output_file = 'analysis_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Analiz sonuclari '{output_file}' dosyasina kaydedildi.")
    print()
    
    # Özet bilgi
    print("=" * 60)
    print("ÖZET")
    print("=" * 60)
    print(f"Toplam Yorum: {sentiment_stats['total']}")
    print(f"Atmosfer: {json_report['general_sentiment']['atmosphere']}")
    print(f"Konumlandırma: {json_report['contextual_analysis']['positioning']}")
    print(f"Tespit Edilen Topic Sayısı: {len(topics)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
