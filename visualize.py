"""
Görselleştirme Modülü - YouTube Yorum Analizi Grafikleri
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import bigrams
import nltk

# NLTK verilerini indir
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Grafik stili ayarları
sns.set_style("darkgrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Renk paleti (Kurumsal - Mavi/Gri tonları)
COLORS = {
    'positive': '#2E86AB',  # Mavi
    'negative': '#A23B72',  # Mor-Kırmızı
    'neutral': '#6C757D',   # Gri
    'opportunity': '#28A745',  # Yeşil
    'crisis': '#DC3545',    # Kırmızı
    'background': '#F8F9FA'  # Açık Gri
}

# Charts klasörünü oluştur
os.makedirs('charts', exist_ok=True)


def load_data():
    """Veri setlerini yükle."""
    print("Veri setleri yükleniyor...")
    
    # JSON analiz sonuçlarını yükle
    with open('analysis_results.json', 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    
    # CSV yorumları yükle
    df_comments = pd.read_csv('yorumlar_backup.csv', encoding='utf-8')
    comments = df_comments['comment'].dropna().tolist()
    
    print(f"  - {len(comments)} yorum yüklendi")
    print(f"  - Analiz sonuçları yüklendi\n")
    
    return analysis_data, comments


def preprocess_text(text):
    """Metni temizle."""
    if not isinstance(text, str):
        return ""
    
    # HTML etiketlerini temizle
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    text = re.sub(r'&#\d+;', '', text)
    
    # URL'leri temizle
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize_text(text, remove_stopwords=True):
    """Metni tokenize et."""
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha() and len(token) > 1]
    
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    return tokens


def plot_sentiment_distribution(analysis_data):
    """Sentiment dağılımı donut chart."""
    print("1. Sentiment Dağılımı grafiği oluşturuluyor...")
    
    sentiment = analysis_data['general_sentiment']
    
    labels = ['Pozitif', 'Negatif', 'Nötr']
    sizes = [
        sentiment['positive']['percentage'],
        sentiment['negative']['percentage'],
        sentiment['neutral']['percentage']
    ]
    colors = [COLORS['positive'], COLORS['negative'], COLORS['neutral']]
    
    # Donut chart
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Dış halka
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    
    # İç halka (boşluk için)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)
    
    # Ortaya toplam yorum sayısı
    total = sentiment['total_comments']
    ax.text(0, 0, f'{total:,}\nYorum', 
            ha='center', va='center', 
            fontsize=16, weight='bold',
            color='#2C3E50')
    
    ax.set_title('Tüketici Duygu Durumu Dağılımı', 
                fontsize=16, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('charts/sentiment_distribution.png', 
                bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    
    print("  [OK] sentiment_distribution.png kaydedildi\n")


def plot_crisis_vs_opportunity(analysis_data):
    """Kriz vs Fırsat algısı bar chart."""
    print("2. Kriz vs Fırsat grafiği oluşturuluyor...")
    
    contextual = analysis_data['contextual_analysis']
    
    # En sık geçen kelimeleri al
    crisis_words = contextual['crisis']['words']
    opportunity_words = contextual['opportunity']['words']
    
    # Top 10 kelimeyi al
    top_crisis = sorted(crisis_words.items(), key=lambda x: x[1], reverse=True)[:10]
    top_opportunity = sorted(opportunity_words.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Grafik için veri hazırla
    crisis_labels = [word for word, _ in top_crisis]
    crisis_values = [count for _, count in top_crisis]
    
    opportunity_labels = [word for word, _ in top_opportunity]
    opportunity_values = [count for _, count in top_opportunity]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Kriz kelimeleri
    bars1 = ax1.barh(range(len(crisis_labels)), crisis_values, 
                     color=COLORS['crisis'], alpha=0.8)
    ax1.set_yticks(range(len(crisis_labels)))
    ax1.set_yticklabels(crisis_labels, fontsize=10)
    ax1.set_xlabel('Frekans', fontsize=11, weight='bold')
    ax1.set_title('Kriz/Risk Algısı\n(Top 10 Kelime)', 
                  fontsize=13, weight='bold', pad=15)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Değerleri çubukların üzerine yaz
    for i, (bar, val) in enumerate(zip(bars1, crisis_values)):
        ax1.text(val + max(crisis_values)*0.01, i, str(val), 
                va='center', fontsize=9, weight='bold')
    
    # Fırsat kelimeleri
    bars2 = ax2.barh(range(len(opportunity_labels)), opportunity_values, 
                     color=COLORS['opportunity'], alpha=0.8)
    ax2.set_yticks(range(len(opportunity_labels)))
    ax2.set_yticklabels(opportunity_labels, fontsize=10)
    ax2.set_xlabel('Frekans', fontsize=11, weight='bold')
    ax2.set_title('Fırsat/Gelişim Algısı\n(Top 10 Kelime)', 
                  fontsize=13, weight='bold', pad=15)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # Değerleri çubukların üzerine yaz
    for i, (bar, val) in enumerate(zip(bars2, opportunity_values)):
        ax2.text(val + max(opportunity_values)*0.01, i, str(val), 
                va='center', fontsize=9, weight='bold')
    
    plt.suptitle('AI Teknolojisinin Algılanan Risk ve Fırsat Dengesi', 
                fontsize=15, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('charts/crisis_vs_opportunity.png', 
                bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    
    print("  [OK] crisis_vs_opportunity.png kaydedildi\n")


def plot_top_fears(analysis_data, comments):
    """Tüketici korkuları horizontal bar chart."""
    print("3. Tüketici Korkuları grafiği oluşturuluyor...")
    
    analyzer = SentimentIntensityAnalyzer()
    
    # Negatif yorumları filtrele
    negative_comments = []
    for comment in comments:
        cleaned = preprocess_text(comment)
        if len(cleaned) < 3:
            continue
        scores = analyzer.polarity_scores(cleaned)
        if scores['compound'] <= -0.05:
            negative_comments.append(cleaned)
    
    # Bigram'ları çıkar
    all_bigrams = []
    for comment in negative_comments[:5000]:  # İlk 5000 negatif yorum
        tokens = tokenize_text(comment, remove_stopwords=True)
        if len(tokens) >= 2:
            comment_bigrams = list(bigrams(tokens))
            # "ai" ve "slop" içeren bigram'ları filtrele (çok sık geçiyor)
            filtered_bigrams = [f"{w1} {w2}" for w1, w2 in comment_bigrams 
                              if not (w1 == 'ai' and w2 == 'slop')]
            all_bigrams.extend(filtered_bigrams)
    
    # En sık geçen bigram'ları bul
    bigram_counter = Counter(all_bigrams)
    
    # "ai slop", "dead internet" gibi önemli terimleri manuel ekle
    important_terms = {
        'dead internet': 0,
        'fake content': 0,
        'human creativity': 0,
        'real content': 0,
        'misinformation': 0
    }
    
    # Bu terimleri yorumlarda say
    for comment in negative_comments:
        comment_lower = comment.lower()
        if 'dead internet' in comment_lower:
            important_terms['dead internet'] += 1
        if 'fake content' in comment_lower or 'fake' in comment_lower:
            important_terms['fake content'] += 1
        if 'human creativity' in comment_lower or 'human made' in comment_lower:
            important_terms['human creativity'] += 1
        if 'real content' in comment_lower or 'real' in comment_lower:
            important_terms['real content'] += 1
        if 'misinformation' in comment_lower:
            important_terms['misinformation'] += 1
    
    # Bigram counter'a ekle
    for term, count in important_terms.items():
        if count > 0:
            bigram_counter[term] = count
    
    # Top 15'i al
    top_fears = bigram_counter.most_common(15)
    
    # Grafik
    fig, ax = plt.subplots(figsize=(12, 8))
    
    labels = [fear for fear, _ in top_fears]
    values = [count for _, count in top_fears]
    
    bars = ax.barh(range(len(labels)), values, 
                   color=COLORS['negative'], alpha=0.8)
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Frekans', fontsize=12, weight='bold')
    ax.set_title('Tüketici Korkuları - En Sık Geçen Kavramlar\n(Negatif Yorumlardan)', 
                fontsize=14, weight='bold', pad=15)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Değerleri çubukların üzerine yaz
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + max(values)*0.01, i, str(val), 
               va='center', fontsize=9, weight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/top_fears.png', 
                bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    
    print("  [OK] top_fears.png kaydedildi\n")


def plot_negative_wordcloud(comments):
    """Negatif yorumlardan kelime bulutu."""
    print("4. Negatif Kelime Bulutu oluşturuluyor...")
    
    analyzer = SentimentIntensityAnalyzer()
    
    # Negatif yorumları topla
    negative_text = []
    for comment in comments:
        cleaned = preprocess_text(comment)
        if len(cleaned) < 3:
            continue
        scores = analyzer.polarity_scores(cleaned)
        if scores['compound'] <= -0.05:
            negative_text.append(cleaned)
    
    # Tüm negatif yorumları birleştir
    all_negative_text = ' '.join(negative_text)
    
    # Kelime bulutu oluştur
    wordcloud = WordCloud(
        width=1600,
        height=900,
        background_color='#1a1a1a',  # Koyu arka plan
        colormap='Reds',  # Kırmızı tonları
        max_words=200,
        relative_scaling=0.5,
        min_font_size=10,
        max_font_size=100,
        stopwords=set(stopwords.words('english')).union({'ai', 'slop', 'video', 'youtube', 'channel'})
    ).generate(all_negative_text)
    
    # Grafik
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='#1a1a1a')
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Negatif Yorumlardan Kelime Bulutu\n(Tüketici Endişeleri ve Korkuları)', 
                fontsize=16, weight='bold', pad=20, color='white')
    
    plt.tight_layout()
    plt.savefig('charts/negative_wordcloud.png', 
                bbox_inches='tight', facecolor='#1a1a1a', dpi=300)
    plt.close()
    
    print("  [OK] negative_wordcloud.png kaydedildi\n")


def main():
    """Ana fonksiyon."""
    print("=" * 60)
    print("GÖRSELLEŞTİRME MODÜLÜ")
    print("=" * 60)
    print()
    
    # Verileri yükle
    analysis_data, comments = load_data()
    
    # Grafikleri oluştur
    plot_sentiment_distribution(analysis_data)
    plot_crisis_vs_opportunity(analysis_data)
    plot_top_fears(analysis_data, comments)
    plot_negative_wordcloud(comments)
    
    print("=" * 60)
    print("Tüm grafikler başarıyla oluşturuldu!")
    print("Grafikler 'charts/' klasöründe kaydedildi.")
    print("=" * 60)


if __name__ == "__main__":
    main()
