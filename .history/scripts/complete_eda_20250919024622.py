import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

plt.style.use('ggplot')
sns.set_palette("viridis")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 200)

BASE_DIR = Path(_file_).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
EDA_OUTPUT = BASE_DIR / "eda_output"
EDA_OUTPUT.mkdir(exist_ok=True)

def generate_eda_report():
    report_content = "# Comprehensive EDA Report for Medical RAG Project\n\n"
    
    report_content += analyze_outbreak_data()
    report_content += analyze_symptoms_data()
    report_content += analyze_misinformation_data()
    
    report_path = EDA_OUTPUT / 'eda_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"EDA report generated successfully at: {report_path}")

def analyze_outbreak_data():
    section_content = "\n## 1. Outbreaks Dataset Analysis\n\n"
    try:
        df = pd.read_csv(DATA_RAW / 'outbreaks_data.csv', low_memory=False)
        section_content += f"*Dataset Shape:* {df.shape}\n\n"
        
        numeric_cols = ['Illnesses', 'Hospitalizations', 'Deaths']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        section_content += "### Missing Values\n"
        missing_values = df.isnull().sum()
        missing_sorted = missing_values[missing_values > 0].sort_values(ascending=False)
        if not missing_sorted.empty:
            for col, count in missing_sorted.items():
                section_content += f"- ⁠ {col} ⁠: {count} missing values ({count/len(df)*100:.1f}%)\n"
        else:
            section_content += "- No missing values found.\n"

        section_content += "\n### Key Distributions\n"
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Outbreak Data Distributions', fontsize=18)
        
        df['State'].value_counts().head(10).plot(kind='bar', ax=axes[0, 0], title='Top 10 States by Outbreaks')
        df['Etiology'].value_counts().head(10).plot(kind='bar', ax=axes[0, 1], title='Top 10 Pathogens')
        df['Illnesses'][df['Illnesses'] > 0].plot(kind='hist', ax=axes[1, 0], bins=50, log=True, title='Distribution of Illnesses (Log Scale)')
        axes[1,0].set_xlabel("Number of Illnesses")
        df['Setting'].value_counts().head(10).plot(kind='barh', ax=axes[1, 1], title='Top 10 Outbreak Settings')

        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
            for label in ax.get_xticklabels():
                label.set_ha('right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(EDA_OUTPUT / 'outbreak_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        section_content += f"![Outbreak Distributions](outbreak_distributions.png)\n\n"
    except Exception as e:
        section_content += f"Error analyzing outbreak data: {str(e)}\n"
    
    return section_content

def analyze_symptoms_data():
    section_content = "\n## 2. Symptoms Dataset Analysis\n\n"
    try:
        df = pd.read_csv(DATA_RAW / 'symptoms_data.csv')
        section_content += f"*Dataset Shape:* {df.shape}\n\n"
        
        section_content += "### Analysis of Symptoms\n"
        fig, axes = plt.subplots(2, 1, figsize=(12, 16), gridspec_kw={'hspace': 0.4})
        fig.suptitle('Symptom Data Analysis', fontsize=18)

        df['Disease'].value_counts().head(15).plot(kind='barh', ax=axes[0], title='Disease Profiles in Dataset')
        axes[0].invert_yaxis()

        symptom_cols = [c for c in df.columns if c.startswith('Symptom_')]
        melted_symptoms = df.melt(id_vars=['Disease'], value_vars=symptom_cols, value_name='Symptom')
        melted_symptoms.dropna(subset=['Symptom'], inplace=True)
        
        melted_symptoms['Symptom'].value_counts().head(15).plot(kind='barh', ax=axes[1], title='Top 15 Most Common Symptoms Overall')
        axes[1].invert_yaxis()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(EDA_OUTPUT / 'symptoms_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        section_content += f"![Symptom Distributions](symptoms_distributions.png)\n\n"
        
    except Exception as e:
        section_content += f"Error analyzing symptoms data: {str(e)}\n"
    
    return section_content

def analyze_misinformation_data():
    section_content = "\n## 3. Misinformation Dataset Analysis\n\n"
    try:
        df = pd.read_csv(DATA_RAW / 'misinformation_data.csv')
        section_content += f"*Dataset Shape:* {df.shape}\n\n"
        
        df.dropna(subset=['text', 'title'], inplace=True)
        section_content += f"*Shape after dropping empty text/title:* {df.shape}\n\n"
        
        section_content += "### Class Distribution (Real vs. Fake)\n"
        class_dist_pct = df['label'].value_counts(normalize=True) * 100
        section_content += f"- Fake (label 0): {class_dist_pct.get(0, 0):.1f}%\n"
        section_content += f"- Real (label 1): {class_dist_pct.get(1, 0):.1f}%\n"

        section_content += "\n### Word Clouds: Fake vs. Real News\n"
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Common Words in Fake vs. Real News', fontsize=18)

        fake_text = ' '.join(df[df['label'] == 0]['text'].astype(str))
        fake_wc = WordCloud(width=800, height=600, background_color='black', colormap='Reds', max_words=100).generate(fake_text)
        axes[0].imshow(fake_wc, interpolation='bilinear')
        axes[0].axis('off')
        axes[0].set_title('Top Words in Fake News')
        
        real_text = ' '.join(df[df['label'] == 1]['text'].astype(str))
        real_wc = WordCloud(width=800, height=600, background_color='white', colormap='Greens', max_words=100).generate(real_text)
        axes[1].imshow(real_wc, interpolation='bilinear')
        axes[1].axis('off')
        axes[1].set_title('Top Words in Real News')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(EDA_OUTPUT / 'misinformation_wordclouds.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        section_content += f"![Misinformation Word Clouds](misinformation_wordclouds.png)\n\n"
        
    except Exception as e:
        section_content += f"Error analyzing misinformation data: {str(e)}\n"
    
    return section_content

if _name_ == "_main_":
    generate_eda_report()
