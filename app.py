import gradio as gr
import pandas as pd
import torch
import os
import re
import string
import nltk
import emoji
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel

nltk.download('stopwords')
stop_words_list = stopwords.words('turkish')

# Ön işleme adımlarını yapmak için fonksiyonumuzu tanımlıyoruz.
def preprocess_text(text):
    # Küçük harflere çevirme
    text = text.lower()
    # Satır sonu karakterlerini kaldırma
    text = re.sub(r'\n', ' ', text)
    # Rakamları kaldırma
    text = re.sub(r'\d', '', text)
    # Noktalama işaretlerini kaldırma
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Stop-words'leri kaldırma
    words = text.split()
    words = [word for word in words if not word in stop_words_list]
    # Tekrarlanan karakterlerin kaldırılması
    words = [re.sub(r'(.)\1{1,}', r'\1\1', word) for word in words]
    # Tekrarlanan boşlukların kaldırılması
    words = [word.strip() for word in words if len(word.strip()) > 1]
    
    text = " ".join(words)
    return text

class BertClassifier(torch.nn.Module):
  def __init__(self, dropout=0.5):
    super(BertClassifier, self).__init__()
    
    self.bert = BertModel.from_pretrained("dbmdz/bert-base-turkish-uncased")
    self.dropout = torch.nn.Dropout(dropout)
    # Kullandığımız önceden eğilmiş model "base" sınıfına ait bir BERT modelidir. Yani;
    # 12 layers of Transformer encoder, 12 attention heads, 768 hidden size, 110M parameters.
    # 768, BERT-base modelindeki hidden size'yi, 5 ise veri setimizdeki toplam kategori sayısını temsil ediyor.
    self.linear = torch.nn.Linear(768, 5)
    self.relu = torch.nn.ReLU()

  def forward(self, input_id, mask):
    # _ değişkeni dizideki tüm belirteçlerin gömme vektörlerini içerir.
    # pooled_output değişkeni [CLS] belirtecinin gömme vektörünü içerir.
    # Metin sınıflandırma için polled_output değişkenini girdi olarak kullanmak yeterlidir.

    # Attention mask, bir belirtecin gercek bir kelimemi yoksa dolgu mu olduğunu tanımlar.
    # Eğer gerçek bir kelime ise attention_mask=1, eğer dolgu ise attention_mask=0 olacaktır.
    # return_dict, değeri "True ise" bir BERT modeli tahmin, eğitim veya değerlendirme sırasında ortaya çıkan
    # loss, logits, hidden_states ve attentions dan oluşan bir tuple oluşturacaktır.
    _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
    dropout_output = self.dropout(pooled_output)
    linear_output = self.linear(dropout_output)
    final_layer = self.relu(linear_output)

    return final_layer
    
model = BertClassifier()
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
model.load_state_dict(torch.load('tubitak2.pt', map_location=torch.device('cpu')))



def predict_text(model, sentence):
  device = torch.device("cpu")
  #model = model.cuda()
  # Prediction işlemi sırasında model ağırlıklarını değiştirmeyeceğimiz modelin gradyanlara ihtiyacı yoktur
  # "no_grad" fonksiyonu ile gradyan hesaplarını devre dışı bırakıyoruz.
  with torch.no_grad():
    # text = Modeli eğitmek için kullanılacak veri setindeki "clean_text" sütunundaki her bir satır.
    # padding = Her bir diziyi belirttiğimiz maksimum uzunluga kadar doldurmak için.
    # max_length = Her bir dizinin maksimum uzunluğu
    # truncation = Eğer değeri "True" ise dizimiz maksimum uzunluğu aşar ise onu keser.
    # return_tensors = Döndürelecek tensörlerin türü. Pytorch kullandığımız için "pt" yazıyoruz. Tensorflow kullansaydık "tf" yazmamız gerekirdi.
    input_id = tokenizer(sentence, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
    
    # Attention mask, bir belirtecin gercek bir kelimemi yoksa dolgu mu olduğunu tanımlar.
    # Eğer gerçek bir kelime ise attention_mask=1, eğer dolgu ise attention_mask=0 olacaktır.
    mask = input_id['attention_mask'].to(device)
    
    # squeeze() fonksiyonu ile "input_ids" özelliğindeki tensörlerin boyutu 1 olan boyutları
    # kaldırarak, tensörün boyutunu azaltıyoruz.
    input_id = input_id['input_ids'].squeeze(1).to(device)
    
    # Modelin eğitim verileri üzerindeki tahminlerinin sonuçları saklanır.
    output = model(input_id, mask)

    categories =  {
        0: 'HAM',
        1: 'SPAM',
    }

    # Kategorik sınıfı döndür.
    return categories.get(output.argmax(dim=1).item()) 
    
import re

# Ön işleme adımlarını yapmak için fonksiyonumuzu tanımlıyoruz.
def preprocess_text(text):
    # Küçük harflere çevirme
    text = text.lower()
    # Satır sonu karakterlerini kaldırma
    text = re.sub(r'\n', ' ', text)
    # Rakamları kaldırma
    text = re.sub(r'\d', '', text)
    # Noktalama işaretlerini kaldırma
    import string
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Stop-words'leri kaldırma
    words = text.split()
    words = [word for word in words if not word in stop_words_list]
    # Tekrarlanan karakterlerin kaldırılması
    words = [re.sub(r'(.)\1{1,}', r'\1\1', word) for word in words]
    # Tekrarlanan boşlukların kaldırılması
    words = [word.strip() for word in words if len(word.strip()) > 1]
    
    text = " ".join(words)
    return text

def predict(df):
    # TODO:
    regex = r'@\w+\s?'
    df['clean_text'] = df['text'].apply(lambda x: re.sub(regex, '', x))
    df['clean_text'] = df['clean_text'].apply(preprocess_text)
    for i in range(len(df)):
      df.loc[i, 'label'] = predict_text(model, df['text'][i])

    return df
    
def get_file(file):
    output_file = "output_GAT0R.csv"

    # For windows users, replace path seperator
    file_name = file.name.replace("\\", "/")
    df = pd.read_csv(file_name, sep="|")
    predict(df)
    
    df.to_csv(output_file, index=False, sep="|")

    return output_file
    
#  Launch the interface with user password
iface = gr.Interface(get_file, "file", "file")

if __name__ == "__main__":
    iface.launch(debug=True)
