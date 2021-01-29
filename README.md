# Dacon 소설 작가 분류 AI 경진대회
- ##### 대회진행기간 : 2020.10.29 ~ 2020.12.04 
- ##### 참여기간     : 2020.11.23 ~ 2020.12.04
- ##### 평가기준     : logloss
- ##### 최종결과     : 상위 21% (private : 58위 / 287)
- https://dacon.io/competitions/official/235670/leaderboard/

![final_record](https://user-images.githubusercontent.com/54063179/106223881-9f3fc280-6225-11eb-8572-6ae19c1d052b.PNG)







## 진행과정

## (1) 데이터 불러오기 및 전처리

- 데이터 전처리 과정은 대회에서 제공해주는 가이드 라인을 적용하였습니다.

- 데이터 불러오기

  ```python
  # 데이터 불러오기
  
  raw_train = pd.read_csv('/content/drive/MyDrive/[데이콘] 소설 작가 분류 AI 경진대회/data/train.csv')
  raw_test = pd.read_csv('/content/drive/MyDrive/[데이콘] 소설 작가 분류 AI 경진대회/data/test_x.csv')
  sample_submission = pd.read_csv('/content/drive/MyDrive/[데이콘] 소설 작가 분류 AI 경진대회/data/sample_submission.csv')
  ```

- 텍스트 전처리

  - 토큰화는 음절단위(한글자씩으로)로 진행하였습니다.

  ```python
  # 텍스트 전처리(토큰화 + 패딩화)
  
  def alpha_num(text):
      return re.sub(r'[^A-Za-z0-9 ]', '', text)
  
  
  def remove_stopwords(text):
      final_text = []
      for i in text.split():
          if i.strip().lower() not in stopwords:
              final_text.append(i.strip())
      return " ".join(final_text)
  
  
  stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", 
               "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", 
               "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", 
               "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", 
               "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", 
               "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", 
               "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", 
               "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", 
               "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", 
               "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", 
               "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
  ```

  ```python
  train['text'] = train['text'].str.lower().apply(alpha_num).apply(remove_stopwords)
  test['text'] = test['text'].str.lower().apply(alpha_num).apply(remove_stopwords)
  ```

  ```python
  vocab_size = 20000
  padding_type='post'
  max_length = 500
  ```

  ```python
  tokenizer = Tokenizer(num_words = vocab_size)
  tokenizer.fit_on_texts(X_train)
  word_index = tokenizer.word_index
  
  train_sequences = tokenizer.texts_to_sequences(X_train)
  test_sequences = tokenizer.texts_to_sequences(X_test)
  
  x_train = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)
  x_test = pad_sequences(test_sequences, padding=padding_type, maxlen=max_length)
  print(x_train.shape, x_test.shape)
  ```

## (2) 초기 딥러닝 모델 적용

- `CNN`과 `LSTM`으로 초기 모델을 구축하였습니다.

- `CNN`과 `LSTM` 모두 5-Fold를 적용하여 최종 Test set을 예측하였습니다.

  5-Fold

  ```python
  from sklearn.model_selection import StratifiedKFold
  cv = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
  
  validation_pred = np.zeros((y.shape[0], n_class))
  test_pred = np.zeros((test.shape[0], n_class))
  
  i = 0
  for train_idx, val_idx in tqdm_notebook(cv.split(x_train, y)):
      print("{}-Fold" .format(i+1))
      X_train = x_train[train_idx]
      y_train = y[train_idx]
  
      X_validation = x_train[val_idx]
      y_validation = y[val_idx]
  
      CNN = get_model()
  
      es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2, restore_best_weights=True)
  
      CNN.fit(X_train, y_train,
               epochs           = 20,
               callbacks        = [es],
               batch_size       = 64,
               validation_data  = (X_validation, y_validation))
      
      validation_pred[val_idx, :] = CNN.predict(X_validation)
      test_pred += (CNN.predict(x_test) / 5)
      print('')
  
      i += 1
  ```

  

  ##### LSTM

  ```python
  def get_model():
      model = Sequential()
      model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
      model.add(Dropout(0.5))
      model.add(Bidirectional(LSTM(128, return_sequences=True)))
      model.add(Bidirectional(LSTM(128, return_sequences=False)))
      model.add(Dense(n_class, activation = 'softmax'))
  
      model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
  
      return model
  ```

  ##### 1D_CNN

  - `CNN`의 경우 Pooling과정에서 `MaxPooling`을 많이 사용한다고 알려져 있습니다.
  - 하지만, 제가 구현한 모델의 경우는 `GlobalMaxPooling1D`이 더욱 성능이 좋았습니다.

  ```python
  def get_model() :
      from tensorflow.keras import Sequential
      from tensorflow.keras.layers import Dense, Embedding, LSTM, GlobalMaxPooling1D, Conv1D, Dropout, Bidirectional, Flatten, MaxPool1D, GlobalAveragePooling1D, AveragePooling1D
      import tensorflow as tf
  
      model = Sequential()
      model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
      model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation=mish, strides=1))
      model.add(GlobalAveragePooling1D())
      model.add(Flatten())
      model.add(Dropout(0.5))
      model.add(Dense(128, activation=mish))
      model.add(Dense(n_class, activation='softmax'))
      model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.01))
      return model
  
  ```

##### 그러나,

- 대회에서 제공한 전처리와 단순한 딥러닝 모델로써는 아래와 같이 logloss가 약 0.5를 웃도는 좋지 않은 성능을 내었습니다.

- 위 모델 구조에서 활성화 함수를 relu에서 mish로 변경하거나`Conv1D`, `LSTM`, `Dense`의 노드 개수만 바꾸는 정도로 눈에띄는 성능 향상을 기대할 수 없었습니다.

  ![그림2](https://user-images.githubusercontent.com/54063179/106223941-baaacd80-6225-11eb-973f-68875429a2e8.PNG)

  ![그림1](https://user-images.githubusercontent.com/54063179/106223901-aa92ee00-6225-11eb-8f0d-c002bde59160.PNG)



#### 더 이상 위의 구조에서 모델의 성능을 높이는 것은 무리라고 생각하였고, 다른 방법을 찾아야 했습니다.



## (3) Word Embedding 사용 : FastText, Glove

- 모델의 성능을 더욱 높이기 위해 구글링하였고 '자연어처리의 성능은 Embedding과정에 달려있다'는 글을 접하였습니다.
- 그래서 모델의 성능을 높이기 위해 토큰의 단위인 단어의 연관성을 고려하여 Embedding하는 `FastText`, `Glove` 기법 적용해보았습니다.

#### FastText

```python
!pip install fasttext
import fasttext
import fasttext.util

print(f"== LOAD fasttext START at {datetime.datetime.now()}")
ft = fasttext.load_model('/content/drive/MyDrive/FastText/cc.en.300.bin')
print(f"== LOAD fasttext   END at {datetime.datetime.now()}")

embedding_dim = 300
embedding_matrix = np.zeros( (len(word_index)+1, embedding_dim) )

# 임베딩테이블 만들기
embedding_dim = 300
embedding_matrix = np.zeros( (len(word_index)+1, embedding_dim) )
for word, idx in word_index.items():
    embedding_vector = ft.get_word_vector(word)
    if embedding_vector is not None :
        embedding_matrix[idx] = embedding_vector
```



#### Glove
- Glove는 model이 아니라 Vector를 사용하였습니다.
```python
import numpy as np
embedding_dict = dict()
f = open('/content/drive/MyDrive/Glove/glove.6B.300d.txt', encoding="utf8")

for line in f:
    word_vector = line.split()
    word = word_vector[0]
    word_vector_arr = np.asarray(word_vector[1:], dtype='float32')
    embedding_dict[word] = word_vector_arr
f.close()
print('%s개의 Embedding vector가 있습니다.' % len(embedding_dict))


embedding_dim = 300
embedding_matrix = np.zeros( (len(word_index)+1, embedding_dim) )


# 임베딩테이블 만들기
for word, idx in word_index.items():
    embedding_vector = embedding_dict.get(word)

    if embedding_vector is not None :
        embedding_matrix[idx] = embedding_vector
    else :
        embedding_matrix[idx] = np.zeros((1, embedding_dim))


```



#### Word Embedding을 적용한 1D-CNN 코드
 - LSTM보다 1D-CNN의 코드가 더욱 좋았기 떄문에 1D-CNN 코드를 보여드립니다.

```python
def get_model() :

    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Embedding, GlobalMaxPooling1D, Conv1D, Dropout, Flatten, MaxPool1D, GlobalAveragePooling1D, Flatten

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix[0:vocab_size]], input_length=max_length))  ### Fast Text 또는 Glove를 적용
    model.add(Dropout(0.2))
    model.add(Conv1D(50, 3, padding='same', activation=mish, strides=1))
    model.add(GlobalAveragePooling1D())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(50, activation=mish))
    model.add(Dropout(0.2))
    model.add(Dense(n_class, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.002))

    return model

```



#### Word Embedding 적용 결과
![after_embedding](https://user-images.githubusercontent.com/54063179/106223841-86cfa800-6225-11eb-8588-df10f33b7f31.PNG)


- 임베딩을 적용하기 전 약 0.5대의 logloss를 보였지만 임베딩 적용 후 약 0.3점 대로 눈에띄는 향상을 보여주었습니다.





## (4) 최종 모형
- 최종모형은 dim이 100인 **Glove Embedding**을 적용한 1D-CNN입니다.
- 최종모형은 최종제출 `logloss : 0.3255139553`을 기록하였습니다.
- 모델의 구조는 다음과 같습니다
```python
def get_model() :

    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Embedding, GlobalMaxPooling1D, Conv1D, Dropout, Flatten, MaxPool1D, GlobalAveragePooling1D, Flatten

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix[0:vocab_size]], input_length=max_length))  ### Fast Text 또는 Glove를 적용
    model.add(Dropout(0.2))
    model.add(Conv1D(50, 3, padding='same', activation=mish, strides=1))
    model.add(GlobalAveragePooling1D())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(50, activation=mish))
    model.add(Dropout(0.2))
    model.add(Dense(n_class, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.002))

    return model
```



## (5) 느낀점
- 처음 참여해보는 자연어처리 공모전이었고 딥러닝 또한 처음 다루어 보았기에 기초적인 1D-CNN과 LSTM만을 다루어본 것이 아쉽습니다. 조금 더 공부해서 Bert나 VDCNN같은 성능이 좋은 최신 모델을 다루어 보고싶습니다.
- 자연어 처리에서 임베딩 과정이 성능 향상에 매우 중요하다는 사실을 깨달았습니다.