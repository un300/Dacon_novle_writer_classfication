# Dacon 소설 작가 분류 AI 경진대회

- ##### 참여기간 : 2020.11.23 ~ 2020.12.04

- ##### 평가기준 : logloss





## 1. 진행과정

### (1) 데이터 불러오기 및 전처리

- 데이터 전처리 과정은 대회에서 제공해주는 가이드 라인을 적용하였습니다.

- 데이터 불러오기

  ```python
  # 데이터 불러오기
  
  raw_train = pd.read_csv('/content/drive/MyDrive/[데이콘] 소설 작가 분류 AI 경진대회/data/train.csv')
  raw_test = pd.read_csv('/content/drive/MyDrive/[데이콘] 소설 작가 분류 AI 경진대회/data/test_x.csv')
  sample_submission = pd.read_csv('/content/drive/MyDrive/[데이콘] 소설 작가 분류 AI 경진대회/data/sample_submission.csv')
  ```

- 텍스트 전처리

  - 토큰화는 음절단위로 진행하였습니다.

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

## (2) 딥러닝 모델 적용

- **악플탐지봇**을 구현하면서 배운 `CNN`과 `LSTM`을 이용하여 초기 모델을 구축하였습니다.

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

##### 하지만,

