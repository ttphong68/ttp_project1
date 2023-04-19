
from Library_Functions import *
#----------------------------------------------------------------------------------------------------
# Part 1: Build project
#----------------------------------------------------------------------------------------------------

# Load data
# print('Loading data.....')
df = df_parquet()

### Load model

# Đọc model LogisticRegression()
model = load_model_Sentiment_1()
# Đọc model CountVectorizer()
tfidf = load_model_tf()

# Data pre - processing
# df.drop(['Unnamed: 0'],axis=1,inplace=True)
# remove duplicate
df.drop_duplicates(inplace=True)
# remove missing values
df.dropna(inplace=True)

# split data into train and test
# print('Split data into train and test...........')
X=df['comment']
y=df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_cv = tfidf.transform(X_test)
y_pred = model.predict(X_test_cv)
cm = confusion_matrix(y_test, y_pred)


#----------------------------------------------------------------------------------------------------
# Part 2: Build app
#----------------------------------------------------------------------------------------------------
# Overview

st.image('download.jpg')

st.title("Trung tâm tin học - ĐH KHTN")
st.header('Data Science and Machine Learning Certificate')
st.subheader('Sentiment analysis of Vietnamese comments on Shopee')

# st.video('https://www.youtube.com/watch?v=q3nSSZNOg38&list=PLFTWPHJsZXVVnckL0b3DYmHjjPiRB5mqX')

menu = ['Tổng quan', 'Xây dựng mô hình', 'Dự đoán mới']
choice = st.sidebar.selectbox('Menu', menu)

#----------------------------------------------------------------------------------------------------
if choice == 'Tổng quan':
#----------------------------------------------------------------------------------------------------
    st.subheader('1.Tổng quan')
    st.write('''**Yêu cầu** : Xây dựng hệ thống hỗ trợ phân loại các phản hồi của khách hàng thành các nhóm : tích cực, tiêu cực trung tính dựa trên dữ liệu dạng văn bản.
    ''')
    st.write('''**Mục tiêu/ Vấn đề** : Xây dựng mô hình dự đoán giúp người bán hàng có thể biết được những phản hồi nhanh chóng của khách hàng về sản phẩm hay dịch vụ của họ ( tích cực, tiêu cực hay trung tính ), điều này giúp cho người bán biết được tình hình kinh doanh, hiểu được ý kiến của khách hàng từ đó giúp họ cải thiện hơn trong dịch vụ, sản phẩm.
    ''')
    st.write('''
    **Hướng dẫn chi tiết** :
    - Hiểu được vấn đề
    - Import các thư viện cần thiết và hiểu cách sử dụng
    - Đọc dữ liệu được cung cấp
    - Thực hiện EDA (Exploratory Data Analysis – Phân tích Khám phá Dữ liệu) cơ bản ( sử dụng Pandas Profifing Report )
    - Tiền xử lý dữ liệu : Làm sạch, tạo tính năng mới , lựa chọn tính năng cần thiết....
    ''')
    st.write('''
    **Bước 1** : Business Understanding

    **Bước 2** : Data Understanding ==> Giải quyết bài toán Sentiment analysis trong E-commerce bằng thuật toán nhóm Supervised Learning - Classification : Naive Bayes, KNN, Logictic Regression...

    **Bước 3** : Data Preparation/ Prepare : Chuẩn hóa tiếng việt, viết các hàm xử lý dữ liệu thô...

    **Xử lý tiếng việt** : ''')

    st.write('''
    **1.Tiền xử lý dữ liệu thô** :''')

    st.write('''
    - Chuyển text về chữ thường
    - Loại bỏ các ký tự đặc biệt nếu có
    - Thay thế emojicon/ teencode bằng text tương ứng
    - Thay thế một số punctuation và number bằng khoảng trắng
    - Thay thế các từ sai chính tả bằng khoảng trắng
    - Thay thế loạt khoảng trắng bằng một khoảng trắng''')
    
    st.write('''**2.Chuẩn hóa Unicode tiếng Việt** :''')
    st.write('''**3.Tokenizer văn bản tiếng Việt bằng thư viện underthesea** :''')
    st.write('''**4.Xóa các stopword tiếng Việt** :''')
    st.write('''**Bước 4&5: Modeling & Evaluation/ Analyze & Report**''')
    st.write('''**Xây dựng các Classification model dự đoán**''')
    
    st.write('''
    - Naïve Bayes\n
    - Logistic Regression\n
    - Tree Algorithms…\n
    - Thực hiện/ đánh giá kết quả các Classification model\n
    - R-squared\n
    - Acc, precision, recall, f1,…''')
    
    st.write('''**Kết luận**''')
    st.write('''**Bước 6: Deployment & Feedback/ Act**''')
    st.write('''Đưa ra những cải tiến phù hợp để nâng cao sự hài lòng của khách hàng, thu hút sự chú ý của khách hàng mới''')
    
    st.write('''
    Đây là dự án về phân tích cảm xúc của các bình luận của người Việt trên Shopee.
    Bộ dữ liệu được thu thập từ trang web Shopee (Do Cô cung cấp).
    Tập dữ liệu chứa 2 cột: xếp hạng và bình luận.
    Thang đánh giá là từ 1 đến 5.
    Cột Comment là nhận xét của khách hàng sau khi họ mua hàng trên Shopee.
    Mục tiêu của dự án này là xây dựng một mô hình để dự đoán cảm xúc của các bình luận.
    Cảm xúc tích cực hay tiêu cực.
    ''')
    st.write('''
    Bộ dữ liệu có 2 lớp: Tích cực và tiêu cực. 
    ''')
    st.write('''
    Mô hình được xây dựng với Logistic Regression và sử dụng oversampling data:
    - Mô hình có độ chính xác 87% ( Accuracy ).
    - Mô hình có độ chính xác 94% ( precision ) cho lớp tích cực ( positive class ).
    - Mô hình có độ chính xác 85% ( recall ) cho lớp tích cực ( positive class ).
    - Mô hình có độ chính xác 71% ( precision ) cho lớp tiêu cực ( negative class ).
    - Mô hình có độ chính xác 87% ( recall ) cho lớp tiêu cực ( negative class ). 
    ''')
    st.subheader('2.Giáo viên hướng dẫn')
    st.write('''
    **Cô : Khuất Thùy Phương**
    ''')
    st.subheader('3.Học viên thực hiện')
    st.write('''
    **HV : Thái Thanh Phong - Nguyễn Hoàng Long**
    ''')
#----------------------------------------------------------------------------------------------------
elif choice == 'Xây dựng mô hình':
#----------------------------------------------------------------------------------------------------
    st.subheader('Xây dựng mô hình')
    st.write('#### Tiền xử lý dữ liệu')
    st.write('##### Hiển thị dữ liệu')
    st.table(df.head())
    # plot bar chart for sentiment
    st.write('##### Biểu đồ Bar cho biểu thị tình cảm')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df['sentiment'].value_counts().index, df['sentiment'].value_counts().values)
    # add percentage labels to bars (color: black), format: 2 decimal places, center labels
    for p in ax.patches:
        ax.annotate('{:.2f}%'.format(p.get_height()/len(df)*100), (p.get_x()+0.4, p.get_height()+1000), color='black', ha='center')

    ax.set_xticks(df['sentiment'].value_counts().index)
    ax.set_xticklabels(['Tiêu cực', 'Tích cực'])
    ax.set_ylabel('Số lượng bình luận')
    ax.set_title('Biểu đồ Bar cho biểu thị tình cảm')
    st.pyplot(fig)
    
    # ## Negative
    st.write('##### Wordcloud Cho bình luận tiêu cực')
    st.image('sentiment_0.png')
    
    # ## Positive
    st.write('##### Wordcloud Cho bình luận tích cực')
    st.image('sentiment_1.png')

    st.write('#### Xây dựng mô hình và đánh giá:')
    st.write('##### Confusion matrix')
    st.table(cm)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig)    

    st.write('##### Classification report')
    st.table(classification_report(y_test, y_pred, output_dict=True))

    st.write('##### Accuracy')
    # show accuracy as percentage with 2 decimal places
    st.write(f'{accuracy_score(y_test, y_pred) * 100:.2f}%')
#----------------------------------------------------------------------------------------------------    
elif choice == 'Dự đoán mới':
#----------------------------------------------------------------------------------------------------
    st.subheader('Dự đoán mới')
    st.write('''
    Nhập vào một bình luận và mô hình sẽ dự đoán tình cảm của bình luận. 
    ''')
    # menu = ["Nhập bình luận", "Tải tệp Excel", "Tải tệp CSV", "Bình luận bằng giọng nói", "Nói chuyện với chatGPT"]
    # menu = ["Nhập bình luận", "Tải tệp Excel", "Tải tệp CSV"]
    menu = ["Nhập bình luận", "Tải tệp Excel", "Tải tệp CSV", "Tải tệp âm thanh"]

    choice = st.selectbox("Menu",menu)
    if choice == "Nhập bình luận":
        comment = st.text_input('Nhập vào một bình luận')
        if st.button('Dự đoán'):
            if comment != '':
                comment = text_transform(comment)
                comment = tfidf.transform([comment])
                y_predict = model.predict(comment)

                if y_predict[0] == 1:
                    st.write('Tình cảm của bình luận là tích cực')
                else:
                    st.write('Tình cảm của bình luận là tiêu cực')
            else:
                st.write('Nhập vào một bình luận')
    elif choice == "Tải tệp Excel":
        st.write('Bạn chọn upload excel')
        uploaded_file = st.file_uploader("Bạn vui lòng chọn file: ")
        if uploaded_file is not None:
            # check file type not excel
            if uploaded_file.name.split('.')[-1] != 'xlsx':
                st.write('File không đúng định dạng, vui lòng chọn file excel')
            
            elif uploaded_file.name.split('.')[-1] == 'xlsx':
                    
                # load data csv
                df_upload = pd.read_excel(uploaded_file)
                
                # predict sentiment of review
                # list result
                list_result = []
                for i in range(len(df_upload)):
                    comment = df_upload['review_text'][i]
                    comment = text_transform(comment)
                    comment = tfidf.transform([comment])
                    y_predict = model.predict(comment)
                    list_result.append(y_predict[0])
                    
                # apppend list result to dataframe
                df_upload['sentiment'] = list_result
                df_after_predict = df_upload.copy()
                # change sentiment to string
                y_class = {0:'Tình cảm của bình luận là tiêu cực', 1:'Tình cảm của bình luận là tích cực'}
                df_after_predict['sentiment']  = [y_class[i] for i in df_after_predict.sentiment]
                
                # show table result
                st.subheader("Result & Statistics :")
                # get 5 first row
                st.write("5 bình luận đầu tiên: ")
                st.table(df_after_predict.iloc[:,[0,1]].head())
                                
                # show wordcloud
                st.subheader("Biểu đồ Wordcloud trích xuất đặc trưng theo nhóm sentiment: ")
                cmt0 = df_after_predict[df_after_predict['sentiment']=='Tình cảm của bình luận là tiêu cực']
                cmt1 = df_after_predict[df_after_predict['sentiment']=='Tình cảm của bình luận là tích cực']
                cmt0 = cmt0['review_text'].str.cat(sep=' ')
                cmt1 = cmt1['review_text'].str.cat(sep=' ')
                wc0 = WordCloud(background_color='black', max_words=200, stopwords=stopwords_lst, width=1600,height=800, max_font_size=200)
                wc0.generate(cmt0)
                wc1 = WordCloud(background_color='black', max_words=200, stopwords=stopwords_lst, width=1600,height=800, max_font_size=200)
                wc1.generate(cmt1)
                fig, ax = plt.subplots(1,2,figsize=(20, 10))
                ax[0].imshow(wc0,interpolation='bilinear')
                ax[0].axis('off')
                ax[0].set_title('Tình cảm của bình luận là tiêu cực')
                ax[1].imshow(wc1,interpolation='bilinear')
                ax[1].axis('off')
                ax[1].set_title('Tình cảm của bình luận là tích cực')
                st.pyplot(fig)
                
                # show plot bar chart of sentiment
                st.subheader("Biểu đồ cột thể hiện số lượng bình luận theo nhóm sentiment: ")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax = sns.countplot(x='sentiment', data=df_after_predict)
                # add percentage in center of bar chart
                for p in ax.patches:
                    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(df_after_predict)),
                                (p.get_x()+0.4, p.get_height()+0.03), 
                                color='black', ha='center')
                st.pyplot(fig)
                
                # download file excel
                st.subheader("Tải tệp excel kết quả dự đoán: ")
                output = io.BytesIO()
                writer = pd.ExcelWriter(output)
                df_after_predict.to_excel(writer, sheet_name='Sheet1', index=False)
                writer.save()
                output.seek(0)
                
                st.download_button('Download', data=output, file_name='result_excel.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    
    elif choice == "Tải tệp CSV":
        st.write('Bạn chọn upload csv')
        uploaded_file = st.file_uploader("Bạn vui lòng chọn file: ")
        if uploaded_file is not None:
            # check file type if not csv
            if uploaded_file.name.split('.')[-1] != 'csv':
                st.write('File không đúng định dạng, vui lòng chọn file csv')
            elif uploaded_file.name.split('.')[-1] == 'csv':
                            
                # load data csv
                df_upload = pd.read_csv(uploaded_file)
                df_upload.drop('Unnamed: 0', axis=1, inplace=True)
                
                # predict sentiment of review
                # list result
                list_result = []
                for i in range(len(df_upload)):
                    comment = df_upload['review_text'][i]
                    comment = text_transform(comment)
                    comment = tfidf.transform([comment])
                    y_predict = model.predict(comment)
                    list_result.append(y_predict[0])
                    
                # apppend list result to dataframe
                df_upload['sentiment'] = list_result
                df_after_predict = df_upload.copy()
                # change sentiment to string
                y_class = {0:'Tình cảm của bình luận là tiêu cực', 1:'Tình cảm của bình luận là tích cực'}
                df_after_predict['sentiment']  = [y_class[i] for i in df_after_predict.sentiment]
                
                # show table result
                st.subheader("Result & Statistics :")
                # get 5 first row
                st.write("5 bình luận đầu tiên: ")
                st.table(df_after_predict.iloc[:,[0,1]].head())
                                
                # show wordcloud
                st.subheader("Biểu đồ Wordcloud trích xuất đặc trưng theo nhóm sentiment: ")
                cmt0 = df_after_predict[df_after_predict['sentiment']=='Tình cảm của bình luận là tiêu cực']
                cmt1 = df_after_predict[df_after_predict['sentiment']=='Tình cảm của bình luận là tích cực']
                cmt0 = cmt0['review_text'].str.cat(sep=' ')
                cmt1 = cmt1['review_text'].str.cat(sep=' ')
                wc0 = WordCloud(background_color='black', max_words=200, stopwords=stopwords_lst, width=1600,height=800, max_font_size=200)
                wc0.generate(cmt0)
                wc1 = WordCloud(background_color='black', max_words=200, stopwords=stopwords_lst, width=1600,height=800, max_font_size=200)
                wc1.generate(cmt1)
                fig, ax = plt.subplots(1,2,figsize=(20, 10))
                ax[0].imshow(wc0,interpolation='bilinear')
                ax[0].axis('off')
                ax[0].set_title('Tình cảm của bình luận là tiêu cực')
                ax[1].imshow(wc1,interpolation='bilinear')
                ax[1].axis('off')
                ax[1].set_title('Tình cảm của bình luận là tích cực')
                st.pyplot(fig)
                
                # show plot bar chart of sentiment
                st.subheader("Biểu đồ cột thể hiện số lượng bình luận theo nhóm sentiment: ")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax = sns.countplot(x='sentiment', data=df_after_predict)
                # add percentage in center of bar chart
                for p in ax.patches:
                    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(df_after_predict)),
                                (p.get_x()+0.4, p.get_height()+0.03), 
                                color='black', ha='center')
                st.pyplot(fig)
                
                # download file csv
                st.subheader("Tải tệp csv kết quả dự đoán: ")
                output = io.BytesIO()
                df_after_predict.to_csv(output, index=False)
                output.seek(0)
                st.download_button('Download', data=output, file_name='result_csv.csv', mime='text/csv')

    elif choice == "Tải tệp âm thanh" :
        st.write('Click link bên dưới để ghi âm bình luận')
        st.header("https://vocalremover.org/vi/voice-recorder")
        # Tạo một khu vực để tải lên tệp âm thanh
        audio_file = st.file_uploader("Tải lên tệp âm thanh", type=["wav", "mp3"])

        # Tạo một nút để phát âm
        if st.button("Phát âm file đã tải"):
            if audio_file is not None:
                # Đọc dữ liệu âm thanh
                audio_bytes = audio_file.read()
                # Phát âm thanh
                # st.audio(audio_bytes, format='audio/ogg')
                st.audio(audio_bytes)
            else:
                st.warning("Hãy tải lên một tệp âm thanh để phát âm")

        # Tạo một nút để bắt đầu chuyển đổi giọng nói thành văn bản
        if st.button("Chuyển đổi giọng nói thành bình luận và dự đoán"):

            if audio_file is not None:
                # Sử dụng thư viện SpeechRecognition để chuyển đổi tệp âm thanh thành văn bản
                r = sr.Recognizer()
                audio_data = sr.AudioFile(audio_file)
                with audio_data as source:
                    audio = r.record(source)
                    text = r.recognize_google(audio, language="vi-VN")
                
                # Hiển thị văn bản được chuyển đổi
                st.write('Câu bình luận của bạn :'+text)
                # Dự đoán
                text = text_transform(text)
                text = tfidf.transform([text])
                y_predict = model.predict(text)

                if y_predict[0] == 1:
                    sentiment = 'Tình cảm của bình luận là tích cực'
                    st.write(sentiment)
                else:
                    sentiment = 'Tình cảm của bình luận là tiêu cực'
                    st.write(sentiment)
                    # text_to_speech(sentiment)

            else:
                st.warning("Hãy tải lên một tệp âm thanh để chuyển đổi")

