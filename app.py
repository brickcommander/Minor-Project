import pandas as pd
import streamlit as st

import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import LSTMsentiment
import datetime

# st.sidebar.title("Sentiment Analyzer")
# selectedModel = st.sidebar.selectbox("Select the Model", ['LSTM', 'NaiveBayes', 'RandomForestClassifier'])
# if selectedModel == 'LSTM':
#     sentence = st.text_input('Write your sentence here:')
#     if sentence is not None:
#         st.write("Emotion:", "testing")


st.sidebar.title("Whatsapp Chat Analyzer")

from matplotlib.font_manager import FontProperties

prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')
plt.rcParams['font.family'] = prop.get_family()

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    temp = df.copy()
    # st.dataframe(df)

    #fetch unique users
    user_list = df['user'].unique().tolist()
    if "group_notification" in user_list:
        user_list.remove("group_notification")
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    date_ = st.sidebar.date_input("Select the day")
    st.sidebar.subheader("Time Limit")
    col1, col2 = st.sidebar.columns(2)
    starting_time = ""
    ending_time = ""
    with col1:
        starting_time = st.sidebar.time_input("Start")
    with col2:
        ending_time = st.sidebar.time_input("End")

    flag = int(0)
    if st.sidebar.button("Sentiment Analysis"):
        flag = int(1)
        y = date_.year
        m = date_.month
        d = date_.day
        df2 = df.copy()
        if selected_user != 'Overall':
            df2 = df2[df2['user'] == selected_user]
        df2 = df2[df2['year'] == y]
        df2 = df2[df2['month_num'] == m]
        header_stat = ""
        months = ["Unknown",
                  "January",
                  "Febuary",
                  "March",
                  "April",
                  "May",
                  "June",
                  "July",
                  "August",
                  "September",
                  "October",
                  "November",
                  "December"]
        month_name_ = months[m]
        sh = str(starting_time.hour)
        sm = str(starting_time.hour)
        eh = str(ending_time.hour)
        em = str(ending_time.minute)
        if len(sh) == 1:
            sh = "0"+sh
        if len(sm) == 1:
            sm = "0"+sm
        if len(eh) == 1:
            eh = "0" + eh
        if len(em) == 1:
            em = "0" + em
        starting_time_hour = starting_time.hour
        starting_time_minute = starting_time.minute
        ending_time_hour = ending_time.hour
        ending_time_minute = ending_time.minute
        Stime_p = "AM"
        Etime_p = "AM"
        if starting_time_hour > 12:
            Stime_p = "PM"
        if ending_time_hour > 12:
            Etime_p = "PM"

        if starting_time.hour >= ending_time.hour:
            header_stat = (f"Emotions from {sh}:{sm} {Stime_p} on {d} {month_name_} to {eh}:{em} {Etime_p} on {d} {month_name_}")
            df2 = df2[((df2['day'] == d) & (df2['hour'] >= starting_time.hour)) | ((df2['day'] == d+1) & (df2['hour'] <= ending_time.hour))]
        else:
            header_stat = (f"Emotions on {d} {month_name_} from {sh}:{sm} {Stime_p} to {eh}:{em} {Etime_p}")
            df2 = df2[df2['day'] == d]
            df2 = df2[(df2['hour'] >= starting_time.hour) & (df2['hour'] <= ending_time.hour)]

        # print(date_)
        # print(starting_time)
        # print(ending_time)
        # print(df2['message'].tolist())
        # print(type(df2['message'].tolist()))

        df = df2.copy()

        dff = pd.DataFrame()
        if df2.empty:
            st.title("No conversation took place in this time period.")
        else:
            output_ = LSTMsentiment.predictTheEmotion(df2['message'].tolist())
            sentimentS = ["Anger", "Fear", "Joy", "Love", "Sadness", "Surprise"]
            count_sentimentS = [0, 0, 0, 0, 0, 0]
            for senti in output_:
                for j in range(6):
                    if senti == sentimentS[j]:
                        count_sentimentS[j] += 1
            dff[0] = sentimentS
            dff[1] = count_sentimentS
            fig, ax = plt.subplots()
            st.title(header_stat)
            ax.pie(dff[1].head(), labels=dff[0].head(), autopct="%0.2f")
            ax.legend(dff[0].head())
            st.pyplot(fig)

    if (st.sidebar.button("Show Analysis") or flag==1) and df.empty==0:
        if flag == 1:
            flag = 0
        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4) # creating 4 columns on the page of app
        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Total Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # Monthly Timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily Timeline
        daily_timeline = helper.daily_timeline(selected_user, df)
        if daily_timeline.empty == 0:
            st.title("Daily Timeline")
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color="black")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Activity Map

        busy_day = helper.week_activity_map(selected_user, df)
        busy_month = helper.month_activity_map(selected_user, df)
        if busy_month.empty == 0:
            st.title("Activity Map")
            col1, col2 = st.columns(2)
            with col1:
                st.header("Most busy day")
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values)
                plt.xticks(rotation="vertical")
                st.pyplot(fig)
            with col2:
                st.header("Most busy month")
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color="orange")
                plt.xticks(rotation="vertical")
                st.pyplot(fig)

        user_heatmap = helper.activity_heatmap(selected_user, df)
        if user_heatmap.empty:
            pass
        else:
            st.title("Weekly Activity Map")
            fig, ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)

        #finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            x, new_df = helper.most_busy_users(df)
            if new_df.empty:
                pass
            else:
                st.title('Most Busy Users')
                fig, ax = plt.subplots()
                col1, col2 = st.columns(2)
                with col1:
                    ax.bar(x.index, x.values, color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    st.dataframe(new_df)

        # WordCloud
        df_wc = helper.create_wordcloud(selected_user, df)
        if df_wc is not None:
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user, df)
        if most_common_df.empty:
            pass
        else:
            fig, ax = plt.subplots()
            ax.barh(most_common_df[0], most_common_df[1])
            plt.xticks(rotation='vertical')
            st.title('Most Common Words')
            st.pyplot(fig)

        # st.dataframe(most_common_df)

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        # st.dataframe(emoji_df)

        if emoji_df.empty:
            pass
        else:
            st.title("Emoji Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(emoji_df)
            with col2:
                fig, ax = plt.subplots()
                ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
                st.pyplot(fig)

        df = temp.copy()
    elif flag == 1:
        flag = 0
    else:
        st.title("No conversation took place in this time period.")