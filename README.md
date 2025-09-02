# Bitcoin Price Analysis and Prediction System

A comprehensive machine learning application for Bitcoin price analysis, prediction, and sentiment analysis using LSTM neural networks and Natural Language Processing.

![Bitcoin Analysis](Analysis_Project/immm.jpg)

## 🚀 Features

### 📈 Price Analysis & Prediction
- **Historical Data Analysis**: Comprehensive Bitcoin price data from 2010 to present
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands
- **3-Day Price Prediction**: LSTM neural network for short-term forecasting
- **Interactive Visualizations**: Real-time charts and graphs

### 💭 Sentiment Analysis
- **Real-time Twitter Analysis**: Live sentiment analysis of Bitcoin-related tweets
- **Machine Learning Classification**: Random Forest model for sentiment prediction
- **Text Preprocessing**: Advanced NLP pipeline with NLTK
- **Sentiment Trends**: Track public opinion impact on price movements

### 🖥️ Interactive Dashboard
- **Streamlit Web Interface**: User-friendly dashboard with multiple sections
- **Real-time Data**: Integration with CoinGecko API for live Bitcoin prices
- **Model Performance Metrics**: Detailed evaluation and accuracy reporting
- **Responsive Design**: Works on desktop and mobile devices

## 🛠️ Technologies Used

- **Machine Learning**: TensorFlow/Keras, Scikit-learn
- **Data Analysis**: Pandas, NumPy, TA-Lib
- **Visualization**: Matplotlib, Seaborn, mplfinance
- **Web Framework**: Streamlit
- **APIs**: CoinGecko API, Twitter API (Tweepy)
- **NLP**: NLTK, TF-IDF Vectorization
- **Data Storage**: CSV files, Pre-trained model files

## 📁 Project Structure

```
machine_learning/
└── Analysis_Project/
    ├── FINAL_IDS.py                    # Main Streamlit application
    ├── BTC.csv                         # Historical Bitcoin price data
    ├── tweets.csv                      # Twitter sentiment dataset
    ├── btc_lstm_model.h5              # Pre-trained LSTM model
    ├── random_forest_sentiment_model.pkl # Sentiment analysis model
    ├── tfidf_vectorizer.pkl           # Text vectorizer
    ├── requirements.txt               # Python dependencies
    ├── Bitcoin_proposal.pdf           # Project documentation
    └── immm.jpg                       # Bitcoin image asset
```

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Internet connection for live data fetching

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/usman9522/machine_learning.git
   cd machine_learning/Analysis_Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (automatically handled by the application)
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## 🎯 Usage

### Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run FINAL_IDS.py
   ```

2. **Access the dashboard**
   - Open your web browser
   - Navigate to `http://localhost:8501`

### Application Sections

#### 1. **Introduction**
- Project overview and Bitcoin background
- Visual introduction to the analysis

#### 2. **Data Overview**
- Dataset statistics and summary
- Column descriptions and data types
- Sample data preview

#### 3. **Exploratory Data Analysis (EDA)**
- Bitcoin price history visualization
- Technical indicator analysis
- Market trend identification

#### 4. **ML Model Performance**
- LSTM model evaluation metrics
- Actual vs Predicted price comparisons
- Model accuracy and error analysis

#### 5. **Sentiment Analysis**
- Real-time Twitter sentiment analysis
- Live tweet fetching and classification
- Sentiment trend visualization

#### 6. **3-Day Prediction**
- Short-term Bitcoin price forecasting
- Live data integration with CoinGecko API
- Prediction confidence intervals

#### 7. **Conclusion**
- Key insights and findings
- Future improvement suggestions

## 🤖 Model Information

### LSTM Price Prediction Model
- **Architecture**: 2-layer LSTM with Dense layers
- **Input Features**: Close price, SMA_7, EMA_12, EMA_26, RSI, MACD, Signal_Line
- **Window Size**: 30 days
- **Prediction Horizon**: 3 days
- **Training**: 80% train, 20% test split

### Sentiment Analysis Model
- **Algorithm**: Random Forest Classifier
- **Features**: TF-IDF vectorized tweet text
- **Preprocessing**: Text cleaning, stopword removal, punctuation handling
- **Classes**: Positive, Negative, Neutral sentiment

## 📊 Data Sources

- **Bitcoin Price Data**: Historical data from 2010-07-17 to present
- **Live Price Data**: CoinGecko API integration
- **Sentiment Data**: Twitter API for real-time tweet analysis
- **Technical Indicators**: Calculated using TA-Lib library

## 🔧 Configuration

### API Keys Required
- **Twitter API**: Update bearer token in the sentiment analysis section
- **CoinGecko API**: No authentication required (rate-limited)

### Model Parameters
- LSTM epochs: 20
- Batch size: 32
- Optimizer: Adam
- Loss function: Mean Squared Error

## 📈 Performance Metrics

The application provides comprehensive performance evaluation:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R-squared Score**
- **Prediction Accuracy Percentage**

## 🔮 Future Enhancements

- [ ] Extended prediction horizons (7, 14, 30 days)
- [ ] Additional technical indicators (Fibonacci, Ichimoku)
- [ ] Multi-asset support (Ethereum, other cryptocurrencies)
- [ ] Improved sentiment analysis with BERT/transformer models
- [ ] Portfolio optimization features
- [ ] Real-time alerts and notifications
- [ ] Mobile application development
- [ ] Integration with trading platforms

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 📧 Contact

For questions, suggestions, or collaborations:
- **Repository**: [machine_learning](https://github.com/usman9522/machine_learning)
- **Issues**: [GitHub Issues](https://github.com/usman9522/machine_learning/issues)

## 🙏 Acknowledgments

- CoinGecko for providing free cryptocurrency API
- Twitter for sentiment data access
- TensorFlow and Scikit-learn communities
- Streamlit for the excellent web framework
- TA-Lib for technical analysis indicators

---

**⚠️ Disclaimer**: This application is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Cryptocurrency investments carry significant risks, and past performance does not guarantee future results.