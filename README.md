# Financial Market Analysis Dashboard

This project provides an AI driven dashboard for analyzing financial markets and creating investment forecasts. It offers interactive visuals and automatic insights in Turkish and English.

## Project Features / Proje Özellikleri
- Market overview and trend analysis
- Stock analysis with technical indicators
- Portfolio forecasting
- News and sentiment analysis

## Prerequisites / Gereksinimler
1. Python 3.11 or higher installed
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment Configuration
1. Copy the example environment file and edit it:
```bash
cp .env.example .env
```
2. Open `.env` and fill in your API keys and database credentials:
   - `ALPHA_VANTAGE_API_KEY`
   - `FINNHUB_API_KEY`
   - `NEWS_API_KEY`
   - `DEEPSEEK_API_KEY`
   - `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`
   - other optional settings

## Running the App / Uygulamanın Çalıştırılması
Start the Streamlit dashboard:
```bash
streamlit run app.py
```
The app will read the variables from your `.env` file to connect to the database and external APIs.

## Contribution
Contributions are welcome! Please open an issue or submit a pull request for improvements.

## License
This project is distributed under the MIT License.

