# 🔒 PII Detection and Redaction Agent

A sophisticated AI-powered agent built with LangGraph that automatically detects Personally Identifiable Information (PII) in CSV files and produces redacted versions while maintaining data utility.

## 🚀 Features

- **Intelligent PII Detection**: Combines regex pattern matching with LLM analysis for comprehensive PII identification
- **Smart Redaction Strategies**: Applies context-aware masking rules (partial email, phone masking, IP subnetting, etc.)
- **LangGraph Framework**: Built on LangGraph for robust, stateful agent workflows
- **Multiple PII Types**: Detects emails, phone numbers, IP addresses, dates of birth, SSNs, credit cards, and more
- **Configurable Rules**: Customizable masking strategies and detection patterns
- **Secure Processing**: Uses hashing for sensitive data previews when communicating with LLMs

## 🏗️ Architecture

The agent follows a sophisticated workflow using LangGraph's state management:

```
Load CSV → Regex Scan → LLM Classification → Consolidate → Mask & Save
```

### Core Components

- **State Management**: Centralized state object (`AgentState`) that flows through the graph
- **Detection Nodes**: Specialized functions for different analysis stages
- **Masking Engine**: Intelligent redaction strategies based on PII type
- **Output Generation**: Produces redacted CSV and detailed findings JSON

## 📋 Prerequisites

- Python 3.8+
- Google Gemini API key
- Required Python packages (see Installation)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd pii-detection-agent
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv myenv
   # On Windows
   myenv\Scripts\activate
   # On macOS/Linux
   source myenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -U langgraph langchain langchain-google-genai pandas pydantic python-dotenv
   ```

4. **Set up API key**
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY="your_google_api_key_here"
   ```

## 🚀 Usage

### Basic Usage

```bash
python main.py
```

The script will:
1. Create sample data in `./data/customers.csv`
2. Run the PII detection workflow
3. Generate output files in `./out/` directory

### Output Files

- **`masked.csv`**: Redacted version of the input CSV with PII data masked
- **`findings.json`**: Detailed analysis results in JSON format
- **`report.md`**: Human-readable summary report (optional)

## 🔧 Configuration

### PII Detection Patterns

The agent includes pre-configured regex patterns for common PII types:

```python
pii_regex = {
    "EMAIL": r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    "PHONE": r"(?:(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?|\d{3})[-.\s]?\d{3}[-.\s]?\d{4})",
    "SSN": r"\b\d{3}-?\d{2}-?\d{4}\b",
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
    "IP": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "DOB": r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
}
```

### Masking Strategies

Different PII types use specialized masking approaches:

- **EMAIL**: `john.doe@example.com` → `j***@***.com`
- **PHONE**: `+1-555-123-4567` → `***-***-4567`
- **IP**: `192.168.1.100` → `192.168.1.0`
- **DOB**: `1990-05-15` → `1990`
- **NAME**: `John Doe` → `Name Redacted`

## 🧠 How It Works

### 1. Data Loading
- Reads CSV file and caches dataframe
- Extracts column metadata and sample data

### 2. Regex Pattern Matching
- Scans sample rows using predefined patterns
- Identifies potential PII columns
- Collects examples for analysis

### 3. LLM Classification (Optional)
- Uses Google Gemini to enhance detection
- Analyzes column names and patterns
- Provides confidence scores and rationale

### 4. Consolidation
- Merges findings from multiple sources
- Resolves conflicts and duplicates
- Generates comprehensive report

### 5. Data Masking
- Applies appropriate masking strategies
- Preserves data structure and utility
- Generates redacted output files

## 🔒 Security Features

- **No Raw Data Exposure**: Only hashed previews sent to LLM APIs
- **Configurable Sampling**: Limits data exposure during analysis
- **Audit Trail**: Comprehensive logging of all operations
- **Secure Fallbacks**: Graceful degradation when LLM unavailable

## 📊 Supported PII Types

| PII Type | Detection Method | Masking Strategy |
|----------|------------------|------------------|
| **EMAIL** | Regex + LLM | Partial masking (first char + domain TLD) |
| **PHONE** | Regex + LLM | Last 4 digits visible |
| **IP ADDRESS** | Regex + LLM | Subnet masking (last octet = 0) |
| **DATE OF BIRTH** | Regex + LLM | Year only |
| **SSN** | Regex + LLM | Consistent hash token |
| **CREDIT CARD** | Regex + LLM | Consistent hash token |
| **NAME** | LLM inference | Complete redaction |
| **ADDRESS** | LLM inference | Complete redaction |

## 🚧 Limitations & Considerations

- **LLM Dependency**: Requires Google Gemini API for enhanced detection
- **Pattern Coverage**: Regex patterns may not catch all edge cases
- **Performance**: Large files processed in samples for efficiency
- **False Positives**: Conservative approach may flag non-PII columns

## 🔮 Future Enhancements

- [ ] **Conditional Routing**: Smart branching based on detection confidence
- [ ] **Parallel Processing**: Concurrent regex and LLM analysis
- [ ] **Human-in-the-Loop**: Approval workflows for high-confidence detections
- [ ] **Custom Pattern Learning**: Adaptive pattern recognition
- [ ] **Jurisdiction-Specific Rules**: GDPR, CCPA, HIPAA compliance
- [ ] **Real-time Monitoring**: Streaming data processing capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangGraph**: For the robust agent framework
- **Google Gemini**: For intelligent PII classification
- **Pandas**: For efficient data processing
- **Pydantic**: For data validation and serialization

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review the code examples

---

**⚠️ Disclaimer**: This tool is designed for educational and development purposes. Always test thoroughly in your specific environment and ensure compliance with relevant data protection regulations before using in production.
