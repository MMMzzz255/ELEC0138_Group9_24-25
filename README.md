# Cyber Attack Simulation  
A practical demonstration of DDoS and phishing attack methodologies for educational purposes.  

**⚠️ Disclaimer**:  
These tools are **strictly for authorized security testing, academic research, or defensive training**. Unauthorized use is illegal and unethical.  

---

## DDoS Attack Simulation  
A toolkit demonstrating SYN/UDP flood and HTTP slowloris attacks using `hping3` and `slowhttptest`.  

### Components:  
**Tool Setup**:  
- `hping3`: Generates high-volume SYN/UDP flood attacks.  
- `slowhttptest`: Executes slow HTTP requests to exhaust server resources.  
- Target server: Simple Python HTTP server for attack validation.  

**Attack Execution**:  
1. **SYN Flood**:  
   ```bash  
   sudo hping3 -S -p 8000 --flood <TARGET_IP>  # Replace <TARGET_IP>
  











# Cyber Threat Defense

A full-stack system to detect and defend against phishing websites and DDoS network attacks using machine learning techniques.

This repository includes:
DDOS Defense Model:
- Frontend: Flask web application
  - A Flask web application for DDoS detection
  - logging of auto-defense activities
- Machine Learning:
  - ML models comprising Random Forest and SVM for DDoS detection
  - Random Forest and SVM models for DDoS detection
  - Evaluation metrics and confusion matrix for model performance
- Testcase Generation:
  - Test cases for DDoS detection
  - A dataset of DDoS attack and normal traffic data
- Usage:
  - Instructions for running the Flask app
  - Instructions for using the trained models


Phishing Defense Model:
- Frontend: Flask web application
  - A Flask web application for Phishing detection
- Machine Learning:
  - ML models comprising Random Forest and SVM for Phishing detection
  - Random Forest and SVM models for Phishing detection
  - Evaluation metrics and confusion matrix for model performance
- Example Safe URL:
  - Example of a safe URL with detailed analysis
- Usage:
  - Instructions for running the Flask app
  - Instructions for using the trained models
---

## Project Layout

```
Defense/
├── ddos_defense/
│   ├── app/                # Flask web app for DDoS detection
│   ├── data/               # Raw and preprocessed DDoS datasets
│   ├── model/              # Trained models (SVM & RF)
│   ├── result/             # Confusion matrices & metrics
│   ├── defense_log/        # Auto-defense activity logs
│   └── testing/            # Test samples
├── phishing_defense/
│   ├── app/                # Flask web app for Phishing detection
│   ├── data/               # Phishing URL datasets
│   ├── model/              # Trained models (SVM & RF)
│   ├── result/             # Evaluation results
│   └── testing/            # Phishing test cases
```

- DDOS Defense Testing:
  - Use testing_ddos.py to create the test cases for DDoS detection. 
  The test cases are generated based on the dataset in the data folder.
