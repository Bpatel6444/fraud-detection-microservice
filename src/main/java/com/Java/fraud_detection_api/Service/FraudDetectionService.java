package com.Java.fraud_detection_api.Service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.Java.fraud_detection_api.Dto.TransactionRequest;

@Service
public class FraudDetectionService {

    @Autowired
    private MlClientService mlClientService;

    // Optional: Keep rule-based quick check
    public String checkTransaction(Double amount, Integer hourOfDay) {
        boolean isHighAmount = amount != null && amount > 100000.0;
        boolean isUnusualTime = hourOfDay != null && (hourOfDay >= 1 && hourOfDay <= 5);

        if (isHighAmount && isUnusualTime) {
            return "RED ALERT: Highly likely fraudulent (High amount at unusual hour).";
        } else if (isHighAmount) {
            return "YELLOW ALERT: Potential fraud due to high amount.";
        } else if (isUnusualTime) {
            return "YELLOW ALERT: Potential fraud due to unusual time.";
        } else {
            return "GREEN: Transaction appears legitimate.";
        }
    }

    // New method: call Python ML service
    public String checkTransaction(TransactionRequest request) {
        return mlClientService.checkTransactionWithML(request.getAmount(), request.getHourOfDay());
    }
}
