package com.Java.fraud_detection_api.Service;

import org.springframework.stereotype.Service;

import com.Java.fraud_detection_api.Dto.TransactionRequest;

@Service // This tells Spring this is a service component
public class FraudDetectionService {

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

    public String checkTransaction(TransactionRequest request) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'checkTransaction'");
    }
}
