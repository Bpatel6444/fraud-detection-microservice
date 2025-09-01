package com.Java.fraud_detection_api.Service;

import org.springframework.stereotype.Service;
import com.Java.fraud_detection_api.Dto.TransactionRequest;

@Service // This tells Spring this is a service component
public class FraudDetectionService {

    public String checkTransaction(Double amount, Integer hourOfDay) {
        /*
         * SIMPLE RULE-BASED FRAUD DETECTION
         * Rule 1: Very high amount transactions are suspicious.
         * Rule 2: Transactions between 1 AM and 5 AM are suspicious.
         * Rule 3: Combine the rules for a higher risk score.
         */

        boolean isHighAmount = amount > 100000.0; // Rule 1: Flag transactions over ₹1 lakh
        boolean isUnusualTime = (hourOfDay >= 1) && (hourOfDay <= 5); // Rule 2: Flag transactions between 1 AM - 5 AM

        // Rule 3: Decision Making
        if (isHighAmount && isUnusualTime) {
            return "RED ALERT: Highly likely to be fraudulent (High amount at unusual hour).";
        } else if (isHighAmount) {
            return "YELLOW ALERT: Potentially fraudulent due to high amount.";
        } else if (isUnusualTime) {
            return "YELLOW ALERT: Potentially fraudulent due to unusual time.";
        } else {
            return "GREEN: Transaction appears legitimate.";
        }
    }

    // Overload for DTO usage
    public String checkTransaction(TransactionRequest request) {
        return checkTransaction(request.getAmount(), request.getHourOfDay());
    }
}