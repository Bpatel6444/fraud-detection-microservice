package com.Java.fraud_detection_api.Service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import com.Java.fraud_detection_api.Dto.TransactionRequest;

import java.time.Instant;
import java.time.ZoneId;
import java.util.Map;

@Service
public class FraudDetectionService {

    @Autowired
    private MlClientService mlClientService;

    // Rule-based check (fallback if ML service is down)
    public String checkTransaction(Double amount, Instant timestamp, String transactionId, 
                                 String merchantId, String customerId, Map<String, Double> pcaFeatures) {
        int hour = timestamp.atZone(ZoneId.systemDefault()).getHour();
        boolean isHighAmount = amount != null && amount > 1000.0;
        boolean isUnusualTime = hour >= 1 && hour <= 5;
        
        // Check PCA features for anomalies (simplified)
        boolean hasAnomalousPca = false;
        if (pcaFeatures != null) {
            for (Double value : pcaFeatures.values()) {
                if (Math.abs(value) > 3.0) {  // Simple threshold for anomaly
                    hasAnomalousPca = true;
                    break;
                }
            }
        }

        if (isHighAmount && isUnusualTime && hasAnomalousPca) {
            return "{\"riskLevel\": \"VERY_HIGH\", \"message\": \"Multiple fraud indicators detected\"}";
        } else if (isHighAmount && isUnusualTime) {
            return "{\"riskLevel\": \"HIGH\", \"message\": \"High amount at unusual time\"}";
        } else if (isHighAmount && hasAnomalousPca) {
            return "{\"riskLevel\": \"HIGH\", \"message\": \"High amount with anomalous features\"}";
        } else if (isHighAmount) {
            return "{\"riskLevel\": \"MEDIUM\", \"message\": \"High amount transaction\"}";
        } else if (isUnusualTime) {
            return "{\"riskLevel\": \"LOW\", \"message\": \"Unusual time transaction\"}";
        } else if (hasAnomalousPca) {
            return "{\"riskLevel\": \"MEDIUM\", \"message\": \"Anomalous features detected\"}";
        } else {
            return "{\"riskLevel\": \"LOW\", \"message\": \"Transaction appears legitimate\"}";
        }
    }

    // ML-based check
    public String checkTransaction(TransactionRequest request) {
        // First try ML service
        String mlResult = mlClientService.checkTransactionWithML(
            request.getAmount(),
            request.getTimestamp().toString(),
            request.getTransactionId(),
            request.getMerchantId(),
            request.getCustomerId(),
            request.getPcaFeatures()
        );
        
        // Fallback to rule-based if ML service fails
        if (mlResult.contains("error") || mlResult.contains("unavailable")) {
            return checkTransaction(
                request.getAmount(),
                request.getTimestamp(),
                request.getTransactionId(),
                request.getMerchantId(),
                request.getCustomerId(),
                request.getPcaFeatures()
            );
        }
        
        return mlResult;
    }
}